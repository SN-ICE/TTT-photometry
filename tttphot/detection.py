"""Source detection, seeing measurement and selection of well-behaved stars."""

from __future__ import annotations

from typing import Optional

import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.modeling import fitting, models
from astropy.table import Table
from photutils.detection import DAOStarFinder

from .utils import edge_margin_px, get_logger

log = get_logger("tttphot.detection")


def detect_sources(image, cfg, fwhm_px: Optional[float] = None,
                   threshold_sigma: Optional[float] = None) -> Table:
    """Run DAOStarFinder on the background-subtracted frame.

    Returns a table with at least ``xcentroid``, ``ycentroid``, ``flux``,
    ``peak``, plus ``ra``/``dec`` when a WCS is present.
    """
    if image.bkg_map is None:
        image.estimate_background(cfg)

    if fwhm_px is None:
        fwhm_px = float(cfg.get("detection.fwhm_init_arcsec", 2.0)) / image.meta.pixel_scale
    if threshold_sigma is None:
        threshold_sigma = float(cfg.get("detection.threshold_sigma", 5.0))

    threshold = threshold_sigma * float(np.nanmedian(image.bkg_rms))
    finder = DAOStarFinder(
        fwhm=float(fwhm_px),
        threshold=threshold,
        sharplo=float(cfg.get("detection.sharplo", 0.2)),
        sharphi=float(cfg.get("detection.sharphi", 1.0)),
        roundlo=float(cfg.get("detection.roundlo", -0.75)),
        roundhi=float(cfg.get("detection.roundhi", 0.75)),
        exclude_border=True,
    )
    sources = finder(image.data_sub, mask=image.mask)
    if sources is None or len(sources) == 0:
        log.warning("%s: no sources detected at %.1f sigma", image.meta.path.name, threshold_sigma)
        return Table()

    sources = sources[np.argsort(-np.asarray(sources["flux"]))]
    sources["id"] = np.arange(len(sources))

    if image.wcs is not None:
        sky = image.wcs.pixel_to_world(sources["xcentroid"], sources["ycentroid"])
        sources["ra"] = sky.ra.deg
        sources["dec"] = sky.dec.deg

    sources.meta["threshold_sigma"] = threshold_sigma
    sources.meta["fwhm_px_used"] = float(fwhm_px)
    log.debug("%s: %d sources at %.1f sigma", image.meta.path.name, len(sources), threshold_sigma)
    return sources


def neighbour_distance(sources: Table) -> np.ndarray:
    """Distance (px) from each source to its nearest neighbour."""
    if len(sources) < 2:
        return np.full(len(sources), np.inf)
    xy = np.column_stack([np.asarray(sources["xcentroid"]), np.asarray(sources["ycentroid"])])
    try:
        from scipy.spatial import cKDTree
        dist, _ = cKDTree(xy).query(xy, k=2)
        return dist[:, 1]
    except Exception:  # pragma: no cover
        d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
        np.fill_diagonal(d, np.inf)
        return d.min(axis=1)


def select_psf_stars(image, sources: Table, cfg, fwhm_px: float,
                     max_stars: int = 40, min_snr: float = 20.0) -> Table:
    """Bright, unsaturated, isolated, on-chip stars suitable for PSF work."""
    if len(sources) == 0:
        return sources

    ny, nx = image.data.shape
    x = np.asarray(sources["xcentroid"], dtype=float)
    y = np.asarray(sources["ycentroid"], dtype=float)
    peak = np.asarray(sources["peak"], dtype=float)

    isolation = float(cfg.get("photometry.isolation_scale", 3.0)) * fwhm_px
    margin = max(edge_margin_px(cfg, image.meta.pixel_scale),
                 int(np.ceil(float(cfg.get("photometry.psf.stamp_scale", 5.0)) * fwhm_px)))

    sky_rms = float(np.nanmedian(image.bkg_rms))
    approx_snr = peak / max(sky_rms, 1e-6)

    keep = (
        (x > margin) & (x < nx - margin) & (y > margin) & (y < ny - margin)
        & (peak < 0.85 * (image.meta.saturation - image.qa.get("sky_adu", 0.0)))
        & (approx_snr > min_snr)
        & (neighbour_distance(sources) > isolation)
    )
    crmask = getattr(image, "crmask", None)
    if crmask is not None and np.any(crmask):
        from . import cosmics
        near = cosmics.contaminated(
            crmask, x, y,
            float(cfg.get("image.cosmic_rays.reject_within_fwhm", 1.0)) * fwhm_px)
        if near.any():
            log.debug("%s: %d PSF candidates dropped next to a cosmic ray",
                      image.meta.path.name, int((near & keep).sum()))
        keep &= ~near

    stars = sources[keep]
    if len(stars) == 0:
        log.warning("%s: no isolated PSF stars survived the cuts", image.meta.path.name)
        return stars

    stars = stars[np.argsort(-np.asarray(stars["flux"]))]
    return stars[:max_stars]


def _fit_gaussian_fwhm(stamp: np.ndarray) -> Optional[float]:
    """Fit a circular 2D Gaussian to a background-subtracted stamp."""
    ny, nx = stamp.shape
    if not np.all(np.isfinite(stamp)) or np.nanmax(stamp) <= 0:
        return None
    y, x = np.mgrid[0:ny, 0:nx]
    model = models.Gaussian2D(
        amplitude=float(np.nanmax(stamp)), x_mean=nx / 2.0, y_mean=ny / 2.0,
        x_stddev=max(nx / 6.0, 1.0), y_stddev=max(ny / 6.0, 1.0),
    )
    model.theta.fixed = True
    fitter = fitting.LevMarLSQFitter()
    try:
        fit = fitter(model, x, y, stamp, maxiter=200)
    except Exception:
        return None
    sx, sy = abs(float(fit.x_stddev.value)), abs(float(fit.y_stddev.value))
    if not np.isfinite(sx) or not np.isfinite(sy) or sx <= 0 or sy <= 0:
        return None
    # Guard against runaway fits on cosmic rays or blends.
    if abs(float(fit.x_mean.value) - nx / 2.0) > nx / 4.0:
        return None
    if abs(float(fit.y_mean.value) - ny / 2.0) > ny / 4.0:
        return None
    return float(2.3548200 * np.sqrt(sx * sy))


def filter_by_profile(image, stars: Table, fwhm_px: float, cfg,
                      tolerance: Optional[float] = None) -> Table:
    """Keep only sources whose own fitted width matches the frame's seeing.

    Galaxies, blends and cosmic rays all survive the brightness and isolation
    cuts but ruin an empirical PSF, so they are removed here by measuring each
    candidate individually.
    """
    if len(stars) == 0:
        return stars
    if tolerance is None:
        tolerance = float(cfg.get("photometry.psf.fwhm_tolerance", 0.25))

    data = image.data_sub
    half = int(np.ceil(3.0 * fwhm_px))
    widths = np.full(len(stars), np.nan)
    for k, row in enumerate(stars):
        xc, yc = float(row["xcentroid"]), float(row["ycentroid"])
        x0, x1 = int(round(xc)) - half, int(round(xc)) + half + 1
        y0, y1 = int(round(yc)) - half, int(round(yc)) + half + 1
        if x0 < 0 or y0 < 0 or x1 > data.shape[1] or y1 > data.shape[0]:
            continue
        if np.any(image.mask[y0:y1, x0:x1]):
            continue
        value = _fit_gaussian_fwhm(data[y0:y1, x0:x1])
        if value is not None:
            widths[k] = value

    keep = np.isfinite(widths) & (np.abs(widths - fwhm_px) <= tolerance * fwhm_px)
    out = stars[keep]
    out["fwhm_px"] = widths[keep]
    if len(out) < len(stars):
        log.debug("%s: %d/%d candidates are point-like", image.meta.path.name,
                  len(out), len(stars))
    return out


def measure_fwhm(image, cfg, sources: Optional[Table] = None) -> float:
    """Median stellar FWHM in pixels, iterating once on the detection FWHM.

    Sets ``image.fwhm_px`` / ``image.fwhm_arcsec`` and returns the FWHM in px.
    """
    scale = image.meta.pixel_scale
    fwhm_guess = float(cfg.get("detection.fwhm_init_arcsec", 2.0)) / scale
    fwhm_min = float(cfg.get("detection.fwhm_min_arcsec", 0.8)) / scale
    fwhm_max = float(cfg.get("detection.fwhm_max_arcsec", 8.0)) / scale
    min_stars = int(cfg.get("detection.min_stars_for_fwhm", 6))
    max_stars = int(cfg.get("detection.max_stars_for_fwhm", 40))

    if sources is None:
        sources = detect_sources(image, cfg, fwhm_px=fwhm_guess)

    # Iterate detection and fitting, but do not simply keep the last answer.
    # Re-detecting with a *smaller* kernel can cut the sample to a handful of
    # sources dominated by blends and galaxies: on a 1.1" TTT3 frame the first
    # pass gave 3.81 px from 27 stars (10-90 per cent range 3.73-3.91) and the
    # second 5.29 px from 10 sources ranging 3.7-13.4. Keeping that second
    # value made every real star look 28 per cent too sharp to the profile
    # test, which then rejected all but two of them, leaving no empirical PSF
    # and a one-star aperture correction off by 0.4 mag. So each pass uses a
    # clipped median, and the pass with the most fitted stars wins.
    fwhm = fwhm_guess
    best = None                                   # (n_values, fwhm, scatter)
    for iteration in range(3):
        stars = select_psf_stars(image, sources, cfg, fwhm, max_stars=max_stars, min_snr=10.0)
        half = int(np.ceil(3.0 * fwhm))
        values = []
        data = image.data_sub
        for row in stars:
            xc, yc = float(row["xcentroid"]), float(row["ycentroid"])
            x0, x1 = int(round(xc)) - half, int(round(xc)) + half + 1
            y0, y1 = int(round(yc)) - half, int(round(yc)) + half + 1
            if x0 < 0 or y0 < 0 or x1 > data.shape[1] or y1 > data.shape[0]:
                continue
            if np.any(image.mask[y0:y1, x0:x1]):
                continue
            value = _fit_gaussian_fwhm(data[y0:y1, x0:x1])
            if value is not None and fwhm_min <= value <= fwhm_max:
                values.append(value)

        if not values:
            if best is None:
                log.warning("%s: FWHM measurement failed; keeping %.2f px",
                            image.meta.path.name, fwhm)
            break
        if len(values) >= 3:
            _, new_fwhm, scatter = sigma_clipped_stats(np.asarray(values), sigma=2.5, maxiters=5)
        else:
            new_fwhm, scatter = float(np.median(values)), float(np.std(values))
        new_fwhm = float(new_fwhm)
        if best is None or len(values) > best[0]:
            best = (len(values), new_fwhm, float(scatter))

        converged = abs(new_fwhm - fwhm) / max(fwhm, 1e-6) < 0.05
        fwhm = new_fwhm
        if converged:
            break
        sources = detect_sources(image, cfg, fwhm_px=fwhm)

    if best is not None:
        if best[0] < min_stars:
            log.warning("%s: FWHM from only %d stars", image.meta.path.name, best[0])
        fwhm = best[1]
        image.qa["fwhm_nstars"] = best[0]
        image.qa["fwhm_scatter_px"] = best[2]

    fwhm = float(np.clip(fwhm, fwhm_min, fwhm_max))
    image.fwhm_px = fwhm
    image.fwhm_arcsec = fwhm * scale
    image.qa["fwhm_px"] = fwhm
    image.qa["fwhm_arcsec"] = image.fwhm_arcsec
    return fwhm
