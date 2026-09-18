"""Aperture and PSF photometry, aperture corrections and limiting magnitudes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from astropy.nddata import NDData
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.table import Table
from photutils.aperture import (ApertureStats, CircularAnnulus, CircularAperture,
                                aperture_photometry)
from photutils.psf import EPSFBuilder, PSFPhotometry, extract_stars

from .detection import neighbour_distance
from .utils import edge_margin_px, get_logger, odd

log = get_logger("tttphot.photometry")


@dataclass
class ApCorResult:
    apcor_mag: float          # magnitude to ADD to the aperture magnitude
    apcor_err: float
    n_stars: int
    radii_px: np.ndarray
    growth: np.ndarray        # median enclosed-flux fraction at each radius


# --------------------------------------------------------------------------
# aperture photometry
# --------------------------------------------------------------------------
def measure_apertures(image, positions: Sequence[tuple[float, float]], radius_px: float,
                      cfg, annulus_px: Optional[tuple[float, float]] = None,
                      local_bkg: bool = True) -> Table:
    """Circular-aperture photometry with a sigma-clipped annulus background.

    Returns a table with ``flux`` (background-subtracted counts in the
    aperture), ``flux_err``, ``local_bkg``, ``npix`` and quality flags.
    """
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    if image.error is None:
        image.build_error_map()

    aperture = CircularAperture(positions, r=float(radius_px))

    if local_bkg and annulus_px is not None:
        annulus = CircularAnnulus(positions, r_in=float(annulus_px[0]), r_out=float(annulus_px[1]))
        stats = ApertureStats(image.data, annulus, mask=image.mask,
                              sigma_clip=SigmaClip(sigma=3.0, maxiters=10))
        bkg_per_pix = np.atleast_1d(np.asarray(stats.median, dtype=float))
        bkg_std = np.atleast_1d(np.asarray(stats.std, dtype=float))
        n_bkg = np.atleast_1d(np.asarray(stats.sum_aper_area.value, dtype=float))
        data = image.data
    else:
        # Use the global 2D background model instead of a local annulus.
        bkg_per_pix = np.zeros(len(positions))
        bkg_std = np.full(len(positions), float(np.nanmedian(image.bkg_rms)))
        n_bkg = np.full(len(positions), np.inf)
        data = image.data_sub

    phot = aperture_photometry(data, aperture, error=image.error, mask=image.mask)
    area = np.atleast_1d(aperture.area_overlap(data, mask=image.mask))

    raw = np.asarray(phot["aperture_sum"], dtype=float)
    raw_err = np.asarray(phot["aperture_sum_err"], dtype=float)
    flux = raw - bkg_per_pix * area

    # Uncertainty on the background level propagated into the aperture.
    with np.errstate(divide="ignore", invalid="ignore"):
        bkg_term = np.where(np.isfinite(n_bkg) & (n_bkg > 0),
                            area * bkg_std / np.sqrt(np.maximum(n_bkg, 1.0)), 0.0)
    flux_err = np.sqrt(np.nan_to_num(raw_err) ** 2 + np.nan_to_num(bkg_term) ** 2)

    out = Table()
    out["x"] = positions[:, 0]
    out["y"] = positions[:, 1]
    out["flux"] = flux
    out["flux_err"] = flux_err
    out["local_bkg"] = bkg_per_pix
    out["bkg_rms"] = bkg_std
    out["npix"] = area
    out["aperture_radius_px"] = float(radius_px)

    # Flags: 1 = masked pixel inside the aperture, 2 = saturated, 4 = off-image
    flags = np.zeros(len(positions), dtype=int)
    peak = np.full(len(positions), np.nan)
    ny, nx = image.data.shape
    for k, (x, y) in enumerate(positions):
        x0, x1 = int(np.floor(x - radius_px)), int(np.ceil(x + radius_px)) + 1
        y0, y1 = int(np.floor(y - radius_px)), int(np.ceil(y + radius_px)) + 1
        if x0 < 0 or y0 < 0 or x1 > nx or y1 > ny:
            flags[k] |= 4
            continue
        sub = image.data[y0:y1, x0:x1]
        if np.any(image.mask[y0:y1, x0:x1]):
            flags[k] |= 1
        if np.any(sub >= image.meta.saturation):
            flags[k] |= 2
        finite = sub[np.isfinite(sub)]
        if finite.size:
            peak[k] = float(finite.max())
    out["flags"] = flags
    # Brightest pixel in the aperture. Saturation is a cliff, but a CCD stops
    # being linear well below it, so a calibration star can be unsaturated and
    # still too bright to trust. Reported rather than flagged, because the cut
    # belongs to whoever is calibrating -- the target itself must not be
    # dropped for being bright.
    out["peak"] = peak
    return out


def curve_of_growth(image, stars: Table, fwhm_px: float, cfg) -> ApCorResult:
    """Median enclosed-flux curve of bright isolated stars, and the aperture
    correction from the science aperture to the 'total' radius."""
    scales = np.asarray(cfg.get("photometry.curve_of_growth_scales",
                                [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]), dtype=float)
    ap_scale = float(cfg.get("photometry.aperture_scale", 1.2))
    total_scale = float(cfg.get("photometry.apcor_scale", 4.0))
    ann_scale = cfg.get("photometry.annulus_scale", [5.0, 8.0])
    min_stars = int(cfg.get("photometry.min_apcor_stars", 5))

    scales = np.unique(np.concatenate([scales, [ap_scale, total_scale]]))
    radii = scales * fwhm_px
    annulus = (ann_scale[0] * fwhm_px, ann_scale[1] * fwhm_px)

    if len(stars) == 0:
        return ApCorResult(0.0, np.nan, 0, radii, np.full(radii.size, np.nan))

    positions = np.column_stack([np.asarray(stars["xcentroid"], dtype=float),
                                 np.asarray(stars["ycentroid"], dtype=float)])

    fluxes = np.full((len(positions), radii.size), np.nan)
    flags = np.zeros(len(positions), dtype=int)
    for j, radius in enumerate(radii):
        tab = measure_apertures(image, positions, radius, cfg, annulus_px=annulus)
        fluxes[:, j] = np.asarray(tab["flux"], dtype=float)
        flags |= np.asarray(tab["flags"], dtype=int)

    i_total = int(np.argmin(np.abs(scales - total_scale)))
    i_ap = int(np.argmin(np.abs(scales - ap_scale)))

    total = fluxes[:, i_total]
    good = (flags == 0) & np.isfinite(total) & (total > 0)
    good &= np.all(np.isfinite(fluxes), axis=1)
    if good.sum() < min_stars:
        log.warning("%s: only %d stars for the aperture correction", image.meta.path.name, int(good.sum()))
    if good.sum() == 0:
        return ApCorResult(0.0, np.nan, 0, radii, np.full(radii.size, np.nan))

    growth = np.nanmedian(fluxes[good] / total[good, None], axis=0)

    ratio = fluxes[good, i_total] / fluxes[good, i_ap]
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    if ratio.size == 0:
        return ApCorResult(0.0, np.nan, 0, radii, growth)

    clipped = sigma_clipped_stats(-2.5 * np.log10(ratio), sigma=3.0, maxiters=5)
    apcor = float(clipped[1])                       # median, negative (brighter)
    apcor_err = float(clipped[2] / max(np.sqrt(ratio.size), 1.0))
    return ApCorResult(apcor, apcor_err, int(ratio.size), radii, growth)


# --------------------------------------------------------------------------
# PSF photometry
# --------------------------------------------------------------------------
def _model_fwhm(psf_model, half: int = 15) -> float:
    """FWHM of a PSF model sampled on the real (not oversampled) pixel grid."""
    from .detection import _fit_gaussian_fwhm
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    try:
        stamp = psf_model.evaluate(x, y, flux=1.0, x_0=0.0, y_0=0.0)
    except Exception:
        return float("nan")
    value = _fit_gaussian_fwhm(np.asarray(stamp, dtype=float))
    return float(value) if value is not None else float("nan")


def build_epsf(image, stars: Table, fwhm_px: float, cfg):
    """Build an empirical PSF from isolated stars. Returns (model, n_stars)."""
    min_stars = int(cfg.get("photometry.psf.min_stars", 5))
    oversampling = int(cfg.get("photometry.psf.oversampling", 2))
    stamp_scale = float(cfg.get("photometry.psf.stamp_scale", 5.0))
    maxiters = int(cfg.get("photometry.psf.maxiters", 6))
    # photutils' 'quartic'/'quadratic' smoothing kernels are derived for a
    # specific oversampling and badly distort the EPSF at other values -- on
    # TJO frames they collapse it to a near-delta function. None is safe.
    smoothing = cfg.get("photometry.psf.smoothing_kernel", None)
    if isinstance(smoothing, str) and smoothing.lower() in ("none", "null", ""):
        smoothing = None

    if len(stars) < min_stars:
        log.warning("%s: %d stars is too few to build an EPSF", image.meta.path.name, len(stars))
        return None, len(stars)

    size = odd(stamp_scale * fwhm_px)
    half = size // 2
    ny, nx = image.data.shape
    x = np.asarray(stars["xcentroid"], dtype=float)
    y = np.asarray(stars["ycentroid"], dtype=float)
    inside = (x > half + 1) & (x < nx - half - 1) & (y > half + 1) & (y < ny - half - 1)
    if inside.sum() < min_stars:
        return None, int(inside.sum())

    table = Table({"x": x[inside], "y": y[inside]})
    nddata = NDData(data=image.data_sub, mask=image.mask)
    try:
        cutouts = extract_stars(nddata, table, size=size)
        builder = EPSFBuilder(oversampling=oversampling, maxiters=maxiters,
                              progress_bar=False, smoothing_kernel=smoothing)
        epsf, _ = builder(cutouts)
        # Sanity check: an EPSF whose width disagrees wildly with the measured
        # seeing is a failed build, not a PSF.
        model_fwhm = _model_fwhm(epsf)
        if np.isfinite(model_fwhm) and not (0.5 * fwhm_px < model_fwhm < 2.0 * fwhm_px):
            log.warning("%s: EPSF width %.2f px disagrees with the seeing %.2f px;"
                        " discarding the PSF model", image.meta.path.name, model_fwhm, fwhm_px)
            return None, int(inside.sum())
    except Exception as exc:
        log.warning("%s: EPSF construction failed (%s)", image.meta.path.name, exc)
        return None, int(inside.sum())
    return epsf, int(inside.sum())


def psf_photometry_at(image, psf_model, positions: Sequence[tuple[float, float]],
                      fwhm_px: float, cfg, recentre: Optional[bool] = None) -> Table:
    """Fit the PSF at given positions (forced, or with a bounded recentring)."""
    if psf_model is None:
        return Table()
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    if image.error is None:
        image.build_error_map()

    if recentre is None:
        recentre = bool(cfg.get("photometry.psf.recentre", True))
    max_shift = float(cfg.get("photometry.psf.recentre_max_px", 2.0))
    fit_shape = odd(2.0 * float(cfg.get("photometry.psf.fit_shape_scale", 1.5)) * fwhm_px)

    model = psf_model.copy()
    xy_bounds = max_shift if recentre else 1e-3

    init = Table({"x": positions[:, 0], "y": positions[:, 1]})
    photometry = PSFPhotometry(
        model, fit_shape, finder=None, grouper=None,
        aperture_radius=max(3.0, 1.5 * fwhm_px), xy_bounds=xy_bounds, progress_bar=False,
    )
    try:
        result = photometry(image.data_sub, error=image.error, mask=image.mask, init_params=init)
    except Exception as exc:
        log.warning("%s: PSF fit failed (%s)", image.meta.path.name, exc)
        return Table()

    out = Table()
    out["x"] = np.asarray(result["x_fit"], dtype=float)
    out["y"] = np.asarray(result["y_fit"], dtype=float)
    out["flux"] = np.asarray(result["flux_fit"], dtype=float)
    err = result["flux_err"] if "flux_err" in result.colnames else np.full(len(result), np.nan)
    out["flux_err"] = np.asarray(err, dtype=float)
    if "qfit" in result.colnames:
        out["qfit"] = np.asarray(result["qfit"], dtype=float)
    if "cfit" in result.colnames:
        out["cfit"] = np.asarray(result["cfit"], dtype=float)
    if "flags" in result.colnames:
        out["psf_flags"] = np.asarray(result["flags"], dtype=int)
    out["dx"] = out["x"] - positions[:, 0]
    out["dy"] = out["y"] - positions[:, 1]
    return out


# --------------------------------------------------------------------------
# depth
# --------------------------------------------------------------------------
def limiting_magnitude(image, fwhm_px: float, zp: float, cfg,
                       apcor_mag: float = 0.0, sigma: Optional[float] = None) -> float:
    """N-sigma limiting magnitude for a point source in the science aperture."""
    if sigma is None:
        sigma = float(cfg.get("calibration.limiting_sigma", 3.0))
    radius = float(cfg.get("photometry.aperture_scale", 1.2)) * fwhm_px
    npix = np.pi * radius ** 2
    sky_rms = float(np.nanmedian(image.bkg_rms))
    noise = sky_rms * np.sqrt(npix)
    if not np.isfinite(noise) or noise <= 0:
        return float("nan")
    # The zero point is defined for flux per second, so the noise must be too.
    return float(-2.5 * np.log10(sigma * noise / max(image.meta.exptime, 1e-6))
                 + zp + apcor_mag)


def empirical_limit(image, fwhm_px: float, zp: float, cfg, apcor_mag: float = 0.0,
                    n_apertures: int = 300, sigma: Optional[float] = None,
                    rng_seed: int = 12345) -> float:
    """Limiting magnitude from the scatter of blank-sky apertures.

    More honest than the analytic estimate because it captures correlated noise
    left over from flat fielding and background subtraction.
    """
    if sigma is None:
        sigma = float(cfg.get("calibration.limiting_sigma", 3.0))
    radius = float(cfg.get("photometry.aperture_scale", 1.2)) * fwhm_px
    ny, nx = image.data.shape
    margin = int(np.ceil(radius)) + edge_margin_px(cfg, image.meta.pixel_scale)
    rng = np.random.default_rng(rng_seed)

    xs = rng.uniform(margin, nx - margin, size=n_apertures * 4)
    ys = rng.uniform(margin, ny - margin, size=n_apertures * 4)
    positions = np.column_stack([xs, ys])
    apertures = CircularAperture(positions, r=radius)
    phot = aperture_photometry(image.data_sub, apertures, mask=image.mask)
    sums = np.asarray(phot["aperture_sum"], dtype=float)

    # Drop apertures that landed on a source: keep the quiet 80 per cent.
    finite = np.isfinite(sums)
    sums = sums[finite]
    if sums.size < 20:
        return limiting_magnitude(image, fwhm_px, zp, cfg, apcor_mag, sigma)
    lo, hi = np.percentile(sums, [10, 90])
    quiet = sums[(sums >= lo) & (sums <= hi)]
    _, _, std = sigma_clipped_stats(quiet, sigma=3.0, maxiters=5)
    if not np.isfinite(std) or std <= 0:
        return limiting_magnitude(image, fwhm_px, zp, cfg, apcor_mag, sigma)
    return float(-2.5 * np.log10(sigma * std / max(image.meta.exptime, 1e-6))
                 + zp + apcor_mag)


def psf_to_aperture_correction(image, psf_model, stars: Table, fwhm_px: float, cfg,
                               apcor_mag: float = 0.0) -> float:
    """Offset that puts PSF magnitudes on the aperture-corrected system.

    The zero point is fitted to aperture photometry, so PSF fluxes -- which use
    no aperture correction of their own -- need this term before the same zero
    point can be applied to them. Returns a magnitude to ADD to the PSF
    instrumental magnitude.
    """
    if psf_model is None or stars is None or len(stars) == 0:
        return 0.0
    min_stars = int(cfg.get("photometry.min_apcor_stars", 5))
    ann = cfg.get("photometry.annulus_scale", [5.0, 8.0])
    ap_radius = float(cfg.get("photometry.aperture_scale", 1.2)) * fwhm_px

    positions = np.column_stack([np.asarray(stars["xcentroid"], dtype=float),
                                 np.asarray(stars["ycentroid"], dtype=float)])
    aper = measure_apertures(image, positions, ap_radius, cfg,
                             annulus_px=(ann[0] * fwhm_px, ann[1] * fwhm_px))
    fit = psf_photometry_at(image, psf_model, positions, fwhm_px, cfg)
    if len(fit) != len(aper):
        return 0.0

    aper_flux = np.asarray(aper["flux"], dtype=float)
    psf_flux = np.asarray(fit["flux"], dtype=float)
    good = ((np.asarray(aper["flags"], dtype=int) == 0) & (aper_flux > 0) & (psf_flux > 0)
            & np.isfinite(aper_flux) & np.isfinite(psf_flux))
    if good.sum() < min_stars:
        log.debug("%s: only %d stars for the PSF-to-aperture correction",
                  image.meta.path.name, int(good.sum()))
        if good.sum() == 0:
            return 0.0

    # aperture magnitude (aperture-corrected) minus PSF magnitude
    delta = (-2.5 * np.log10(aper_flux[good]) + apcor_mag) - (-2.5 * np.log10(psf_flux[good]))
    _, median, _ = sigma_clipped_stats(delta, sigma=3.0, maxiters=5)
    if not np.isfinite(median) or abs(median) > 1.0:
        return 0.0
    return float(median)


def blank_aperture_noise(image, radius_px: float, cfg, n_apertures: int = 400,
                         rng_seed: int = 20260907) -> float:
    """Scatter of source-free apertures of the given radius, in ADU.

    This is the honest sky noise for a source of that size: unlike the sum of
    the per-pixel variances it includes the pixel-to-pixel correlation left by
    resampling and PSF matching, which otherwise makes difference-image errors
    look several times better than they are.
    """
    ny, nx = image.data.shape
    margin = int(np.ceil(radius_px)) + 2
    if min(ny, nx) < 4 * margin:
        return float("nan")
    rng = np.random.default_rng(rng_seed)
    xs = rng.uniform(margin, nx - margin, size=n_apertures)
    ys = rng.uniform(margin, ny - margin, size=n_apertures)
    apertures = CircularAperture(np.column_stack([xs, ys]), r=float(radius_px))
    phot = aperture_photometry(image.data_sub, apertures, mask=image.mask)
    sums = np.asarray(phot["aperture_sum"], dtype=float)
    sums = sums[np.isfinite(sums)]
    if sums.size < 20:
        return float("nan")
    lo, hi = np.percentile(sums, [5, 95])
    quiet = sums[(sums >= lo) & (sums <= hi)]
    _, _, std = sigma_clipped_stats(quiet, sigma=3.0, maxiters=5)
    return float(std) if np.isfinite(std) and std > 0 else float("nan")


def combine_errors(flux: float, propagated_err: float, sky_sigma: float, gain: float) -> float:
    """Total flux error: empirical sky term plus the source's own shot noise."""
    if not np.isfinite(sky_sigma) or sky_sigma <= 0:
        return float(propagated_err)
    source = np.sqrt(max(float(flux), 0.0) / max(gain, 1e-6))
    empirical = float(np.hypot(sky_sigma, source))
    if not np.isfinite(propagated_err) or propagated_err <= 0:
        return empirical
    return float(max(empirical, propagated_err))
