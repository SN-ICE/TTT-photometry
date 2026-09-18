"""Photometric calibration: match field stars to a reference catalogue and fit
a zero point (optionally with a colour term)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u

from . import catalogs as cat
from . import detector
from .photometry import measure_apertures
from .utils import get_logger

log = get_logger("tttphot.calibration")


@dataclass
class ZeroPoint:
    band: str
    zp: float                       # instrumental -> catalogue, for flux per second
    zp_err: float
    color_term: float
    color_term_err: float
    color_pivot: float
    n_stars: int
    rms: float
    provider: str
    system: str
    color_name: str = ""
    stars: Optional[Table] = None
    relaxed: bool = False
    ok: bool = True
    message: str = ""
    meta: dict = field(default_factory=dict)

    def apply(self, inst_mag, color: Optional[float] = None) -> float:
        """Catalogue magnitude for an instrumental magnitude (per second)."""
        value = np.asarray(inst_mag, dtype=float) + self.zp
        if color is not None and np.isfinite(self.color_term) and np.isfinite(color):
            value = value + self.color_term * (color - self.color_pivot)
        return value


def _sigma_clipped_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray, fit_slope: bool,
                       nsigma: float = 3.0, maxiters: int = 5):
    """Weighted straight-line fit with iterative sigma clipping.

    Returns (intercept, slope, err_intercept, err_slope, rms, keep_mask).
    """
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    slope = 0.0
    slope_err = 0.0
    intercept = float("nan")
    intercept_err = float("nan")
    rms = float("nan")

    for _ in range(maxiters):
        if keep.sum() < (3 if fit_slope else 2):
            break
        design = np.column_stack([np.ones(keep.sum()), x[keep]]) if fit_slope \
            else np.ones((int(keep.sum()), 1))
        weights = w[keep]
        wd = design * weights[:, None]
        try:
            cov = np.linalg.inv(design.T @ wd)
        except np.linalg.LinAlgError:
            break
        beta = cov @ (wd.T @ y[keep])
        intercept = float(beta[0])
        slope = float(beta[1]) if fit_slope else 0.0
        errs = np.sqrt(np.diag(cov))
        intercept_err = float(errs[0])
        slope_err = float(errs[1]) if fit_slope else 0.0

        model = intercept + slope * x
        resid = y - model
        scatter = np.std(resid[keep]) if keep.sum() > 2 else np.nan
        rms = float(scatter)
        if not np.isfinite(scatter) or scatter == 0:
            break
        new_keep = keep & (np.abs(resid) < nsigma * scatter)
        if new_keep.sum() == keep.sum():
            break
        keep = new_keep

    return intercept, slope, intercept_err, slope_err, rms, keep


class _Relaxed:
    """A read-only view of the configuration with a few keys overridden."""

    def __init__(self, cfg, overrides: dict):
        self._cfg = cfg
        self._overrides = overrides

    def get(self, dotted, default=None):
        if dotted in self._overrides:
            return self._overrides[dotted]
        return self._cfg.get(dotted, default)

    def resolve_path(self, value):
        return self._cfg.resolve_path(value)


# Tried in order when the strict cuts leave too few calibration stars. Each
# step trades catalogue precision for numbers: a zero point averaged over ten
# stars of 0.1 mag beats one from three stars of 0.03, because the scatter
# averages down while the systematics mostly do not.
RELAXATION = [
    {},
    {"calibration.max_cat_err": 0.10, "calibration.min_snr": 15.0},
    {"calibration.max_cat_err": 0.20, "calibration.min_snr": 10.0,
     "calibration.match_radius_arcsec": 2.0},
    {"calibration.max_cat_err": 0.30, "calibration.min_snr": 8.0,
     "calibration.match_radius_arcsec": 2.5,
     "calibration.mag_range": [10.0, 20.5], "calibration.color_term": "none"},
]


def calibrate_image(image, sources: Table, fwhm_px: float, cfg, cache_root: Path,
                    apcor_mag: float = 0.0, refresh: bool = False) -> ZeroPoint:
    """Fit a zero point, relaxing the star-selection cuts if it comes up short.

    Two things are traded off. Loosening the cuts admits noisier catalogue
    entries but more of them; accepting fewer stars keeps the entries clean but
    leaves the mean poorly determined and outlier rejection blind. Numbers win,
    so every relaxation level is tried at the full ``min_stars`` before any of
    them is tried at the reduced floor -- and at the floor the colour term is
    dropped, since three stars cannot constrain a slope.
    """
    min_stars = int(cfg.get("calibration.min_stars", 5))
    floor = int(cfg.get("calibration.min_stars_floor", 3))
    relax = bool(cfg.get("calibration.relax_when_short", True))

    ladder = RELAXATION if relax else [{}]
    targets = [min_stars] if (not relax or floor >= min_stars) else [min_stars, floor]

    last = None
    for wanted in targets:
        for level, overrides in enumerate(ladder):
            settings = dict(overrides)
            settings["calibration.min_stars"] = wanted
            if wanted < min_stars:
                # A handful of stars cannot constrain a colour term as well as
                # a zero point, and trying makes both worse.
                settings["calibration.color_term"] = "none"
            view = _Relaxed(cfg, settings) if settings else cfg

            result = _calibrate_once(image, sources, fwhm_px, view, cache_root,
                                     apcor_mag=apcor_mag, refresh=refresh)
            last = result
            if not result.ok:
                continue

            result.meta["relaxation_level"] = level
            result.meta["min_stars_used"] = wanted
            result.relaxed = level > 0 or wanted < min_stars
            if result.relaxed:
                # Few stars means the sigma clipping had little to reject and
                # the mean rests on a short average; say so in the error.
                inflation = float(np.sqrt(max(min_stars, 1) / max(result.n_stars, 1)))
                if result.n_stars < min_stars:
                    result.zp_err = float(result.zp_err * max(inflation, 1.0))
                log.info("%s %s: zero point needed relaxed cuts "
                         "(level %d, %d stars%s)", image.meta.path.name, result.band,
                         level, result.n_stars,
                         ", no colour term" if wanted < min_stars else "")
            return result

    return last if last is not None else ZeroPoint(
        image.meta.band, float("nan"), float("nan"), 0.0, 0.0, 0.5, 0,
        float("nan"), "", image.meta.system, ok=False, message="no attempt succeeded")


def _drop_variables(table: Table, cfg, ra: float, dec: float, radius: float,
                    cache_root: Path, refresh: bool) -> tuple[Table, bool]:
    """Remove known variable stars from a calibration catalogue.

    Returns ``(table, screened)``. ``screened`` is False when the variability
    lookup could not be made at all -- a SIMBAD outage, say. That is not the
    same as "this field has no variables", and it used to be indistinguishable:
    both came back as an untouched table, so a network hiccup silently put the
    Mira back into a three-star zero point with nothing in the output to say so.
    """
    if not bool(cfg.get("calibration.screen_variables", True)) or len(table) == 0:
        return table, True
    known = cat.variable_stars(ra, dec, radius, cache_root, refresh=refresh)
    if known is None:
        return table, False
    if len(known) == 0:
        return table, True
    tol = float(cfg.get("calibration.variable_match_arcsec", 2.0))
    here = SkyCoord(np.asarray(table["ra"]) * u.deg, np.asarray(table["dec"]) * u.deg)
    there = SkyCoord(np.asarray(known["ra"]) * u.deg, np.asarray(known["dec"]) * u.deg)
    idx, sep, _ = here.match_to_catalog_sky(there)
    hit = sep.arcsec < tol
    if hit.any():
        log.info("dropped %d calibration star(s) matching a known variable "
                 "(%s)", int(hit.sum()),
                 ", ".join(sorted({str(known["otype"][i]) for i in idx[hit]})[:4]))
    return table[~hit], True


def _calibrate_once(image, sources: Table, fwhm_px: float, cfg, cache_root: Path,
                    apcor_mag: float = 0.0, refresh: bool = False) -> ZeroPoint:
    """One pass at the zero point with the cuts exactly as given."""
    band = image.meta.band
    providers = list(cfg.get("calibration.catalogs", ["gaia_synth"]))
    radius = detector.catalog_radius_arcmin(cfg, image=image)
    match_radius = float(cfg.get("calibration.match_radius_arcsec", 1.5))
    min_snr = float(cfg.get("calibration.min_snr", 20.0))
    min_stars = int(cfg.get("calibration.min_stars", 5))
    nsigma = float(cfg.get("calibration.sigma_clip", 3.0))
    mode = str(cfg.get("calibration.color_term", "fit")).lower()
    pivot = float(cfg.get(f"calibration.color_pivot.{band}", 0.5) or 0.5)
    zp_floor = float(cfg.get("calibration.zp_error_floor", 0.01))
    local_path = cfg.resolve_path(cfg.get("calibration.local_catalog"))

    fail = ZeroPoint(band, float("nan"), float("nan"), 0.0, 0.0, pivot, 0,
                     float("nan"), "", image.meta.system, ok=False)

    if image.wcs is None:
        fail.message = "no WCS"
        return fail
    if len(sources) == 0:
        fail.message = "no detected sources"
        return fail

    centre = image.center_coord()
    ra0, dec0 = float(centre.ra.deg), float(centre.dec.deg)

    # -- instrumental magnitudes of the field stars -------------------------
    ann = cfg.get("photometry.annulus_scale", [5.0, 8.0])
    positions = np.column_stack([np.asarray(sources["xcentroid"], dtype=float),
                                 np.asarray(sources["ycentroid"], dtype=float)])
    phot = measure_apertures(image, positions, float(cfg.get("photometry.aperture_scale", 1.2)) * fwhm_px,
                             cfg, annulus_px=(ann[0] * fwhm_px, ann[1] * fwhm_px))
    flux = np.asarray(phot["flux"], dtype=float)
    flux_err = np.asarray(phot["flux_err"], dtype=float)
    flags = np.asarray(phot["flags"], dtype=int)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(flux_err > 0, flux / flux_err, 0.0)
        inst = np.where(flux > 0, -2.5 * np.log10(flux / image.meta.exptime), np.nan) + apcor_mag
        inst_err = np.where(flux > 0, 1.0857362 * flux_err / flux, np.nan)

    usable = (flags == 0) & np.isfinite(inst) & (snr > min_snr)

    # A CCD stops being linear well below saturation, so an unsaturated star can
    # still be too bright to calibrate on. The target is never subject to this:
    # it is only ever applied to the field stars that set the zero point.
    nonlinear_frac = float(cfg.get("calibration.nonlinear_frac", 0.85) or 0)
    if nonlinear_frac > 0 and "peak" in phot.colnames:
        peak = np.asarray(phot["peak"], dtype=float)
        sky = float(image.qa.get("sky_adu", 0.0) or 0.0)
        ceiling = sky + nonlinear_frac * (image.meta.saturation - sky)
        too_bright = np.isfinite(peak) & (peak >= ceiling)
        if too_bright.any():
            log.debug("%s: %d calibration stars above the linear ceiling (%.0f ADU)",
                      image.meta.path.name, int((too_bright & usable).sum()), ceiling)
        usable &= ~too_bright

    if usable.sum() < min_stars:
        fail.message = f"only {int(usable.sum())} stars above SNR {min_snr:.0f}"
        return fail

    det = SkyCoord(np.asarray(sources["ra"])[usable] * u.deg,
                   np.asarray(sources["dec"])[usable] * u.deg)
    inst_u, inst_err_u = inst[usable], inst_err[usable]
    x_u, y_u = positions[usable, 0], positions[usable, 1]

    # -- try each catalogue in turn ----------------------------------------
    messages = []
    for provider in providers:
        bands = cat.PROVIDER_BANDS.get(provider)
        if bands is not None and band not in bands:
            messages.append(f"{provider}: no {band} band")
            continue
        try:
            table = cat.get_catalog(provider, ra0, dec0, radius, cache_root,
                                    refresh=refresh, local_path=local_path)
        except Exception as exc:
            messages.append(f"{provider}: {exc}")
            log.warning("%s catalogue unavailable: %s", provider, exc)
            continue

        clean = cat.clean_catalog(table, band, cfg)
        clean, screened = _drop_variables(clean, cfg, ra0, dec0, radius,
                                          cache_root, refresh)
        if len(clean) < min_stars:
            messages.append(f"{provider}: {len(clean)} usable {band} entries")
            continue

        ref = SkyCoord(np.asarray(clean["ra"], dtype=float) * u.deg,
                       np.asarray(clean["dec"], dtype=float) * u.deg)
        idx, sep, _ = det.match_to_catalog_sky(ref)
        matched = sep.arcsec < match_radius
        if matched.sum() < min_stars:
            messages.append(f"{provider}: {int(matched.sum())} matches within {match_radius}\"")
            continue

        rows = clean[idx[matched]]
        cat_mag = np.asarray(rows[band], dtype=float)
        ecol = f"e_{band}"
        cat_err = np.asarray(rows[ecol], dtype=float) if ecol in rows.colnames \
            else np.full(len(rows), 0.02)
        cat_err = np.where(np.isfinite(cat_err) & (cat_err > 0), cat_err, 0.02)

        m_inst = inst_u[matched]
        m_inst_err = inst_err_u[matched]

        color_pair = cat.color_columns(clean, band)
        color_name = ""
        if color_pair is not None:
            c1 = np.asarray(rows[color_pair[0]], dtype=float)
            c2 = np.asarray(rows[color_pair[1]], dtype=float)
            color = c1 - c2
            color_name = f"{color_pair[0]}-{color_pair[1]}"
        else:
            color = np.zeros(len(rows))

        # delta = m_cat - m_inst  should equal ZP + C * (colour - pivot)
        delta = cat_mag - m_inst
        sigma = np.sqrt(m_inst_err ** 2 + cat_err ** 2)
        weight = 1.0 / np.maximum(sigma, 1e-3) ** 2
        xvar = color - pivot

        fit_slope = (mode == "fit") and color_pair is not None and \
            np.nanstd(xvar[np.isfinite(xvar)]) > 0.05 and np.isfinite(xvar).sum() >= min_stars + 2
        if mode == "fixed":
            fixed_c = float(cfg.get(f"calibration.color_term_fixed.{band}", 0.0) or 0.0)
            delta = delta - fixed_c * np.where(np.isfinite(xvar), xvar, 0.0)

        xclean = np.where(np.isfinite(xvar), xvar, 0.0)
        zp, cterm, zp_err, cterm_err, rms, keep = _sigma_clipped_fit(
            xclean, delta, weight, fit_slope, nsigma=nsigma)

        # A colour term far outside the physically plausible range means the
        # fit is being driven by noise or by a short colour baseline, not by a
        # real bandpass mismatch. Fall back to a pure zero-point fit.
        cmax = float(cfg.get("calibration.color_term_max", 0.6))
        if fit_slope and np.isfinite(cterm) and abs(cterm) > cmax:
            log.warning("%s %s: colour term %.2f exceeds the plausible range (|C| < %.2f);"
                        " refitting with no colour term",
                        image.meta.path.name, band, cterm, cmax)
            zp, cterm, zp_err, cterm_err, rms, keep = _sigma_clipped_fit(
                xclean, delta, weight, False, nsigma=nsigma)

        if mode == "fixed":
            cterm, cterm_err = fixed_c, 0.0

        n_used = int(keep.sum())
        if n_used < min_stars or not np.isfinite(zp):
            messages.append(f"{provider}: fit failed with {n_used} stars")
            continue

        # The formal error only reflects the quoted photometric and catalogue
        # uncertainties. When the stars scatter more than that, the scatter is
        # the honest uncertainty, so take whichever is larger.
        scatter_err = rms / np.sqrt(max(n_used, 1)) if np.isfinite(rms) else 0.0
        zp_err = float(np.hypot(max(np.nan_to_num(zp_err), scatter_err), zp_floor))

        stars = Table({
            "x": x_u[matched], "y": y_u[matched],
            "ra": det.ra.deg[matched], "dec": det.dec.deg[matched],
            "mag_inst": m_inst, "mag_inst_err": m_inst_err,
            "mag_cat": cat_mag, "mag_cat_err": cat_err,
            "color": color, "delta": delta,
            "residual": delta - (zp + cterm * np.where(np.isfinite(xvar), xvar, 0.0)),
            "used": keep,
        })

        log.info("%s %s: ZP = %.3f +- %.3f (%s, %d/%d stars, rms %.3f, C = %.3f)",
                 image.meta.path.name, band, zp, zp_err, provider, n_used, len(rows), rms, cterm)

        return ZeroPoint(band=band, zp=float(zp), zp_err=zp_err,
                         color_term=float(cterm), color_term_err=float(cterm_err),
                         color_pivot=pivot, n_stars=n_used, rms=float(rms),
                         provider=provider, system=image.meta.system,
                         color_name=color_name, stars=stars, ok=True,
                         meta={"n_matched": int(matched.sum()), "n_catalog": len(clean),
                               "variables_screened": bool(screened)})

    fail.message = "; ".join(messages) or "no catalogue produced a usable fit"
    log.warning("%s %s: calibration failed (%s)", image.meta.path.name, band, fail.message)
    return fail
