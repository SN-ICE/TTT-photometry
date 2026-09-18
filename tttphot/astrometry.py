"""WCS refinement.

Frames arrive with a header astrometric solution (TTT: TAN-SIP from the
observatory pipeline), which is close but not necessarily tight against Gaia. Since every aperture, PSF fit
and difference-image stamp is placed through the WCS, it is worth re-fitting it
against the same catalogue used for the photometric calibration before anything
else happens.

Three fit modes, in increasing freedom:

``shift``   translation only (CRPIX), the safest with few matched stars
``linear``  full TAN: translation, rotation, scale and shear
``sip``     TAN plus SIP distortion polynomials of the configured degree
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points
import astropy.units as u

from .utils import get_logger

log = get_logger("tttphot.astrometry")


@dataclass
class AstrometryResult:
    wcs: Optional[WCS]
    mode: str
    n_stars: int
    rms_before: float          # arcsec
    rms_after: float           # arcsec
    shift_arcsec: float        # how far the solution moved on average
    ok: bool = True
    message: str = ""
    meta: dict = field(default_factory=dict)


def _match(detected: SkyCoord, reference: SkyCoord, radius_arcsec: float):
    """Nearest-neighbour match, keeping only mutually unique pairs."""
    if len(detected) == 0 or len(reference) == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0)
    idx, sep, _ = detected.match_to_catalog_sky(reference)
    keep = sep.arcsec < radius_arcsec

    # Drop reference sources claimed by more than one detection.
    order = np.argsort(sep.arcsec)
    seen, unique = set(), np.zeros(len(detected), dtype=bool)
    for k in order:
        if not keep[k]:
            continue
        if idx[k] in seen:
            continue
        seen.add(int(idx[k]))
        unique[k] = True

    det_idx = np.where(unique)[0]
    return det_idx, idx[det_idx], sep.arcsec[det_idx]


def refine_wcs(image, sources: Table, reference: Table, cfg) -> AstrometryResult:
    """Re-fit the WCS of *image* against a reference catalogue.

    *sources* must carry ``xcentroid``/``ycentroid``; *reference* must carry
    ``ra``/``dec`` in degrees.
    """
    mode = str(cfg.get("astrometry.mode", "linear")).lower()
    radius = float(cfg.get("astrometry.match_radius_arcsec", 3.0))
    final_radius = float(cfg.get("astrometry.final_match_radius_arcsec", 1.0))
    min_stars = int(cfg.get("astrometry.min_stars", 8))
    sip_degree = int(cfg.get("astrometry.sip_degree", 2))
    max_shift = float(cfg.get("astrometry.max_shift_arcsec", 30.0))
    max_rms = float(cfg.get("astrometry.max_rms_arcsec", 2.0))
    iterations = int(cfg.get("astrometry.iterations", 2))
    clip = float(cfg.get("astrometry.sigma_clip", 3.0))

    fail = AstrometryResult(None, mode, 0, float("nan"), float("nan"), 0.0, ok=False)

    if image.wcs is None:
        fail.message = "no starting WCS"
        return fail
    if mode in ("off", "none"):
        fail.message = "disabled"
        return fail
    if len(sources) < min_stars:
        fail.message = f"only {len(sources)} detections"
        return fail
    if reference is None or len(reference) < min_stars:
        fail.message = "reference catalogue too small"
        return fail

    x = np.asarray(sources["xcentroid"], dtype=float)
    y = np.asarray(sources["ycentroid"], dtype=float)
    ref = SkyCoord(np.asarray(reference["ra"], dtype=float) * u.deg,
                   np.asarray(reference["dec"], dtype=float) * u.deg)

    wcs = image.wcs.deepcopy()
    rms_before = float("nan")
    rms_after = float("nan")
    n_used = 0

    for iteration in range(max(iterations, 1)):
        tolerance = radius if iteration == 0 else final_radius
        detected = wcs.pixel_to_world(x, y)
        det_idx, ref_idx, sep = _match(detected, ref, tolerance)
        if len(det_idx) < min_stars:
            if iteration == 0:
                fail.message = f"only {len(det_idx)} matches within {tolerance}\""
                return fail
            break

        if iteration == 0:
            rms_before = float(np.sqrt(np.mean(sep ** 2)))

        # Reject the tail before fitting: a single mismatched pair can drag a
        # linear or SIP solution a long way.
        if len(sep) > min_stars + 2:
            cut = np.median(sep) + clip * (1.4826 * np.median(np.abs(sep - np.median(sep))))
            good = sep <= max(cut, 0.3)
            if good.sum() >= min_stars:
                det_idx, ref_idx, sep = det_idx[good], ref_idx[good], sep[good]

        try:
            if mode == "shift":
                fitted = _fit_shift(wcs, x[det_idx], y[det_idx], ref[ref_idx])
            else:
                fitted = fit_wcs_from_points(
                    (x[det_idx], y[det_idx]), ref[ref_idx],
                    proj_point=wcs.pixel_to_world(image.data.shape[1] / 2.0,
                                                  image.data.shape[0] / 2.0),
                    projection="TAN",
                    sip_degree=sip_degree if mode == "sip" else None,
                )
        except Exception as exc:
            fail.message = f"fit failed: {exc}"
            log.warning("%s: astrometry %s", image.meta.path.name, fail.message)
            return fail

        residual = fitted.pixel_to_world(x[det_idx], y[det_idx]).separation(ref[ref_idx]).arcsec
        rms_after = float(np.sqrt(np.mean(residual ** 2)))
        n_used = int(len(det_idx))
        wcs = fitted

    if not np.isfinite(rms_after):
        fail.message = "no solution"
        return fail

    # How far did the solution actually move the frame?
    ny, nx = image.data.shape
    corners_x = np.array([0.25, 0.75, 0.25, 0.75, 0.5]) * nx
    corners_y = np.array([0.25, 0.25, 0.75, 0.75, 0.5]) * ny
    shift = float(np.mean(image.wcs.pixel_to_world(corners_x, corners_y)
                          .separation(wcs.pixel_to_world(corners_x, corners_y)).arcsec))

    if shift > max_shift:
        fail.message = f"solution moved {shift:.1f}\" (limit {max_shift:.0f}\"); rejected"
        log.warning("%s: %s", image.meta.path.name, fail.message)
        return fail
    if rms_after > max_rms:
        fail.message = f"residual {rms_after:.2f}\" exceeds {max_rms:.2f}\"; rejected"
        log.warning("%s: %s", image.meta.path.name, fail.message)
        return fail
    if np.isfinite(rms_before) and rms_after > rms_before:
        fail.message = (f"refit made it worse ({rms_before:.3f}\" -> {rms_after:.3f}\"); "
                        "keeping the original WCS")
        log.info("%s: %s", image.meta.path.name, fail.message)
        return fail

    log.info("%s: WCS refined (%s, %d stars, rms %.3f\" -> %.3f\", moved %.2f\")",
             image.meta.path.name, mode, n_used, rms_before, rms_after, shift)
    return AstrometryResult(wcs=wcs, mode=mode, n_stars=n_used, rms_before=rms_before,
                            rms_after=rms_after, shift_arcsec=shift, ok=True)


def _fit_shift(wcs: WCS, x: np.ndarray, y: np.ndarray, ref: SkyCoord) -> WCS:
    """Translation-only correction, applied through CRPIX."""
    ref_x, ref_y = wcs.world_to_pixel(ref)
    dx = float(np.median(ref_x - x))
    dy = float(np.median(ref_y - y))
    fitted = wcs.deepcopy()
    fitted.wcs.crpix = [wcs.wcs.crpix[0] - dx, wcs.wcs.crpix[1] - dy]
    return fitted

