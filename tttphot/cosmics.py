"""Cosmic ray rejection.

A cosmic ray is a sharp, few-pixel spike that no optical PSF could produce, and
it is indistinguishable from a faint source in a single frame by brightness
alone. Exposures of minutes on a chip of several megapixels carry hundreds to
thousands of hits (on TJO/MEIA3, where this was tuned, roughly three thousand
per 300 s frame).

Left alone they do three things, in increasing order of harm: they inflate the
background RMS, they get picked as PSF stars and corrupt the empirical PSF, and
occasionally one lands inside the target aperture and produces a spuriously
bright epoch that nothing downstream flags.

Detection is L.A.Cosmic (van Dokkum 2001) via astroscrappy, which separates
cosmic rays from stars by their Laplacian edge sharpness rather than their
brightness.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .utils import get_logger

log = get_logger("tttphot.cosmics")

_WARNED = False


def available() -> bool:
    try:
        import astroscrappy  # noqa: F401
        return True
    except ImportError:
        return False


def protect_mask(data: np.ndarray, saturation: float, nsigma: float = 5.0,
                 min_star_pixels: int = 4) -> np.ndarray:
    """Pixels the detector must not flag as cosmic rays.

    Saturated and non-finite pixels, plus the cores of *extended* bright
    objects. The size condition is the point: a bright pixel on its own is as
    likely to be a cosmic ray as a star core, so protecting every bright pixel
    would stop the detector finding anything. Only bright pixels belonging to a
    connected group of at least ``min_star_pixels`` are protected, which is a
    real source and not a spike.
    """
    from scipy.ndimage import label

    finite = np.isfinite(data)
    mask = ~finite
    if saturation and np.isfinite(saturation):
        mask |= data >= saturation

    values = data[finite]
    if values.size:
        med = float(np.median(values))
        sigma = max(float(np.median(np.abs(values - med))) * 1.4826, 1.0)
        bright = finite & (data > med + nsigma * sigma)
        if bright.any():
            labels, n = label(bright)
            if n:
                counts = np.bincount(labels.ravel())
                big = np.flatnonzero(counts >= int(min_star_pixels))
                big = big[big > 0]
                if big.size:
                    mask |= np.isin(labels, big)
    return mask


def remove(data: np.ndarray, cfg, gain: float, readnoise: float,
           saturation: float, fwhm_px: float,
           extra_mask: Optional[np.ndarray] = None):
    """Return ``(cleaned, cosmic_ray_mask, info)``.

    On any failure the frame comes back untouched with an empty mask: a missing
    cosmic ray costs one bad pixel, a broken frame costs the epoch.
    """
    global _WARNED
    info: dict = {"cosmic_rays": 0, "cosmic_ray_fraction": 0.0, "applied": False}
    if not bool(cfg.get("image.cosmic_rays.enabled", True)):
        return data, np.zeros(data.shape, bool), info

    try:
        import astroscrappy
    except ImportError:
        if not _WARNED:
            _WARNED = True
            log.warning("astroscrappy is not installed, so cosmic rays are left in "
                        "(pip install astroscrappy). Set image.cosmic_rays.enabled: "
                        "false to stop this message.")
        return data, np.zeros(data.shape, bool), info

    keep = protect_mask(data, saturation,
                        nsigma=float(cfg.get("image.cosmic_rays.protect_nsigma", 5.0)),
                        min_star_pixels=int(cfg.get("image.cosmic_rays.min_star_pixels", 4)))
    if extra_mask is not None:
        keep = keep | np.asarray(extra_mask, bool)

    finite = np.isfinite(data)
    work = np.where(finite, data, float(np.nanmedian(data[finite])) if finite.any() else 0.0)
    # detect_cosmics multiplies the frame by the gain and works in electrons
    # throughout, and its satlevel is compared against that electron-scale
    # array (astroscrappy.pyx: `cleanarr = indat * gain`, then
    # `satpixels = data >= satlevel`). Passing the ADU level made the threshold
    # a factor of `gain` too low -- 3.96 on TTT3 -- so every pixel above a
    # quarter of full well was treated as saturated and shielded from cleaning,
    # on exactly the stars the PSF and the aperture correction are built from.
    gain_e = float(gain) or 1.0
    try:
        crmask, cleaned = astroscrappy.detect_cosmics(
            work.astype(np.float32), inmask=keep,
            gain=gain_e, readnoise=float(readnoise) or 0.0,
            satlevel=float(saturation) * gain_e,
            sigclip=float(cfg.get("image.cosmic_rays.sigclip", 4.5)),
            sigfrac=float(cfg.get("image.cosmic_rays.sigfrac", 0.3)),
            objlim=float(cfg.get("image.cosmic_rays.objlim", 10.0)),
            niter=int(cfg.get("image.cosmic_rays.iterations", 4)),
            sepmed=True, cleantype="medmask", fsmode="median",
            psfmodel="gauss", psffwhm=max(float(fwhm_px), 2.0),
            verbose=False)
    except Exception as exc:
        log.warning("cosmic ray rejection failed (%s); keeping the frame as it is", exc)
        return data, np.zeros(data.shape, bool), info

    crmask = np.asarray(crmask, bool) & ~keep
    fraction = float(crmask.mean())
    limit = float(cfg.get("image.cosmic_rays.max_fraction", 0.02))
    if fraction > limit:
        # Flagging this much of a frame is not a cosmic ray rate, it is a
        # misfire -- undersampled stars being shredded, most likely.
        log.warning("cosmic ray rejection flagged %.2f%% of the frame (limit %.2f%%); "
                    "leaving the frame alone", 100 * fraction, 100 * limit)
        return data, np.zeros(data.shape, bool), info

    grow = int(cfg.get("image.cosmic_rays.dilate", 1))
    if grow > 0 and crmask.any():
        from scipy.ndimage import binary_dilation
        crmask = binary_dilation(crmask, iterations=grow)

    out = np.asarray(cleaned, dtype=float)
    out[~finite] = np.nan
    info.update(cosmic_rays=int(crmask.sum()), cosmic_ray_fraction=float(crmask.mean()),
                applied=True)
    return out, crmask, info


def contaminated(crmask: np.ndarray, x, y, radius_px: float) -> np.ndarray:
    """True for positions within *radius_px* of a flagged pixel.

    A source can sit beside a cosmic ray rather than on it and still have its
    flux and shape spoiled, so proximity is the test, not coincidence.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if crmask is None or not np.any(crmask) or radius_px <= 0:
        return np.zeros(x.shape, bool)
    from scipy.ndimage import distance_transform_edt

    distance = distance_transform_edt(~np.asarray(crmask, bool))
    ny, nx = crmask.shape
    xi = np.clip(np.round(x).astype(int), 0, nx - 1)
    yi = np.clip(np.round(np.atleast_1d(np.asarray(y, dtype=float))).astype(int), 0, ny - 1)
    return distance[yi, xi] < float(radius_px)
