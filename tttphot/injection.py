"""Limiting magnitude by putting fake stars in and seeing which come back.

The analytic limit asks what flux would give N sigma against the sky RMS. It is
a statement about the noise, not about the pipeline: it assumes detection and
photometry behave ideally at the threshold, and it uses a single global sky RMS
where what matters is the sky *at the transient*, on top of its host.

Injection answers the question actually being asked -- at what magnitude does
this frame, measured by this code, recover a source half the time -- and it
does so where the supernova sits, on the same galaxy background, using the
frame's own empirical PSF. The two disagree most exactly where the analytic
number is least trustworthy: faint epochs on a bright host.

The recoveries also come with a free by-product. Each one gives a measured flux
and a quoted uncertainty against a known truth, so the spread of
(measured - true)/quoted is a direct test of whether the error bars are honest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import get_logger

log = get_logger("tttphot.injection")


@dataclass
class InjectionLimit:
    limit_mag: float = float("nan")
    limit_err: float = float("nan")
    n_sites: int = 0
    error_scale: float = float("nan")   # >1 means the quoted errors are optimistic
    n_calibration: int = 0
    message: str = ""

    def as_dict(self) -> dict:
        return {"inject_limit_mag": self.limit_mag,
                "inject_limit_err": self.limit_err,
                "inject_sites": self.n_sites,
                "error_scale": self.error_scale}


def _sites(image, x0: float, y0: float, fwhm_px: float, n: int,
           r_min: float, r_max: float, rng) -> list[tuple[float, float]]:
    """Positions on a Fibonacci spiral in an annulus around the target.

    An annulus, because the point is to sample the background the transient
    actually sits on -- its own host, at its own radius -- rather than empty sky
    somewhere else in the frame. Area-uniform, so the inner radii are not
    oversampled.
    """
    ny, nx = image.data.shape
    golden = np.pi * (3.0 - np.sqrt(5.0))
    margin = 3.0 * fwhm_px
    out = []
    for k in range(4 * n):
        frac = (k + 0.5) / (4 * n)
        r = np.sqrt(r_min ** 2 + frac * (r_max ** 2 - r_min ** 2))
        a = k * golden
        x = x0 + r * np.cos(a) + rng.uniform(-0.5, 0.5)
        y = y0 + r * np.sin(a) + rng.uniform(-0.5, 0.5)
        if not (margin < x < nx - margin and margin < y < ny - margin):
            continue
        half = int(np.ceil(2.0 * fwhm_px))
        sub = image.mask[int(y) - half:int(y) + half + 1,
                         int(x) - half:int(x) + half + 1]
        if sub.size == 0 or np.any(sub):
            continue
        out.append((float(x), float(y)))
        if len(out) >= n:
            break
    return out


def _render(epsf, shape, x: float, y: float, flux: float) -> np.ndarray:
    """The frame's own empirical PSF, unit flux scaled, at a sub-pixel position."""
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    try:
        return np.asarray(epsf.evaluate(xx, yy, flux, x, y), dtype=float)
    except Exception:
        return np.zeros(shape, dtype=float)


def measure_limit(image, epsf, fwhm_px: float, zp: float, cfg,
                  target_xy, apcor_mag: float = 0.0,
                  seed: int = 20260910) -> InjectionLimit:
    """Magnitude at which half the injected sources come back."""
    from . import photometry as ph

    out = InjectionLimit()
    if not bool(cfg.get("calibration.injection.enabled", True)):
        out.message = "disabled"
        return out
    if epsf is None or not np.isfinite(zp):
        out.message = "no PSF model or no zero point"
        return out

    rng = np.random.default_rng(seed)
    n_sites = int(cfg.get("calibration.injection.n_sites", 40))
    snr_limit = float(cfg.get("calibration.injection.snr", 3.0))
    target_frac = float(cfg.get("calibration.injection.completeness", 0.5))
    r_min = float(cfg.get("calibration.injection.min_radius_fwhm", 2.0)) * fwhm_px
    r_max = float(cfg.get("calibration.injection.max_radius_fwhm", 6.0)) * fwhm_px

    sites = _sites(image, float(target_xy[0]), float(target_xy[1]), fwhm_px,
                   n_sites, r_min, r_max, rng)
    out.n_sites = len(sites)
    if len(sites) < int(cfg.get("calibration.injection.min_sites", 12)):
        out.message = f"only {len(sites)} usable injection sites"
        return out

    radius = float(cfg.get("photometry.aperture_scale", 1.2)) * fwhm_px
    ann = cfg.get("photometry.annulus_scale", [5.0, 8.0])
    annulus = (ann[0] * fwhm_px, ann[1] * fwhm_px)
    exptime = max(float(image.meta.exptime), 1e-6)
    truth: list[tuple[float, float, float]] = []

    # The sky annulus starts 5 FWHM from the site, where the injected PSF has
    # essentially no flux, so it is measured once on the clean frame and reused
    # for every site at every magnitude. A single frame makes some 800
    # measurements here (40 sites x up to 21 magnitudes), and each one was
    # sigma-clipping its own annulus. Measuring it before the injection is also
    # the more honest sky: the real background does not contain the fake source.
    site_background = ph.annulus_background(image, sites, annulus)

    def recovered_fraction(mag: float) -> float:
        """Inject one source per site at *mag*, measure it back, count the wins."""
        flux = 10.0 ** (-0.4 * (mag - zp - apcor_mag)) * exptime
        if not np.isfinite(flux) or flux <= 0:
            return 0.0
        hits = 0
        half = int(np.ceil(3.0 * fwhm_px))
        for k, (x, y) in enumerate(sites):
            x0, x1 = int(x) - half, int(x) + half + 1
            y0, y1 = int(y) - half, int(y) + half + 1
            patch = image.data[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            model = _render(epsf, patch.shape, x - x0, y - y0, flux)
            saved = patch.copy()
            image.data[y0:y1, x0:x1] = patch + model
            here = tuple(values[k:k + 1] for values in site_background)
            try:
                tab = ph.measure_apertures(image, [(x, y)], radius, cfg,
                                           annulus_px=annulus, background=here)
                f = float(tab["flux"][0])
                e = float(tab["flux_err"][0])
            finally:
                image.data[y0:y1, x0:x1] = saved
            if not np.isfinite(f) or not np.isfinite(e) or e <= 0:
                continue
            if f > 0 and f / e >= snr_limit:
                hits += 1
            truth.append((flux, f, e))
        return hits / max(len(sites), 1)

    # Bracket downwards from something comfortably detected, then bisect.
    start = float(cfg.get("calibration.injection.start_mag", 0.0)) or None
    if start is None:
        sky = float(np.nanmedian(image.bkg_rms))
        npix = np.pi * radius ** 2
        start = -2.5 * np.log10(10.0 * sky * np.sqrt(npix) / exptime) + zp + apcor_mag
    step = float(cfg.get("calibration.injection.step_mag", 0.5))
    max_steps = int(cfg.get("calibration.injection.max_steps", 14))

    lo_m, lo_c = start, recovered_fraction(start)
    if lo_c < target_frac:
        for _ in range(max_steps):          # too faint a guess: walk brighter
            lo_m -= step
            lo_c = recovered_fraction(lo_m)
            if lo_c >= target_frac:
                break
        else:
            out.message = "never reached the target completeness"
            return out
    hi_m, hi_c = lo_m, lo_c
    for _ in range(max_steps):              # walk faint until it drops out
        hi_m += step
        hi_c = recovered_fraction(hi_m)
        if hi_c < target_frac:
            break
    else:
        out.message = "completeness never fell below the target"
        return out

    for _ in range(int(cfg.get("calibration.injection.bisect_steps", 6))):
        if hi_m - lo_m < 0.02:
            break
        mid = 0.5 * (lo_m + hi_m)
        c = recovered_fraction(mid)
        if c >= target_frac:
            lo_m, lo_c = mid, c
        else:
            hi_m, hi_c = mid, c

    span = hi_c - lo_c
    w = (target_frac - lo_c) / span if abs(span) > 1e-9 else 0.5
    out.limit_mag = float(lo_m + np.clip(w, 0.0, 1.0) * (hi_m - lo_m))
    # Binomial uncertainty on the completeness, mapped through the bracket.
    sigma_c = np.sqrt(max(target_frac * (1 - target_frac), 1e-6) / max(len(sites), 1))
    out.limit_err = float(abs(hi_m - lo_m) * sigma_c / max(abs(span), 1e-3))

    # Are the quoted errors honest? z should have unit spread if they are.
    if truth:
        arr = np.array(truth)
        z = (arr[:, 1] - arr[:, 0]) / arr[:, 2]
        z = z[np.isfinite(z)]
        if z.size >= int(cfg.get("calibration.injection.min_for_error_scale", 20)):
            out.n_calibration = int(z.size)
            out.error_scale = float(np.median(np.abs(z - np.median(z))) * 1.4826)
    return out
