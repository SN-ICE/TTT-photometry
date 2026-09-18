"""Difference imaging.

PSF matching follows Alard & Lupton (1998): the convolution kernel that turns
the reference into the science frame is expanded on a basis of Gaussians
modulated by polynomials, so that the matching problem becomes linear and can
be solved exactly by least squares. A low-order polynomial in (x, y) absorbs
the differential background.

    science  ~  sum_n  c_n * (reference * B_n)  +  background(x, y)

Nothing here needs an external binary: it is numpy + scipy.fft only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
import astropy.units as u
from scipy.signal import fftconvolve

from . import detector

from .utils import get_logger

log = get_logger("tttphot.subtraction")


@dataclass
class DifferenceResult:
    difference: np.ndarray            # science - convolved reference, science units
    variance: np.ndarray              # per-pixel variance of the difference
    wcs: WCS
    science_stamp: np.ndarray
    # Whichever frame was convolved, resampled onto the science grid: the
    # matched reference, or -- in convolve_science mode -- the science frame
    # degraded to the reference PSF. Label it by `mode`, not by its name.
    reference_stamp: np.ndarray
    kernel: np.ndarray                # kernel at the centre of the stamp
    kernel_sum: float                 # photometric scaling reference -> science
    flux_scale: float                 # divide difference fluxes by this
    mode: str                         # 'convolve_reference' | 'convolve_science'
    n_stamps: int
    chi2: float                       # reduced chi2 of the difference vs its noise model
    rms: float
    ok: bool = True
    message: str = ""
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# kernel basis
# --------------------------------------------------------------------------
def gaussian_basis(half_size: int, sigmas: Sequence[float],
                   degrees: Sequence[int]) -> list[np.ndarray]:
    """Alard & Lupton basis: Gaussians times 2D polynomials, mean-subtracted
    for every term except the first so the basis is close to orthogonal."""
    y, x = np.mgrid[-half_size:half_size + 1, -half_size:half_size + 1].astype(float)
    basis: list[np.ndarray] = []
    for sigma, degree in zip(sigmas, degrees):
        gauss = np.exp(-(x ** 2 + y ** 2) / (2.0 * float(sigma) ** 2))
        gauss /= gauss.sum()
        for total in range(int(degree) + 1):
            for i in range(total + 1):
                j = total - i
                term = gauss * (x ** i) * (y ** j)
                if basis:
                    term = term - term.sum() * basis[0]   # keep sum(K) tied to the first term
                basis.append(term)
    return basis


def _polynomial_terms(shape, order: int) -> list[np.ndarray]:
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    xx = (xx - nx / 2.0) / max(nx / 2.0, 1.0)
    yy = (yy - ny / 2.0) / max(ny / 2.0, 1.0)
    terms = []
    for total in range(int(order) + 1):
        for i in range(total + 1):
            terms.append((xx ** i) * (yy ** (total - i)))
    return terms


# --------------------------------------------------------------------------
# At most this many candidates are examined per threshold. The list is sorted
# brightest first and the good ones are taken from the top, so the tail is only
# ever reached on a field where almost everything is being rejected.
_MAX_CANDIDATES = 600


def _accept_stamps(found, science: np.ndarray, mask: np.ndarray, n_stamps: int,
                   half: int, peak_limit: float, avoid_xy=None,
                   avoid_radius: float = 0.0) -> list[tuple[int, int]]:
    """The cuts a candidate has to survive to become a kernel stamp."""
    found = found[np.argsort(-np.asarray(found["flux"]))][:_MAX_CANDIDATES]
    ny, nx = science.shape
    picks: list[tuple[int, int]] = []
    for row in found:
        x, y = int(round(float(row["xcentroid"]))), int(round(float(row["ycentroid"])))
        if x - half < 0 or y - half < 0 or x + half + 1 > nx or y + half + 1 > ny:
            continue
        if np.any(mask[y - half:y + half + 1, x - half:x + half + 1]):
            continue
        if float(row["peak"]) > 0.9 * peak_limit:
            continue
        if avoid_xy is not None and np.hypot(float(row["xcentroid"]) - avoid_xy[0],
                                             float(row["ycentroid"]) - avoid_xy[1]) < avoid_radius:
            continue                        # the transient itself, or a knot right on it
        if any(abs(x - px) < half and abs(y - py) < half for px, py in picks):
            continue
        if np.nanmax(science[y - half:y + half + 1, x - half:x + half + 1]) <= 0:
            continue
        picks.append((x, y))
        if len(picks) >= n_stamps:
            break
    return picks


def _select_stamps(science: np.ndarray, mask: np.ndarray, fwhm_px: float,
                   n_stamps: int, half: int, avoid_xy=None,
                   avoid_radius: float = 0.0) -> list[tuple[int, int]]:
    """Positions of bright, isolated, unsaturated sources to fit the kernel on.

    The detection threshold is lowered until there are enough stamps that
    *survive the cuts*, not until there are enough candidates. Counting
    candidates was measurably wrong: the ladder stopped as soon as a rung
    returned twelve of them, whatever ``n_stamps`` asked for, and then four to
    eight of those twelve were thrown away for having a masked pixel -- a
    saturation halo, a cosmic ray -- inside the stamp box. On the sparse
    SN 2026abwp field 30 sigma gave 15-17 candidates and 7-10 stamps against the
    40 requested, while 5 sigma had 47-60 candidates waiting one rung below.

    That matters more than it sounds. The kernel's overall flux scale is what
    sets how much reference is subtracted, the host there is 2.6-4.5 times the
    transient inside the aperture, and so a few per cent of scatter in the
    kernel sum becomes a tenth of a magnitude in the light curve: two r frames
    nine minutes apart came out 0.21 mag (5.4 sigma) apart, of which the
    kernel-sum difference alone accounted for 95 per cent.

    Candidates are taken brightest first, so a deeper threshold can only add
    fainter stars after the bright ones have been used -- it never displaces a
    good stamp with a worse one.
    """
    from photutils.detection import DAOStarFinder

    _, median, std = sigma_clipped_stats(science, mask=mask, sigma=3.0)
    # A stamp with nothing unmasked left in it gives a masked std, which
    # DAOStarFinder rejects from deep inside photutils with a message about
    # quantities. Say the useful thing here instead and let the caller's
    # "too few pixels to solve for the kernel" be what the log shows.
    if not np.isfinite(std) or std <= 0:
        return []

    peak_limit = float(np.nanpercentile(science, 99.999))
    best: list[tuple[int, int]] = []
    for threshold in (30.0, 15.0, 8.0, 5.0, 3.0):
        try:
            candidates = DAOStarFinder(fwhm=max(fwhm_px, 1.5), threshold=threshold * std,
                                       exclude_border=True)(science - median, mask=mask)
        except Exception:
            break
        if candidates is None or len(candidates) == 0:
            continue
        picks = _accept_stamps(candidates, science, mask, n_stamps, half,
                               peak_limit, avoid_xy, avoid_radius)
        if len(picks) > len(best):
            best = picks
        if len(best) >= n_stamps:
            break
    if len(best) < n_stamps:
        log.debug("only %d of the %d requested kernel stamps survived the cuts",
                  len(best), n_stamps)
    return best


def solve_kernel(science: np.ndarray, reference: np.ndarray, mask: np.ndarray,
                 fwhm_px: float, cfg,
                 exclude_xy: Optional[tuple] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Least-squares PSF matching taking *reference* into *science*.

    The kernel may vary over the stamp: each basis function is multiplied by a
    2D polynomial of order ``subtraction.kernel.spatial_order`` (0 = constant).

    Returns ``(model, kernel_centre, info)`` where *model* is the matched
    reference (the differential-background polynomial included) and
    *kernel_centre* is the kernel evaluated at the centre of the stamp, used for
    noise propagation and QA. The background term used to be returned separately
    as a fourth, full-frame array that no caller ever looked at.
    """
    half = int(cfg.get("subtraction.kernel.half_size_px", 9))
    sigmas = list(cfg.get("subtraction.kernel.gauss_sigmas_px", [0.7, 1.5, 3.0]))
    degrees = list(cfg.get("subtraction.kernel.gauss_degrees", [4, 2, 0]))
    bkg_order = int(cfg.get("subtraction.kernel.background_order", 1))
    spatial_order = int(cfg.get("subtraction.kernel.spatial_order", 0))
    n_stamps = int(cfg.get("subtraction.kernel.n_stamps", 40))
    stamp_half = int(cfg.get("subtraction.kernel.stamp_half_px", 20))
    nsigma = float(cfg.get("subtraction.kernel.sigma_clip", 3.0))
    max_iter = int(cfg.get("subtraction.kernel.max_iter", 3))

    # The pixel sizes above were tuned on frames with a ~5 px FWHM. A kernel
    # must be wide enough to carry the seeing difference, and the stamps wide
    # enough to hold a star, so on coarser-sampled frames everything grows in
    # proportion. Never shrinks: below the tuning FWHM the defaults stand.
    if bool(cfg.get("subtraction.kernel.scale_with_fwhm", True)):
        tuned = float(cfg.get("subtraction.kernel.tuned_fwhm_px", 5.0) or 5.0)
        grow = max(float(fwhm_px) / tuned, 1.0) if np.isfinite(fwhm_px) else 1.0
        if grow > 1.0:
            half = int(np.ceil(half * grow))
            sigmas = [float(v) * grow for v in sigmas]
            stamp_half = int(np.ceil(stamp_half * grow))
            log.debug("kernel scaled x%.2f for a %.1f px FWHM (half-size %d px)",
                      grow, fwhm_px, half)

    basis = gaussian_basis(half, sigmas, degrees)
    spatial = _polynomial_terms(science.shape, spatial_order)
    bkg_terms = _polynomial_terms(science.shape, bkg_order)
    n_kernel_columns = len(basis) * len(spatial)
    n_columns = n_kernel_columns + len(bkg_terms)
    # With a spatially constant kernel the single spatial term is all ones, so
    # multiplying by it only makes a copy.
    unit_spatial = spatial_order == 0 and len(spatial) == 1

    ref = np.nan_to_num(reference, nan=0.0, posinf=0.0, neginf=0.0)

    # The transient must never inform the kernel. It is in the science frame and
    # not in the reference, so a stamp on it asks the kernel to conjure its flux
    # out of the host: the solve tilts (a kernel centroid of 0.9 px on one
    # SN 2026abgb frame) and the matched host is over-subtracted exactly where
    # the supernova is measured. It sits on a bright host, so the star finder
    # happily picks it: on every 2026abgb frame a stamp landed within 1 px of
    # it. Keep every stamp, and every fitted pixel, clear of it by a stamp's
    # half-width plus a few FWHM.
    exclusion = None
    radius = 0.0
    if exclude_xy is not None and all(np.isfinite(exclude_xy)):
        radius = float(cfg.get("subtraction.kernel.exclude_target_fwhm", 2.0)) * float(fwhm_px)
        yy_e, xx_e = np.mgrid[0:science.shape[0], 0:science.shape[1]]
        exclusion = np.hypot(xx_e - float(exclude_xy[0]), yy_e - float(exclude_xy[1])) < radius
    # Only the transient's own pixels are kept out. Rejecting every stamp that
    # overlaps it (a disc of stamp_half + 3 FWHM) was tried and was worse: on
    # SN 2026abgb it also threw out the host nucleus and nearby knots, and the
    # kernel sums scattered by 5 per cent about the zero-point prediction
    # instead of 1-2.
    picks = _select_stamps(science, mask, fwhm_px, n_stamps, stamp_half,
                           avoid_xy=exclude_xy if exclusion is not None else None,
                           avoid_radius=radius)
    fit_mask = np.zeros(science.shape, dtype=bool)
    if picks:
        for x, y in picks:
            fit_mask[y - stamp_half:y + stamp_half + 1, x - stamp_half:x + stamp_half + 1] = True
    else:
        log.warning("no isolated stars for kernel fitting; using the whole stamp")
        fit_mask[:] = True

    fit_mask &= ~mask
    if exclusion is not None:
        fit_mask &= ~exclusion
    fit_mask &= np.isfinite(science) & np.isfinite(reference)
    # Never let the edges, where the convolution is incomplete, drive the fit.
    fit_mask[:half + 1, :] = False
    fit_mask[-half - 1:, :] = False
    fit_mask[:, :half + 1] = False
    fit_mask[:, -half - 1:] = False

    fit_index = np.flatnonzero(fit_mask)
    if fit_index.size < 10 * n_columns:
        raise RuntimeError(f"too few pixels ({int(fit_index.size)}) to solve for the kernel")

    # The design matrix holds only the pixels that are actually fitted, and it
    # is built one basis function at a time so that at most one convolution
    # exists at full stamp size. Keeping every column at full size instead --
    # 22 convolutions, 22 copies of them, and a fresh model image per iteration
    # -- cost roughly 450 MB at the default rebinning and some 4 GB with
    # image.rebin: 1, which is more than most laptops have to spare.
    design = np.empty((fit_index.size, n_columns), dtype=float)
    column = 0
    for function in basis:
        convolved = fftconvolve(ref, function, mode="same")
        flat = convolved.ravel()
        for poly in spatial:
            design[:, column] = (flat[fit_index] if unit_spatial
                                 else (convolved * poly).ravel()[fit_index])
            column += 1
        del convolved, flat
    for term in bkg_terms:
        design[:, column] = term.ravel()[fit_index]
        column += 1
    observed = science.ravel()[fit_index]

    keep = np.ones(fit_index.size, dtype=bool)
    coefficients = np.zeros(n_columns)
    rms = float("nan")
    for _ in range(max_iter):
        coefficients, *_ = np.linalg.lstsq(design[keep], observed[keep], rcond=None)
        residual = observed - design @ coefficients
        scatter = float(np.std(residual[keep]))
        rms = scatter
        if not np.isfinite(scatter) or scatter <= 0:
            break
        new_keep = np.abs(residual) < nsigma * scatter
        if new_keep.sum() < 10 * n_columns or new_keep.sum() == keep.sum():
            break
        keep = new_keep

    # Kernel at the centre of the stamp: the spatial polynomials are built so
    # that the centre pixel has P_0 = 1 and every higher term = 0.
    centre = tuple(n // 2 for n in science.shape)
    weights = coefficients[:n_kernel_columns].reshape(len(basis), len(spatial))
    kernel_centre = sum(
        float(np.dot(weights[n], [p[centre] for p in spatial])) * b
        for n, b in enumerate(basis)
    )

    # The matched image, rebuilt from the coefficients rather than carried
    # through the fit. Convolution is linear, so with a constant kernel the
    # whole basis expansion collapses to a single convolution by that kernel.
    model = np.zeros(science.shape, dtype=float)
    for coefficient, term in zip(coefficients[n_kernel_columns:], bkg_terms):
        if coefficient:
            model += coefficient * term
    if unit_spatial:
        model += fftconvolve(ref, kernel_centre, mode="same")
    else:
        index = 0
        for function in basis:
            convolved = fftconvolve(ref, function, mode="same")
            for poly in spatial:
                coefficient = coefficients[index]
                index += 1
                if coefficient:
                    model += coefficient * (convolved * poly)
            del convolved

    yk, xk = np.mgrid[:kernel_centre.shape[0], :kernel_centre.shape[1]]
    ksum = float(kernel_centre.sum())
    kc = kernel_centre.shape[0] // 2
    info = {
        "kernel_dx": float(((xk - kc) * kernel_centre).sum() / ksum) if ksum else float("nan"),
        "kernel_dy": float(((yk - kc) * kernel_centre).sum() / ksum) if ksum else float("nan"),
        "target_excluded": exclusion is not None,
        "n_stamps": len(picks),
        "n_pixels": int(keep.sum()),
        "n_basis": len(basis),
        "n_columns": n_columns,
        "spatial_order": spatial_order,
        "resid_rms": rms,
        "kernel_sum": float(kernel_centre.sum()),
        "background_coefficients": [float(c) for c in coefficients[n_kernel_columns:]],
    }
    return model, kernel_centre, info


# --------------------------------------------------------------------------
def subtract(image, reference, cfg, coord: SkyCoord,
             fwhm_px: Optional[float] = None) -> DifferenceResult:
    """Resample the reference onto the science frame and PSF-match it away."""
    from reproject import reproject_interp

    size_arcmin = detector.stamp_arcmin(cfg, image)
    fwhm_px = float(fwhm_px or image.fwhm_px or 4.0)

    # --- science stamp ---------------------------------------------------
    size = (size_arcmin * u.arcmin, size_arcmin * u.arcmin)
    sci_cut = Cutout2D(image.data_sub, coord, size, wcs=image.wcs, mode="partial",
                       fill_value=np.nan)
    mask_cut = Cutout2D(image.mask.astype(float), coord, size, wcs=image.wcs,
                        mode="partial", fill_value=1.0).data > 0.5
    var_cut = Cutout2D(image.error ** 2, coord, size, wcs=image.wcs, mode="partial",
                       fill_value=np.nan).data
    science = np.asarray(sci_cut.data, dtype=float)
    stamp_wcs = sci_cut.wcs
    mask = mask_cut | ~np.isfinite(science)

    empty = DifferenceResult(np.zeros_like(science), np.zeros_like(science), stamp_wcs,
                             science, np.zeros_like(science), np.zeros((1, 1)), np.nan,
                             np.nan, "none", 0, np.nan, np.nan, ok=False)

    # --- reference resampled onto the science stamp ----------------------
    try:
        ref_data, footprint = reproject_interp(
            (np.nan_to_num(reference.data, nan=np.nan), reference.wcs),
            stamp_wcs, shape_out=science.shape)
    except Exception as exc:
        empty.message = f"reprojection failed: {exc}"
        return empty

    ref_valid = np.isfinite(ref_data) & (footprint > 0.5)
    coverage = float(ref_valid.mean())
    if coverage < 0.5:
        empty.message = f"reference covers only {coverage:.0%} of the stamp"
        return empty

    # Reference pixels carry the reference's own sky; remove it.
    _, ref_sky, ref_rms = sigma_clipped_stats(ref_data[ref_valid], sigma=3.0)
    ref_data = np.where(ref_valid, ref_data - ref_sky, np.nan)
    mask = mask | ~ref_valid

    # Flux-conserving resampling changes surface brightness with pixel area.
    area_ratio = (image.meta.pixel_scale / max(reference.pixel_scale, 1e-6)) ** 2

    # --- which frame gets convolved? -------------------------------------
    # What the reference's own sampling implies about its PSF, in science
    # pixels. Survey images are sampled at roughly 2.5 pixels per FWHM: a crude
    # estimate, but an honest one, and a far better starting point for the fit
    # than the science seeing, which tells us nothing about the reference.
    expected_ref_fwhm = 2.5 * reference.pixel_scale / max(image.meta.pixel_scale, 1e-6)
    ref_fwhm = _stamp_fwhm(np.nan_to_num(ref_data), mask, fwhm_px,
                           seed=expected_ref_fwhm)
    ref_fwhm_measured = bool(np.isfinite(ref_fwhm))
    if not ref_fwhm_measured:
        # Do not fall through to convolve_reference here. That would quietly ask
        # the kernel to *deconvolve* the reference, which it cannot do: it comes
        # back with a ringing kernel and a "difference" that is essentially the
        # science frame, host galaxy and all. An unmeasurable PSF is not
        # evidence of a sharp one, so fall back on the sampling estimate.
        ref_fwhm = expected_ref_fwhm
        log.warning("%s: could not measure the reference PSF; assuming %.2f px "
                    "from its %.2f\"/px sampling", image.meta.path.name, ref_fwhm,
                    reference.pixel_scale)

    mode = "convolve_reference"
    science_fit, reference_fit = science, ref_data
    flip_ratio = float(cfg.get("subtraction.convolve_science_ratio", 1.25))
    if ref_fwhm > flip_ratio * fwhm_px:
        # The reference is blurrier: matching it to the science would be a
        # deconvolution. Degrade the science instead and difference there.
        mode = "convolve_science"
        science_fit, reference_fit = ref_data, science
        log.info("reference seeing (%.2f px) is worse than the science (%.2f px):"
                 " convolving the science frame instead", ref_fwhm, fwhm_px)
    log.info("%s: PSF match -- reference %.2f px (%s), science %.2f px, %s",
             image.meta.path.name, ref_fwhm,
             "measured" if ref_fwhm_measured else "assumed from sampling",
             fwhm_px, mode)

    work_mask = mask | ~np.isfinite(science_fit) | ~np.isfinite(reference_fit)
    try:
        target_xy = tuple(float(v) for v in stamp_wcs.world_to_pixel(coord))
        matched, kernel, info = solve_kernel(
            np.nan_to_num(science_fit), np.nan_to_num(reference_fit), work_mask,
            fwhm_px if mode == "convolve_reference" else max(ref_fwhm, fwhm_px), cfg,
            exclude_xy=target_xy)
    except Exception as exc:
        empty.message = f"kernel solution failed: {exc}"
        log.warning("%s: %s", image.meta.path.name, empty.message)
        return empty

    difference = science_fit - matched

    ref_var = np.full(science.shape, float(ref_rms) ** 2)
    if reference.variance is not None:
        try:
            rv, _ = reproject_interp((reference.variance, reference.wcs), stamp_wcs,
                                     shape_out=science.shape)
            ref_var = np.where(np.isfinite(rv), rv, ref_var)
        except Exception:
            pass
    ref_var = ref_var * area_ratio

    kernel_power = fftconvolve(ref_var, kernel ** 2, mode="same")
    if mode == "convolve_reference":
        variance = np.nan_to_num(var_cut) + np.abs(kernel_power)
        flux_scale = 1.0
    else:
        # The science frame was convolved, so its fluxes were multiplied by
        # sum(kernel); divide measured difference fluxes by that.
        sci_power = fftconvolve(np.nan_to_num(var_cut), kernel ** 2, mode="same")
        variance = np.abs(sci_power) + ref_var
        flux_scale = float(kernel.sum())
        difference = -difference          # keep "positive = transient in science"

    variance = np.where(np.isfinite(variance) & (variance > 0), variance, np.nan)
    difference = np.where(mask, np.nan, difference)

    inner = ~mask & np.isfinite(difference) & np.isfinite(variance)
    diff_rms = float(sigma_clipped_stats(difference[inner], sigma=3.0)[2]) if inner.any() else np.nan
    # Reduced chi-square of the difference against its own noise model: 1.0
    # means the subtraction left nothing but photon noise behind.
    if inner.any():
        pull = difference[inner] / np.sqrt(variance[inner])
        mean_pull, _, std_pull = sigma_clipped_stats(pull, sigma=5.0, maxiters=5)
        chi2 = float(mean_pull ** 2 + std_pull ** 2)
    else:
        chi2 = float("nan")

    result = DifferenceResult(
        difference=difference, variance=variance, wcs=stamp_wcs,
        science_stamp=science, reference_stamp=matched, kernel=kernel,
        kernel_sum=float(kernel.sum()), flux_scale=flux_scale, mode=mode,
        n_stamps=int(info["n_stamps"]), chi2=chi2, rms=float(info["resid_rms"]),
        ok=True,
    )
    result.meta.update(info)
    result.meta.update({
        "reference_source": reference.source,
        "reference_band": reference.source_band or reference.band,
        "reference_fwhm_px": float(ref_fwhm),
        "reference_fwhm_measured": ref_fwhm_measured,
        "science_fwhm_px": float(fwhm_px),
        "coverage": coverage,
        "difference_rms": diff_rms,
        "reference_n_images": reference.n_images,
        "reference_mjd_min": reference.mjd_range[0],
        "reference_mjd_max": reference.mjd_range[1],
        "area_ratio": area_ratio,
    })
    log.info("%s: difference built (%s, %d stamps, kernel sum %.3f, chi2 %.2f, rms %.1f)",
             image.meta.path.name, mode, result.n_stamps, result.kernel_sum, result.chi2,
             result.rms)
    return result


def _stamp_fwhm(data: np.ndarray, mask: np.ndarray, fallback: float,
                min_stars: int = 5, seed: Optional[float] = None) -> float:
    """Median stellar FWHM of a stamp, in its own pixels.

    Iterated from a deliberately sharp starting guess. An earlier version took
    the science frame's FWHM as both the detection kernel and the fitting-box
    size, which on a deep reference let the fit wander onto blends: the
    "measured" reference width then simply tracked the science seeing, and the
    decision about which frame to convolve became noise.

    The fitting box must be able to hold the PSF it is measuring, and the box
    is set by the current guess. A width well above the guess was therefore
    thrown out as implausible, and with too few widths left the function
    returned "unknown" on its first pass -- precisely for the references much
    blurrier than the science, which are the ones the caller most needs to
    recognise. So a round that ends short of *min_stars* now widens the guess
    and tries again rather than giving up, and *seed* lets the caller start
    from the resolution it expects instead of from the science seeing.
    """
    from photutils.detection import DAOStarFinder
    from .detection import _fit_gaussian_fwhm

    try:
        _, median, std = sigma_clipped_stats(data, mask=mask, sigma=3.0)
        if not np.isfinite(std) or std <= 0:
            return float("nan")
    except Exception:
        return float("nan")

    start = seed if (seed is not None and np.isfinite(seed) and seed > 0) else fallback
    # The box is 2.5 guesses wide and stars within a box of the edge are
    # skipped, so the guess cannot grow without limit on a small stamp.
    guess_max = float(np.clip(min(data.shape) / 16.0, 6.0, 25.0))
    guess = float(np.clip(start, 2.0, guess_max))
    result = float("nan")

    for _ in range(6):
        half = int(np.ceil(2.5 * guess))
        widths = []
        try:
            finder = DAOStarFinder(fwhm=guess, threshold=15.0 * std, exclude_border=True)
            found = finder(data - median, mask=mask)
        except Exception:
            return result

        if found is not None and len(found):
            found = found[np.argsort(-np.asarray(found["flux"]))][:60]
            peak_limit = np.nanpercentile(data, 99.99)
            for row in found:
                x, y = int(round(float(row["xcentroid"]))), int(round(float(row["ycentroid"])))
                if x - half < 0 or y - half < 0 or x + half + 1 > data.shape[1] \
                        or y + half + 1 > data.shape[0]:
                    continue
                if np.any(mask[y - half:y + half + 1, x - half:x + half + 1]):
                    continue
                if float(row["peak"]) > 0.9 * peak_limit:
                    continue
                value = _fit_gaussian_fwhm(data[y - half:y + half + 1, x - half:x + half + 1])
                # A fitted width filling the box is not trustworthy -- but that
                # says the box is too small, not that the star is bad.
                if value is not None and 0.8 < value < 0.8 * half:
                    widths.append(value)

        if len(widths) < min_stars:
            if guess < guess_max:
                guess = float(min(guess * 1.8, guess_max))
                continue
            return result

        _, new_guess, _ = sigma_clipped_stats(np.array(widths), sigma=2.5, maxiters=5)
        if not np.isfinite(new_guess) or new_guess <= 0:
            return result
        result = float(new_guess)
        if abs(new_guess - guess) / guess < 0.10:
            break
        guess = float(np.clip(new_guess, 1.0, 40.0))

    return result
