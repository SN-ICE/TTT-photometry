"""Per-band detector patterns: illumination gradients and interference fringes.

Red-band frames from thinned detectors carry a fringe pattern -- interference of night
sky emission lines inside the thinned CCD -- of about +-2 per cent of the sky
with a wavelength near 100 px, plus a smooth ~2 per cent illumination gradient
that is present in every band. Both are fixed in *detector* coordinates, so
they can be measured by median-combining many source-masked frames and then
removed from each new frame.

The two components are corrected differently, because they are physically
different things:

* the smooth component is a sensitivity (flat-field) residual -- multiplicative,
  it scales source flux as well as sky, so the frame is *divided* by it;
* the ripples are sky emission that never went through the source's optical
  path -- additive, so a scaled copy is *subtracted*, with the scale fitted on
  each frame because fringe strength follows the sky lines rather than the
  total sky level.

Patterns are stored per band under ``calib/pattern_<band>.fits`` on a coarse
block grid, which both keeps them small and averages down their noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

from .detector import Detector, identify, prepare_raw
from .utils import band_token, ensure_dir, get_logger

log = get_logger("tttphot.patterns")


@dataclass
class Pattern:
    """A measured detector pattern, held on a block grid."""

    illumination: np.ndarray      # multiplicative, median 1
    fringe: np.ndarray            # additive, in units of the sky level, median 0
    block: int
    band: str
    shape: tuple[int, int]        # full-frame shape the pattern was built from
    n_frames: int = 0
    header: fits.Header = None
    detector: Optional[Detector] = None   # the readout this was measured on

    @property
    def fringe_amplitude(self) -> float:
        return float(np.ptp(np.nanpercentile(self.fringe, [1, 99])))

    @property
    def illumination_amplitude(self) -> float:
        return float(np.ptp(np.nanpercentile(self.illumination, [1, 99])))

    def expand(self, component: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        """Interpolate a block-grid component back to full frame resolution."""
        from scipy.ndimage import zoom

        factor = (shape[0] / component.shape[0], shape[1] / component.shape[1])
        try:
            out = zoom(component, factor, order=3, grid_mode=True, mode="nearest")
        except TypeError:      # scipy < 1.6 has no grid_mode
            out = zoom(component, factor, order=3, mode="nearest")
        if out.shape != tuple(shape):
            padded = np.full(shape, np.nanmedian(component), dtype=float)
            ny = min(out.shape[0], shape[0])
            nx = min(out.shape[1], shape[1])
            padded[:ny, :nx] = out[:ny, :nx]
            out = padded
        return out


# --------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------
def _source_mask(data: np.ndarray, nsigma: float = 1.5, npixels: int = 5,
                 dilate: int = 12) -> np.ndarray:
    """Segmentation-based mask of everything that is not blank sky."""
    from astropy.convolution import convolve
    from photutils.segmentation import (detect_sources, detect_threshold,
                                        make_2dgaussian_kernel)
    from photutils.utils import circular_footprint

    try:
        threshold = detect_threshold(data, nsigma=nsigma)
        kernel = make_2dgaussian_kernel(3.0, size=5)
        segments = detect_sources(convolve(data, kernel), threshold, npixels=npixels)
        if segments is None:
            return ~np.isfinite(data)
        return segments.make_source_mask(footprint=circular_footprint(radius=dilate)) \
            | ~np.isfinite(data)
    except Exception as exc:      # pragma: no cover
        log.warning("segmentation failed (%s); falling back to a sigma cut", exc)
        _, median, std = sigma_clipped_stats(data, sigma=3.0)
        return (data > median + 2.0 * std) | ~np.isfinite(data)


def _block_median(data: np.ndarray, block: int) -> np.ndarray:
    ny, nx = data.shape
    by, bx = ny // block, nx // block
    trimmed = data[:by * block, :bx * block]
    cube = trimmed.reshape(by, block, bx, block)
    import warnings
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN blocks are expected
        return np.nanmedian(cube, axis=(1, 3))


def balance_fields(entries, max_frames: int) -> list:
    """Pick up to *max_frames* frames, spread evenly over the fields available.

    ``entries`` are objects with ``.path`` and ``.object_name``. Taking the
    first N in time instead lets one target fill every slot -- a 2023 campaign
    of 393 frames leaves no room for anything later -- and a pattern measured
    on one field quietly absorbs that field's own sky, then subtracts it back
    out of every frame of that field. Round-robin keeps every field
    represented whatever their relative sizes.
    """
    groups: dict = {}
    for entry in entries:
        groups.setdefault(getattr(entry, "object_name", "") or "?", []).append(entry)
    order = sorted(groups)
    picked, i = [], 0
    while len(picked) < max_frames:
        added = False
        for name in order:
            if i < len(groups[name]):
                picked.append(groups[name][i])
                added = True
                if len(picked) >= max_frames:
                    break
        if not added:
            break
        i += 1
    return picked


def build_pattern(paths: Sequence[Path], band: str, block: int = 16,
                  smooth_blocks: float = 24.0, min_frames: int = 8,
                  max_frames: int = 60, outlier_sigma: float = 30.0,
                  max_amplitude: float = 0.25,
                  instrument: Optional[dict] = None) -> Optional[Pattern]:
    """Median-combine source-masked, sky-normalised frames in detector space.

    Frames should be dithered (a few tens of pixels is plenty) and
    drawn from more than one field, so that no astronomical source survives the
    median. Use :func:`balance_fields` to choose them.
    """
    paths = list(paths)[:max_frames]
    if len(paths) < min_frames:
        log.warning("only %d frames for the %s pattern (need %d)", len(paths), band, min_frames)
        return None

    planes, used, shape = [], [], None
    binning, offset, detector = 1, (0, 0), None
    for path in paths:
        try:
            with fits.open(path, memmap=False) as hdul:
                hdu = next((h for h in hdul if getattr(h, "data", None) is not None), hdul[0])
                data = np.asarray(hdu.data, dtype=float)
                head = hdu.header
            sat = ((instrument or {}).get("defaults", {}) or {}).get("saturation_adu")
            data, head, _ = prepare_raw(data, head, instrument,
                                        saturation=float(sat) if sat else None)
            here = identify(head, instrument=instrument)
            if detector is None:
                detector = here
                binning = int(here.binning)
                offset = (int(head.get("OFFSET1", 0) or 0), int(head.get("OFFSET2", 0) or 0))
        except Exception as exc:
            log.warning("cannot read %s: %s", Path(path).name, exc)
            continue
        if here != detector:
            # Mixing readouts here would average an unrelated fringe field into
            # the pattern. Group the frames before calling this, not after.
            log.warning("%s is %s, not %s -- skipped",
                        Path(path).name, here.label, detector.label)
            continue
        if shape is None:
            shape = data.shape

        mask = _source_mask(data)
        _, sky, noise = sigma_clipped_stats(data, mask=mask, sigma=3.0)
        if not np.isfinite(sky) or sky <= 0:
            continue
        # Bad columns divided by a near-zero flat reach +-1e7 ADU on MEIA3.
        # A block median mostly survives them, but the source mask and the
        # block statistics do not, and a single such block turns the pattern
        # into nonsense. Drop anything far outside the sky distribution.
        if np.isfinite(noise) and noise > 0:
            wild = np.abs(data - sky) > outlier_sigma * noise
            if wild.any():
                mask = mask | wild
        normalised = np.where(mask, np.nan, data) / sky
        planes.append(_block_median(normalised, block).astype(np.float32))
        used.append(Path(path))

    if len(planes) < min_frames:
        log.warning("only %d usable frames for the %s pattern", len(planes), band)
        return None

    cube = np.array(planes, dtype=float)
    import warnings
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pattern = np.nanmedian(cube, axis=0)
        coverage = np.sum(np.isfinite(cube), axis=0)

    # Fill any block that never saw blank sky.
    median_level = float(np.nanmedian(pattern))
    pattern = np.where(np.isfinite(pattern), pattern, median_level)
    pattern /= median_level

    from scipy.ndimage import gaussian_filter
    illumination = gaussian_filter(pattern, sigma=float(smooth_blocks), mode="nearest")
    illumination /= float(np.median(illumination))
    fringe = pattern - illumination
    fringe -= float(np.median(fringe))

    header = fits.Header()
    header["BAND"] = band
    header["INSTRUME"] = detector.instrument if detector else "unknown"
    header["DETECTOR"] = detector.token if detector else "unknown"
    header["NFRAMES"] = len(planes)
    header["BLOCK"] = block
    header["SMOOTHB"] = smooth_blocks
    header["FULLNY"] = shape[0]
    header["FULLNX"] = shape[1]
    header["XBINNING"] = binning
    header["OFFSET1"] = offset[0]
    header["OFFSET2"] = offset[1]
    header["MINCOV"] = int(coverage.min())
    header["ILLUMPP"] = float(np.ptp(np.nanpercentile(illumination, [1, 99])))
    header["FRNGPP"] = float(np.ptp(np.nanpercentile(fringe, [1, 99])))

    log.info("%s pattern from %d frames: illumination %.3f%% pp, fringe %.3f%% pp",
             band, len(planes), 100 * header["ILLUMPP"], 100 * header["FRNGPP"])

    # A detector pattern is a few per cent. Anything larger is not the
    # detector: it is an unmasked defect, or one bright field that the median
    # failed to remove. Refuse it rather than write something that will be
    # subtracted from every frame.
    worst = max(float(header["ILLUMPP"]), float(header["FRNGPP"]))
    if max_amplitude and worst > max_amplitude:
        log.error("%s pattern rejected: %.1f%% peak-to-peak is far too large for a "
                  "detector pattern. Usually one bright field dominating the median "
                  "(add frames of other fields) or unmasked bad pixels.",
                  band, 100 * worst)
        return None

    return Pattern(illumination=illumination, fringe=fringe, block=block, band=band,
                   shape=shape, n_frames=len(planes), header=header, detector=detector)


# --------------------------------------------------------------------------
# storage
# --------------------------------------------------------------------------
def pattern_path(calib_root: Path, band: str, detector: Optional[Detector] = None) -> Path:
    """Where a pattern lives.

    The detector is part of the name because a pattern is only meaningful on
    the readout it was measured on, and one archive can hold several. Patterns
    written before this existed have no detector in the name; see
    :func:`load_pattern` for how those are still honoured.
    """
    root = ensure_dir(Path(calib_root))
    stem = f"pattern_{band_token(band)}"
    if detector is not None:
        stem += f"_{detector.token}"
    return root / f"{stem}.fits"


def save_pattern(pattern: Pattern, calib_root: Path) -> Path:
    path = pattern_path(calib_root, pattern.band, pattern.detector)
    hdus = fits.HDUList([
        fits.PrimaryHDU(data=pattern.illumination.astype(np.float32),
                        header=pattern.header),
        fits.ImageHDU(data=pattern.fringe.astype(np.float32), name="FRINGE"),
    ])
    # Truncate and rewrite in place rather than unlink-and-recreate, so that
    # rebuilding works on filesystems that allow writing but not deleting.
    with open(path, "wb") as handle:
        hdus.writeto(handle)
    log.info("wrote %s", path)
    return path


def load_pattern(calib_root: Path, band: str,
                 detector: Optional[Detector] = None) -> Optional[Pattern]:
    """The pattern for this band on this detector, or None.

    A pattern named for the detector wins. Failing that, a legacy
    ``pattern_<band>.fits`` from before patterns were keyed this way is
    accepted, but only if its recorded geometry matches -- an unlabelled
    pattern is not a pattern for every camera.
    """
    path = pattern_path(calib_root, band, detector)
    if not path.exists() and detector is not None:
        legacy = pattern_path(calib_root, band, None)
        if not legacy.exists():
            return None
        path = legacy
    if not path.exists():
        return None
    try:
        with fits.open(path, memmap=False) as hdul:
            illumination = np.asarray(hdul[0].data, dtype=float)
            header = hdul[0].header
            fringe = np.asarray(hdul["FRINGE"].data, dtype=float) \
                if "FRINGE" in hdul else np.zeros_like(illumination)
        stored = None
        if header.get("INSTRUME"):
            stored = Detector(instrument=str(header["INSTRUME"]).strip(),
                              ny=int(header.get("FULLNY", 0) or 0),
                              nx=int(header.get("FULLNX", 0) or 0),
                              binning=int(header.get("XBINNING", 1) or 1))
        if detector is not None:
            if stored is not None and stored != detector:
                log.warning("%s was measured on %s, not %s -- ignored",
                            path.name, stored.label, detector.label)
                return None
            # A legacy pattern names no camera, but it does record its shape.
            want = (int(header.get("FULLNY", 0) or 0), int(header.get("FULLNX", 0) or 0))
            if stored is None and want != (0, 0) and want != (detector.ny, detector.nx):
                log.debug("%s was measured on a %dx%d readout, not %s -- ignored",
                          path.name, want[1], want[0], detector.label)
                return None
        return Pattern(illumination=illumination, fringe=fringe,
                       block=int(header.get("BLOCK", 16)), band=band,
                       shape=(int(header.get("FULLNY", 0)), int(header.get("FULLNX", 0))),
                       n_frames=int(header.get("NFRAMES", 0)), header=header,
                       detector=stored or detector)
    except Exception as exc:
        log.warning("could not read %s: %s", path, exc)
        return None


# --------------------------------------------------------------------------
# applying
# --------------------------------------------------------------------------
def fit_fringe_scale(data: np.ndarray, fringe_full: np.ndarray,
                     nsigma: float = 2.5, maxiters: int = 5) -> float:
    """Least-squares amplitude of the fringe pattern in one frame, in ADU.

    Fitted rather than assumed, because fringe strength follows the sky
    emission lines and not the total sky level -- moonlight adds continuum that
    raises the sky without adding fringes.
    """
    good = np.isfinite(data) & np.isfinite(fringe_full)
    if good.sum() < 1000:
        return 0.0
    y = data[good]
    x = fringe_full[good]
    x = x - np.mean(x)

    scale = 0.0
    keep = np.ones(y.size, dtype=bool)
    for _ in range(maxiters):
        if keep.sum() < 1000:
            break
        yk = y[keep] - np.median(y[keep])
        xk = x[keep]
        denom = float(np.dot(xk, xk))
        if denom <= 0:
            break
        scale = float(np.dot(xk, yk) / denom)
        residual = yk - scale * xk
        std = float(np.std(residual))
        if not np.isfinite(std) or std <= 0:
            break
        new_keep = keep.copy()
        new_keep[keep] = np.abs(residual) < nsigma * std
        if new_keep.sum() == keep.sum():
            break
        keep = new_keep
    return scale


def geometry_matches(pattern: Pattern, header, shape) -> tuple[bool, str]:
    """A pattern is tied to a readout geometry; refuse to stretch it onto another."""
    if pattern.header is None:
        return True, ""
    want_bin = int(pattern.header.get("XBINNING", 1) or 1)
    have_bin = int(identify(header).binning or want_bin)
    if have_bin != want_bin:
        return False, f"binning {have_bin} does not match the pattern's {want_bin}"
    offsets = (int(header.get("OFFSET1", 0) or 0), int(header.get("OFFSET2", 0) or 0))
    want_offsets = (int(pattern.header.get("OFFSET1", 0) or 0),
                    int(pattern.header.get("OFFSET2", 0) or 0))
    if offsets != want_offsets:
        return False, f"readout offset {offsets} does not match the pattern's {want_offsets}"
    want_shape = (int(pattern.header.get("FULLNY", 0) or 0),
                  int(pattern.header.get("FULLNX", 0) or 0))
    if want_shape != (0, 0) and tuple(shape) != want_shape:
        return False, f"frame shape {tuple(shape)} does not match the pattern's {want_shape}"
    return True, ""


def preview(pattern: Pattern, path) -> Optional[Path]:
    """Save a two-panel PNG of the illumination and fringe components."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
        for ax, (image, title) in zip(axes, [
                (pattern.illumination, "illumination (multiplicative, divided out)"),
                (pattern.fringe, "fringes (additive, scaled and subtracted)")]):
            lo, hi = np.nanpercentile(image, [1, 99])
            handle = ax.imshow(image, origin="lower", vmin=lo, vmax=hi, cmap="RdBu_r")
            ax.set_title(f"{title}\npeak-to-peak {100 * (hi - lo):.2f}%", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(handle, ax=ax, fraction=0.046)
        fig.suptitle(f"{pattern.band}-band detector pattern -- {pattern.n_frames} frames")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        path = Path(path)
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path
    except Exception as exc:      # pragma: no cover
        log.warning("pattern preview failed: %s", exc)
        return None


def apply_pattern(data: np.ndarray, pattern: Pattern, cfg=None,
                  fit_scale: bool = True) -> tuple[np.ndarray, dict]:
    """Remove the illumination gradient and the fringes from one frame.

    Returns the corrected data and a dictionary of what was done.
    """
    shape = data.shape
    illumination = pattern.expand(pattern.illumination, shape)
    illumination = np.where(np.isfinite(illumination) & (illumination > 0.5),
                            illumination, 1.0)
    corrected = data / illumination

    info = {"pattern_band": pattern.band, "pattern_nframes": pattern.n_frames,
            "illumination_pp": pattern.illumination_amplitude,
            "fringe_pp": pattern.fringe_amplitude}

    if np.any(pattern.fringe != 0.0):
        fringe = pattern.expand(pattern.fringe, shape)
        fringe = np.where(np.isfinite(fringe), fringe, 0.0)
        _, sky, _ = sigma_clipped_stats(corrected, sigma=3.0)
        scale = fit_fringe_scale(corrected, fringe) if fit_scale else float(sky)

        # Sanity check on what is actually being removed rather than on the
        # scale itself: fringe strength follows the sky emission lines, so a
        # dark frame can legitimately want a scale several times its own
        # continuum sky. What must not happen is subtracting a pattern
        # comparable to the sky itself.
        removed = abs(scale) * pattern.fringe_amplitude
        limit = float(cfg.get("image.pattern.max_fringe_fraction", 0.5)) * abs(sky) \
            if cfg is not None else 0.5 * abs(sky)
        if not np.isfinite(scale) or (np.isfinite(sky) and removed > limit):
            log.warning("fringe amplitude %.1f ADU is implausible against a sky of %.1f; "
                        "not subtracting", removed, sky)
            scale = 0.0

        # Verify on the frame itself that subtracting actually flattens it.
        # Under a bright twilight or moon gradient the fit can latch onto
        # structure that is not the fringe pattern, and then subtracting hurts.
        if scale != 0.0:
            gain_ratio = _flatness_gain(corrected, pattern, scale)
            min_gain = float(cfg.get("image.pattern.min_flatness_gain", 1.05)) \
                if cfg is not None else 1.05
            info["fringe_flatness_gain"] = gain_ratio
            if np.isfinite(gain_ratio) and gain_ratio < min_gain:
                log.info("fringe subtraction would not flatten this frame "
                         "(gain %.2fx < %.2fx); skipping it", gain_ratio, min_gain)
                scale = 0.0
        corrected = corrected - scale * fringe
        info["fringe_scale_adu"] = float(scale)
        info["fringe_removed_pp_adu"] = float(abs(scale) * pattern.fringe_amplitude)
    else:
        info["fringe_scale_adu"] = 0.0
        info["fringe_removed_pp_adu"] = 0.0

    return corrected, info


def _flatness_gain(data: np.ndarray, pattern: "Pattern", scale: float) -> float:
    """How much flatter the frame gets on the pattern's own grid, as a ratio.

    Measured on block medians so that photon noise, which the fringe term
    cannot remove, does not dilute the comparison.
    """
    grid = _block_median(data, pattern.block)
    fringe = pattern.fringe
    if grid.shape != fringe.shape:
        ny = min(grid.shape[0], fringe.shape[0])
        nx = min(grid.shape[1], fringe.shape[1])
        grid, fringe = grid[:ny, :nx], fringe[:ny, :nx]
    good = np.isfinite(grid) & np.isfinite(fringe)
    if good.sum() < 64:
        return float("nan")
    before = sigma_clipped_stats(grid[good], sigma=3.0)[2]
    after = sigma_clipped_stats((grid - scale * fringe)[good], sigma=3.0)[2]
    if not np.isfinite(after) or after <= 0:
        return float("nan")
    return float(before / after)


def large_scale_scatter(data: np.ndarray, block: int = 32) -> float:
    """Robust scatter of the source-masked sky on large scales, relative to sky.

    Used to check that a pattern correction actually flattened the frame.
    """
    mask = _source_mask(data)
    _, sky, _ = sigma_clipped_stats(data, mask=mask, sigma=3.0)
    if not np.isfinite(sky) or sky == 0:
        return float("nan")
    blocks = _block_median(np.where(mask, np.nan, data), block)
    blocks = blocks[np.isfinite(blocks)]
    if blocks.size < 16:
        return float("nan")
    return float(np.std(blocks) / sky)
