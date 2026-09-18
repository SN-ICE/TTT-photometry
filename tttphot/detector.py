"""Detector configurations: which camera, read out how.

An archive is rarely a single instrument. TTT runs more than one telescope and
camera, and the same camera can be read out at different binnings. A *detector
configuration* is the tuple (instrument, array shape the pipeline works on,
binning), and anything measured on the detector -- fringe patterns,
illumination residuals -- is keyed by it.

This module also owns the step between the FITS file and the pixels the
pipeline measures (:func:`prepare_raw`): an optional border trim, masking of
known bad regions, and software rebinning. The shape that step produces is the
shape a detector configuration records, so that indexing (headers only) and
image loading (pixels) agree on which detector a frame belongs to.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from typing import Iterable, Optional

from .utils import first_header_value, get_logger, slugify

log = get_logger("tttphot.detector")

_WARNED_ASYMMETRIC: set = set()


@dataclass(frozen=True)
class Detector:
    """How one frame was read out. Hashable, so it can key a dict."""

    instrument: str
    ny: int
    nx: int
    binning: int = 1

    @property
    def token(self) -> str:
        """Filesystem-safe key, e.g. ``qhy600-ttt3_1595x1064_b6``."""
        name = slugify(self.instrument).lower() or "unknown"
        token = f"{name}_{self.nx}x{self.ny}"
        if self.binning != 1:
            token += f"_b{self.binning}"
        return token

    @property
    def label(self) -> str:
        """Short human-readable form for logs and tables."""
        text = f"{self.instrument} {self.nx}x{self.ny}"
        if self.binning != 1:
            text += f" bin{self.binning}"
        return text

    def field_arcmin(self, pixel_scale_arcsec: float) -> float:
        """Width of the field in arcmin, given the frame's pixel scale."""
        return max(self.nx, self.ny) * float(pixel_scale_arcsec) / 60.0

    def matches_shape(self, shape) -> bool:
        return tuple(int(v) for v in shape[:2]) == (self.ny, self.nx)


def _int(value, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def trim_spec(instrument: Optional[dict]) -> Optional[dict]:
    """The ``trim`` block of an instrument definition, if it has one."""
    if not instrument:
        return None
    spec = (instrument.get("defaults", {}) or {}).get("trim") or instrument.get("trim")
    return spec or None


def trim_slices(header, instrument: Optional[dict]):
    """(y_slice, x_slice) to apply to this frame, or None if it needs no trim.

    A frame is trimmed only when it is still the full chip -- astrometry.net
    records the size it solved in IMAGEW/IMAGEH, so an already-trimmed frame is
    recognisable and left alone. That lets one archive hold both.
    """
    spec = trim_spec(instrument)
    if not spec:
        return None
    ny, nx = _int(header.get("NAXIS2")), _int(header.get("NAXIS1"))
    full_x, full_y = _int(header.get("IMAGEW"), nx), _int(header.get("IMAGEH"), ny)
    if (nx, ny) != (full_x, full_y):
        return None                      # already trimmed by something else
    x0, x1 = (int(v) for v in spec.get("x", (0, 0)))
    y0, y1 = (int(v) for v in spec.get("y", (0, 0)))
    if x0 + x1 >= nx or y0 + y1 >= ny or (x0 or x1 or y0 or y1) == 0:
        return None
    return slice(y0, ny - y1), slice(x0, nx - x1)


def apply_trim(data, header, instrument: Optional[dict]):
    """Cut an unusable border off a frame, keeping the WCS valid.

    Only frames that are still the full chip are trimmed (see
    :func:`trim_slices`). CRPIX moves with the crop; the SIP coefficients are
    polynomials in (x - CRPIX), so they stay valid without being touched.
    """
    cut = trim_slices(header, instrument)
    if cut is None:
        return data, header, None
    ys, xs = cut
    header = header.copy()
    for key, shift in (("CRPIX1", xs.start), ("CRPIX2", ys.start)):
        if key in header:
            header[key] = float(header[key]) - shift
    data = data[ys, xs]
    header["NAXIS1"], header["NAXIS2"] = data.shape[1], data.shape[0]
    header["TTTTRIM"] = (f"x{xs.start}:{xs.stop} y{ys.start}:{ys.stop}",
                         "border removed by tttphot")
    return data, header, (ys, xs)


# --------------------------------------------------------------------------
# binning
# --------------------------------------------------------------------------
def _defaults(instrument: Optional[dict]) -> dict:
    return (instrument or {}).get("defaults", {}) or {}


def camera_binning(header, instrument: Optional[dict] = None) -> tuple[int, int]:
    """(x, y) binning the camera applied, from the header."""
    hmap = (instrument or {}).get("header", {}) or {}
    bx = _int(first_header_value(header, hmap.get("binning_x", ["XBINNING", "XFACTOR"]), 1), 1)
    by = _int(first_header_value(header, hmap.get("binning_y", ["YBINNING", "YFACTOR"]), bx), bx)
    return max(bx, 1), max(by, 1)


def header_pixel_scale(header, instrument: Optional[dict] = None) -> float:
    """Pixel scale of the stored frame in arcsec, from the CD/CDELT cards if
    present, else the instrument's unbinned scale times the camera binning."""
    cd11, cd12 = header.get("CD1_1"), header.get("CD1_2")
    cd21, cd22 = header.get("CD2_1"), header.get("CD2_2")
    try:
        if cd11 is not None and cd22 is not None:
            det = abs(float(cd11) * float(cd22) - float(cd12 or 0.0) * float(cd21 or 0.0))
            if det > 0:
                return math.sqrt(det) * 3600.0
        if header.get("CDELT1") is not None and header.get("CDELT2") is not None:
            return math.sqrt(abs(float(header["CDELT1"]) * float(header["CDELT2"]))) * 3600.0
    except (TypeError, ValueError):
        pass
    bx, by = camera_binning(header, instrument)
    return float(_defaults(instrument).get("pixel_scale", 0.4)) * math.sqrt(bx * by)


def rebin_factor(header, instrument: Optional[dict] = None) -> int:
    """The software rebinning factor this frame gets on load.

    ``defaults.rebin`` in the instrument file: an integer, or ``auto`` for the
    factor that brings the pixel scale closest to ``rebin_target_arcsec``. A
    frame that has already been rebinned (TTTREBIN in its header) gets 1.
    """
    if header.get("TTTREBIN") is not None:
        return 1
    spec = _defaults(instrument).get("rebin", 1)
    if spec is None:
        return 1
    if isinstance(spec, str):
        if spec.strip().lower() != "auto":
            return max(_int(spec, 1), 1)
        target = float(_defaults(instrument).get("rebin_target_arcsec", 0.35) or 0.35)
        scale = header_pixel_scale(header, instrument)
        if not (scale > 0):
            return 1
        return max(int(round(target / scale)), 1)
    return max(_int(spec, 1), 1)


def _rebin_array(data: np.ndarray, factor: int, saturation: Optional[float] = None) -> np.ndarray:
    """Block-SUM by *factor*, dropping any remainder rows/columns.

    Summing keeps fluxes in the camera's own ADU, so zero points stay directly
    comparable with un-rebinned frames and with the observatory's ZP card: the
    sky and a star's total counts are exactly what they were, only spread over
    fewer pixels. A block with NaNs is scaled up from its finite pixels (NaN if
    fewer than half are finite). A block containing a saturated pixel is set to
    at least factor**2 x saturation, so the saturation mask downstream -- whose
    threshold is scaled the same way -- still finds it rather than seeing a
    saturated star core diluted by its neighbours.
    """
    ny, nx = data.shape
    my, mx = ny // factor, nx // factor
    blocks = data[:my * factor, :mx * factor].reshape(my, factor, mx, factor)
    finite = np.isfinite(blocks)
    count = finite.sum(axis=(1, 3))
    n = factor * factor
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(finite, blocks, 0.0).sum(axis=(1, 3)) * (n / count)
    out[count < 0.5 * n] = np.nan
    if saturation is not None and np.isfinite(saturation):
        peak = np.where(finite, blocks, -np.inf).max(axis=(1, 3))
        hot = peak >= float(saturation)
        out[hot] = np.maximum(out[hot], float(saturation) * n)
    return out


def _rebin_header(header, factor: int, shape):
    """Rescale the WCS of a frame block-averaged by *factor*.

    Pixel (1-based FITS) coordinates map as p' = (p - 0.5) / f + 0.5, the CD
    matrix grows by f, and a SIP term of total order n = p + q -- a polynomial
    in pixel offsets returning a pixel offset -- scales by f ** (n - 1). The
    inverse (AP/BP) terms scale the same way.
    """
    header = header.copy()
    f = float(factor)
    for key in ("CRPIX1", "CRPIX2"):
        if key in header:
            header[key] = (float(header[key]) - 0.5) / f + 0.5
    for key in ("CD1_1", "CD1_2", "CD2_1", "CD2_2", "CDELT1", "CDELT2"):
        if key in header:
            header[key] = float(header[key]) * f
    for poly in ("A", "B", "AP", "BP"):
        order = header.get(f"{poly}_ORDER")
        if order is None:
            continue
        for p in range(int(order) + 1):
            for q in range(int(order) + 1 - p):
                key = f"{poly}_{p}_{q}"
                if key in header:
                    header[key] = float(header[key]) * f ** (p + q - 1)
    header["NAXIS2"], header["NAXIS1"] = int(shape[0]), int(shape[1])
    header["TTTREBIN"] = (int(factor), "software rebinning applied by tttphot (sum)")
    return header


def apply_bad_regions(data: np.ndarray, instrument: Optional[dict]) -> np.ndarray:
    """Set the instrument's known-bad regions to NaN ([y0, y1, x0, x1], native px)."""
    regions = _defaults(instrument).get("bad_regions") or []
    if not regions:
        return data
    data = np.array(data, dtype=float, copy=True)
    for region in regions:
        try:
            y0, y1, x0, x1 = (int(v) for v in region)
        except (TypeError, ValueError):
            continue
        data[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = np.nan
    return data


def prepare_raw(data: np.ndarray, header, instrument: Optional[dict],
                saturation: Optional[float] = None):
    """FITS pixels -> the pixels the pipeline measures.

    Trim, mask known bad regions, rebin. Returns ``(data, header, info)``, the
    header updated to describe the new array (shape and WCS).
    """
    info: dict = {"trimmed": False, "rebin": 1}
    if header.get("TTTREBIN") is not None:
        return data, header, info                  # already prepared
    data, header, cut = apply_trim(data, header, instrument)
    info["trimmed"] = cut is not None
    data = apply_bad_regions(data, instrument)
    factor = rebin_factor(header, instrument)
    if factor > 1:
        data = _rebin_array(np.asarray(data, dtype=float), factor, saturation)
        header = _rebin_header(header, factor, data.shape)
        info["rebin"] = factor
    return data, header, info


def identify(header, instrument_map: Optional[dict] = None,
             instrument: Optional[dict] = None) -> Detector:
    """The detector configuration a FITS header describes.

    ``instrument_map`` optionally renames the raw INSTRUME value, for archives
    where the same camera was written under more than one spelling. The shape
    and binning reported are those of the array the pipeline works on, after
    any trim and software rebinning; ``binning`` is the camera binning times
    the software factor.
    """
    raw = str(header.get("INSTRUME", "") or "unknown").strip()
    if instrument_map:
        raw = str(instrument_map.get(raw, raw)).strip()
    ny = _int(header.get("NAXIS2"), 0)
    nx = _int(header.get("NAXIS1"), 0)
    bx, by = camera_binning(header, instrument)
    if by != bx and (raw, bx, by) not in _WARNED_ASYMMETRIC:
        # The detector token records one binning factor, so an asymmetric
        # readout would be labelled by its x factor alone and could be merged
        # with a different configuration in group_by_detector and in the
        # kernel-sum grouping. No TTT frame has done this yet; say so if one does.
        _WARNED_ASYMMETRIC.add((raw, bx, by))
        log.warning("%s is binned %dx%d; the detector key records only %d, so two "
                    "readouts could be grouped together. Check the pixel scale and "
                    "the gain for these frames.", raw or "this camera", bx, by, bx)
    already = _int(header.get("TTTREBIN"), 0)
    if already:
        return Detector(instrument=raw or "unknown", ny=ny, nx=nx, binning=bx * already)
    cut = trim_slices(header, instrument)
    if cut is not None:
        ys, xs = cut
        ny, nx = ys.stop - ys.start, xs.stop - xs.start
    factor = rebin_factor(header, instrument)
    if factor > 1:
        ny, nx = ny // factor, nx // factor
    return Detector(instrument=raw or "unknown", ny=ny, nx=nx, binning=bx * factor)


def group_by_detector(entries: Iterable, header_of) -> dict:
    """Split an iterable of anything into ``{Detector: [entry, ...]}``.

    ``header_of`` maps an entry to its FITS header.
    """
    groups: dict = {}
    for entry in entries:
        header = header_of(entry)
        if header is None:
            continue
        groups.setdefault(identify(header), []).append(entry)
    return groups


# --------------------------------------------------------------------------
# sizes that should follow the frame
# --------------------------------------------------------------------------
def field_radius_arcmin(nx: int, ny: int, pixel_scale_arcsec: float) -> float:
    """Radius of the smallest circle containing the whole frame, in arcmin."""
    return math.hypot(float(nx), float(ny)) / 2.0 * float(pixel_scale_arcsec) / 60.0


def resolve_size(value, frame_arcmin: float, fraction: float = 1.0,
                 lo: Optional[float] = None, hi: Optional[float] = None,
                 default: float = 9.0, step: float = 1.0,
                 rounding: str = "up") -> float:
    """Read a config field that may be a number or the word ``auto``.

    A fixed arcmin figure is right only for one field of view. Readouts and
    cameras can span a factor of three in field, so a catalogue radius tuned for the
    middle one fetches four times too much sky for the smallest and misses
    half the field of the largest. ``auto`` scales the figure to the frame
    actually being measured; an explicit number still wins, so a value that
    was tuned by hand keeps working.
    """
    if value is None:
        return float(default)
    if isinstance(value, str):
        if value.strip().lower() not in ("auto", "frame"):
            try:
                return float(value)
            except ValueError:
                log.warning("%r is neither a number nor 'auto'; using %.1f",
                            value, float(default))
                return float(default)
        size = float(frame_arcmin) * float(fraction)
        # Downloads are cached under a key that includes the size, so a size
        # that drifts with every frame would mean a fresh download per frame.
        # Snapping to whole arcmin keeps one cache entry per detector.
        if step and step > 0:
            size = (math.ceil(size / step) if rounding == "up"
                    else math.floor(size / step)) * float(step)
        if lo is not None:
            size = max(size, float(lo))
        if hi is not None:
            size = min(size, float(hi))
        return size
    return float(value)


def frame_arcmin(image=None, detector: Optional[Detector] = None,
                 pixel_scale: Optional[float] = None, default: float = 12.3) -> float:
    """Width of a frame in arcmin, from an image or from a detector + scale."""
    if image is not None:
        try:
            return max(image.data.shape) * float(image.meta.pixel_scale) / 60.0
        except Exception:
            pass
    if detector is not None and pixel_scale:
        return max(detector.nx, detector.ny) * float(pixel_scale) / 60.0
    return float(default)


def stamp_arcmin(cfg, image=None, detector: Optional[Detector] = None,
                 pixel_scale: Optional[float] = None) -> float:
    """Working stamp size around the target, honouring ``auto``.

    Three quarters of the frame keeps a margin for the reprojection edges and
    for dithering, and the clamp stops a small subframe asking for a stamp
    wider than itself or a wide frame dragging down a huge template.
    """
    return resolve_size(cfg.get("subtraction.cutout_arcmin", "auto"),
                        frame_arcmin(image, detector, pixel_scale),
                        fraction=0.75, lo=4.0, hi=12.0, default=9.0,
                        step=1.0, rounding="down")


def catalog_radius_arcmin(cfg, key: str = "calibration.search_radius_arcmin",
                          image=None, detector: Optional[Detector] = None,
                          pixel_scale: Optional[float] = None,
                          default: float = 9.0) -> float:
    """Radius of sky to fetch a reference catalogue over, honouring ``auto``.

    ``auto`` is the circle that just contains the frame, with 5 per cent margin
    and rounded up to a whole arcmin. Every caller goes through here: the
    download is cached under a key that includes the radius, so a run and the
    ``fetch-catalog`` that pre-caches for it must agree to the digit or the
    pre-fetch is silently wasted.
    """
    ny = nx = 2048
    scale = 0.36
    if image is not None:
        try:
            ny, nx = image.data.shape[:2]
            scale = float(image.meta.pixel_scale)
        except Exception:
            pass
    elif detector is not None and pixel_scale:
        ny, nx = detector.ny, detector.nx
        scale = float(pixel_scale)
    return resolve_size(cfg.get(key, "auto"), field_radius_arcmin(nx, ny, scale),
                        fraction=1.05, lo=3.0, hi=30.0, default=default,
                        step=1.0, rounding="up")
