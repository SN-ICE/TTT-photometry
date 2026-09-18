"""Orchestration: from a directory of FITS files to a calibrated light curve.

The run is deliberately memory-frugal. Images are never all held at once:

  phase 0  read headers only and build an index of the epochs
  phase 1  build (or load from cache) one reference image per band
  phase 2  process one science image at a time, writing a row per method
  phase 3  assemble, plot, write
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
import astropy.units as u

from . import astrometry as astrom
from . import detector as det_mod
from .detector import Detector, identify
from . import catalogs as cat
from . import lightcurve as lc
from . import nameresolve
from . import photometry as ph
from . import plots
from . import zeropoints
from . import patterns as pat
from . import templates as tpl
from . import subtraction as sub
from .calibration import ZeroPoint, calibrate_image
from .detection import detect_sources, filter_by_profile, measure_fwhm, select_psf_stars
from .image import ScienceImage
from .utils import (UserError, edge_margin_px, ensure_dir, first_header_value,
                    get_logger, slugify)

log = get_logger("tttphot.pipeline")


# --------------------------------------------------------------------------
@dataclass
class EpochIndex:
    path: Path
    mjd: float
    band: str
    object_name: str
    exptime: float
    detector: Optional[Detector] = None


def index_images(paths: Iterable[Path], instrument: dict,
                 target: Optional[str] = None, cfg=None) -> list[EpochIndex]:
    """Read headers only and build the epoch list.

    When *cfg* is given, each frame's instrument definition is chosen from its
    own INSTRUME header, so an archive spanning a camera change is indexed
    correctly rather than through one camera's assumptions.
    """
    chooser = getattr(cfg, "instrument_for", None) if cfg is not None else None
    index: list[EpochIndex] = []
    for path in sorted(paths):
        try:
            header = fits.getheader(path)
        except Exception as exc:
            log.warning("cannot read %s: %s", path, exc)
            continue
        this = chooser(header) if callable(chooser) else instrument
        hmap = this.get("header", {})
        filters = this.get("filters", {})
        raw = str(first_header_value(header, hmap.get("filter", ["FILTER"]), "")).strip()
        band = filters.get(raw, filters.get(raw.replace("SDSS ", ""), raw))
        mjd = first_header_value(header, hmap.get("mjd", ["MJD-OBS"]))
        if mjd is None:
            jd = first_header_value(header, hmap.get("jd", ["JD"]))
            mjd = float(jd) - 2400000.5 if jd is not None else float("nan")
        name = str(first_header_value(header, hmap.get("object", ["OBJECT"]), "")).strip()
        if target and name and name.lower() != target.lower():
            continue
        index.append(EpochIndex(Path(path), float(mjd), band, name,
                                float(first_header_value(header, hmap.get("exptime", ["EXPTIME"]), 1.0)),
                                identify(header, instrument=this)))
    return sorted(index, key=lambda e: (e.band, e.mjd))


def normalise_name(name: str) -> str:
    """Comparable form of a transient name ('SN 2026abgb', 'AT2026abgb' -> '2026abgb').

    TTT writes the bare TNS designation into OBJECT, while a target is usually
    called SN... or AT... here, so the two must meet in the middle.
    """
    from .utils import normalise_name as _normalise
    return _normalise(name)


def _fits_under(directory: Path, patterns, recursive: bool) -> list[Path]:
    from .config import PIPELINE_ROOT
    out: list[Path] = []
    pipeline_root = PIPELINE_ROOT.resolve()
    for pattern in patterns:
        found = directory.rglob(pattern) if recursive else directory.glob(pattern)
        for path in found:
            try:
                resolved = path.resolve()
            except OSError:
                continue
            # Never pick up the pipeline's own cached templates or outputs.
            if resolved == pipeline_root or pipeline_root in resolved.parents:
                continue
            if any(part.startswith(".") for part in path.relative_to(directory).parts):
                continue
            out.append(path)
    return sorted(set(out))


def find_images(cfg, target: str) -> list[Path]:
    """Locate the FITS files for *target* under ``paths.data_root``.

    In order:

    1. a directory named after the target (``<data_root>/SN2026abgb/`` or the
       bare designation, ``<data_root>/2026abgb/``) -- everything in it;
    2. otherwise every FITS file anywhere under ``data_root`` whose OBJECT
       header names the target, so frames can simply be dropped into the TTT
       folder as they arrive. Files whose name contains the designation (TTT
       puts it there) are checked first; only if none match is every header
       read.
    """
    root = Path(cfg.data_root).expanduser()
    patterns = cfg.get("paths.glob", ["*.fits", "*.fit", "*.fits.gz", "*.fts"])
    wanted = normalise_name(target)
    variants = [target, wanted, f"SN{wanted}", f"AT{wanted}", f"SN {wanted}"]
    for name in dict.fromkeys(variants):
        directory = root / name
        if directory.is_dir():
            found = _fits_under(directory, patterns, recursive=True)
            if found:
                return found

    if not root.is_dir():
        return []
    candidates = _fits_under(root, patterns, recursive=True)
    by_name = [p for p in candidates if wanted and wanted in normalise_name(p.stem)
               or wanted in p.stem.lower()]
    instrument = cfg.get("instrument_config", {}) or {}

    def matches(path: Path) -> bool:
        try:
            header = fits.getheader(path)
        except Exception:
            return False
        chooser = getattr(cfg, "instrument_for", None)
        this = chooser(header) if callable(chooser) else instrument
        keys = (this.get("header", {}) or {}).get("object", ["OBJECT"])
        value = first_header_value(header, keys, "")
        return normalise_name(value) == wanted

    selected = [p for p in by_name if matches(p)]
    if not selected:
        selected = [p for p in candidates if p not in by_name and matches(p)]
    return sorted(selected)


def resolve_target_coord(cfg, index: Sequence[EpochIndex], instrument: dict) -> tuple:
    """SN position. Returns (coord, was_explicit).

    An explicit ``target.ra``/``target.dec`` wins. Otherwise the header is
    tried: target coordinates if the instrument writes any, else the pointing
    (for TTT, RA/DEC are the centre of the frame, which need not be the
    transient at all). Either is only a starting guess for the name resolution
    and the refinement in ``refine_target_coord``.
    """
    ra = cfg.get("target.ra")
    dec = cfg.get("target.dec")
    if ra is not None and dec is not None:
        if isinstance(ra, str) and (":" in ra or " " in str(ra).strip()):
            return SkyCoord(str(ra), str(dec), unit=(u.hourangle, u.deg)), True
        return SkyCoord(float(ra), float(dec), unit="deg"), True

    from .image import _sexagesimal_coord

    ras, decs, what = [], [], "the telescope pointing"
    for keys, label in ((("target_ra", "target_dec")), "the header target position"), \
                       ((("pointing_ra", "pointing_dec")), "the centre of the frame"):
        for entry in index[: min(len(index), 20)]:
            header = fits.getheader(entry.path)
            this = cfg.instrument_for(header) if hasattr(cfg, "instrument_for") else instrument
            hmap = this.get("header", {}) or {}
            coord = _sexagesimal_coord(
                first_header_value(header, hmap.get(keys[0], []) or []),
                first_header_value(header, hmap.get(keys[1], []) or []),
                hmap.get("coord_unit", "deg"))
            if coord is None:
                continue
            ras.append(coord.ra.deg)
            decs.append(coord.dec.deg)
        if ras:
            what = label
            break
    if not ras:
        raise UserError("cannot determine the target position; set target.ra / target.dec "
                        "in targets/<NAME>.yaml or configure TNS credentials")
    coord = SkyCoord(float(np.median(ras)) * u.deg, float(np.median(decs)) * u.deg)
    log.info("starting from %s (%s)", what, coord.to_string("hmsdms", precision=1))
    return coord, False


def refine_target_coord(coord: SkyCoord, cfg, index: Sequence[EpochIndex], instrument: dict,
                        n_epochs: int = 5, patterns: Optional[dict] = None,
                        astrometry_ref=None) -> tuple:
    """Centroid the transient near *coord* on a few epochs and adopt the median.

    Returns (coord, offset_arcsec). The refinement is rejected if it wants to
    move further than ``target.refine_radius_arcsec``, which stops it locking
    onto a neighbouring star when the supernova is undetected.
    """
    from photutils.centroids import centroid_quadratic

    max_shift = float(cfg.get("target.refine_radius_arcsec", 3.0))
    positions, offsets = [], []

    # Sample across the whole campaign rather than taking the first few frames,
    # which would all come from one band and one night.
    stride = max(1, len(index) // max(n_epochs * 4, 1))
    for entry in list(index)[::stride]:
        if len(positions) >= n_epochs:
            break
        try:
            image = prepare_image(entry.path, cfg, instrument,
                                  pattern=pattern_for(patterns, entry.band, entry.detector),
                                  astrometry_ref=astrometry_ref)
            if not image.contains(coord, margin=40):
                continue
            fwhm = image.fwhm_px or 4.0
            x0, y0 = image.pixel_of(coord)
            half = int(np.ceil(max(2.0 * fwhm, max_shift / image.meta.pixel_scale)))
            xi, yi = int(round(x0)), int(round(y0))
            stamp = image.data_sub[yi - half:yi + half + 1, xi - half:xi + half + 1]
            if stamp.size == 0 or not np.isfinite(stamp).all():
                continue
            peak = float(np.nanmax(stamp))
            if peak < 10.0 * float(np.nanmedian(image.bkg_rms)):
                continue                       # nothing convincing to centre on
            cx, cy = centroid_quadratic(stamp)
            if not (np.isfinite(cx) and np.isfinite(cy)):
                continue
            dx, dy = cx - half, cy - half
            shift = float(np.hypot(dx, dy) * image.meta.pixel_scale)
            if shift > max_shift:
                continue
            positions.append(image.wcs.pixel_to_world(x0 + dx, y0 + dy))
            offsets.append(shift)
        except Exception as exc:
            log.debug("position refinement skipped %s: %s", entry.path.name, exc)

    if len(positions) < 2:
        log.warning("could not refine the target position; using it as given")
        return coord, float("nan")

    refined = SkyCoord(np.median([c.ra.deg for c in positions]) * u.deg,
                       np.median([c.dec.deg for c in positions]) * u.deg)
    offset = float(coord.separation(refined).arcsec)
    log.info("refined the target position by %.2f\" using %d epochs -> %s",
             offset, len(positions), refined.to_string("hmsdms", precision=2))
    return refined, offset


# --------------------------------------------------------------------------
def prepare_image(path: Path, cfg, instrument: dict, pattern=None,
                  astrometry_ref=None) -> ScienceImage:
    """Load a frame and get it ready for photometry.

    Order matters: the detector pattern is removed before the background is
    measured (otherwise the fringes end up in the background model), and the
    WCS is refined before anything places an aperture through it.
    """
    image = ScienceImage.from_file(path, cfg, instrument, pattern=pattern)
    image.estimate_background(cfg)
    image.build_error_map()
    measure_fwhm(image, cfg)

    if astrometry_ref is not None and str(cfg.get("astrometry.mode", "linear")).lower() \
            not in ("off", "none"):
        try:
            sources = detect_sources(image, cfg, fwhm_px=image.fwhm_px)
            result = astrom.refine_wcs(image, sources, astrometry_ref, cfg)
            image.astrometry = result
            if result.ok and result.wcs is not None:
                image.wcs = result.wcs
                image.qa.update({"astrom_rms_arcsec": result.rms_after,
                                 "astrom_rms_before_arcsec": result.rms_before,
                                 "astrom_nstars": result.n_stars,
                                 "astrom_shift_arcsec": result.shift_arcsec,
                                 "astrom_mode": result.mode})
            else:
                image.qa["astrom_message"] = result.message
        except Exception as exc:
            log.warning("%s: WCS refinement failed (%s)", path.name, exc)

    return image


def load_patterns(cfg, bands: Sequence[str],
                  detectors: Optional[Iterable[Detector]] = None) -> dict:
    """Stored detector patterns, keyed by ``(band, detector)``.

    A pattern belongs to one readout, so a run spanning two of them needs two
    patterns per band. Missing ones are reported rather than silently skipped:
    an uncorrected z-band frame is about a magnitude shallower than a corrected
    one, which is worth knowing before reading the light curve.
    """
    if not bool(cfg.get("image.pattern.enabled", True)):
        return {}
    calib_root = _calib_root(cfg)
    detectors = list(detectors or [None])
    out: dict = {}
    missing: list[str] = []
    for detector in detectors:
        for band in bands:
            pattern = pat.load_pattern(calib_root, band, detector)
            if pattern is None:
                missing.append(f"{band} on {detector.label if detector else 'this detector'}")
                continue
            out[(band, detector)] = pattern
            log.info("%s on %s: stored pattern (%d frames, illumination %.2f%% pp, "
                     "fringe %.2f%% pp)", band,
                     detector.label if detector else "the detector", pattern.n_frames,
                     100 * pattern.illumination_amplitude, 100 * pattern.fringe_amplitude)
    if missing:
        log.warning("no detector pattern for %s -- those frames keep their fringes. "
                    "Build them with 'tttphot build-pattern'.", "; ".join(missing))
    if not out:
        log.info("no stored detector patterns; run 'tttphot build-pattern' to make them")
    return out


def pattern_for(patterns: dict, band: str, detector: Optional[Detector] = None):
    """Look a pattern up, tolerating a dict keyed the old way (band only)."""
    if not patterns:
        return None
    if (band, detector) in patterns:
        return patterns[(band, detector)]
    return patterns.get(band)


def _calib_root(cfg) -> Path:
    from .config import PIPELINE_ROOT
    value = cfg.get("image.pattern.calib_root", "calib")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PIPELINE_ROOT / path)


def load_astrometry_reference(cfg, coord: SkyCoord, cache_root: Path,
                              refresh: bool = False,
                              field_arcmin: Optional[float] = None):
    """Reference catalogue for the WCS refinement, with fallbacks.

    One catalogue serves the whole run, so when the run spans detectors it is
    sized for the largest of them.
    """
    mode = str(cfg.get("astrometry.mode", "linear")).lower()
    if mode in ("off", "none"):
        return None

    providers = [cfg.get("astrometry.catalog", "gaia_astrometry")]
    providers += list(cfg.get("astrometry.fallback_catalogs", []) or [])
    radius = det_mod.resolve_size(cfg.get("astrometry.search_radius_arcmin", "auto"),
                                  field_arcmin if field_arcmin else 9.6,
                                  fraction=1.05, lo=3.0, hi=30.0, default=10.0,
                                  step=1.0, rounding="up")
    local_path = cfg.resolve_path(cfg.get("calibration.local_catalog"))

    for provider in providers:
        try:
            table = cat.get_catalog(provider, float(coord.ra.deg), float(coord.dec.deg),
                                    radius, cache_root, refresh=refresh,
                                    local_path=local_path)
            if table is not None and len(table) >= int(cfg.get("astrometry.min_stars", 8)):
                log.info("astrometric reference: %s (%d sources)", provider, len(table))
                return table
        except Exception as exc:
            log.warning("astrometric reference %s unavailable: %s", provider, exc)
    log.warning("no astrometric reference catalogue; keeping the header WCS")
    return None


def build_references(cfg, index: Sequence[EpochIndex], instrument: dict, coord: SkyCoord,
                     target: str, cache_root: Path, bands: Sequence[str],
                     refresh: bool = False, patterns: Optional[dict] = None,
                     astrometry_ref=None, stamp_arcmin: Optional[float] = None) -> dict:
    """One reference image per band, built once and reused for every epoch."""
    references: dict[str, Optional[tpl.ReferenceImage]] = {}
    max_images = int(cfg.get("subtraction.ttt_stack.max_images", 8))
    mjd_min = cfg.get("subtraction.ttt_stack.mjd_min")
    mjd_max = cfg.get("subtraction.ttt_stack.mjd_max")

    for band in bands:
        order = cfg.get(f"subtraction.reference_sources.{band}") or \
            ["file", "ttt_stack", "panstarrs"]
        siblings: list[ScienceImage] = []
        if "ttt_stack" in order:
            if mjd_min is None and mjd_max is None:
                log.info("band %s: no template MJD window set, skipping ttt_stack "
                         "(set subtraction.ttt_stack.mjd_min/mjd_max to use the "
                         "target's own late-time epochs)", band)
            else:
                selection = [e for e in index if e.band == band
                             and (mjd_min is None or e.mjd >= float(mjd_min))
                             and (mjd_max is None or e.mjd <= float(mjd_max))]
                if len(selection) > max_images:
                    selection = selection[:max_images]
                for entry in selection:
                    try:
                        sibling = prepare_image(
                            entry.path, cfg, instrument,
                            pattern=pattern_for(patterns, band, entry.detector),
                            astrometry_ref=astrometry_ref)
                        # Every epoch of the stack is held at once. The stack
                        # reads the pixels, the mask, the background model and
                        # the WCS; the error map, the background RMS and the
                        # cosmic-ray mask are three more full-frame arrays per
                        # epoch that nothing downstream of here looks at.
                        sibling.error = None
                        sibling.bkg_rms = None
                        sibling.crmask = None
                        siblings.append(sibling)
                    except Exception as exc:
                        log.warning("template epoch %s failed: %s", entry.path.name, exc)

        try:
            references[band] = tpl.get_reference(
                _FakeImage(band), cfg, cache_root, coord, siblings, target,
                refresh=refresh, stamp_arcmin=stamp_arcmin)
            references[band] = tpl.register_reference(references[band], astrometry_ref, cfg)
        except Exception as exc:
            log.warning("reference for band %s failed: %s", band, exc)
            references[band] = None
        siblings.clear()
    return references


class _FakeImage:
    """Minimal stand-in so templates.get_reference can be called before any
    science frame is loaded (it only needs the band and the MJD to exclude)."""

    class _Meta:
        def __init__(self, band):
            self.band = band
            self.mjd = float("nan")

    def __init__(self, band: str):
        self.meta = self._Meta(band)


# --------------------------------------------------------------------------
def process_image(image: ScienceImage, cfg, coord: SkyCoord, cache_root: Path,
                  reference: Optional[tpl.ReferenceImage], outdir: Path,
                  target: str, refresh: bool = False) -> list[dict]:
    """All measurements for one frame: returns one row per photometric method."""
    band = image.meta.band
    fwhm = image.fwhm_px or 4.0
    ann = cfg.get("photometry.annulus_scale", [5.0, 8.0])
    annulus = (ann[0] * fwhm, ann[1] * fwhm)
    ap_radius = float(cfg.get("photometry.aperture_scale", 1.2)) * fwhm

    common = dict(
        target=target, mjd=image.meta.mjd, band=band, system=image.meta.system,
        fwhm_arcsec=image.fwhm_arcsec, sky_adu=image.qa.get("sky_adu"),
        sky_rms_adu=image.qa.get("sky_rms_adu"), exptime=image.meta.exptime,
        airmass=image.meta.airmass, aperture_radius_px=ap_radius,
        quality=str(image.meta.extra.get("quality", "")), image=image.meta.path.name,
        astrom_rms_arcsec=image.qa.get("astrom_rms_arcsec", np.nan),
        astrom_nstars=image.qa.get("astrom_nstars", np.nan),
        detector=identify(image.header).label,
        # The camera on its own, because the per-camera zero-point transfer
        # needs it and used to recover it by splitting `detector` on whitespace
        # -- a silent coupling to a label format defined in another module.
        camera=identify(image.header).instrument,
        pixel_scale=image.meta.pixel_scale,
        rebin=image.qa.get("rebin", 1),
        gain_eff=image.meta.gain,
        sky_noise_ratio=image.qa.get("sky_noise_ratio", np.nan),
        obs_zp=image.meta.extra.get("obs_zp", np.nan),
        obs_zp_err=image.meta.extra.get("obs_zp_err", np.nan),
        obs_maglim=image.meta.extra.get("obs_maglim", np.nan),
        obs_seeing_arcsec=image.meta.extra.get("obs_seeing", np.nan),
        pattern_applied=bool(image.qa.get("pattern_applied", False)),
        fringe_scale_adu=image.qa.get("pattern_fringe_scale_adu", np.nan),
    )

    sources = detect_sources(image, cfg, fwhm_px=fwhm)
    psf_stars = select_psf_stars(image, sources, cfg, fwhm,
                                 max_stars=int(cfg.get("detection.max_stars_for_fwhm", 40)),
                                 min_snr=float(cfg.get("calibration.min_snr", 20.0)))
    psf_stars = filter_by_profile(image, psf_stars, fwhm, cfg)
    apcor = ph.curve_of_growth(image, psf_stars, fwhm, cfg)
    zp = calibrate_image(image, sources, fwhm, cfg, cache_root,
                         apcor_mag=apcor.apcor_mag, refresh=refresh)

    common.update(
        apcor=apcor.apcor_mag, apcor_err=apcor.apcor_err,
        zp=zp.zp, zp_err=zp.zp_err, color_term=zp.color_term,
        color_term_err=zp.color_term_err, cal_provider=zp.provider,
        cal_nstars=zp.n_stars, cal_rms=zp.rms,
    )

    limit = ph.empirical_limit(image, fwhm, zp.zp, cfg, apcor.apcor_mag) if zp.ok else np.nan


    cal_flags = 0
    if reference is not None:
        lo, hi = reference.mjd_range
        if np.isfinite(lo) and np.isfinite(hi) and lo - 1e-6 <= image.meta.mjd <= hi + 1e-6:
            cal_flags |= lc.FLAG_IN_TEMPLATE
            log.warning("%s is itself part of the reference image; its difference is null "
                        "by construction", image.meta.path.name)
    if not zp.ok:
        cal_flags |= lc.FLAG_NO_ZEROPOINT
    else:
        if zp.n_stars < 2 * int(cfg.get("calibration.min_stars", 5)):
            cal_flags |= lc.FLAG_FEW_CAL_STARS
        if np.isfinite(zp.rms) and zp.rms > float(cfg.get("calibration.max_rms", 0.25)):
            cal_flags |= lc.FLAG_POOR_CALIBRATION
        if getattr(zp, "relaxed", False):
            cal_flags |= lc.FLAG_RELAXED_CALIBRATION
        if not zp.meta.get("variables_screened", True):
            cal_flags |= lc.FLAG_VARIABLES_UNSCREENED
            log.warning("%s %s: the variable-star screen could not be run, so the "
                        "zero point may rest on a variable", image.meta.path.name, band)
    common["flags_base"] = cal_flags

    sn_xy = image.pixel_of(coord)
    rows: list[dict] = []

    # ---------------------------------------------------------------- direct
    tab = ph.measure_apertures(image, [sn_xy], ap_radius, cfg, annulus_px=annulus)
    sky_sigma = ph.blank_aperture_noise(image, ap_radius, cfg) \
        if bool(cfg.get("photometry.empirical_errors", True)) else float("nan")
    aper_err = ph.combine_errors(float(tab["flux"][0]), float(tab["flux_err"][0]),
                                 sky_sigma, image.meta.gain)
    rows.append(_photometry_row(common, "aperture", float(tab["flux"][0]),
                                aper_err, zp, apcor.apcor_mag,
                                image, x=sn_xy[0], y=sn_xy[1],
                                flags=int(tab["flags"][0]), limit=limit, cfg=cfg))

    epsf, n_psf = ph.build_epsf(image, psf_stars, fwhm, cfg)

    # Injection measures the depth of this frame as this code actually measures
    # it, on the background the transient really sits on. Where it disagrees
    # with the blank-aperture estimate, it is the one to believe.
    if zp.ok and bool(cfg.get("calibration.injection.enabled", True)):
        from . import injection
        try:
            sx, sy = image.wcs.world_to_pixel(coord)
            inj = injection.measure_limit(image, epsf, fwhm, zp.zp, cfg,
                                          (float(sx), float(sy)),
                                          apcor_mag=apcor.apcor_mag)
            common.update(inj.as_dict())
            if np.isfinite(inj.limit_mag):
                log.info("%s: injected limit %.2f (blank-aperture %.2f)%s",
                         image.meta.path.name, inj.limit_mag, limit,
                         f", errors x{inj.error_scale:.2f}"
                         if np.isfinite(inj.error_scale) else "")
                limit = inj.limit_mag
            elif inj.message:
                log.debug("%s: no injected limit (%s)", image.meta.path.name, inj.message)
        except Exception as exc:
            log.warning("%s: injection limit failed (%s); keeping the blank-aperture "
                        "estimate", image.meta.path.name, exc)
    psf_corr = 0.0
    if epsf is not None and bool(cfg.get("photometry.psf.enabled", True)):
        psf_corr = ph.psf_to_aperture_correction(image, epsf, psf_stars, fwhm, cfg,
                                                 apcor_mag=apcor.apcor_mag)
        fit = ph.psf_photometry_at(image, epsf, [sn_xy], fwhm, cfg)
        if len(fit):
            psf_err = ph.combine_errors(float(fit["flux"][0]), float(fit["flux_err"][0]),
                                        sky_sigma, image.meta.gain)
            rows.append(_photometry_row(common, "psf", float(fit["flux"][0]),
                                        psf_err, zp, psf_corr, image,
                                        x=float(fit["x"][0]), y=float(fit["y"][0]),
                                        dx=float(fit["dx"][0]), dy=float(fit["dy"][0]),
                                        flags=int(tab["flags"][0]), limit=limit, cfg=cfg))

    # ------------------------------------------------------------ difference
    diff = None
    if reference is not None and bool(cfg.get("subtraction.enabled", True)):
        diff = sub.subtract(image, reference, cfg, coord, fwhm_px=fwhm)
        if diff.ok:
            rows.extend(_difference_rows(diff, image, cfg, coord, common, zp, limit,
                                         science_epsf=epsf, science_apcor=apcor.apcor_mag,
                                         science_psf_corr=psf_corr, sources=sources,
                                         science_flags=int(tab["flags"][0])))

    if bool(cfg.get("output.write_qa_plots", True)):
        plots.epoch_qa(Path(outdir) / "qa" /
                       f"{slugify(image.meta.path.stem)}.png",
                       image, zp, apcor, diff=diff, sn_xy=sn_xy, cutout=coord,
                       stamp_px=int(float(cfg.get("output.stamp_arcsec", 60.0)) /
                                    max(image.meta.pixel_scale, 1e-3) / 2))

    preference = str(cfg.get("output.primary_method", "auto"))
    best = lc.choose_primary(rows, preference, _host_guard(cfg))
    if best is not None:
        best["primary"] = True
    return rows


def _difference_rows(diff, image, cfg, coord, common, zp, limit,
                     science_epsf=None, science_apcor: float = 0.0,
                     science_psf_corr: float = 0.0, sources=None,
                     science_flags: int = 0) -> list[dict]:
    """Aperture and PSF photometry on the difference image.

    When the reference was the frame that got convolved, the difference keeps
    the science PSF exactly, so the PSF model and aperture correction already
    measured on the full science frame are reused -- far more reliable than
    rebuilding them from the handful of stars inside the stamp.

    *science_flags* are the aperture flags from the science frame. Only the
    saturation bit is carried over: difference pixels scatter about zero, so a
    saturation test on the stamp can never fire, and a saturated transient would
    otherwise reach the light curve through an unflagged difference row. Masked
    pixels need no such help -- they propagate as NaN into the difference and
    the stamp's own mask finds them.
    """
    rows: list[dict] = []
    science_flags = int(science_flags) & lc.FLAG_SATURATED
    stamp = _DifferenceImage(diff, image)
    fwhm = _matched_fwhm(diff, image)
    ann = cfg.get("photometry.annulus_scale", [5.0, 8.0])
    annulus = (ann[0] * fwhm, ann[1] * fwhm)
    ap_radius = float(cfg.get("photometry.aperture_scale", 1.2)) * fwhm

    x, y = diff.wcs.world_to_pixel(coord)
    sn_xy = (float(x), float(y))

    max_chi2 = float(cfg.get("subtraction.max_chi2", 2.5))
    # A missing chi2 is not a passing chi2. The kernel solve produces one only
    # when it converged on enough stamps, so NaN means the PSF match could not
    # be assessed at all -- exactly the case most likely to be wrong. Reading it
    # as "fine" let three SN2022xkq epochs through with 0.7 to 1.1 mag errors
    # and no flag.
    bad_subtraction = (lc.FLAG_BAD_SUBTRACTION
                       if (not np.isfinite(diff.chi2) or diff.chi2 > max_chi2)
                       else 0)
    ref_band = diff.meta.get("reference_band")
    if ref_band and ref_band != image.meta.band:
        bad_subtraction |= lc.FLAG_CROSS_BAND_TEMPLATE

    # The kernel chi2 above is measured on bright field stars. This is measured
    # on the shape of what is left behind, which is what distinguishes a
    # misregistered reference from a merely mis-scaled one.
    from . import diffquality
    field_xy = []
    if sources is not None and len(sources) and diff.wcs is not None:
        try:
            fx, fy = diff.wcs.world_to_pixel(SkyCoord(
                np.asarray(sources["ra"]) * u.deg, np.asarray(sources["dec"]) * u.deg))
            field_xy = np.column_stack([fx, fy])
        except Exception:
            field_xy = []
    quality = diffquality.assess(diff, field_xy, fwhm, cfg, target_xy=sn_xy)
    if quality.verdict == "fail":
        bad_subtraction |= lc.FLAG_DIPOLE_RESIDUAL | lc.FLAG_BAD_SUBTRACTION
    elif quality.verdict == "downgrade":
        bad_subtraction |= lc.FLAG_DIPOLE_RESIDUAL
    if quality.message:
        log.info("%s: difference %s -- %s", image.meta.path.name,
                 quality.verdict, quality.message)

    diff_meta = dict(
        ref_source=diff.meta.get("reference_source"),
        ref_band=diff.meta.get("reference_band"),
        ref_nimages=diff.meta.get("reference_n_images"),
        diff_mode=diff.mode, diff_chi2=diff.chi2,
        diff_kernel_sum=diff.kernel_sum, diff_nstamps=diff.n_stamps,
        diff_kernel_dx=diff.meta.get("kernel_dx", np.nan),
        diff_kernel_dy=diff.meta.get("kernel_dy", np.nan),
        **quality.as_dict(),
    )

    # Aperture correction measured on the matched image, which shares the PSF
    # of the difference image.
    matched = _MatchedImage(diff, image)
    reuse_science_psf = (diff.mode == "convolve_reference" and science_epsf is not None)

    matched_stars, apcor = None, None
    if reuse_science_psf:
        apcor_mag = float(science_apcor)
    else:
        from .detection import detect_sources as _detect, select_psf_stars as _select
        try:
            matched_sources = _detect(matched, cfg, fwhm_px=fwhm)
            matched_stars = _select(matched, matched_sources, cfg, fwhm, max_stars=30,
                                    min_snr=20.0)
            matched_stars = filter_by_profile(matched, matched_stars, fwhm, cfg)
            apcor = ph.curve_of_growth(matched, matched_stars, fwhm, cfg)
            if apcor.n_stars < int(cfg.get("photometry.min_apcor_stars", 5)):
                apcor = None      # too few stars to trust; fall back to no correction
        except Exception:
            matched_stars, apcor = None, None
        apcor_mag = apcor.apcor_mag if apcor is not None else 0.0
    scale = diff.flux_scale if np.isfinite(diff.flux_scale) and diff.flux_scale > 0 else 1.0

    tab = ph.measure_apertures(stamp, [sn_xy], ap_radius, cfg, annulus_px=annulus)
    sky_sigma = ph.blank_aperture_noise(stamp, ap_radius, cfg) \
        if bool(cfg.get("photometry.empirical_errors", True)) else float("nan")
    diff_err = ph.combine_errors(float(tab["flux"][0]), float(tab["flux_err"][0]),
                                 sky_sigma, image.meta.gain * scale)
    # Imperfect host subtraction around the transient, as an extra flux error.
    host = diffquality.host_residual(diff.difference, diff.reference_stamp, sn_xy, fwhm, cfg,
                                     noise_per_aperture=sky_sigma)
    ap_flux = float(tab["flux"][0])
    diff_meta["diff_host_residual"] = (host["sigma"] / ap_flux
                                       if np.isfinite(ap_flux) and ap_flux > 0 else np.nan)
    diff_meta["diff_host_nsites"] = host["n"]
    # Kept apart from the statistical error: it widens mag_err but does not
    # decide whether the transient is detected (a bright SN on a poorly
    # subtracted host is still there; only its magnitude is less certain).
    host_sys = float(host["sigma"]) * float(cfg.get("subtraction.host_residual.error_scale", 0.5)) \
        if np.isfinite(host["sigma"]) else 0.0
    rows.append(_photometry_row({**common, **diff_meta}, "difference_aperture",
                                float(tab["flux"][0]) / scale,
                                diff_err / scale, zp, apcor_mag, image,
                                x=sn_xy[0], y=sn_xy[1],
                                flags=int(tab["flags"][0]) | bad_subtraction | science_flags,
                                limit=limit, flux_err_sys=host_sys / scale, cfg=cfg))

    if not bool(cfg.get("photometry.psf.enabled", True)):
        return rows

    if reuse_science_psf:
        epsf, psf_corr = science_epsf, float(science_psf_corr)
    elif matched_stars is not None and len(matched_stars):
        epsf, _ = ph.build_epsf(matched, matched_stars, fwhm, cfg)
        psf_corr = ph.psf_to_aperture_correction(matched, epsf, matched_stars, fwhm, cfg,
                                                 apcor_mag=apcor_mag) if epsf is not None else 0.0
    else:
        epsf, psf_corr = None, 0.0

    if epsf is not None:
        fit = ph.psf_photometry_at(stamp, epsf, [sn_xy], fwhm, cfg)
        if len(fit):
            dpsf_err = ph.combine_errors(float(fit["flux"][0]), float(fit["flux_err"][0]),
                                         sky_sigma, image.meta.gain * scale)
            psf_sys = 0.0
            if host_sys > 0 and np.isfinite(ap_flux) and ap_flux > 0:
                # the same fractional residual, on the PSF (total-flux) scale
                psf_sys = host_sys * abs(float(fit["flux"][0])) / ap_flux
            rows.append(_photometry_row({**common, **diff_meta}, "difference_psf",
                                        float(fit["flux"][0]) / scale,
                                        dpsf_err / scale, zp, psf_corr,
                                        image, x=float(fit["x"][0]), y=float(fit["y"][0]),
                                        dx=float(fit["dx"][0]), dy=float(fit["dy"][0]),
                                        flags=int(tab["flags"][0]) | bad_subtraction | science_flags,
                                        limit=limit, flux_err_sys=psf_sys / scale, cfg=cfg))
    return rows


def _matched_fwhm(diff, image) -> float:
    value = diff.meta.get("reference_fwhm_px", np.nan)
    science = image.fwhm_px or 4.0
    if diff.mode == "convolve_science" and np.isfinite(value):
        return float(value)
    return float(science)


class _StampBase:
    """Adapter presenting a stamp array with the ScienceImage interface."""

    def __init__(self, data, variance, wcs, image, mask=None):
        self.data = np.nan_to_num(data, nan=0.0)
        self.wcs = wcs
        self.meta = image.meta
        self.header = image.header
        self.mask = (~np.isfinite(data)) if mask is None else mask
        self.bkg_map = np.zeros_like(self.data)
        with np.errstate(invalid="ignore"):
            rms = np.sqrt(np.abs(variance))
        median_rms = float(np.nanmedian(rms[np.isfinite(rms)])) if np.isfinite(rms).any() else 1.0
        self.bkg_rms = np.where(np.isfinite(rms) & (rms > 0), rms, median_rms)
        self.error = self.bkg_rms.copy()
        self.fwhm_px = image.fwhm_px
        self.fwhm_arcsec = image.fwhm_arcsec
        self.qa = {"sky_adu": 0.0, "sky_rms_adu": median_rms}

    @property
    def data_sub(self):
        return self.data

    def build_error_map(self):
        return self.error

    def pixel_of(self, coord):
        x, y = self.wcs.world_to_pixel(coord)
        return float(x), float(y)

    def center_coord(self):
        ny, nx = self.data.shape
        return self.wcs.pixel_to_world(nx / 2.0, ny / 2.0)


class _DifferenceImage(_StampBase):
    def __init__(self, diff, image):
        super().__init__(diff.difference, diff.variance, diff.wcs, image)


class _MatchedImage(_StampBase):
    def __init__(self, diff, image):
        super().__init__(diff.reference_stamp, diff.variance, diff.wcs, image)


def _photometry_row(common: dict, method: str, flux: float, flux_err: float,
                    zp: ZeroPoint, apcor_mag: float, image, x=np.nan, y=np.nan,
                    dx=np.nan, dy=np.nan, flags: int = 0, limit: float = np.nan,
                    flux_err_sys: float = 0.0, cfg=None) -> dict:
    """Turn a flux measurement into a calibrated row.

    *flux_err* is the statistical error and alone sets the SNR, hence whether
    the row is a detection or a limit. *flux_err_sys* (same units) is added in
    quadrature to ``mag_err`` only.
    """
    common = dict(common)
    flags = int(flags) | int(common.pop("flags_base", 0))
    exptime = max(image.meta.exptime, 1e-6)
    snr = flux / flux_err if (np.isfinite(flux_err) and flux_err > 0) else np.nan
    detection_snr = float(cfg.get("output.detection_snr", 3.0)) if cfg is not None else 3.0

    mag = np.nan
    mag_err = np.nan
    is_limit = True
    if zp.ok and np.isfinite(flux) and flux > 0 and np.isfinite(snr) and snr >= detection_snr:
        inst = -2.5 * np.log10(flux / exptime) + apcor_mag
        mag = inst + zp.zp
        sys = flux_err_sys if np.isfinite(flux_err_sys) else 0.0
        mag_err = float(np.hypot(1.0857362 * np.hypot(flux_err, sys) / flux, zp.zp_err))
        is_limit = False

    return lc.new_row(
        **common, method=method, mag=mag, mag_err=mag_err, is_limit=is_limit,
        mag_offset=apcor_mag, zp_source=("fit" if zp.ok else ""),
        flux=flux / exptime, flux_err=flux_err / exptime, snr=snr,
        flux_err_sys=(flux_err_sys / exptime if np.isfinite(flux_err_sys) else np.nan),
        x=x, y=y, dx=dx, dy=dy, flags=flags, limit_mag=limit,
    )


# --------------------------------------------------------------------------
def _host_guard(cfg):
    """``output.host_guard_mag`` as a float, or None when switched off."""
    value = cfg.get("output.host_guard_mag", lc.HOST_GUARD_MAG)
    if value in (None, False) or str(value).lower() in ("off", "none", "false"):
        return None
    return float(value)


def _kernel_sum_comparable(frame: pd.DataFrame, cfg) -> tuple:
    """Kernel sums put on a scale on which different epochs can be compared.

    Returns ``(values, group_keys)``, both indexed like *frame*.

    The raw kernel sum is the ratio of science to reference counts per pixel,
    so it depends on everything that sets the science frame's counts: the
    camera's sensitivity (the zero point), the exposure time, the pixel area
    (a rebinned pixel collects four times the sky), and on which way round the
    kernel was solved (in ``convolve_science`` mode it maps science onto the
    reference, so it is the reciprocal). On a single-camera archive those are
    nearly constant and the raw sum could be compared directly. TTT mixes
    cameras on the same target -- on SN 2026abgb the QHY600 frames come out at
    0.12-0.17 in g and the iKon936 frames at 0.78-0.92 -- and a raw comparison
    flagged every frame of the minority camera as a failed subtraction.

    Dividing out exposure time, 10**(0.4 zp) and pixel area leaves a number set
    only by the reference, the same for every camera (and for every night,
    since transparency moves the zero point and the kernel sum together).
    Epochs with no zero point cannot be normalised; they are compared only with
    uncalibrated epochs of the same detector configuration.
    """
    ks = pd.to_numeric(frame["diff_kernel_sum"], errors="coerce")
    mode = frame.get("diff_mode", pd.Series("", index=frame.index)).astype(str)
    with np.errstate(divide="ignore", invalid="ignore"):
        eff = np.where(mode == "convolve_science", 1.0 / ks, ks)
    eff = pd.Series(eff, index=frame.index)

    scale = pd.to_numeric(frame.get("pixel_scale", pd.Series(np.nan, index=frame.index)),
                          errors="coerce")
    if scale.isna().any():
        # Tables written before the pixel_scale column: the aperture radius is
        # aperture_scale x FWHM in pixels, and the FWHM is also given in arcsec.
        ap_scale = float(cfg.get("photometry.aperture_scale", 1.2))
        radius = pd.to_numeric(frame.get("aperture_radius_px"), errors="coerce")
        fwhm = pd.to_numeric(frame.get("fwhm_arcsec"), errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = scale.fillna(ap_scale * fwhm / radius)

    zp = pd.to_numeric(frame.get("zp"), errors="coerce")
    exptime = pd.to_numeric(frame.get("exptime"), errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        norm = eff / (exptime * 10.0 ** (0.4 * zp) * scale ** 2)
    calibrated = np.isfinite(norm) & (norm > 0)

    detector = frame.get("detector", pd.Series("", index=frame.index)).astype(str)
    values = pd.Series(np.where(calibrated, norm, eff), index=frame.index)
    keys = pd.Series(np.where(calibrated, "calibrated", detector), index=frame.index)
    return values, keys


def _flag_collapsed_kernels(frame, cfg):
    """Flag difference measurements whose kernel solve collapsed.

    The kernel sum is the flux scaling from reference to science. It is set by
    the depth and units of the template, so there is no absolute value to test
    against -- but every epoch of a band against the same reference should get
    roughly the same one, once the camera, exposure and pixel size are divided
    out (see ``_kernel_sum_comparable``). An epoch that comes out far away has
    not found a different scaling, it has failed to find one, and the flux
    measured through it means nothing.

    On SN2022xkq (TJO) this is what separated the difference measurements that
    agree with an independent reduction from those off by more than 0.3 mag:
    median kernel sum 20.3 against 0.027. It also explains their suspiciously
    *low* chi2 (0.36 against 5.5) -- a collapsed solve leaves an inflated noise
    model, so the frame looks well fitted precisely because it was not fitted.
    """
    if not bool(cfg.get("subtraction.check_kernel_sum", True)) or frame.empty:
        return frame
    if "diff_kernel_sum" not in frame.columns:
        return frame
    tol = float(cfg.get("subtraction.kernel_sum_tolerance", 3.0))
    if tol <= 1:
        return frame

    values, keys = _kernel_sum_comparable(frame, cfg)
    is_diff = frame["method"].astype(str).str.startswith("difference")
    ref_source = frame.get("ref_source", pd.Series("", index=frame.index)).fillna("")
    flagged = 0
    sub_frame = frame[is_diff]
    for key, idx in sub_frame.groupby([sub_frame["band"], ref_source[is_diff],
                                       keys[is_diff]]).groups.items():
        vals = values.loc[idx]
        good = vals[np.isfinite(vals) & (vals > 0)]
        # One image contributes two rows (PSF and aperture), so count images.
        n_images = frame.loc[good.index, "image"].nunique() if "image" in frame else len(good)
        if n_images < 5:
            continue
        ref = float(np.median(good))
        if not np.isfinite(ref) or ref <= 0:
            continue
        bad = idx[(~np.isfinite(vals)) | (vals <= 0)
                  | (vals > ref * tol) | (vals < ref / tol)]
        if len(bad):
            frame.loc[bad, "flags"] = (frame.loc[bad, "flags"].fillna(0).astype(int)
                                       | lc.FLAG_BAD_SUBTRACTION | lc.FLAG_KERNEL_SUM)
            flagged += len(bad)
            log.info("%s %s (%s): %d of %d difference rows have a kernel sum more than "
                     "x%.1f from the band median; flagged as bad subtractions",
                     key[0], key[1] or "reference",
                     "normalised" if key[2] == "calibrated" else key[2],
                     len(bad), len(idx), tol)
    if flagged:
        frame = lc.reselect_primary(frame, str(cfg.get("output.primary_method", "auto")),
                                     _host_guard(cfg))
    return frame


def run(cfg, target: str, bands: Optional[Sequence[str]] = None,
        max_images: Optional[int] = None, refresh: bool = False,
        skip_subtraction: bool = False, append: bool = False,
        skip: int = 0) -> dict:
    """Full pipeline for one target. Returns a dict of output paths."""
    started = time.time()
    instrument = cfg.get("instrument_config", {})
    outdir = ensure_dir(Path(cfg.work_root) / target)
    cache_root = ensure_dir(cfg.cache_root)

    paths = find_images(cfg, target)
    if not paths:
        raise UserError(f"no FITS files found for {target} under {cfg.data_root}")
    full_index = index_images(paths, instrument, target=None, cfg=cfg)
    if bands:
        full_index = [e for e in full_index if e.band in bands]
    # Templates are always built from every available epoch, even when the run
    # itself is restricted with --max-images.
    index = full_index[int(skip):] if skip else list(full_index)
    if max_images:
        index = index[:max_images]
    if not index:
        raise UserError("no epochs left after filtering")

    present = sorted({e.band for e in index})
    log.info("%s: %d images, bands %s, MJD %.3f-%.3f", target, len(index), ",".join(present),
             min(e.mjd for e in index), max(e.mjd for e in index))

    coord, explicit = resolve_target_coord(cfg, full_index, instrument)

    mode = str(cfg.get("target.resolve", "auto")).lower()
    if mode == "always" or (mode == "auto" and not explicit):
        found = nameresolve.resolve(target, cfg)
        if found is not None:
            if explicit:
                moved = coord.separation(found.coord).arcsec
                log.info("%s moves the position by %.2f\"", found.source.upper(), moved)
            coord, explicit = found.coord, True
            if bool(cfg.get("target.resolve_write", True)):
                try:
                    nameresolve.update_target_file(target, found)
                except Exception as exc:
                    log.warning("could not update the target file: %s", exc)
        elif not explicit:
            log.info("name resolution found nothing; falling back to the telescope pointing")

    detectors = sorted({e.detector for e in index if e.detector is not None},
                       key=lambda d: d.token)
    if len(detectors) > 1:
        log.info("%d detector configurations in this run: %s", len(detectors),
                 "; ".join(f"{d.label} ({sum(1 for e in index if e.detector == d)} frames)"
                           for d in detectors))
    patterns = load_patterns(cfg, present, detectors)
    # Size the astrometric reference for the widest frame in the run. The
    # nominal plate scale from each detector's instrument file is enough here:
    # this only sets how much sky to download.
    def _nominal_scale(det) -> float:
        inst = cfg.instrument_for({"INSTRUME": det.instrument}) \
            if hasattr(cfg, "instrument_for") else instrument
        return float((inst.get("defaults", {}) or {}).get("pixel_scale", 0.4)) * det.binning

    widest = max((det_mod.field_radius_arcmin(d.nx, d.ny, _nominal_scale(d))
                  for d in detectors), default=10.0)
    # One reference stamp per band for the whole run, sized for its widest frame.
    reference_stamp = max((det_mod.stamp_arcmin(cfg, detector=d, pixel_scale=_nominal_scale(d))
                           for d in detectors), default=det_mod.stamp_arcmin(cfg))
    astrometry_ref = load_astrometry_reference(cfg, coord, cache_root, refresh=refresh,
                                               field_arcmin=widest)

    refine = str(cfg.get("target.refine", "auto")).lower()
    offset = float("nan")
    if refine == "always" or (refine == "auto" and not explicit):
        coord, offset = refine_target_coord(coord, cfg, full_index, instrument,
                                            patterns=patterns, astrometry_ref=astrometry_ref)
    elif not explicit:
        log.warning("using the telescope pointing as the SN position without refinement; "
                    "set target.ra / target.dec for precise forced photometry")
    log.info("SN position: %s", coord.to_string("hmsdms", precision=2))

    references: dict = {}
    if bool(cfg.get("subtraction.enabled", True)) and not skip_subtraction:
        references = build_references(cfg, full_index, instrument, coord, target, cache_root,
                                      present, refresh=refresh, patterns=patterns,
                                      astrometry_ref=astrometry_ref,
                                      stamp_arcmin=reference_stamp)

    rows: list[dict] = []
    failures: list[dict] = []
    for k, entry in enumerate(index, 1):
        log.info("[%d/%d] %s", k, len(index), entry.path.name)
        try:
            image = prepare_image(entry.path, cfg, instrument,
                                  pattern=pattern_for(patterns, entry.band, entry.detector),
                                  astrometry_ref=astrometry_ref)
            if image.wcs is None:
                raise RuntimeError("no astrometric solution in the header, so the target "
                                   "cannot be placed (the observatory pipeline failed on "
                                   "this frame; its sky level is %.0f ADU)"
                                   % float(image.qa.get("sky_adu", np.nan)))
            if not image.contains(coord, margin=edge_margin_px(cfg, image.meta.pixel_scale)):
                centre = image.center_coord()
                raise RuntimeError("the target position falls outside this frame, whose "
                                   "centre is %.1f arcmin away (%s)" % (
                                       centre.separation(coord).arcmin,
                                       centre.to_string("hmsdms", precision=0)))
            rows.extend(process_image(image, cfg, coord, cache_root,
                                      references.get(entry.band), outdir, target,
                                      refresh=refresh))
        except Exception as exc:
            log.error("%s failed: %s", entry.path.name, exc)
            failures.append({"image": entry.path.name, "error": str(exc)})
        finally:
            image = None

    if not rows:
        raise UserError("no photometry could be measured")

    frame = lc.to_frame(rows)
    frame = _flag_collapsed_kernels(frame, cfg)

    if bool(cfg.get("calibration.transfer_zeropoints", True)):
        frame, transfer_report = zeropoints.transfer(frame, cfg)
        if transfer_report.get("recovered"):
            frame = lc.reselect_primary(frame, str(cfg.get("output.primary_method", "auto")),
                                     _host_guard(cfg))
    else:
        transfer_report = {}

    paths_out = lc.write(frame, outdir, target, append=append,
                         preference=str(cfg.get("output.primary_method", "auto")))
    if append:
        import pandas as pd
        frame = pd.read_csv(paths_out["all"])

    if bool(cfg.get("output.write_qa_plots", True)):
        plots.lightcurve(outdir / f"{target}_lightcurve.png", frame, target)
        plots.run_diagnostics(outdir / f"{target}_diagnostics.png", frame, target)

    summary = {
        "target": target,
        "config_files": cfg.get("_config_files", []),
        "n_images": len(index),
        "n_rows": len(frame),
        "n_failures": len(failures),
        "failures": failures,
        "bands": present,
        "coord": coord.to_string("hmsdms", precision=3),
        "position_refined_by_arcsec": None if not np.isfinite(offset) else round(offset, 3),
        "position_from_config": bool(explicit),
        "elapsed_s": round(time.time() - started, 1),
        "zeropoint_transfer": {
            "recovered": len(transfer_report.get("recovered", [])),
            "unrecovered": len(transfer_report.get("unrecovered", [])),
            "band_offsets": transfer_report.get("offsets", {}),
        },
        "references": {
            band: (None if ref is None else {
                "source": ref.source,
                "band": ref.source_band or ref.band,
                "path": str(ref.path) if ref.path else None,
                "pixel_scale_arcsec": round(float(ref.pixel_scale), 4),
                "n_images": ref.n_images,
                "registered_shift_arcsec": (round(float(ref.registration.shift_arcsec), 3)
                                            if getattr(ref, "registration", None) else None),
            })
            for band, ref in references.items()
        },
    }
    with open(outdir / "run_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    cfg.dump(outdir / "config_used.yaml")

    log.info("done in %.1f s -- %s", summary["elapsed_s"], outdir)
    print("\n" + lc.summarise(frame) + "\n")
    return {**paths_out, "outdir": outdir, "summary": summary, "frame": frame}
