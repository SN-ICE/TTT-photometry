"""Reference ("template") images for difference imaging.

Three sources, tried in the order given by ``subtraction.reference_sources``:

``file``       an explicit FITS path per band, set in the target file. Best when
               you have a proper pre- or post-explosion image.
``ttt_stack``  a stack of the target's own TTT epochs, selected by MJD range.
               Same telescope, same filters, same pixel scale -- by far the
               cleanest option once the supernova has faded.
``panstarrs``  a PS1 DR2 stacked cutout from MAST (g,r,i,z,y; Dec > -30).
``hips``       any public survey published as a HiPS, rendered to a FITS cutout
               by the CDS hips2fits service. This is the general escape hatch:
               it is all-sky, it returns exactly the grid you ask for (so no
               survey tiling shows through), and it reaches surveys that have
               no cutout API of their own -- including DSS2 blue and red, which
               are the only wide-field photographic B- and R-like images
               covering the whole sky. Configured per band in YAML.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import astropy.units as u

from . import detector
from .utils import band_token, ensure_dir, get_logger, slugify

log = get_logger("tttphot.templates")


@dataclass
class ReferenceImage:
    data: np.ndarray
    wcs: WCS
    band: str                         # the science band this reference serves
    source: str                       # 'file' | 'ttt_stack' | 'panstarrs'
    pixel_scale: float
    source_band: Optional[str] = None  # the band the reference was actually taken in
    header: fits.Header = field(default_factory=fits.Header)
    mask: Optional[np.ndarray] = None
    variance: Optional[np.ndarray] = None
    n_images: int = 1
    mjd_range: tuple[float, float] = (float("nan"), float("nan"))
    path: Optional[Path] = None
    # Set by register_reference when the WCS is re-tied to the astrometric
    # catalogue; an AstrometryResult, declared here rather than attached to the
    # instance out of nowhere.
    registration: object = None

    def cutout(self, coord: SkyCoord, size_arcmin: float) -> "ReferenceImage":
        size = (size_arcmin * u.arcmin, size_arcmin * u.arcmin)
        cut = Cutout2D(self.data, coord, size, wcs=self.wcs, mode="partial", fill_value=np.nan)
        mask = None
        if self.mask is not None:
            mask = Cutout2D(self.mask.astype(float), coord, size, wcs=self.wcs,
                            mode="partial", fill_value=1.0).data > 0.5
        var = None
        if self.variance is not None:
            var = Cutout2D(self.variance, coord, size, wcs=self.wcs,
                           mode="partial", fill_value=np.nan).data
        return ReferenceImage(cut.data, cut.wcs, self.band, self.source, self.pixel_scale,
                              self.source_band, self.header, mask, var, self.n_images,
                              self.mjd_range, self.path)


# --------------------------------------------------------------------------
def from_file(path, band: str) -> ReferenceImage:
    path = Path(path).expanduser()
    with fits.open(path, memmap=False) as hdul:
        hdu = next((h for h in hdul if getattr(h, "data", None) is not None), hdul[0])
        data = np.asarray(hdu.data, dtype=float)
        header = hdu.header.copy()
    wcs = WCS(header).celestial
    scale = float(np.mean(proj_plane_pixel_scales(wcs)) * 3600.0)
    log.info("reference for %s: file %s", band, path.name)
    return ReferenceImage(data, wcs, band, "file", scale, source_band=band,
                          header=header, path=path)


def build_ttt_stack(images: Sequence, cfg, coord: SkyCoord, band: str,
                    exclude_mjd: Optional[float] = None,
                    cache_root: Optional[Path] = None,
                    target: str = "target") -> Optional[ReferenceImage]:
    """Median-combine the target's own TTT epochs into a template.

    *images* are ``ScienceImage`` objects already restricted to one band. The
    epochs actually used are chosen by ``subtraction.ttt_stack.mjd_min/max``.
    """
    from reproject import reproject_interp

    mjd_min = cfg.get("subtraction.ttt_stack.mjd_min")
    mjd_max = cfg.get("subtraction.ttt_stack.mjd_max")
    min_images = int(cfg.get("subtraction.ttt_stack.min_images", 2))
    combine = str(cfg.get("subtraction.ttt_stack.combine", "median")).lower()
    exclude_science = bool(cfg.get("subtraction.ttt_stack.exclude_science", True))
    size_arcmin = detector.stamp_arcmin(
        cfg, next((im for im in images if im.meta.band == band), None)) * 1.6

    chosen = []
    for im in images:
        if im.meta.band != band:
            continue
        if mjd_min is not None and im.meta.mjd < float(mjd_min):
            continue
        if mjd_max is not None and im.meta.mjd > float(mjd_max):
            continue
        if exclude_science and exclude_mjd is not None and abs(im.meta.mjd - exclude_mjd) < 1e-6:
            continue
        if im.wcs is None or not im.contains(coord, margin=32):
            continue
        chosen.append(im)

    # Frames from different cameras (or rebinning) have different ADU per unit
    # flux and different pixel areas; a median across them is meaningless. Keep
    # the detector configuration that contributes most epochs.
    if chosen:
        labels = [detector.identify(im.header).label for im in chosen]
        best = max(set(labels), key=labels.count)
        if len(set(labels)) > 1:
            log.info("ttt_stack %s: epochs span %d detectors; stacking only the %d from %s",
                     band, len(set(labels)), labels.count(best), best)
        chosen = [im for im, lab in zip(chosen, labels) if lab == best]

    if len(chosen) < min_images:
        log.info("ttt_stack %s: only %d usable epochs (need %d)", band, len(chosen), min_images)
        return None

    cache_key = None
    if cache_root is not None:
        mjds = sorted(round(im.meta.mjd, 5) for im in chosen)
        # A stable digest, not hash(): Python randomises the hash of a str per
        # process (PEP 456), so a key built from it changes on every run -- the
        # cache could never be read back and each run left another stack behind.
        key = f"{target}|{band}|" + ",".join(f"{m:.5f}" for m in mjds)
        digest = hashlib.md5(key.encode()).hexdigest()[:10]
        cache_key = ensure_dir(Path(cache_root) / "templates") / \
            f"{target}_{band_token(band)}_tttstack_{digest}.fits"
        if cache_key.exists():
            log.info("reference for %s: cached TTT stack %s", band, cache_key.name)
            ref = from_file(cache_key, band)
            ref.source = "ttt_stack"
            ref.source_band = band
            ref.n_images = len(chosen)
            ref.mjd_range = (min(m for m in mjds), max(m for m in mjds))
            return ref

    # Reproject every epoch onto the WCS of the sharpest one.
    reference_image = min(chosen, key=lambda im: im.fwhm_px if im.fwhm_px else 1e9)
    base_cut = Cutout2D(reference_image.data, coord,
                        (size_arcmin * u.arcmin, size_arcmin * u.arcmin),
                        wcs=reference_image.wcs, mode="partial", fill_value=np.nan)
    target_wcs, shape = base_cut.wcs, base_cut.shape

    planes = []
    for im in chosen:
        data = im.data.astype(float).copy()
        data[im.mask] = np.nan
        # Put every epoch on a common zero level before combining.
        if im.bkg_map is not None:
            data = data - im.bkg_map
        else:
            _, med, _ = sigma_clipped_stats(data, sigma=3.0)
            data = data - med
        array, _ = reproject_interp((data, im.wcs), target_wcs, shape_out=shape)
        planes.append(array)

    cube = np.array(planes, dtype=float)
    with np.errstate(invalid="ignore"):
        if combine == "mean":
            stack = np.nanmean(cube, axis=0)
        else:
            stack = np.nanmedian(cube, axis=0)
        counts = np.sum(np.isfinite(cube), axis=0)
    mask = counts == 0
    stack = np.where(mask, np.nan, stack)

    header = target_wcs.to_header()
    header["NCOMBINE"] = len(chosen)
    header["COMBINE"] = combine
    header["MJDMIN"] = float(min(im.meta.mjd for im in chosen))
    header["MJDMAX"] = float(max(im.meta.mjd for im in chosen))
    header["FILTER"] = band

    if cache_key is not None:
        try:
            fits.PrimaryHDU(data=stack.astype(np.float32), header=header).writeto(cache_key, overwrite=True)
        except Exception as exc:  # pragma: no cover
            log.warning("could not cache TTT stack: %s", exc)

    scale = float(np.mean(proj_plane_pixel_scales(target_wcs)) * 3600.0)
    log.info("reference for %s: TTT stack of %d epochs (MJD %.2f-%.2f)", band, len(chosen),
             header["MJDMIN"], header["MJDMAX"])
    return ReferenceImage(stack, target_wcs, band, "ttt_stack", scale, source_band=band,
                          header=header, mask=mask, n_images=len(chosen),
                          mjd_range=(header["MJDMIN"], header["MJDMAX"]), path=cache_key)


# --------------------------------------------------------------------------
# HiPS surveys worth knowing about, and the pixel scale to render them at.
# Rendering much finer than the survey's own resolution only interpolates.
HIPS_NATIVE_SCALE = {
    "DSS2": 1.00, "DSS": 1.00, "SDSS9": 0.40, "SDSS": 0.40,
    "PanSTARRS": 0.25, "DECaLS": 0.262, "ZTF": 1.01, "2MASS": 1.00,
}

# Shorthands so a target file can say 'dss2-blue' instead of the full HiPS id.
HIPS_ALIASES = {
    "dss2-blue": "CDS/P/DSS2/blue",
    "dss2-red": "CDS/P/DSS2/red",
    "dss2-nir": "CDS/P/DSS2/NIR",
    "sdss9-u": "CDS/P/SDSS9/u", "sdss9-g": "CDS/P/SDSS9/g",
    "sdss9-r": "CDS/P/SDSS9/r", "sdss9-i": "CDS/P/SDSS9/i",
    "sdss9-z": "CDS/P/SDSS9/z",
    "ps1-g": "CDS/P/PanSTARRS/DR1/g", "ps1-r": "CDS/P/PanSTARRS/DR1/r",
    "ps1-i": "CDS/P/PanSTARRS/DR1/i", "ps1-z": "CDS/P/PanSTARRS/DR1/z",
    "ps1-y": "CDS/P/PanSTARRS/DR1/y",
}

_HIPS_ENDPOINTS = (
    "https://alasky.cds.unistra.fr/hips-image-services/hips2fits",
    "https://alaskybis.cds.unistra.fr/hips-image-services/hips2fits",
)


def hips_scale_for(survey: str) -> float:
    for key, scale in HIPS_NATIVE_SCALE.items():
        if key.lower() in survey.lower():
            return scale
    return 1.0


def _nearby_cached(directory: Path, prefix: str, suffix: str, ra: float, dec: float,
                   size_px: int, scale_arcsec: float, within_arcsec: float) -> Optional[Path]:
    """A cached cutout of the same kind centred close enough to *ra*, *dec*.

    Cutouts are cached under their centre, to 5 decimals. A target position
    that moves a few arcsec -- the TNS position replacing a first guess, a
    refined centroid, a hand edit of the target file -- would otherwise miss
    the cache and download the same field again. The file names are
    ``<prefix><ra><dec>_<suffix>_<size>.fits`` (suffix may be empty). A file
    qualifies if it is at least as large and, on each axis, the requested
    square sits inside it give or take *within_arcsec*; the nearest wins.
    """
    import re

    if within_arcsec is None or within_arcsec <= 0 or not directory.is_dir():
        return None
    tail = f"_{suffix}" if suffix else ""
    pattern = re.compile(re.escape(prefix) + r"(\d+\.\d+)([+-]\d+\.\d+)" + re.escape(tail)
                         + r"_(\d+)\.fits$")
    best, best_sep = None, np.inf
    for path in directory.glob(f"{prefix}*.fits"):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        ra_c, dec_c, size_c = float(match.group(1)), float(match.group(2)), int(match.group(3))
        if size_c < size_px:
            continue
        dra = (((ra_c - ra) + 180.0) % 360.0 - 180.0) * np.cos(np.radians(dec)) * 3600.0
        ddec = (dec_c - dec) * 3600.0
        slack = (size_c - size_px) * scale_arcsec / 2.0 + within_arcsec
        if abs(dra) > slack or abs(ddec) > slack:
            continue
        sep = float(np.hypot(dra, ddec))
        if sep < best_sep:
            best, best_sep = path, sep
    if best is not None:
        log.info("reusing cached cutout %s (centred %.1f\" from the requested position)",
                 best.name, best_sep)
    return best


def fetch_hips_cutout(coord: SkyCoord, band: str, survey: str, size_arcmin: float,
                      cache_root: Path, pixel_scale: Optional[float] = None,
                      max_pixels: int = 4000, refresh: bool = False,
                      timeout: float = 600.0,
                      reuse_within_arcsec: float = 0.0) -> Optional[ReferenceImage]:
    """Render a HiPS survey to a FITS cutout through the CDS hips2fits service.

    The output grid is whatever we ask for, so unlike a tiled cutout service
    this never comes back with a blank strip where a tile ended.
    """
    import requests

    survey = HIPS_ALIASES.get(str(survey).strip().lower(), str(survey).strip())
    if pixel_scale is None or float(pixel_scale) <= 0:
        pixel_scale = hips_scale_for(survey)
    pixel_scale = float(pixel_scale)

    size_px = int(round(size_arcmin * 60.0 / pixel_scale))
    if size_px > max_pixels:
        pixel_scale *= size_px / max_pixels
        size_px = int(max_pixels)
        log.info("rendering %s at %.3f\"/px so the cutout fits in %d px",
                 survey, pixel_scale, size_px)
    fov_deg = size_px * pixel_scale / 3600.0

    ra, dec = float(coord.ra.deg), float(coord.dec.deg)
    slug_survey = slugify(survey.replace("/", "-"))
    cache = ensure_dir(Path(cache_root) / "templates") / \
        f"hips_{slug_survey}_{ra:.5f}{dec:+.5f}_{size_px}.fits"
    if not cache.exists() and not refresh:
        cache = _nearby_cached(cache.parent, f"hips_{slug_survey}_", "", ra, dec, size_px,
                               pixel_scale, reuse_within_arcsec) or cache

    if cache.exists() and not refresh:
        log.info("reference for %s: cached HiPS cutout %s", band, cache.name)
        ref = from_file(cache, band)
        ref.source = "hips"
        ref.source_band = band
        return ref

    params = {
        "hips": survey, "width": size_px, "height": size_px,
        "fov": f"{fov_deg:.8f}", "projection": "TAN", "coordsys": "icrs",
        "ra": f"{ra:.7f}", "dec": f"{dec:.7f}", "format": "fits",
    }
    log.info("requesting a %d x %d px cutout of %s at %.5f %+.5f (%.2f'/side, %.3f\"/px)",
             size_px, size_px, survey, ra, dec, fov_deg * 60, pixel_scale)

    content = None
    for endpoint in _HIPS_ENDPOINTS:
        try:
            response = requests.get(endpoint, params=params, timeout=timeout)
            response.raise_for_status()
            if len(response.content) < 10000:
                log.warning("hips2fits returned only %d bytes for %s -- "
                            "check the survey id", len(response.content), survey)
                continue
            content = response.content
            break
        except Exception as exc:
            log.warning("hips2fits at %s failed: %s", endpoint.split("/")[2], exc)

    if content is None:
        return None

    cache.write_bytes(content)
    try:
        with fits.open(cache, memmap=False) as hdul:
            hdu = next((h for h in hdul if getattr(h, "data", None) is not None), hdul[0])
            data = np.asarray(hdu.data, dtype=float)
            header = hdu.header.copy()
    except Exception as exc:
        log.warning("hips2fits returned something unreadable for %s: %s", survey, exc)
        return None

    if not np.isfinite(data).any():
        log.warning("the %s cutout is entirely blank -- does that survey cover "
                    "%.5f %+.5f?", survey, ra, dec)
        return None

    header["HIPS"] = (survey, "HiPS survey rendered by hips2fits")
    header["TTTBAND"] = (band, "science band this reference serves")
    fits.PrimaryHDU(data=data.astype(np.float32), header=header).writeto(cache, overwrite=True)

    wcs = WCS(header).celestial
    scale = float(np.mean(proj_plane_pixel_scales(wcs)) * 3600.0)
    log.info("reference for %s: %s (%d px, %.3f\"/px, %.0f%% valid)",
             band, survey, size_px, scale, 100 * np.isfinite(data).mean())
    return ReferenceImage(data, wcs, band, "hips", scale, source_band=survey,
                          header=header, path=cache)


_PS1_SCALE = 0.25          # arcsec/px


def _ps1_skycells(ra: float, dec: float, band: str, size_px: int,
                  mosaic: bool = True) -> list:
    """Every PS1 skycell overlapping the requested box, centre first.

    PS1 stacks are tiled into skycells and ``fitscut`` only ever cuts from one
    of them, so a field near a tile edge comes back with a blank strip. Probing
    the corners as well as the centre finds the neighbours that fill it in.
    """
    import requests

    half_deg = size_px * _PS1_SCALE / 3600.0 / 2.0
    cos_dec = max(np.cos(np.radians(dec)), 1e-6)
    probes = [(ra, dec)]
    if mosaic:
        for dx in (-half_deg, 0.0, half_deg):
            for dy in (-half_deg, 0.0, half_deg):
                if dx == 0.0 and dy == 0.0:
                    continue
                probes.append((ra + dx / cos_dec, dec + dy))

    found = []
    for probe_ra, probe_dec in probes:
        try:
            response = requests.get(
                "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py",
                params={"ra": probe_ra, "dec": probe_dec, "filters": band,
                        "type": "stack"}, timeout=120)
            response.raise_for_status()
            lines = response.text.strip().splitlines()
            if len(lines) < 2:
                continue
            columns = lines[0].split()
            for line in lines[1:]:
                filename = dict(zip(columns, line.split())).get("filename")
                if filename and filename not in found:
                    found.append(filename)
        except Exception as exc:
            log.debug("skycell probe at %.5f %+.5f failed: %s", probe_ra, probe_dec, exc)
    return found


def _ps1_fitscut(ra: float, dec: float, size_px: int, filename: str, timeout: float):
    """One cutout from one skycell. Returns (data, header) or None."""
    import io
    import requests

    try:
        response = requests.get(
            "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi",
            params={"ra": ra, "dec": dec, "size": size_px, "format": "fits",
                    "red": filename},
            timeout=timeout)
        response.raise_for_status()
        if len(response.content) < 10000:
            log.debug("PS1 returned %d bytes for %s -- skipped",
                      len(response.content), filename)
            return None
        with fits.open(io.BytesIO(response.content), memmap=False) as hdul:
            hdu = next((h for h in hdul if getattr(h, "data", None) is not None), hdul[0])
            return np.asarray(hdu.data, dtype=float), hdu.header.copy()
    except Exception as exc:
        log.debug("fitscut failed for %s: %s", filename, exc)
        return None


def _same_grid(a: WCS, b: WCS, shape_a, shape_b, tol_arcsec: float = 0.02) -> bool:
    """True when two WCS describe the same pixel grid, to within *tol_arcsec*."""
    if tuple(shape_a) != tuple(shape_b):
        return False
    try:
        ys = np.array([0, shape_a[0] // 2, shape_a[0] - 1], dtype=float)
        xs = np.array([0, shape_a[1] // 2, shape_a[1] - 1], dtype=float)
        return bool(a.pixel_to_world(xs, ys)
                    .separation(b.pixel_to_world(xs, ys)).arcsec.max() < tol_arcsec)
    except Exception:
        return False


def _combine_skycells(pieces: list) -> tuple:
    """Fill the blank parts of the first cutout from the neighbouring skycells.

    Each skycell carries its own tangent point, so the extra pieces are
    reprojected onto the first one's grid before being used.
    """
    data, header = pieces[0]
    data = data.copy()
    if len(pieces) == 1:
        return data, header

    missing = ~np.isfinite(data)
    if not missing.any():
        return data, header

    from reproject import reproject_interp
    target_wcs = WCS(header).celestial

    for other_data, other_header in pieces[1:]:
        if not missing.any():
            break
        try:
            other_wcs = WCS(other_header).celestial
            if _same_grid(target_wcs, other_wcs, data.shape, other_data.shape):
                resampled = other_data      # identical grid: do not interpolate
            else:
                resampled, _ = reproject_interp((other_data, other_wcs), target_wcs,
                                                shape_out=data.shape)
        except Exception as exc:
            log.debug("could not reproject a neighbouring skycell: %s", exc)
            continue
        usable = missing & np.isfinite(resampled)
        if usable.any():
            data[usable] = resampled[usable]
            missing &= ~usable

    filled = 100.0 * (1.0 - missing.mean())
    log.info("PS1 mosaic: %.0f%% of the cutout has data after filling from %d skycells",
             filled, len(pieces))
    return data, header


def fetch_panstarrs_cutout(coord: SkyCoord, band: str, size_arcmin: float,
                           cache_root: Path, refresh: bool = False,
                           max_size_px: int = 3200,
                           timeout: float = 600.0,
                           mosaic: bool = True,
                           reuse_within_arcsec: float = 0.0) -> Optional[ReferenceImage]:
    """Download a Pan-STARRS1 DR2 stacked cutout and linearise its pixels.

    PS1 pixels are 0.25", so a cutout is roughly (arcmin x 240) pixels a side:
    a 12-arcmin field is a 2880 x 2880 image and a several-tens-of-MB download.
    ``max_size_px`` keeps the request inside what the cutout service will
    actually serve.
    """
    import requests

    if band not in ("g", "r", "i", "z", "y"):
        log.info("Pan-STARRS has no %s-band template", band)
        return None
    if coord.dec.deg < -31.0:
        log.info("Pan-STARRS does not cover Dec %.1f", coord.dec.deg)
        return None

    ra, dec = float(coord.ra.deg), float(coord.dec.deg)
    size_px = int(round(size_arcmin * 60.0 / _PS1_SCALE))
    if size_px > max_size_px:
        log.info("PS1 cutout of %d px capped at %d px (%.1f')", size_px, max_size_px,
                 max_size_px * _PS1_SCALE / 60.0)
        size_px = int(max_size_px)
    cache = ensure_dir(Path(cache_root) / "templates") / \
        f"ps1_{ra:.5f}{dec:+.5f}_{band_token(band)}_{size_px}.fits"
    if not cache.exists() and not refresh:
        cache = _nearby_cached(cache.parent, "ps1_", band_token(band), ra, dec, size_px,
                               _PS1_SCALE, reuse_within_arcsec) or cache

    if cache.exists() and not refresh:
        log.info("reference for %s: cached PS1 cutout %s", band, cache.name)
        ref = from_file(cache, band)
        ref.source = "panstarrs"
        ref.source_band = band
        return ref

    try:
        log.info("requesting a %d x %d px PS1 %s cutout at %.5f %+.5f "
                 "(about %.0f MB, this can take a minute)",
                 size_px, size_px, band, ra, dec, size_px ** 2 * 4 / 1e6)
        skycells = _ps1_skycells(ra, dec, band, size_px, mosaic=mosaic)
        if not skycells:
            log.warning("no PS1 %s stack covers %.5f %+.5f", band, ra, dec)
            return None
        if len(skycells) > 1:
            log.info("the field spans %d PS1 skycells; fetching all of them",
                     len(skycells))

        pieces = []
        for filename in skycells:
            piece = _ps1_fitscut(ra, dec, size_px, filename, timeout)
            if piece is None:
                continue
            # Every skycell carries its own BSOFTEN/BOFFSET, so each one has to
            # be linearised with its own header *before* the pieces are merged.
            # Merging first and linearising with the first cell's constants gave
            # the filled-in regions a wrong flux, silently, on any field near a
            # tile edge -- which is the only case the mosaic exists for.
            piece_data, piece_header = piece
            pieces.append((linearise_ps1(piece_data, piece_header), piece_header))
        if not pieces:
            return None
    except Exception as exc:
        log.warning("PS1 cutout download failed: %s", exc)
        return None

    data, header = _combine_skycells(pieces)
    # Each piece was already linearised above; this is a no-op guard for a
    # cutout that arrived without the asinh keywords at all.
    data = linearise_ps1(data, header)
    # astropy already applied BZERO/BSCALE on read; leaving them in the header
    # would make it apply them again when the cached file is read back, and
    # leaving BSOFTEN would invite a second asinh conversion.
    for key in ("BSCALE", "BZERO", "BLANK", "BSOFTEN", "BOFFSET"):
        header.pop(key, None)
    if header.get("TTTLIN", False):
        header["BUNIT"] = ("linear", "PS1 asinh scaling removed by tttphot")
    fits.PrimaryHDU(data=data.astype(np.float32), header=header).writeto(cache, overwrite=True)

    wcs = WCS(header).celestial
    log.info("reference for %s: PS1 stack cutout (%d px)", band, size_px)
    return ReferenceImage(data, wcs, band, "panstarrs", _PS1_SCALE, source_band=band,
                          header=header, path=cache)


def linearise_ps1(data: np.ndarray, header: fits.Header) -> np.ndarray:
    """Undo the asinh ("boffset/bsoften") scaling of PS1 stack images.

    PS1 stack pixels are stored as asinh magnitudes; the linear flux is
        flux = BOFFSET + BSOFTEN * (10**(0.4*x) - 10**(-0.4*x))
    Images without BSOFTEN are already linear and are returned unchanged.
    """
    if header.get("TTTLIN", False):
        return data                     # already linearised, do not do it twice
    if "BSOFTEN" not in header or "BOFFSET" not in header:
        return data                     # warp images and cutouts are already linear
    bsoften = float(header["BSOFTEN"])
    boffset = float(header["BOFFSET"])
    with np.errstate(over="ignore", invalid="ignore"):
        linear = boffset + bsoften * (10.0 ** (0.4 * data) - 10.0 ** (-0.4 * data))
    header["TTTLIN"] = (True, "PS1 asinh scaling removed")
    log.info("linearised PS1 pixels (BSOFTEN %.4g, BOFFSET %.4g)", bsoften, boffset)
    return linear


# --------------------------------------------------------------------------
def register_reference(ref: ReferenceImage, astrometry_ref, cfg) -> ReferenceImage:
    """Tie a reference image's WCS to the same astrometric catalogue as the
    science frames.

    Every science frame is re-centred on Gaia before anything is measured, but
    a survey template arrives with whatever WCS its own pipeline wrote, and a
    difference image only cares about the two agreeing with each other. They
    need not: the Pan-STARRS stack cutout for SN 2026abgb sits 0.75 px (0.26")
    off Gaia in both axes, while the Pan-STARRS *catalogue* for the same field
    agrees with Gaia to 20 mas. A quarter of an arcsecond is two thirds of a
    TTT pixel, and it leaves an aligned dipole under every star and under the
    host -- exactly what the difference-quality check then reports as a
    misregistered reference.

    The refit uses ``subtraction.register_reference`` as its mode (``shift``
    by default, ``linear``, or ``off``) and the ``astrometry.*`` matching and
    rejection settings, so it is only kept if it improves the residual.
    """
    mode = str(cfg.get("subtraction.register_reference", "shift")).lower()
    if mode in ("off", "none", "false") or astrometry_ref is None or ref is None:
        return ref
    if ref.source == "ttt_stack":
        return ref                      # built from frames already on Gaia
    try:
        from types import SimpleNamespace
        from astropy.table import Table
        from photutils.detection import DAOStarFinder
        from . import astrometry as astrom

        data = np.array(ref.data, dtype=float)
        finite = np.isfinite(data)
        if finite.sum() < 1000:
            return ref
        _, median, std = sigma_clipped_stats(data[finite], sigma=3.0)
        fwhm_px = float(np.clip(1.3 / max(ref.pixel_scale, 1e-3), 2.0, 10.0))
        finder = DAOStarFinder(fwhm=fwhm_px, threshold=10.0 * std, exclude_border=True)
        found = finder(np.where(finite, data - median, 0.0), mask=~finite)
        if found is None or len(found) < int(cfg.get("astrometry.min_stars", 8)):
            log.info("reference for %s: too few stars to register it on the astrometric "
                     "catalogue; keeping its own WCS", ref.band)
            return ref
        found = found[np.argsort(-np.asarray(found["flux"]))][:400]
        sources = Table({"xcentroid": np.asarray(found["xcentroid"], float),
                         "ycentroid": np.asarray(found["ycentroid"], float)})
        shim = SimpleNamespace(wcs=ref.wcs, data=data,
                               meta=SimpleNamespace(path=Path(f"reference_{ref.band}")))
        local = cfg.as_dict()
        local.setdefault("astrometry", {})["mode"] = mode
        from .config import Config
        result = astrom.refine_wcs(shim, sources, astrometry_ref, Config(local))
        if result.ok and result.wcs is not None:
            ref.wcs = result.wcs
            ref.header["TTTREGSH"] = (round(float(result.shift_arcsec), 4),
                                      "arcsec the WCS moved when registered to Gaia")
            log.info("reference for %s: registered on the astrometric catalogue (%s, "
                     "%d stars, rms %.3f\" -> %.3f\", moved %.2f\")", ref.band, mode,
                     result.n_stars, result.rms_before, result.rms_after,
                     result.shift_arcsec)
            ref.registration = result
        else:
            log.info("reference for %s: not re-registered (%s)", ref.band, result.message)
    except Exception as exc:
        log.warning("reference for %s: registration failed (%s); keeping its own WCS",
                    ref.band, exc)
    return ref


# --------------------------------------------------------------------------
def get_reference(image, cfg, cache_root: Path, coord: SkyCoord,
                  siblings: Sequence, target: str,
                  refresh: bool = False,
                  stamp_arcmin: Optional[float] = None) -> Optional[ReferenceImage]:
    """Return the first reference image that a configured source can provide."""
    band = image.meta.band
    order = cfg.get(f"subtraction.reference_sources.{band}") or \
        cfg.get("subtraction.reference_sources.default") or ["file", "ttt_stack", "panstarrs"]
    # One reference serves every epoch in the band, so it is sized for the
    # widest frame in the run rather than for whichever frame arrived first.
    stamp = float(stamp_arcmin) if stamp_arcmin else detector.stamp_arcmin(cfg, image)
    # The reference only has to cover the science stamp plus a margin for the
    # reprojection edges; asking PS1 for much more is a slow, large download.
    ps1_size = float(cfg.get("subtraction.panstarrs.size_arcmin", 0) or 0) or stamp * 1.25
    # A cached cutout centred a little off still covers the stamp if the offset
    # stays inside that margin; 'auto' keeps 20 per cent of the margin in hand.
    reuse = cfg.get("subtraction.reuse_cached_within_arcsec", "auto")
    if reuse in (None, False) or str(reuse).lower() in ("off", "none", "false"):
        reuse = 0.0
    elif str(reuse).lower() == "auto":
        reuse = max(0.8 * (ps1_size - stamp) * 60.0 / 2.0, 0.0)
    else:
        reuse = float(reuse)

    # A band with no survey counterpart (B, V) can borrow the nearest one that
    # does. The PSF-matching kernel fits a single flux scale, so this subtracts
    # cleanly for a smooth host; what it cannot absorb is a colour *gradient*
    # across the galaxy, which leaves structured residuals. Opt in per band.
    fetch_band = cfg.get(f"subtraction.reference_band.{band}") or band

    for source in order:
        try:
            if source == "file":
                path = cfg.resolve_path(cfg.get(f"subtraction.reference_files.{band}"))
                if path:
                    ref = from_file(path, band)
                    ref.source_band = band
                    return ref
            elif source == "ttt_stack":
                ref = build_ttt_stack(siblings, cfg, coord, band,
                                      exclude_mjd=image.meta.mjd, cache_root=cache_root,
                                      target=target)
                if ref is not None:
                    return ref
            elif source == "hips":
                entry = cfg.get(f"subtraction.hips.{band}")
                if not entry:
                    log.info("band %s has no subtraction.hips entry; skipping that source",
                             band)
                    continue
                if isinstance(entry, str):
                    entry = {"survey": entry}
                ref = fetch_hips_cutout(
                    coord, band, entry.get("survey"), ps1_size, cache_root,
                    pixel_scale=entry.get("pixel_scale"),
                    max_pixels=int(cfg.get("subtraction.hips_max_pixels", 4000)),
                    refresh=refresh,
                    timeout=float(cfg.get("subtraction.panstarrs.timeout_s", 600.0)),
                    reuse_within_arcsec=reuse)
                if ref is not None:
                    return ref
            elif source == "panstarrs":
                if fetch_band != band:
                    log.warning("band %s has no Pan-STARRS filter of its own; borrowing "
                                "the %s stack. The kernel fits the overall flux ratio, "
                                "but a colour gradient in the host will leave residuals.",
                                band, fetch_band)
                ref = fetch_panstarrs_cutout(
                    coord, fetch_band, ps1_size, cache_root, refresh=refresh,
                    max_size_px=int(cfg.get("subtraction.panstarrs.max_size_px", 3200)),
                    timeout=float(cfg.get("subtraction.panstarrs.timeout_s", 600.0)),
                    mosaic=bool(cfg.get("subtraction.panstarrs.mosaic", True)),
                    reuse_within_arcsec=reuse)
                if ref is not None:
                    ref.source_band = ref.band
                    ref.band = band
                    return ref
            else:
                log.warning("unknown reference source %r", source)
        except Exception as exc:
            log.warning("reference source %s failed for %s: %s", source, band, exc)

    log.warning(
        "no reference image available for band %s -- those epochs get direct "
        "photometry, with the host still in the aperture. Tried: %s. To give it "
        "one: point subtraction.reference_files.%s at your own template, or add "
        "a subtraction.hips.%s survey, or (for a filter no survey has) set "
        "subtraction.ttt_stack.mjd_min once the supernova has faded.",
        band, ", ".join(order), band, band)
    return None
