"""Image ingest: FITS -> a fully-described science frame.

``ScienceImage`` is the single object the rest of the pipeline works with. It
owns the pixel data, the mask, the WCS, an uncertainty map and the metadata
extracted through the instrument header mapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import astropy.units as u
from photutils.background import Background2D, MedianBackground

from .utils import edge_margin_px, first_header_value, get_logger, header_mjd

log = get_logger("tttphot.image")


@dataclass
class ImageMeta:
    """Everything the pipeline needs to know about a frame, instrument-neutral."""

    path: Path
    object_name: str
    band: str
    system: str                  # 'ab' or 'vega'
    mjd: float
    exptime: float
    gain: float                  # e-/ADU
    readnoise: float             # e-
    saturation: float            # ADU
    pixel_scale: float           # arcsec/px
    target_coord: Optional[SkyCoord] = None
    airmass: Optional[float] = None
    header_filter: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def epoch_id(self) -> str:
        return f"{self.mjd:.5f}_{self.band}"


def effective_gain(header, hmap: dict, defaults: dict, camera_binning=(1, 1),
                   rebin: int = 1) -> tuple[float, float, dict]:
    """Gain (e-/ADU) and read noise (e-) of one pixel of the array being measured.

    Three things change them from the header value. The camera may quote the
    unbinned sensor (``gain_is_unbinned``). A binned pixel may be the average
    of n native pixels rather than the sum (``binning_mode``): an average of n
    pixels has n times the electrons per ADU, and its read noise, in those
    electrons, adds in quadrature to sqrt(n) times the native value. The same
    holds for a stack of averaged sub-exposures (``stack_mode``). The software
    rebinning sums, which leaves e-/ADU alone and adds read noise in
    quadrature: sqrt(rebin**2) = rebin times the per-pixel value.
    """
    gain = float(first_header_value(header, hmap.get("gain", ["GAIN"]), defaults.get("gain", 1.0)))
    rn = float(first_header_value(header, hmap.get("readnoise", ["RDNOISE"]),
                                  defaults.get("readnoise", 0.0)))
    n = 1.0
    bx, by = camera_binning
    if bool(defaults.get("gain_is_unbinned", False)) and bx * by > 1:
        if str(defaults.get("binning_mode", "average")).lower() == "average":
            n *= bx * by
        # summed binning: e-/ADU is unchanged, read noise grows by sqrt(n)
        else:
            rn *= np.sqrt(bx * by)
    nsub = first_header_value(header, hmap.get("n_subframes", []) or [], 1)
    try:
        nsub = max(int(float(nsub)), 1)
    except (TypeError, ValueError):
        nsub = 1
    if nsub > 1 and str(defaults.get("stack_mode", "average")).lower() == "average":
        n *= nsub
    info = {"gain_header": gain, "readnoise_header": rn, "gain_multiplier": n,
            "n_subframes": nsub}
    return gain * n, rn * np.sqrt(n) * float(rebin), info


def _sexagesimal_coord(ra_value, dec_value, unit: str = "deg") -> Optional[SkyCoord]:
    """Header coordinates: sexagesimal strings are read as hours/degrees, plain
    numbers in *unit* ('deg' or 'hourangle') for RA -- TTT writes decimal degrees."""
    if ra_value is None or dec_value is None:
        return None
    try:
        if isinstance(ra_value, str) and (":" in ra_value or " " in ra_value.strip()):
            return SkyCoord(str(ra_value), str(dec_value), unit=(u.hourangle, u.deg))
        ra_unit = u.hourangle if str(unit).lower().startswith("hour") else u.deg
        return SkyCoord(float(ra_value), float(dec_value), unit=(ra_unit, u.deg))
    except Exception as exc:  # pragma: no cover - malformed headers
        log.warning("could not parse target coordinates %r %r: %s", ra_value, dec_value, exc)
        return None


class ScienceImage:
    """A single reduced exposure, ready for photometry."""

    def __init__(self, data: np.ndarray, header: fits.Header, meta: ImageMeta,
                 wcs: Optional[WCS] = None):
        self.data = np.asarray(data, dtype=float)
        self.header = header
        self.meta = meta
        self.wcs = wcs
        self.mask = np.zeros(self.data.shape, dtype=bool)
        self._background = None      # Background2D result
        self.bkg_map: Optional[np.ndarray] = None
        self.bkg_rms: Optional[np.ndarray] = None
        self.error: Optional[np.ndarray] = None
        self.fwhm_px: Optional[float] = None
        self.fwhm_arcsec: Optional[float] = None
        self.astrometry = None       # AstrometryResult, once the WCS is refined
        self.qa: dict = {}

    # ------------------------------------------------------------------ load
    @classmethod
    def from_file(cls, path, cfg, instrument: dict, pattern=None) -> "ScienceImage":
        path = Path(path)
        with fits.open(path, memmap=False) as hdul:
            hdu = next((h for h in hdul if getattr(h, "data", None) is not None), hdul[0])
            data = np.asarray(hdu.data, dtype=float)
            header = hdu.header.copy()

        # The archive can span more than one camera; let the frame pick.
        chooser = getattr(cfg, "instrument_for", None)
        if callable(chooser):
            instrument = chooser(header) or instrument

        hmap = instrument.get("header", {})
        defaults = instrument.get("defaults", {})
        saturation = float(cfg.get("image.saturation_adu")
                           or defaults.get("saturation_adu", 60000))

        # Trim, mask the camera's known bad pixels and rebin, before the WCS is
        # read: the header that comes back describes the new array.
        from . import detector as detector_module
        cam_bx, cam_by = detector_module.camera_binning(header, instrument)
        data, header, prep = detector_module.prepare_raw(data, header, instrument,
                                                         saturation=saturation)
        trimmed = True if prep.get("trimmed") else None
        rebin = int(prep.get("rebin", 1))
        # Rebinning sums pixels, so one rebinned pixel saturates at f**2 times
        # the native level (and _rebin_array marks saturated blocks that way).
        saturation *= rebin ** 2

        raw_filter = str(first_header_value(header, hmap.get("filter", ["FILTER"]), "")).strip()
        band = instrument.get("filters", {}).get(raw_filter)
        if band is None:
            band = instrument.get("filters", {}).get(raw_filter.replace("SDSS ", ""), raw_filter)
            log.warning("filter %r not in the instrument map; using %r", raw_filter, band)
        system = instrument.get("systems", {}).get(band, "ab")

        mjd = header_mjd(header, hmap)

        binning = float(cam_bx * rebin)

        wcs = None
        try:
            candidate = WCS(header)
            if candidate.has_celestial:
                wcs = candidate.celestial
        except Exception as exc:
            log.warning("%s: unusable WCS (%s)", path.name, exc)

        if wcs is not None:
            pixel_scale = float(np.mean(proj_plane_pixel_scales(wcs)) * 3600.0)
        else:
            pixel_scale = float(defaults.get("pixel_scale", 1.0)) * binning

        gain, readnoise, noise_info = effective_gain(header, hmap, defaults,
                                                     (cam_bx, cam_by), rebin)
        exptime = float(first_header_value(header, hmap.get("exptime", ["EXPTIME"]), 1.0))
        unit = hmap.get("coord_unit", "deg")
        target_coord = _sexagesimal_coord(
            first_header_value(header, hmap.get("target_ra", []) or []),
            first_header_value(header, hmap.get("target_dec", []) or []), unit)
        pointing = _sexagesimal_coord(
            first_header_value(header, hmap.get("pointing_ra", []) or []),
            first_header_value(header, hmap.get("pointing_dec", []) or []), unit)
        observatory = {}
        for key in ("obs_zp", "obs_zp_err", "obs_maglim", "obs_seeing"):
            value = first_header_value(header, hmap.get(key, []) or [])
            try:
                observatory[key] = float(value) if value is not None else float("nan")
            except (TypeError, ValueError):
                observatory[key] = float("nan")

        meta = ImageMeta(
            path=path,
            object_name=str(first_header_value(header, hmap.get("object", ["OBJECT"]), path.stem)).strip(),
            band=band,
            system=system,
            mjd=float(mjd),
            exptime=exptime,
            gain=gain,
            readnoise=readnoise,
            saturation=saturation,
            pixel_scale=pixel_scale,
            target_coord=target_coord,
            airmass=first_header_value(header, hmap.get("airmass", ["AIRMASS"])),
            header_filter=raw_filter,
            extra={"quality": first_header_value(header, hmap.get("quality", []) or []),
                   "pointing": pointing, **observatory, **noise_info},
        )

        image = cls(data, header, meta, wcs)
        if trimmed is not None:
            image.qa["trimmed"] = True
        image.qa["rebin"] = rebin

        if pattern is not None:
            from . import patterns as pattern_module
            fits_ok, why = pattern_module.geometry_matches(pattern, header, data.shape)
            if not fits_ok:
                log.warning("%s: not applying the %s pattern -- %s",
                            path.name, pattern.band, why)
                pattern = None

        image.crmask = None
        if pattern is not None:
            from . import patterns as pattern_module
            try:
                image.data, info = pattern_module.apply_pattern(image.data, pattern, cfg)
                image.qa.update({f"pattern_{k}": v for k, v in info.items()})
                image.qa["pattern_applied"] = True
                log.debug("%s: pattern removed (fringe scale %.1f ADU)",
                          path.name, info.get("fringe_scale_adu", 0.0))
            except Exception as exc:
                log.warning("%s: pattern correction failed (%s); using the raw frame",
                            path.name, exc)
                image.qa["pattern_applied"] = False
        else:
            image.qa["pattern_applied"] = False

        # Cosmic rays go after the detector pattern (which needs the real sky)
        # and before the mask and background, both of which they would bias.
        from . import cosmics
        fwhm_guess = float(cfg.get("detection.fwhm_init_arcsec", 2.0)) / max(pixel_scale, 1e-3)
        image.data, crmask, cr_info = cosmics.remove(
            image.data, cfg, gain=meta.gain, readnoise=meta.readnoise,
            saturation=meta.saturation, fwhm_px=fwhm_guess)
        image.crmask = crmask
        image.qa.update(cr_info)
        if cr_info.get("cosmic_rays"):
            log.debug("%s: %d cosmic ray pixels removed (%.3f%%)", path.name,
                      cr_info["cosmic_rays"], 100 * cr_info["cosmic_ray_fraction"])

        image.build_mask(cfg)
        return image

    # ------------------------------------------------------------------ prep
    def build_mask(self, cfg) -> np.ndarray:
        """Mask non-finite, saturated, dead, wildly negative and edge pixels."""
        mask = ~np.isfinite(self.data)
        mask |= self.data >= self.meta.saturation
        for bad in cfg.get("image.bad_value_mask", []) or []:
            mask |= self.data == float(bad)

        # Saturation only bounds the frame from above. Dividing by a flat that
        # is near zero on a bad column sends pixels to large values of *either*
        # sign, and MEIA3 frames carry columns that reach -7e6 ADU. Those pass
        # every test above and then land in an aperture. Cut symmetrically, far
        # enough out that no real sky pixel is ever touched.
        low_sigma = float(cfg.get("image.outlier_sigma", 0) or 0)
        if low_sigma > 0:
            from astropy.stats import sigma_clipped_stats
            finite = np.isfinite(self.data)
            if finite.any():
                _, med, std = sigma_clipped_stats(self.data[finite], sigma=3.0)
                if np.isfinite(med) and np.isfinite(std) and std > 0:
                    floor = med - low_sigma * std
                    wild = self.data < floor
                    if wild.any():
                        from scipy.ndimage import binary_dilation
                        mask |= binary_dilation(wild, iterations=1)
                        self.qa["outlier_pixels"] = int(wild.sum())
                        self.qa["outlier_floor_adu"] = float(floor)

        if getattr(self, "crmask", None) is not None and np.any(self.crmask):
            mask |= np.asarray(self.crmask, bool)

        margin = self._edge_margin(cfg)
        if margin > 0:
            edge = np.zeros_like(mask)
            edge[:margin, :] = True
            edge[-margin:, :] = True
            edge[:, :margin] = True
            edge[:, -margin:] = True
            mask |= edge

        # Saturated stars bleed: grow the saturation mask slightly.
        if np.any(self.data >= self.meta.saturation):
            from scipy.ndimage import binary_dilation
            sat = binary_dilation(self.data >= self.meta.saturation, iterations=2)
            mask |= sat

        self.mask = mask
        self.qa["masked_fraction"] = float(mask.mean())
        return mask

    def _edge_margin(self, cfg) -> int:
        """Edge margin in pixels (see utils.edge_margin_px)."""
        return edge_margin_px(cfg, self.meta.pixel_scale)

    def _background_box(self, cfg) -> int:
        """Background mesh size in pixels.

        Configured in arcsec, for two reasons: the sky structure being
        modelled has an angular scale, and a fixed pixel box gives wildly
        different mesh counts across binnings -- 128 px is 32x32 meshes
        on a 4096 frame but only 8x8 on a 1024 binned one, where a filter_size
        of 5 then spans most of the field and the 'local' background is
        effectively global. The box is also kept small enough to leave at least
        `min_cells` meshes per axis.
        """
        arcsec = cfg.get("image.background.box_arcsec")
        if arcsec:
            scale = float(self.meta.pixel_scale) or 1.0
            box = int(round(float(arcsec) / scale))
        else:
            box = int(cfg.get("image.background.box_size", 128))
        min_cells = int(cfg.get("image.background.min_cells", 12) or 0)
        if min_cells > 0:
            box = min(box, max(min(self.data.shape) // min_cells, 16))
        return max(box, 16)

    def estimate_background(self, cfg) -> None:
        """2D background and its RMS via photutils.Background2D."""
        box = self._background_box(cfg)
        filt = int(cfg.get("image.background.filter_size", 5))
        nsig = float(cfg.get("image.background.sigma_clip", 3.0))
        self.qa["background_box_px"] = box
        try:
            bkg = Background2D(
                self.data,
                box_size=box,
                filter_size=filt,
                mask=self.mask,
                sigma_clip=SigmaClip(sigma=nsig),
                bkg_estimator=MedianBackground(),
                exclude_percentile=30.0,
            )
            self._background = bkg
            self.bkg_map = bkg.background
            self.bkg_rms = bkg.background_rms
        except Exception as exc:
            log.warning("%s: Background2D failed (%s); falling back to a constant sky",
                        self.meta.path.name, exc)
            _, median, std = sigma_clipped_stats(self.data, mask=self.mask, sigma=nsig)
            self.bkg_map = np.full(self.data.shape, median, dtype=float)
            self.bkg_rms = np.full(self.data.shape, std, dtype=float)

        self.qa["sky_adu"] = float(np.nanmedian(self.bkg_map))
        self.qa["sky_rms_adu"] = float(np.nanmedian(self.bkg_rms))
        self._check_noise_model()

    def _check_noise_model(self) -> None:
        """Measured sky scatter over what the gain and read noise predict.

        The gain enters every Poisson error bar and the cosmic-ray detector, and
        for a binned CMOS frame it depends on how the camera combined pixels,
        which the header does not say. The sky tells us: for a bias-subtracted
        frame sigma_sky^2 = sky / gain + (readnoise / gain)^2 in ADU. The
        pixel-to-pixel scatter is used rather than the Background2D RMS, which
        also carries faint sources and sky structure.
        """
        try:
            sky = float(self.qa.get("sky_adu", np.nan))
            gain = float(self.meta.gain)
            rn = float(self.meta.readnoise)
            if not (np.isfinite(sky) and sky > 0 and gain > 0):
                return
            predicted = np.sqrt(sky / gain + (rn / gain) ** 2)
            ny, nx = self.data.shape
            core = (slice(ny // 4, 3 * ny // 4), slice(nx // 4, 3 * nx // 4))
            d = self.data[core]
            m = self.mask[core]
            diff = (d[:, 1:] - d[:, :-1]) / np.sqrt(2.0)
            good = ~(m[:, 1:] | m[:, :-1]) & np.isfinite(diff)
            if good.sum() < 1000:
                return
            _, _, measured = sigma_clipped_stats(diff[good], sigma=3.0, maxiters=5)
            ratio = float(measured / predicted)
            self.qa["sky_noise_measured_adu"] = float(measured)
            self.qa["sky_noise_predicted_adu"] = float(predicted)
            self.qa["sky_noise_ratio"] = ratio
            level = log.warning if not (0.75 <= ratio <= 1.33) else log.debug
            level("%s: sky scatter %.2f ADU against %.2f predicted by gain %.2f e-/ADU and "
                  "read noise %.1f e- (ratio %.2f)%s", self.meta.path.name, measured,
                  predicted, gain, rn, ratio,
                  "" if 0.75 <= ratio <= 1.33 else
                  " -- the gain model looks wrong for this frame; check binning_mode / "
                  "stack_mode in the instrument file")
        except Exception as exc:  # pragma: no cover - diagnostic only
            log.debug("noise-model check skipped: %s", exc)

    def build_error_map(self) -> np.ndarray:
        """Per-pixel 1-sigma uncertainty in ADU.

        sigma^2 = sky_rms^2 (sky shot noise + read noise, measured empirically)
                  + source counts / gain  (Poisson noise of the source itself)
        """
        if self.bkg_rms is None:
            raise RuntimeError("call estimate_background() before build_error_map()")
        source = np.clip(self.data - self.bkg_map, 0.0, None)
        var = self.bkg_rms.astype(float) ** 2 + source / max(self.meta.gain, 1e-6)
        self.error = np.sqrt(var)
        return self.error

    # ------------------------------------------------------------- accessors
    @property
    def data_sub(self) -> np.ndarray:
        """Background-subtracted data."""
        if self.bkg_map is None:
            raise RuntimeError("background not estimated yet")
        return self.data - self.bkg_map

    def pixel_of(self, coord: SkyCoord) -> tuple[float, float]:
        if self.wcs is None:
            raise RuntimeError(f"{self.meta.path.name} has no WCS")
        x, y = self.wcs.world_to_pixel(coord)
        return float(x), float(y)

    def contains(self, coord: SkyCoord, margin: int = 0) -> bool:
        try:
            x, y = self.pixel_of(coord)
        except Exception:
            return False
        ny, nx = self.data.shape
        return (margin <= x < nx - margin) and (margin <= y < ny - margin)

    def center_coord(self) -> SkyCoord:
        ny, nx = self.data.shape
        return self.wcs.pixel_to_world(nx / 2.0, ny / 2.0)

    def __repr__(self) -> str:  # pragma: no cover
        return (f"<ScienceImage {self.meta.path.name} band={self.meta.band} "
                f"mjd={self.meta.mjd:.4f} fwhm={self.fwhm_arcsec}>")
