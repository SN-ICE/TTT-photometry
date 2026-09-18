"""Assembly and serialisation of the photometry table."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .utils import get_logger

log = get_logger("tttphot.lightcurve")

COLUMNS = [
    "target", "mjd", "band", "system", "method", "primary",
    "mag", "mag_err", "limit_mag", "is_limit",
    "flux", "flux_err", "flux_err_sys", "snr",
    "zp", "zp_err", "zp_source", "color_term", "color_term_err",
    "cal_provider", "cal_nstars", "cal_rms", "mag_offset",
    "apcor", "apcor_err", "aperture_radius_px",
    "fwhm_arcsec", "sky_adu", "sky_rms_adu", "exptime", "airmass",
    "astrom_rms_arcsec", "astrom_nstars", "detector", "pixel_scale", "rebin", "gain_eff", "sky_noise_ratio",
    "pattern_applied", "fringe_scale_adu",
    "obs_zp", "obs_zp_err", "obs_maglim", "obs_seeing_arcsec",
    "diff_verdict", "diff_dipole_fraction", "diff_dipole_alignment", "diff_quality_nsources",
    "diff_residual_sigma", "diff_target_dipole",
    "inject_limit_mag", "inject_limit_err", "inject_sites", "error_scale",
    "ref_source", "ref_band", "ref_nimages", "diff_mode", "diff_chi2", "diff_kernel_sum", "diff_nstamps", "diff_kernel_dx", "diff_kernel_dy", "diff_host_residual", "diff_host_nsites",
    "x", "y", "dx", "dy", "flags", "quality", "image",
]

# Order in which methods are preferred when output.primary_method is 'auto'.
METHOD_PRIORITY = ["difference_psf", "difference_aperture", "psf", "aperture"]

# Bits set in the ``flags`` column.
FLAG_MASKED_PIXEL = 1        # a masked pixel fell inside the aperture
FLAG_SATURATED = 2           # saturated pixel inside the aperture
FLAG_OFF_IMAGE = 4           # aperture ran off the frame
FLAG_BAD_SUBTRACTION = 8     # difference image failed its own noise model
FLAG_FEW_CAL_STARS = 16      # zero point rests on very few stars
FLAG_NO_ZEROPOINT = 32       # no photometric calibration for this frame
FLAG_POOR_CALIBRATION = 64   # zero-point fit has a large residual scatter
FLAG_IN_TEMPLATE = 128       # this epoch is part of the reference image itself
FLAG_CROSS_BAND_TEMPLATE = 256   # the template was taken in a different filter
FLAG_ZP_TRANSFERRED = 512        # zero point borrowed from another band, same night
FLAG_APPROX_OFFSET = 1024        # recalculated without the exact magnitude offset
FLAG_RELAXED_CALIBRATION = 2048  # zero point needed loosened star-selection cuts
FLAG_DIPOLE_RESIDUAL = 4096      # the difference leaves dipoles: the reference is misregistered
FLAG_KERNEL_SUM = 8192           # kernel sum far from the band's other epochs: the solve collapsed
FLAG_VARIABLES_UNSCREENED = 16384  # the variable-star screen could not be run for this frame

FLAG_NAMES = {
    FLAG_MASKED_PIXEL: "masked_pixel", FLAG_SATURATED: "saturated",
    FLAG_OFF_IMAGE: "off_image", FLAG_BAD_SUBTRACTION: "bad_subtraction",
    FLAG_FEW_CAL_STARS: "few_cal_stars", FLAG_NO_ZEROPOINT: "no_zeropoint",
    FLAG_POOR_CALIBRATION: "poor_calibration", FLAG_IN_TEMPLATE: "in_template",
    FLAG_CROSS_BAND_TEMPLATE: "cross_band_template",
    FLAG_ZP_TRANSFERRED: "zp_transferred", FLAG_APPROX_OFFSET: "approx_offset",
    FLAG_RELAXED_CALIBRATION: "relaxed_calibration",
    FLAG_DIPOLE_RESIDUAL: "dipole_residual", FLAG_KERNEL_SUM: "kernel_sum_outlier",
    FLAG_VARIABLES_UNSCREENED: "variables_unscreened",
}


def describe_flags(value) -> str:
    """Human-readable flag list, e.g. 'saturated|bad_subtraction'."""
    try:
        value = int(value)
    except (TypeError, ValueError):
        return ""
    return "|".join(name for bit, name in FLAG_NAMES.items() if value & bit)


def new_row(**kwargs) -> dict:
    row = {key: np.nan for key in COLUMNS}
    row["primary"] = False
    row["is_limit"] = False
    row.update(kwargs)
    return row


def _suspect(row: dict) -> bool:
    try:
        flags = int(row.get("flags", 0) or 0)
    except (TypeError, ValueError):
        return False
    # A masked or off-frame aperture is a *biased* flux, not merely a noisy one:
    # aperture_photometry drops the masked pixels from the sum and nothing puts
    # their flux back, so the measurement is systematically faint. It stays
    # usable -- 'only if nothing cleaner exists' -- rather than being discarded.
    #
    # A dipole 'downgrade' is deliberately NOT suspect here. Demoting on it was
    # tried and measured: on SN2022xkq it changed nothing at all (the gain there
    # came from the kernel-sum check), and on SN2023ixf it moved 46 epochs off
    # their difference image with no improvement in either scatter or agreement
    # with an independent reduction -- while giving up host subtraction on M101,
    # which is a real cost the metrics cannot see. The flag stays as an
    # annotation; only a 'fail', which also sets FLAG_BAD_SUBTRACTION, demotes.
    return bool(flags & (FLAG_BAD_SUBTRACTION | FLAG_SATURATED | FLAG_IN_TEMPLATE
                         | FLAG_MASKED_PIXEL | FLAG_OFF_IMAGE))


# A direct measurement brighter than the same epoch's difference measurement by
# more than this carries enough host light to be worse than a difference image
# that merely failed the dipole test (0.2 mag = the host supplies 17 per cent of
# the direct flux). See choose_primary.
HOST_GUARD_MAG = 0.2


def _host_guard_candidate(row: dict, max_chi2: float = 2.5) -> bool:
    """A difference measurement that is flagged only for dipole residuals.

    Its kernel solved (no kernel-sum outlier, a finite chi2 within bounds) and
    its template is in the right filter; what it failed is the field-star
    dipole test. That is a warning about registration, not evidence that the
    flux at the transient is wrong.
    """
    if not str(row.get("method", "")).startswith("difference"):
        return False
    try:
        flags = int(row.get("flags", 0) or 0)
        chi2 = float(row.get("diff_chi2", np.nan))
    except (TypeError, ValueError):
        return False
    if flags & (FLAG_KERNEL_SUM | FLAG_CROSS_BAND_TEMPLATE | FLAG_SATURATED | FLAG_IN_TEMPLATE):
        return False
    if not (np.isfinite(chi2) and chi2 <= max_chi2):
        return False
    return np.isfinite(row.get("mag", np.nan))


def choose_primary(rows: Sequence[dict], preference: str = "auto",
                   host_guard_mag: Optional[float] = HOST_GUARD_MAG) -> Optional[dict]:
    """Pick the row that represents the epoch in the final light curve.

    Preference order is difference-image PSF, difference aperture, direct PSF,
    direct aperture -- but a measurement carrying a hard quality flag (a failed
    subtraction, a saturated aperture, a difference leaving dipole residuals) is
    only used if nothing cleaner exists.

    With one exception. Falling back from a flagged difference to *direct*
    photometry is only an improvement when the host is faint under the
    transient. When the direct measurement comes out brighter than the
    difference by more than ``host_guard_mag``, more than half of its flux is
    galaxy, and a subtraction that merely failed the dipole test is by far the
    better number. On SN 2026abgb this is the difference between 18.5 and
    16.9 mag for the same g epoch. The adopted row keeps its flags.

    The threshold was 0.75 mag at first, which is "more than half the flux is
    host". It let two TTT3/QHY600 r epochs of 2026-09-15 through with direct
    photometry 0.57 and 0.69 mag too bright (16.82, 16.68), while their flagged
    differences (17.39, 17.36) matched the iKon936 frames taken ten minutes
    later (17.34). On that target no epoch in any band has a gap below 0.32 mag,
    so 0.2 changes nothing else.
    """
    usable = [r for r in rows if np.isfinite(r.get("mag", np.nan)) or r.get("is_limit")]
    if not usable:
        return None

    if preference and preference != "auto":
        for row in usable:
            if row["method"] == preference:
                return row

    chosen = None
    for allow_suspect in (False, True):
        for require_detection in (True, False):
            for method in METHOD_PRIORITY:
                for row in usable:
                    if row["method"] != method:
                        continue
                    if not allow_suspect and _suspect(row):
                        continue
                    if require_detection and not np.isfinite(row.get("mag", np.nan)):
                        continue
                    chosen = row
                    break
                if chosen is not None:
                    break
            if chosen is not None:
                break
        if chosen is not None:
            break
    if chosen is None:
        return usable[0]

    if (host_guard_mag is not None and not str(chosen["method"]).startswith("difference")
            and np.isfinite(chosen.get("mag", np.nan))):
        for method in METHOD_PRIORITY[:2]:
            for row in usable:
                if row["method"] == method and _host_guard_candidate(row):
                    if float(row["mag"]) - float(chosen["mag"]) > float(host_guard_mag):
                        return row
    return chosen


def reselect_primary(frame: pd.DataFrame, preference: str = "auto",
                     host_guard_mag: Optional[float] = HOST_GUARD_MAG) -> pd.DataFrame:
    """Recompute the primary row of every image with the usual preference order."""
    frame = frame.copy()
    frame["primary"] = False
    for _, group in frame.groupby("image", sort=False):
        best = choose_primary(group.to_dict("records"), preference, host_guard_mag)
        if best is None:
            continue
        match = group[group["method"] == best["method"]]
        if len(match):
            frame.loc[match.index[0], "primary"] = True
    return frame


def to_frame(rows: Iterable[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    frame = frame[COLUMNS]
    return frame.sort_values(["band", "mjd", "method"]).reset_index(drop=True)


def merge(existing: pd.DataFrame, new: pd.DataFrame,
          preference: str = "auto",
          host_guard_mag: Optional[float] = HOST_GUARD_MAG) -> pd.DataFrame:
    """Combine a previous run's table with new rows, newest measurement wins.

    The primary flag is recomputed per image afterwards: the two runs each
    chose one, and after merging only one may stand.
    """
    keys = ["image", "method"]
    combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=keys, keep="last")
    combined = combined.sort_values(["band", "mjd", "method"]).reset_index(drop=True)
    return reselect_primary(combined, preference, host_guard_mag)


def write(frame: pd.DataFrame, outdir: Path, target: str,
          append: bool = False, preference: str = "auto") -> dict:
    """Write the full table and the primary-only light curve. Returns paths."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_path = outdir / f"{target}_photometry_all.csv"
    lc_path = outdir / f"{target}_lightcurve.csv"

    if append and all_path.exists():
        try:
            previous = pd.read_csv(all_path)
            before = len(frame)
            frame = merge(previous, frame, preference)
            log.info("merged %d new rows into %d existing ones", before, len(previous))
        except Exception as exc:
            log.warning("could not merge with %s: %s", all_path.name, exc)

    frame = frame.copy()
    frame["flags_str"] = frame["flags"].apply(describe_flags)
    frame.to_csv(all_path, index=False, float_format="%.6f")

    primary = frame[frame["primary"].astype(bool)].copy()
    slim = primary[[
        "target", "mjd", "band", "system", "mag", "mag_err", "limit_mag", "is_limit",
        "flux", "flux_err", "snr", "zp", "zp_err", "zp_source", "fwhm_arcsec",
        "astrom_rms_arcsec",
        "detector", "pattern_applied", "method", "diff_verdict",
        "cal_provider", "cal_nstars", "cal_rms", "ref_source", "flags", "flags_str", "image",
    ]]
    slim.to_csv(lc_path, index=False, float_format="%.6f")

    log.info("wrote %s (%d rows) and %s (%d epochs)", all_path.name, len(frame),
             lc_path.name, len(slim))
    return {"all": all_path, "lightcurve": lc_path}


def summarise(frame: pd.DataFrame) -> str:
    """A short human-readable report of what the run produced."""
    primary = frame[frame["primary"].astype(bool)]
    if primary.empty:
        return "no photometry produced"
    lines = [f"{len(primary)} epochs in {primary['band'].nunique()} bands",
             f"MJD {primary['mjd'].min():.3f} to {primary['mjd'].max():.3f}", ""]
    lines.append(f"{'band':>5} {'N':>4} {'det':>4} {'lim':>4} "
                 f"{'brightest':>10} {'faintest':>9} {'median err':>11} {'median ZP':>10}")
    for band, group in primary.groupby("band", sort=False):
        detected = group[~group["is_limit"].astype(bool) & np.isfinite(group["mag"])]
        limits = group[group["is_limit"].astype(bool)]
        bright = detected["mag"].min() if len(detected) else np.nan
        faint = detected["mag"].max() if len(detected) else np.nan
        err = detected["mag_err"].median() if len(detected) else np.nan
        zp = group["zp"].median()
        lines.append(f"{band:>5} {len(group):>4} {len(detected):>4} {len(limits):>4} "
                     f"{bright:>10.3f} {faint:>9.3f} {err:>11.3f} {zp:>10.3f}")
    return "\n".join(lines)
