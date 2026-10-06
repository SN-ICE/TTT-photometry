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
    "astrom_rms_arcsec", "astrom_nstars", "detector", "camera", "pixel_scale", "rebin", "gain_eff", "sky_noise_ratio",
    "pattern_applied", "fringe_scale_adu",
    "obs_zp", "obs_zp_err", "obs_maglim", "obs_seeing_arcsec",
    "diff_verdict", "diff_dipole_fraction", "diff_dipole_alignment", "diff_quality_nsources",
    "diff_residual_sigma", "diff_target_dipole",
    "inject_limit_mag", "inject_limit_err", "inject_sites", "error_scale",
    "ref_source", "ref_band", "ref_nimages", "diff_mode", "diff_chi2", "diff_kernel_sum", "diff_nstamps", "diff_kernel_dx", "diff_kernel_dy", "diff_host_residual", "diff_host_nsites",
    "x", "y", "dx", "dy", "psf_flags", "psf_qfit", "psf_cfit",
    "flags", "quality", "image", "input_fingerprint",
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
FLAG_BAD_PSF = 32768          # PSF fit failed or has no usable parameter uncertainty

# These compromise the target measurement itself. Subtraction QA failures
# (including kernel-sum outliers) are warnings: retain them for inspection.
INVALID_MEASUREMENT_FLAGS = (FLAG_MASKED_PIXEL | FLAG_SATURATED | FLAG_OFF_IMAGE
                             | FLAG_IN_TEMPLATE | FLAG_BAD_PSF | FLAG_NO_ZEROPOINT)
OPEN_SYMBOL_FLAGS = (INVALID_MEASUREMENT_FLAGS | FLAG_BAD_SUBTRACTION
                     | FLAG_DIPOLE_RESIDUAL | FLAG_KERNEL_SUM | FLAG_CROSS_BAND_TEMPLATE
                     | FLAG_FEW_CAL_STARS | FLAG_POOR_CALIBRATION | FLAG_ZP_TRANSFERRED
                     | FLAG_APPROX_OFFSET | FLAG_RELAXED_CALIBRATION)
COLOR_MAX_SEPARATION_MINUTES = 30.0

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
    FLAG_BAD_PSF: "bad_psf",
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
    """Whether target validity, rather than field-level QA, is compromised."""
    try:
        return bool(int(row.get("flags", 0) or 0) & INVALID_MEASUREMENT_FLAGS)
    except (TypeError, ValueError):
        return False


# Kept for compatibility with existing configurations and Python callers.
# Auto selection no longer depends on a difference-versus-direct magnitude gap.
HOST_GUARD_MAG = 0.2


def choose_primary(rows: Sequence[dict], preference: str = "auto",
                   host_guard_mag: Optional[float] = HOST_GUARD_MAG) -> Optional[dict]:
    """Prefer usable difference photometry, retaining subtraction QA warnings.

    Field residuals, chi2 and kernel-sum outliers flag a measurement for review;
    they do not justify substituting host-contaminated direct photometry.
    Target-invalid measurements (e.g. masked pixels or a failed PSF fit) yield
    to valid alternatives. Within each family detections precede finite limits,
    then PSF precedes aperture. An explicit method remains an override.
    ``host_guard_mag`` is accepted for compatibility but has no effect.
    """
    usable = [r for r in rows if
              (not r.get("is_limit", False) and np.isfinite(r.get("mag", np.nan)))
              or (r.get("is_limit", False) and np.isfinite(r.get("limit_mag", np.nan)))]
    if not usable:
        return None
    if preference and preference != "auto":
        for row in usable:
            if row["method"] == preference:
                return row

    valid = [r for r in usable if not _suspect(r)]
    candidates = valid or usable
    differences = [r for r in candidates if str(r["method"]).startswith("difference")]
    candidates = differences or candidates
    return min(candidates, key=lambda r: (
        bool(r.get("is_limit", False)),
        METHOD_PRIORITY.index(r["method"]) if r["method"] in METHOD_PRIORITY else len(METHOD_PRIORITY),
    ))


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
    # Anything not in COLUMNS is dropped on the next line. That is the point --
    # the output schema is fixed -- but it used to happen in silence, so a new
    # diagnostic added upstream and forgotten here vanished without a word.
    dropped = sorted(set(frame.columns) - set(COLUMNS))
    if dropped:
        log.warning("these measured values are not in lightcurve.COLUMNS and will "
                    "not be written: %s", ", ".join(dropped))
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    frame = frame[COLUMNS]
    return frame.sort_values(["band", "mjd", "method"]).reset_index(drop=True)


def merge(existing: pd.DataFrame, new: pd.DataFrame,
          preference: str = "auto",
          host_guard_mag: Optional[float] = HOST_GUARD_MAG) -> pd.DataFrame:
    """Replace every method of a remeasured image with its new measurements.

    The primary flag is recomputed per image afterwards: the two runs each
    chose one, and after merging only one may stand. A method absent from the
    new measurement (e.g. a failed subtraction) must not survive from an older
    reduction. Images absent from ``new`` retain all their previous rows.
    """
    keys = ["image", "method"]
    retained = existing[~existing["image"].isin(new["image"])]
    combined = pd.concat([retained, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=keys, keep="last")
    combined = combined.sort_values(["band", "mjd", "method"]).reset_index(drop=True)
    return reselect_primary(combined, preference, host_guard_mag)


COLOR_COLUMNS = ["target", "color", "system", "mjd", "color_mag", "color_err",
                 "separation_minutes", "flags", "flags_str", "flagged",
                 "mjd_a", "mjd_b", "mag_a", "mag_b", "mag_err_a", "mag_err_b",
                 "image_a", "image_b", "method_a", "method_b", "detector_a", "detector_b"]


def color_curve(frame: pd.DataFrame,
                max_separation_minutes: float = COLOR_MAX_SEPARATION_MINUTES) -> pd.DataFrame:
    """Pair primary detections for g-r, r-i, g-i, closest available pair first.

    Pair only within the same magnitude system and photometry family (direct
    or difference). Each exposure is used at most once per color. Times are
    pair midpoints; errors assume independent magnitude uncertainties. No
    interpolation, limits, or band-display offsets enter the colors.
    """
    maximum = float(max_separation_minutes)
    if not np.isfinite(maximum) or maximum < 0:
        raise ValueError("color_max_separation_minutes must be finite and nonnegative")
    selected = frame.loc[frame["primary"].fillna(False).astype(bool)].copy()
    selected = selected.loc[
        ~selected["is_limit"].fillna(False).astype(bool)
        & np.isfinite(selected["mag"]) & np.isfinite(selected["mag_err"])
        & (selected["mag_err"] >= 0) & np.isfinite(selected["mjd"])
    ]
    selected["family"] = selected["method"].str.startswith("difference")
    result = []
    # Unknown systems cannot safely be combined; groupby drops missing values.
    for (_, system, _), group in selected.groupby(["target", "system", "family"], sort=True):
        for band_a, band_b in (("g", "r"), ("r", "i"), ("g", "i")):
            a = group.loc[group["band"] == band_a].sort_values(["mjd", "image"]).to_dict("records")
            b = group.loc[group["band"] == band_b].sort_values(["mjd", "image"]).to_dict("records")
            if not a or not b:
                continue
            # Limit candidate construction to the nearby time window rather
            # than allocating every pair of a potentially long campaign.
            times_b = np.array([r["mjd"] for r in b])
            edges = []
            tolerance = 1e-7  # minutes: floating-point MJD boundary roundoff
            window = (maximum + tolerance) / 1440.
            for ia, ra in enumerate(a):
                lo, hi = np.searchsorted(times_b, [ra["mjd"] - window, ra["mjd"] + window], side="left")
                hi = min(hi + 1, len(b))
                for ib in range(lo, hi):
                    separation = abs(ra["mjd"] - b[ib]["mjd"]) * 1440.
                    if separation <= maximum + tolerance:
                        edges.append((separation, ia, ib))
            used_a, used_b = set(), set()
            for separation, ia, ib in sorted(edges):
                if ia in used_a or ib in used_b:
                    continue
                used_a.add(ia)
                used_b.add(ib)
                ra, rb = a[ia], b[ib]
                flags = int(ra["flags"] if pd.notna(ra["flags"]) else 0) | int(rb["flags"] if pd.notna(rb["flags"]) else 0)
                row = dict(target=ra["target"], color=f"{band_a}-{band_b}", system=system,
                           mjd=(ra["mjd"] + rb["mjd"]) / 2,
                           color_mag=ra["mag"] - rb["mag"],
                           color_err=np.hypot(ra["mag_err"], rb["mag_err"]),
                           separation_minutes=separation, flags=flags,
                           flags_str=describe_flags(flags), flagged=bool(flags & OPEN_SYMBOL_FLAGS))
                for suffix, source in (("a", ra), ("b", rb)):
                    for key in ("mjd", "mag", "mag_err", "image", "method", "detector"):
                        row[f"{key}_{suffix}"] = source.get(key, np.nan)
                result.append(row)
    return pd.DataFrame(result, columns=COLOR_COLUMNS).sort_values(["color", "mjd"]).reset_index(drop=True)


def write(frame: pd.DataFrame, outdir: Path, target: str,
          append: bool = False, preference: str = "auto",
          color_max_separation_minutes: float = COLOR_MAX_SEPARATION_MINUTES) -> dict:
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

    colors_path = outdir / f"{target}_colors.csv"
    color_curve(frame, color_max_separation_minutes).to_csv(
        colors_path, index=False, float_format="%.6f")

    log.info("wrote %s (%d rows) and %s (%d epochs)", all_path.name, len(frame),
             lc_path.name, len(slim))
    return {"all": all_path, "lightcurve": lc_path, "colors": colors_path}


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
