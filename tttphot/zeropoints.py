"""Recovering zero points for frames whose own calibration failed.

A frame that finds too few catalogue stars gets no zero point, and its
photometry is then uncalibrated even though the flux was measured perfectly
well. The obvious repair -- substitute the band's average zero point -- does
not work: on the SN2026acd campaign the zero point spans 1.4 to 2.6 magnitudes
within a band, because it tracks atmospheric transparency, and calibration
tends to fail on exactly the poor nights where the average is furthest from the
truth.

What *is* stable is the difference between two bands on the same night. The
extinction is largely common to frames taken minutes apart, so it cancels in
the difference: on that campaign the band-to-band offsets hold to 0.02-0.10 mag
against 0.07-0.18 for the zero points themselves. So a missing zero point is
recovered from another band of the same night,

    ZP_missing(night) = ZP_partner(night) + median(ZP_missing - ZP_partner)

with the offset measured over every night where both bands were calibrated, and
the scatter of that offset folded into the uncertainty.
"""

from __future__ import annotations

import itertools
from typing import Optional

import numpy as np
import pandas as pd

from . import lightcurve as lc
from .utils import get_logger

log = get_logger("tttphot.zeropoints")


def observing_night(mjd) -> np.ndarray:
    """Integer label grouping frames taken in one night, not one calendar day."""
    return np.floor(np.asarray(mjd, dtype=float) - 0.3).astype(np.int64)


def _robust(values: np.ndarray) -> tuple:
    """Median, and a spread that does not pretend the tail is not there.

    The 68th percentile of the absolute deviations equals sigma for a Gaussian
    but, unlike a median absolute deviation, grows when a few nights sit far
    from the rest -- which is what non-grey extinction does to a band-to-band
    offset. Using the MAD here produced uncertainties about half the size of
    the errors actually made.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    median = float(np.median(values))
    deviations = np.abs(values - median)
    spread = float(np.percentile(deviations, 68.27)) if deviations.size > 2 \
        else float(1.4826 * np.median(deviations))
    return median, spread


def band_offsets(frame: pd.DataFrame, min_nights: int = 3) -> dict:
    """Median same-night ZP offset between every pair of bands.

    Returns {(a, b): (offset, scatter, n_nights)} meaning ZP_a - ZP_b.
    """
    table = frame.dropna(subset=["zp"]).copy()
    if table.empty:
        return {}
    table["night"] = observing_night(table["mjd"])
    per_night = table.groupby(["night", "band"])["zp"].median().unstack()

    offsets = {}
    for a, b in itertools.permutations(per_night.columns, 2):
        difference = (per_night[a] - per_night[b]).dropna()
        if len(difference) < min_nights:
            continue
        median, scatter = _robust(difference.values)
        if np.isfinite(median):
            offsets[(a, b)] = (median, scatter, int(len(difference)))
    return offsets


def _predictions(band, night, per_night, offsets, min_nights, max_scatter) -> list:
    """One predicted zero point per usable partner band on this night."""
    if night not in per_night.index:
        return []
    row = per_night.loc[night]
    out = []
    for other in per_night.columns:
        if other == band or not np.isfinite(row.get(other, np.nan)):
            continue
        entry = offsets.get((band, other))
        if entry is None:
            continue
        offset, scatter, n = entry
        if n < min_nights or (np.isfinite(max_scatter) and scatter > max_scatter):
            continue
        out.append({"band": other, "zp": float(row[other]) + offset,
                    "scatter": float(scatter), "n": int(n)})
    return sorted(out, key=lambda d: d["scatter"])


def _combine(predictions: list, floor: float = 0.03) -> Optional[dict]:
    """Combine the partners into one zero point, and an honest uncertainty.

    Every partner sees the same atmosphere, so when they agree the night's
    extinction was grey and the transfer is trustworthy; when they disagree it
    was not, and that disagreement is the best available measure of how wrong
    the answer may be. Taking the spread among partners as an error term is
    what makes the uncertainty track the real one on bad nights instead of
    quietly staying small.
    """
    if not predictions:
        return None
    values = np.array([p["zp"] for p in predictions])
    weights = 1.0 / np.maximum([p["scatter"] for p in predictions], 1e-3) ** 2
    zp = float(np.sum(weights * values) / np.sum(weights))

    calibration_error = float(np.sqrt(1.0 / np.sum(weights)))
    if len(values) > 1:
        deviations = np.abs(values - zp)
        spread = float(np.percentile(deviations, 68.27) if len(values) > 2
                       else deviations.max())
    else:
        spread = float(predictions[0]["scatter"])
    return {"zp": zp,
            "err": float(np.sqrt(calibration_error ** 2 + spread ** 2 + floor ** 2)),
            "spread": spread, "n_partners": len(values),
            "partners": ",".join(p["band"] for p in predictions)}


def _partner(band, night, per_night, offsets, min_nights, max_scatter):
    """Kept for the single-partner case; returns the tightest option."""
    options = _predictions(band, night, per_night, offsets, min_nights, max_scatter)
    if not options:
        return None
    best = options[0]
    return (best["band"], best["zp"] - 0.0, best["scatter"], 0.0, best["n"])


def camera_of(frame: pd.DataFrame) -> pd.Series:
    """The camera each row was taken with: the instrument part of ``detector``
    (``QHY600-TTT3``, ``iKon936-TTT3``), without the readout geometry.

    Band-to-band offsets are a property of one camera's filters and detector,
    and a zero point is a property of one camera's sensitivity: on SN 2026abgb
    the iKon936 zero points sit 1.4 mag above the QHY600 ones in the same band
    on the same night. Mixing cameras in one per-night median would turn the
    offsets into noise (g-i scattered by 0.62 mag across nights that way).
    """
    if "detector" not in frame.columns:
        return pd.Series("", index=frame.index)
    return frame["detector"].fillna("").astype(str).str.split().str[0].fillna("")


def transfer(frame: pd.DataFrame, cfg=None, min_nights: int = 3,
             max_scatter: float = 0.25) -> tuple:
    """Fill in missing zero points from another band of the same night and camera.

    Each camera is handled on its own (see ``camera_of``). Returns (frame,
    report); the report's offsets are keyed ``camera:a-b`` when there is more
    than one camera.
    """
    cameras = camera_of(frame)
    names = [c for c in pd.unique(cameras)]
    if len(names) <= 1:
        return _transfer_one(frame, cfg, min_nights, max_scatter)
    pieces, report = [], {"offsets": {}, "recovered": [], "unrecovered": [],
                          "approximate_offset": 0}
    for name in names:
        part, rep = _transfer_one(frame[cameras == name], cfg, min_nights, max_scatter,
                                  label=name or "unknown camera")
        pieces.append(part)
        report["offsets"].update({f"{name}:{k}": v for k, v in rep["offsets"].items()})
        report["recovered"] += rep["recovered"]
        report["unrecovered"] += rep["unrecovered"]
        report["approximate_offset"] += rep["approximate_offset"]
    return pd.concat(pieces).loc[frame.index], report


def _transfer_one(frame: pd.DataFrame, cfg=None, min_nights: int = 3,
                  max_scatter: float = 0.25, label: str = "") -> tuple:
    """Fill in missing zero points from another band of the same night.

    Returns (frame, report). Rows that gain a zero point are recalculated and
    flagged ``zp_transferred``; nothing that already had one is touched.
    """
    floor = 0.03
    if cfg is not None:
        min_nights = int(cfg.get("calibration.transfer_min_nights", min_nights))
        max_scatter = float(cfg.get("calibration.transfer_max_scatter", max_scatter))
        floor = float(cfg.get("calibration.transfer_error_floor", floor))

    frame = frame.copy()
    for column in ("zp_source", "mag_offset"):
        if column not in frame.columns:
            frame[column] = np.nan
    frame["zp_source"] = frame["zp_source"].fillna(
        pd.Series(np.where(np.isfinite(frame["zp"]), "fit", ""), index=frame.index))

    offsets = band_offsets(frame, min_nights=min_nights)
    report = {"offsets": {f"{a}-{b}": (round(v[0], 4), round(v[1], 4), v[2])
                          for (a, b), v in offsets.items()},
              "recovered": [], "unrecovered": [], "approximate_offset": 0}
    if not offsets:
        log.info("%snot enough overlapping nights to measure band offsets",
                 f"{label}: " if label else "")
        return frame, report

    known = frame.dropna(subset=["zp"]).copy()
    known["night"] = observing_night(known["mjd"])
    per_night = known.groupby(["night", "band"])["zp"].median().unstack()

    missing = frame[~np.isfinite(frame["zp"])]
    exact_possible = frame["mag_offset"].notna().any()

    for image, rows in missing.groupby("image"):
        band = rows["band"].iloc[0]
        night = observing_night([rows["mjd"].iloc[0]])[0]
        options = _predictions(band, night, per_night, offsets, min_nights, max_scatter)
        combined = _combine(options, floor=floor)
        if combined is None:
            report["unrecovered"].append(str(image))
            continue
        zp = combined["zp"]
        other = combined["partners"]

        # Fold in how well the partner frames were themselves calibrated.
        partner_err = known[(known["band"].isin([o["band"] for o in options])) &
                            (observing_night(known["mjd"]) == night)]["zp_err"]
        partner_err = float(partner_err.median()) if len(partner_err) else 0.0
        zp_err = float(np.hypot(combined["err"], np.nan_to_num(partner_err)))

        index = frame.index[frame["image"] == image]
        frame.loc[index, "zp"] = zp
        frame.loc[index, "zp_err"] = zp_err
        frame.loc[index, "zp_source"] = f"transferred from {other}"
        frame.loc[index, "cal_provider"] = f"via {other}"
        frame.loc[index, "flags"] = (frame.loc[index, "flags"].fillna(0).astype(int)
                                     & ~lc.FLAG_NO_ZEROPOINT) | lc.FLAG_ZP_TRANSFERRED

        for row_index in index:
            row = frame.loc[row_index]
            offset_mag = row.get("mag_offset", np.nan)
            if not np.isfinite(offset_mag):
                # An older table has no record of the correction that was
                # applied, and it differs by method. The aperture rows kept
                # theirs in 'apcor'; the PSF rows did not, so those are only
                # approximate and are flagged as such.
                if str(row["method"]).endswith("aperture"):
                    offset_mag = float(row.get("apcor", 0.0) or 0.0)
                else:
                    offset_mag = 0.0
                    frame.loc[row_index, "flags"] = int(frame.loc[row_index, "flags"]) \
                        | lc.FLAG_APPROX_OFFSET
                    report["approximate_offset"] += 1

            flux, flux_err = row.get("flux", np.nan), row.get("flux_err", np.nan)
            snr = flux / flux_err if (np.isfinite(flux_err) and flux_err > 0) else np.nan
            frame.loc[row_index, "snr"] = snr
            if np.isfinite(flux) and flux > 0 and np.isfinite(snr) and snr >= 3.0:
                frame.loc[row_index, "mag"] = -2.5 * np.log10(flux) + offset_mag + zp
                sys = row.get("flux_err_sys", 0.0)
                sys = float(sys) if sys is not None and np.isfinite(sys) else 0.0
                frame.loc[row_index, "mag_err"] = float(np.hypot(
                    1.0857362 * np.hypot(flux_err, sys) / flux, zp_err))
                frame.loc[row_index, "is_limit"] = False
            else:
                frame.loc[row_index, "mag"] = np.nan
                frame.loc[row_index, "mag_err"] = np.nan
                frame.loc[row_index, "is_limit"] = True

            sky_rms = row.get("sky_rms_adu", np.nan)
            radius = row.get("aperture_radius_px", np.nan)
            exptime = row.get("exptime", np.nan)
            if np.isfinite(sky_rms) and np.isfinite(radius) and np.isfinite(exptime):
                noise = sky_rms * np.sqrt(np.pi * radius ** 2) / max(exptime, 1e-6)
                if noise > 0:
                    frame.loc[row_index, "limit_mag"] = float(
                        -2.5 * np.log10(3.0 * noise) + offset_mag + zp)

        report["recovered"].append({"image": str(image), "band": str(band),
                                    "from": str(other), "zp": round(float(zp), 4),
                                    "zp_err": round(zp_err, 4),
                                    "partner_spread": round(combined["spread"], 4),
                                    "n_partners": combined["n_partners"]})

    frame["flags_str"] = frame["flags"].fillna(0).astype(int).apply(lc.describe_flags)
    if report["recovered"]:
        log.info("recovered %d frames by transferring a zero point from another band",
                 len(report["recovered"]))
    if report["unrecovered"]:
        log.info("%d frames still have no zero point (no calibrated band that night)",
                 len(report["unrecovered"]))
    if report["approximate_offset"] and not exact_possible:
        log.warning("this table predates the mag_offset column, so PSF rows were "
                    "recalculated without their PSF-to-aperture correction "
                    "(typically a few hundredths of a magnitude); re-run for exact values")
    return frame, report
