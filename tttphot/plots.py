"""QA figures: per-epoch diagnostics and the final light curve."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from astropy.visualization import ZScaleInterval  # noqa: E402

from .lightcurve import FLAG_ZP_TRANSFERRED  # noqa: E402
from .utils import get_logger  # noqa: E402

log = get_logger("tttphot.plots")

BAND_COLORS = {"U": "#7b3fa0", "B": "#3465a4", "V": "#4e9a06", "R": "#cc0000",
               "I": "#8f5902", "u": "#75507b", "g": "#2e8b57", "r": "#d2691e",
               "i": "#a52a2a", "z": "#555753", "y": "#8b0000", "C": "#000000"}
BAND_ORDER = ["U", "B", "V", "R", "I", "u", "g", "r", "i", "z", "y", "C"]


def _zscale(data: np.ndarray):
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0
    try:
        return ZScaleInterval().get_limits(finite)
    except Exception:
        return float(np.nanpercentile(finite, 1)), float(np.nanpercentile(finite, 99))


def _stamp(ax, data, title, xy=None, radius=None):
    vmin, vmax = _zscale(data)
    ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    if xy is not None and radius:
        ax.add_patch(plt.Circle(xy, radius, fill=False, color="#ff7f0e", lw=1.2))
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def epoch_qa(path: Path, image, zp, apcor, diff=None, sn_xy=None, cutout=None,
             stamp_px: int = 60) -> Optional[Path]:
    """One figure per image: stamps, curve of growth, zero-point fit, kernel."""
    try:
        ncols = 4 if diff is not None and diff.ok else 3
        fig, axes = plt.subplots(2, ncols, figsize=(3.1 * ncols, 6.4))
        axes = np.atleast_2d(axes)

        # -- row 1: images --------------------------------------------------
        if sn_xy is not None:
            x, y = sn_xy
            x0, x1 = int(x - stamp_px), int(x + stamp_px)
            y0, y1 = int(y - stamp_px), int(y + stamp_px)
            x0, y0 = max(x0, 0), max(y0, 0)
            science = image.data_sub[y0:y1, x0:x1]
            _stamp(axes[0, 0], science, f"science  {image.meta.band}  MJD {image.meta.mjd:.3f}",
                   xy=(x - x0, y - y0), radius=image.fwhm_px * 1.2 if image.fwhm_px else None)
        else:
            _stamp(axes[0, 0], image.data_sub, "science")

        if diff is not None and diff.ok:
            dx, dy = diff.wcs.world_to_pixel(cutout) if cutout is not None else (None, None)
            half = stamp_px
            def _cut(arr):
                if dx is None:
                    return arr
                a0, a1 = max(int(dx - half), 0), int(dx + half)
                b0, b1 = max(int(dy - half), 0), int(dy + half)
                return arr[b0:b1, a0:a1]
            centre = (min(dx, half), min(dy, half)) if dx is not None else None
            # In convolve_science mode this stamp is the *science* frame blurred
            # to the reference PSF, so it still holds the transient. Calling it
            # "the reference" makes a correct subtraction look like a
            # contaminated template.
            matched_title = ("science blurred to reference PSF"
                             if diff.mode == "convolve_science"
                             else "reference matched to science PSF")
            _stamp(axes[0, 1], _cut(diff.reference_stamp), matched_title,
                   xy=centre, radius=image.fwhm_px * 1.2 if image.fwhm_px else None)
            _stamp(axes[0, 2], _cut(diff.difference), "difference",
                   xy=centre, radius=image.fwhm_px * 1.2 if image.fwhm_px else None)
            _stamp(axes[0, 3], diff.kernel, f"kernel (sum {diff.kernel_sum:.3f})")
        else:
            _stamp(axes[0, 1], image.data, "science (full)")
            message = "no difference image"
            if diff is not None and diff.message:
                message += f"\n{diff.message}"
            axes[0, 2].text(0.5, 0.5, message, ha="center", va="center", fontsize=8,
                            transform=axes[0, 2].transAxes)
            axes[0, 2].set_xticks([]); axes[0, 2].set_yticks([])

        # -- row 2: diagnostics --------------------------------------------
        ax = axes[1, 0]
        if apcor is not None and np.isfinite(apcor.growth).any():
            ax.plot(apcor.radii_px, apcor.growth, "o-", color="#3465a4", ms=3)
            ax.axhline(1.0, color="grey", lw=0.7, ls=":")
            ax.set_xlabel("radius [px]"); ax.set_ylabel("enclosed flux")
            ax.set_title(f"curve of growth ({apcor.n_stars} stars)\n"
                         f"apcor {apcor.apcor_mag:+.3f} mag", fontsize=9)
        else:
            ax.text(0.5, 0.5, "no aperture correction", ha="center", va="center", fontsize=8)

        ax = axes[1, 1]
        if zp is not None and zp.ok and zp.stars is not None and len(zp.stars):
            stars = zp.stars
            used = np.asarray(stars["used"], dtype=bool)
            ax.errorbar(stars["color"][used], stars["delta"][used],
                        yerr=stars["mag_cat_err"][used], fmt="o", ms=3, color="#3465a4",
                        lw=0.6, label="used")
            if (~used).any():
                ax.plot(stars["color"][~used], stars["delta"][~used], "x", ms=4,
                        color="#cc0000", label="clipped")
            grid = np.linspace(np.nanmin(stars["color"]), np.nanmax(stars["color"]), 10)
            ax.plot(grid, zp.zp + zp.color_term * (grid - zp.color_pivot), "-", color="#4e9a06")
            ax.set_xlabel(zp.color_name or "colour"); ax.set_ylabel(r"$m_{cat}-m_{inst}$")
            ax.set_title(f"ZP {zp.zp:.3f}$\\pm${zp.zp_err:.3f}  C {zp.color_term:+.3f}\n"
                         f"{zp.provider}, {zp.n_stars} stars, rms {zp.rms:.3f}", fontsize=9)
            ax.legend(fontsize=7, frameon=False)
        else:
            reason = zp.message if zp is not None else "not attempted"
            ax.text(0.5, 0.5, f"no zero point\n{reason}", ha="center", va="center", fontsize=8)

        ax = axes[1, 2]
        if zp is not None and zp.ok and zp.stars is not None and len(zp.stars):
            stars = zp.stars
            used = np.asarray(stars["used"], dtype=bool)
            ax.plot(stars["mag_cat"][used], stars["residual"][used], "o", ms=3, color="#3465a4")
            if (~used).any():
                ax.plot(stars["mag_cat"][~used], stars["residual"][~used], "x", ms=4, color="#cc0000")
            ax.axhline(0.0, color="grey", lw=0.7)
            ax.set_xlabel("catalogue mag"); ax.set_ylabel("residual")
            ax.set_title("calibration residuals", fontsize=9)
        else:
            ax.set_axis_off()

        if ncols == 4:
            ax = axes[1, 3]
            if diff is not None and diff.ok:
                finite = diff.difference[np.isfinite(diff.difference)]
                ax.hist(finite, bins=80, color="#3465a4",
                        range=np.nanpercentile(finite, [0.5, 99.5]))
                ax.set_yscale("log")
                ax.set_xlabel("difference [ADU]")
                ax.set_title(f"{diff.mode}\n$\\chi^2_\\nu$={diff.chi2:.2f}, "
                             f"{diff.n_stamps} stamps", fontsize=9)
            else:
                ax.set_axis_off()

        fig.suptitle(f"{image.meta.path.name}   FWHM {image.fwhm_arcsec:.2f}\"", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return path
    except Exception as exc:
        log.warning("QA plot failed for %s: %s", image.meta.path.name, exc)
        plt.close("all")
        return None


def lightcurve(path: Path, frame, target: str, offsets: bool = True) -> Optional[Path]:
    """Magnitude versus MJD, one series per band."""
    try:
        primary = frame[frame["primary"].astype(bool)]
        if primary.empty:
            return None
        bands = [b for b in BAND_ORDER if b in set(primary["band"])]
        bands += [b for b in sorted(set(primary["band"])) if b not in bands]

        fig, ax = plt.subplots(figsize=(9, 6.2))
        step = 0.0
        for k, band in enumerate(bands):
            group = primary[primary["band"] == band].sort_values("mjd")
            shift = (len(bands) // 2 - k) * (1.0 if offsets else 0.0)
            color = BAND_COLORS.get(band, "#000000")
            detected = group[~group["is_limit"].astype(bool) & np.isfinite(group["mag"])]
            limits = group[group["is_limit"].astype(bool) & np.isfinite(group["limit_mag"])]
            label = band + (f" {shift:+.0f}" if shift else "")

            # A zero point borrowed from another band is a weaker measurement
            # than a fitted one; draw it open so that is visible on the curve.
            flags = detected["flags"].fillna(0).astype(int) if "flags" in detected else 0
            borrowed = (flags & FLAG_ZP_TRANSFERRED) > 0 if np.ndim(flags) else False
            own = detected[~borrowed] if np.ndim(flags) else detected
            ax.errorbar(own["mjd"], own["mag"] + shift, yerr=own["mag_err"],
                        fmt="o", ms=4.5, color=color, lw=0.8, capsize=2, label=label)
            if np.ndim(flags) and borrowed.any():
                sub = detected[borrowed]
                ax.errorbar(sub["mjd"], sub["mag"] + shift, yerr=sub["mag_err"],
                            fmt="o", ms=5.0, mfc="white", mec=color, ecolor=color,
                            lw=0.8, capsize=2)
            if len(limits):
                ax.plot(limits["mjd"], limits["limit_mag"] + shift, "v", ms=5,
                        mfc="none", color=color)
            step = shift

        ax.invert_yaxis()
        ax.set_xlabel("MJD")
        ax.set_ylabel("magnitude" + (" (+ offset)" if offsets else ""))
        ax.set_title(f"{target} -- TTT")
        ax.grid(alpha=0.25, lw=0.5)
        handles, labels = ax.get_legend_handles_labels()
        if any((frame["flags"].fillna(0).astype(int) & FLAG_ZP_TRANSFERRED) > 0):
            from matplotlib.lines import Line2D
            handles.append(Line2D([], [], marker="o", ls="none", mfc="white",
                                  mec="#555753", color="#555753", ms=5))
            labels.append("zero point from\nanother band")
        ax.legend(handles, labels, ncol=2, fontsize=9, frameon=False)
        fig.tight_layout()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path
    except Exception as exc:
        log.warning("light-curve plot failed: %s", exc)
        plt.close("all")
        return None


def run_diagnostics(path: Path, frame, target: str) -> Optional[Path]:
    """Seeing, zero point and depth over the whole run."""
    try:
        primary = frame[frame["primary"].astype(bool)]
        if primary.empty:
            return None
        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        for band in sorted(set(primary["band"])):
            group = primary[primary["band"] == band].sort_values("mjd")
            color = BAND_COLORS.get(band, "#000000")
            axes[0].plot(group["mjd"], group["fwhm_arcsec"], "o-", ms=3.5, lw=0.7,
                         color=color, label=band)
            axes[1].plot(group["mjd"], group["zp"], "o-", ms=3.5, lw=0.7, color=color)
            axes[2].plot(group["mjd"], group["limit_mag"], "o-", ms=3.5, lw=0.7, color=color)
        axes[0].set_ylabel("FWHM [\"]"); axes[0].legend(ncol=3, fontsize=8, frameon=False)
        axes[1].set_ylabel("zero point")
        axes[2].set_ylabel("3$\\sigma$ limit"); axes[2].invert_yaxis()
        axes[2].set_xlabel("MJD")
        for ax in axes:
            ax.grid(alpha=0.25, lw=0.5)
        fig.suptitle(f"{target} -- run diagnostics")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return path
    except Exception as exc:
        log.warning("diagnostics plot failed: %s", exc)
        plt.close("all")
        return None
