"""Judging a difference image where it matters: at the sources.

The reduced chi-square the kernel solve returns is measured on the stamps used
to fit the kernel, which are bright field stars. It says the PSF match worked
*there*. It does not say the subtraction is clean at the transient, and a frame
can subtract well across the field and badly on a bright host -- which is how
three SN2022xkq epochs came through with errors of 0.7 to 1.1 mag while every
chi-square on that target stayed below 1.8.

The test that catches those is the shape of what is left behind. A residual
from a photometric mismatch is a monopole: the star over- or under-subtracts by
some fraction and leaves a blob of one sign. A residual from a *registration*
error is a dipole -- positive on one side, negative on the other -- because the
same flux has been subtracted a fraction of a pixel away from where it sits.
Dipoles are the signature no flux rescaling can remove, and counting them over
the field stars is a direct measure of whether the alignment is good enough.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .utils import get_logger

log = get_logger("tttphot.diffquality")


@dataclass
class DifferenceQuality:
    verdict: str = "unknown"          # 'pass' | 'downgrade' | 'fail' | 'unknown'
    dipole_fraction: float = float("nan")
    dipole_alignment: float = float("nan")
    n_sources: int = 0
    residual_sigma: float = float("nan")   # bright-star residual, in noise units
    target_dipole: bool = False
    message: str = ""

    def as_dict(self) -> dict:
        return {"diff_verdict": self.verdict,
                "diff_dipole_fraction": self.dipole_fraction,
                "diff_dipole_alignment": self.dipole_alignment,
                "diff_quality_nsources": self.n_sources,
                "diff_residual_sigma": self.residual_sigma,
                "diff_target_dipole": bool(self.target_dipole)}


def _noise(diff: np.ndarray) -> float:
    finite = diff[np.isfinite(diff)]
    if finite.size < 16:
        return float("nan")
    med = float(np.median(finite))
    return float(np.median(np.abs(finite - med))) * 1.4826


def _dipole(stamp: np.ndarray, sigma: float, nsigma: float,
            antisymmetry: float):
    """Is this residual a dipole, and if so along which axis?

    Returns ``(is_dipole, angle)`` with the angle of the positive-to-negative
    axis in radians, or ``(None, nan)`` when the stamp cannot be judged.

    *sigma* must be the noise the pixels are *expected* to have -- the
    difference image's own photon-plus-read variance -- not the scatter they
    actually show. Two wrong choices sit either side of it: the frame's global
    scatter is inflated by real residuals, so a clean subtraction near a bright
    star trips the test; the scatter inside the stamp itself is inflated by the
    very residual being measured, so a badly subtracted frame normalises its
    own evidence away and everything passes.
    """
    good = np.isfinite(stamp)
    if good.sum() < 9:
        return None, float("nan")
    values = np.where(good, stamp, 0.0)
    if not np.isfinite(sigma) or sigma <= 0:
        return None, float("nan")
    scale = float(sigma)

    hi = np.unravel_index(int(np.argmax(values)), values.shape)
    lo = np.unravel_index(int(np.argmin(values)), values.shape)
    pos, neg = float(values[hi]), float(values[lo])
    if pos < nsigma * scale or neg > -nsigma * scale:
        return False, float("nan")
    cy, cx = (np.array(stamp.shape) - 1) / 2.0
    v1 = np.array([hi[0] - cy, hi[1] - cx])
    v2 = np.array([lo[0] - cy, lo[1] - cx])
    if float(v1 @ v2) >= 0:
        return False, float("nan")
    ratio = min(abs(pos), abs(neg)) / max(abs(pos), abs(neg), 1e-12)
    if ratio < antisymmetry:
        return False, float("nan")
    return True, float(np.arctan2(hi[0] - lo[0], hi[1] - lo[1]))


def _alignment(angles) -> float:
    """How well the dipole axes agree, 0 (random) to 1 (all parallel).

    This is what separates a real misregistration from chance. A shift is
    global: every source is subtracted displaced the same way, so every dipole
    points the same way. Residuals thrown up by photon noise near bright stars
    point wherever they like. Angles are doubled because an axis has no head or
    tail -- opposite directions are the same axis.
    """
    a = np.asarray([x for x in angles if np.isfinite(x)], dtype=float)
    if a.size < 3:
        return float("nan")
    return float(np.abs(np.mean(np.exp(2j * a))))


def assess(diff, positions, fwhm_px: float, cfg,
           target_xy: Optional[tuple] = None) -> DifferenceQuality:
    """Score a difference image on the shape of its residuals.

    *positions* are (x, y) of field sources on the difference stamp.
    """
    q = DifferenceQuality()
    if not bool(cfg.get("subtraction.quality.enabled", True)):
        q.verdict = "pass"
        return q

    data = np.asarray(diff.difference, dtype=float)
    # The expected noise, from the propagated variance of the difference. Only
    # if that is unavailable do we fall back to the observed scatter.
    var = getattr(diff, "variance", None)
    sigma_map = None
    if var is not None:
        v = np.asarray(var, dtype=float)
        if v.shape == data.shape and np.any(np.isfinite(v) & (v > 0)):
            sigma_map = np.sqrt(np.where(np.isfinite(v) & (v > 0), v, np.nan))
    sigma = _noise(data)
    if sigma_map is None and (not np.isfinite(sigma) or sigma <= 0):
        q.message = "no noise estimate"
        return q

    nsigma = float(cfg.get("subtraction.quality.dipole_nsigma", 5.0))
    anti = float(cfg.get("subtraction.quality.dipole_antisymmetry", 0.3))
    half = max(int(round(1.5 * float(fwhm_px))), 3)
    ny, nx = data.shape

    def stamp(x, y):
        x0, x1 = int(round(x)) - half, int(round(x)) + half + 1
        y0, y1 = int(round(y)) - half, int(round(y)) + half + 1
        if x0 < 0 or y0 < 0 or x1 > nx or y1 > ny:
            return None, float("nan")
        if sigma_map is None:
            return data[y0:y1, x0:x1], sigma
        local = sigma_map[y0:y1, x0:x1]
        finite = local[np.isfinite(local)]
        return data[y0:y1, x0:x1], (float(np.median(finite)) if finite.size else sigma)

    verdicts, residuals, angles = [], [], []
    for x, y in np.atleast_2d(np.asarray(positions, dtype=float)):
        sub, sig = stamp(x, y)
        if sub is None:
            continue
        got, angle = _dipole(sub, sig, nsigma, anti)
        if got is None:
            continue
        verdicts.append(got)
        if got:
            angles.append(angle)
        residuals.append(float(np.nanmedian(np.abs(sub))) / max(sig, 1e-12))

    q.n_sources = len(verdicts)
    if residuals:
        q.residual_sigma = float(np.median(residuals))
    if target_xy is not None:
        sub, sig = stamp(*target_xy)
        if sub is not None:
            q.target_dipole = bool(_dipole(sub, sig, nsigma, anti)[0])

    min_sources = int(cfg.get("subtraction.quality.min_sources", 8))
    if q.n_sources < min_sources:
        q.message = f"only {q.n_sources} sources to judge on"
        return q

    q.dipole_fraction = float(np.mean(verdicts))
    q.dipole_alignment = _alignment(angles)

    # Both conditions, not either. A handful of dipoles pointing every which way
    # is photon noise near bright sources and is present in every clean
    # subtraction; the same handful all pointing one way is a shift.
    warn = float(cfg.get("subtraction.quality.dipole_warn", 0.15))
    fail = float(cfg.get("subtraction.quality.dipole_fail", 0.30))
    align = float(cfg.get("subtraction.quality.alignment_warn", 0.45))
    coherent = np.isfinite(q.dipole_alignment) and q.dipole_alignment >= align

    if q.dipole_fraction >= fail and coherent:
        q.verdict, q.message = "fail", (
            f"{100 * q.dipole_fraction:.0f}% of sources leave a dipole and they "
            f"align (R={q.dipole_alignment:.2f}) -- the reference is "
            f"misregistered, not merely mismatched in flux")
    elif (q.dipole_fraction >= warn and coherent) or q.target_dipole:
        q.verdict = "downgrade"
        q.message = ("the transient itself leaves a dipole" if q.target_dipole
                     else f"{100 * q.dipole_fraction:.0f}% of sources leave an "
                          f"aligned dipole (R={q.dipole_alignment:.2f})")
    else:
        q.verdict = "pass"
    return q
