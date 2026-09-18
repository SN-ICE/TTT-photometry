"""Small shared helpers: logging, paths, robust statistics, header parsing."""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)-18s %(message)s"
DATE_FORMAT = "%H:%M:%S"

# Chatty third-party loggers. They are silenced with a filter rather than with
# logging.getLogger(name).setLevel(): astroquery and astropy install their own
# Logger subclasses on import, and creating a plain logger under one of those
# names first makes their initialisation fail outright.
NOISY_LOGGERS = ("astropy", "astroquery", "urllib3", "matplotlib", "PIL",
                 "reproject", "fsspec")


class _QuietFilter(logging.Filter):
    """Drop sub-warning records coming from the noisy libraries."""

    def __init__(self, names=NOISY_LOGGERS, level: int = logging.WARNING):
        super().__init__()
        self.names = tuple(names)
        self.level = level

    def filter(self, record: logging.LogRecord) -> bool:
        return not (record.name.split(".")[0] in self.names and record.levelno < self.level)


def quiet_expected_warnings() -> None:
    """Silence warnings that every frame raises and that mean nothing here.

    astropy's 'datfix' and SIP-keyword fixes are reported as FITSFixedWarning on
    every file; and the Levenberg-Marquardt fitter warns whenever a star's
    Gaussian fit stops on the iteration limit, which the calling code already
    handles by rejecting the result.
    """
    import warnings

    from astropy.wcs import FITSFixedWarning
    from astropy.utils.exceptions import AstropyUserWarning

    warnings.filterwarnings("ignore", category=FITSFixedWarning)
    warnings.filterwarnings("ignore", category=AstropyUserWarning,
                            message=".*fit may be unsuccessful.*")


def setup_logging(level: str = "INFO", logfile: Optional[Path] = None,
                  quiet_warnings: bool = True) -> None:
    """Configure root logging once, optionally teeing to *logfile*."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if logfile is not None:
        logfile = Path(logfile)
        logfile.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile, mode="a"))
    quiet = _QuietFilter()
    for handler in handlers:
        handler.addFilter(quiet)

    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=handlers,
        force=True,
    )

    if quiet_warnings:
        try:
            quiet_expected_warnings()
        except Exception:  # pragma: no cover - astropy always provides these
            pass


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def ensure_dir(path: os.PathLike | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def first_header_value(header, keys: Sequence[str], default=None):
    """Return the first present, non-empty header card among *keys*."""
    for key in keys:
        if key in header:
            value = header[key]
            if isinstance(value, str):
                value = value.strip()
                if value == "":
                    continue
            if value is not None:
                return value
    return default


# Filenames must survive a case-insensitive filesystem (macOS, Windows), where
# Johnson I and SDSS i -- or R/r, U/u -- would otherwise be the same file and
# silently overwrite each other. Every band that reaches a path goes through
# this map.
BAND_TOKENS = {
    "U": "U-jc", "B": "B-jc", "V": "V-jc", "R": "R-jc", "I": "I-jc",
    "u": "u-sdss", "g": "g-sdss", "r": "r-sdss", "i": "i-sdss", "z": "z-sdss",
    "y": "y-ps1", "C": "clear",
}


def band_token(band: str) -> str:
    """A filesystem-safe, case-collision-free token for a band name."""
    band = str(band).strip()
    if band in BAND_TOKENS:
        return BAND_TOKENS[band]
    # Unknown band: keep it readable but make the case explicit.
    token = slugify(band)
    return token if token.islower() else token + "-uc"


def slugify(text: str) -> str:
    """Filesystem-safe token (used for filter names such as ``SDSS g``)."""
    text = str(text).strip()
    text = re.sub(r"[^\w.+-]+", "_", text)
    return text.strip("_")


def robust_sigma(values: np.ndarray) -> float:
    """Normalised median absolute deviation, NaN-safe."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    med = np.median(values)
    return float(1.4826 * np.median(np.abs(values - med)))


def weighted_mean(values: Iterable[float], errors: Iterable[float]) -> tuple[float, float]:
    """Inverse-variance weighted mean and its uncertainty."""
    v = np.asarray(list(values), dtype=float)
    e = np.asarray(list(errors), dtype=float)
    good = np.isfinite(v) & np.isfinite(e) & (e > 0)
    if not np.any(good):
        return float("nan"), float("nan")
    w = 1.0 / e[good] ** 2
    mean = float(np.sum(w * v[good]) / np.sum(w))
    err = float(np.sqrt(1.0 / np.sum(w)))
    return mean, err


def flux_to_mag(flux, flux_err=None, zp: float = 0.0, zp_err: float = 0.0):
    """Convert flux (counts) to magnitude; returns (mag, mag_err).

    Non-positive fluxes give NaN magnitudes -- the caller is expected to fall
    back on an upper limit in that case.
    """
    flux = np.asarray(flux, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.where(flux > 0, -2.5 * np.log10(np.abs(flux)) + zp, np.nan)
    if flux_err is None:
        return mag
    flux_err = np.asarray(flux_err, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag_err = np.where(
            flux > 0,
            np.sqrt((1.0857362 * flux_err / np.abs(flux)) ** 2 + zp_err ** 2),
            np.nan,
        )
    return mag, mag_err


def circular_mask(shape, xc: float, yc: float, radius: float) -> np.ndarray:
    ny, nx = shape
    y, x = np.mgrid[0:ny, 0:nx]
    return (x - xc) ** 2 + (y - yc) ** 2 <= radius ** 2


def odd(n: int) -> int:
    """Nearest odd integer >= 3."""
    n = int(round(n))
    if n % 2 == 0:
        n += 1
    return max(3, n)


# TNS name prefixes. A transient is registered as AT and renamed SN (or TDE,
# ...) once classified, so the same object turns up under either.
TNS_PREFIXES = ("sn", "at", "tde", "kn", "frb", "lrn", "ilrt", "ln", "sne")


def normalise_name(name) -> str:
    """Comparable form of a transient name.

    'SN 2026abgb', 'SN2026abgb', 'AT2026abgb', 'AT 2026abgb', '2026abgb' and
    'sn_2026abgb' all become '2026abgb'. Anything that is not a TNS-style
    designation is returned lower-cased with spaces, '_' and '-' removed.
    """
    text = re.sub(r"[\s_\-]+", "", str(name or "")).lower()
    match = re.match(r"^(?:" + "|".join(TNS_PREFIXES) + r")?((?:19|20)\d{2}[a-z]{1,4})$", text)
    return match.group(1) if match else text


def find_by_designation(directory, name, suffixes=(".yaml", ".yml")):
    """The file in *directory* for *name*, whatever prefix either one carries.

    Exact name first (``SN2026abwp.yaml`` for SN2026abwp); failing that, any
    file whose stem is the same designation (``AT2026abwp.yaml``,
    ``2026abwp.yaml``). None if there is none; the first match, sorted, if
    several.
    """
    directory = Path(directory)
    for suffix in suffixes:
        exact = directory / f"{name}{suffix}"
        if exact.exists():
            return exact
    wanted = normalise_name(name)
    if not wanted or not directory.is_dir():
        return None
    matches = sorted(p for p in directory.iterdir()
                     if p.suffix in suffixes and normalise_name(p.stem) == wanted)
    return matches[0] if matches else None

