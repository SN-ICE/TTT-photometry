"""Reference photometric catalogues.

Every provider returns the same thing: an astropy ``Table`` with columns

    ra, dec, <band>, e_<band>, ...          (band = B, V, g, r, i, z, ...)

in the canonical band naming used throughout the pipeline. Results are cached
on disk so a re-run costs no network calls, and so that a field can be
pre-fetched once on a machine with internet access.

Providers
---------
gaia_synth : Gaia DR3 synthetic photometry (GSPC), via the ESA Gaia TAP
             service. Gives Johnson-Kron-Cousins UBVRI *and* SDSS ugriz from a
             single, all-sky, internally consistent system. Primary choice.
panstarrs  : Pan-STARRS1 DR2 mean object catalogue via the MAST API. griz(y),
             Dec > -30 only. Deeper than GSPC.
apass      : AAVSO APASS DR9 via VizieR. B, V (and Sloan g'r'i').
local      : a user-supplied CSV/ECSV file (columns ra, dec, <band>, e_<band>).
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u

from .utils import UserError, ensure_dir, get_logger, slugify

log = get_logger("tttphot.catalogs")

# Bands each provider can deliver.
PROVIDER_BANDS = {
    "gaia_astrometry": [],      # positions only, for the WCS refinement
    "gaia_synth": ["U", "B", "V", "R", "I", "u", "g", "r", "i", "z"],
    "panstarrs": ["g", "r", "i", "z", "y"],
    "apass": ["B", "V", "g", "r", "i"],
    "local": None,          # whatever the file contains
}

# Colour used to fit the colour term of each band.
BAND_COLOR = {
    "U": ("U", "B"),
    "B": ("B", "V"),
    "V": ("B", "V"),
    "R": ("V", "R"),
    "I": ("V", "I"),
    "u": ("u", "g"),
    "g": ("g", "r"),
    "r": ("g", "r"),
    "i": ("r", "i"),
    "z": ("i", "z"),
    "y": ("i", "z"),
}


class CatalogError(UserError):
    """A reference catalogue could not be fetched or has nothing usable."""


def _float_column(table, name, default=float("nan")):
    """A plain float array from a catalogue column, masked entries -> NaN.

    Archive queries return masked columns whenever a value is missing, and the
    mask has to be resolved *before* the column becomes a plain ndarray -- an
    ndarray has no .filled().
    """
    if name not in table.colnames:
        return None
    try:
        values = np.ma.masked_invalid(np.ma.asarray(table[name], dtype=float))
    except (TypeError, ValueError):
        return None
    # np.ma keeps the astropy Column subclass alive; hand back a plain array so
    # no column metadata rides along into the output table.
    return np.asarray(np.ma.filled(values, default), dtype=float)


# --------------------------------------------------------------------------
# caching
# --------------------------------------------------------------------------
# Query centres are snapped to this grid before they become a cache key, so
# every frame of a dithered field shares one download instead of fetching its
# own copy. At 0.01 deg the offset is small against a 9-arcmin search radius.
CACHE_GRID_DEG = 0.01


def _snap(value: float, grid: float = CACHE_GRID_DEG) -> float:
    return float(np.round(value / grid) * grid)


def _cache_path(cache_root: Path, provider: str, ra: float, dec: float,
                radius_arcmin: float, variant: Optional[str] = None) -> Path:
    ra, dec = _snap(ra), _snap(dec)
    key = f"{provider}_{ra:.5f}_{dec:.5f}_{radius_arcmin:.3f}"
    if variant:
        key += f"_{variant}"
    digest = hashlib.md5(key.encode()).hexdigest()[:8]
    tag = f"_{slugify(variant)}" if variant else ""
    name = f"{provider}_{ra:.4f}{dec:+.4f}_r{radius_arcmin:.1f}{tag}_{digest}.ecsv"
    return ensure_dir(cache_root / "catalogs") / name


def load_cached(cache_root: Path, provider: str, ra: float, dec: float,
                radius_arcmin: float, variant: Optional[str] = None) -> Optional[Table]:
    path = _cache_path(cache_root, provider, ra, dec, radius_arcmin, variant)
    if path.exists():
        try:
            table = Table.read(path, format="ascii.ecsv")
            log.info("using cached %s catalogue (%d rows) from %s", provider, len(table), path.name)
            return table
        except Exception as exc:  # pragma: no cover
            log.warning("could not read cache %s: %s", path, exc)
    return None


def save_cache(table: Table, cache_root: Path, provider: str, ra: float, dec: float,
               radius_arcmin: float, variant: Optional[str] = None) -> Path:
    path = _cache_path(cache_root, provider, ra, dec, radius_arcmin, variant)
    try:
        table.write(path, format="ascii.ecsv", overwrite=True)
    except Exception as exc:  # pragma: no cover
        log.warning("could not write cache %s: %s", path, exc)
    return path


# --------------------------------------------------------------------------
# providers
# --------------------------------------------------------------------------
_GAIA_JKC = {"U": "u_jkc", "B": "b_jkc", "V": "v_jkc", "R": "r_jkc", "I": "i_jkc"}
_GAIA_SDSS = {"u": "u_sdss", "g": "g_sdss", "r": "r_sdss", "i": "i_sdss", "z": "z_sdss"}


def fetch_gaia_synth(ra: float, dec: float, radius_arcmin: float,
                     row_limit: int = 20000) -> Table:
    """Gaia DR3 synthetic photometry (GSPC) joined to gaia_source for positions."""
    from astroquery.gaia import Gaia

    prefixes = list(_GAIA_JKC.values()) + list(_GAIA_SDSS.values())
    cols = ["g.source_id", "g.ra", "g.dec", "g.phot_g_mean_mag", "g.bp_rp"]
    for pre in prefixes:
        cols += [f"s.{pre}_mag", f"s.{pre}_flux", f"s.{pre}_flux_error", f"s.{pre}_flag"]

    adql = (
        "SELECT " + ", ".join(cols) + " "
        "FROM gaiadr3.gaia_source AS g "
        "JOIN gaiadr3.synthetic_photometry_gspc AS s ON g.source_id = s.source_id "
        f"WHERE 1 = CONTAINS(POINT('ICRS', g.ra, g.dec), "
        f"CIRCLE('ICRS', {ra:.7f}, {dec:.7f}, {radius_arcmin / 60.0:.7f}))"
    )
    log.info("querying Gaia DR3 GSPC around %.5f %+.5f (r=%.1f')", ra, dec, radius_arcmin)
    previous_limit = Gaia.ROW_LIMIT
    try:
        Gaia.ROW_LIMIT = row_limit
        raw = Gaia.launch_job_async(adql).get_results()
    finally:
        Gaia.ROW_LIMIT = previous_limit
    if raw is None or len(raw) == 0:
        raise CatalogError("Gaia GSPC returned no sources")

    out = Table()
    out["ra"] = _float_column(raw, "ra")
    out["dec"] = _float_column(raw, "dec")
    out["source_id"] = raw["source_id"]
    for source_col, name in (("phot_g_mean_mag", "Gmag"), ("bp_rp", "bp_rp")):
        values = _float_column(raw, source_col)
        if values is not None:
            out[name] = values

    for band, pre in {**_GAIA_JKC, **_GAIA_SDSS}.items():
        mag_col, flux_col, err_col = f"{pre}_mag", f"{pre}_flux", f"{pre}_flux_error"
        if mag_col not in raw.colnames:
            continue
        mag = _float_column(raw, mag_col)
        flux = _float_column(raw, flux_col)
        eflux = _float_column(raw, err_col)
        if flux is not None and eflux is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                err = np.where(flux > 0, 1.0857362 * eflux / flux, np.nan)
        else:
            err = np.full(len(raw), np.nan)
        out[band] = mag
        out[f"e_{band}"] = err
        flag_col = f"{pre}_flag"
        if flag_col in raw.colnames:
            out[f"f_{band}"] = np.asarray(raw[flag_col])

    out.meta["provider"] = "gaia_synth"
    out.meta["systems"] = {b: "vega" for b in _GAIA_JKC}
    out.meta["systems"].update({b: "ab" for b in _GAIA_SDSS})
    return out


def fetch_gaia_astrometry(ra: float, dec: float, radius_arcmin: float,
                          max_mag: float = 20.0, row_limit: int = 50000) -> Table:
    """Gaia DR3 positions for refining the WCS.

    Deliberately separate from ``gaia_synth``: the synthetic-photometry table
    only covers sources brighter than G ~ 17.65, whereas an astrometric fit
    wants as many well-measured positions as it can get. Proper motions are
    applied to bring the catalogue to the epoch of observation only if the
    caller passes an epoch; at this pixel scale the difference matters for a
    handful of high-proper-motion stars at most, and the sigma clipping in the
    fit removes them anyway.
    """
    from astroquery.gaia import Gaia

    adql = (
        "SELECT source_id, ra, dec, pmra, pmdec, phot_g_mean_mag, "
        "astrometric_excess_noise, ruwe "
        "FROM gaiadr3.gaia_source "
        f"WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), "
        f"CIRCLE('ICRS', {ra:.7f}, {dec:.7f}, {radius_arcmin / 60.0:.7f})) "
        f"AND phot_g_mean_mag < {max_mag:.2f}"
    )
    log.info("querying Gaia DR3 positions around %.5f %+.5f (r=%.1f', G < %.1f)",
             ra, dec, radius_arcmin, max_mag)
    previous_limit = Gaia.ROW_LIMIT
    try:
        Gaia.ROW_LIMIT = row_limit
        raw = Gaia.launch_job_async(adql).get_results()
    finally:
        Gaia.ROW_LIMIT = previous_limit
    if raw is None or len(raw) == 0:
        raise CatalogError("Gaia returned no astrometric sources")

    out = Table()
    out["ra"] = np.asarray(raw["ra"], dtype=float)
    out["dec"] = np.asarray(raw["dec"], dtype=float)
    out["source_id"] = raw["source_id"]
    for column in ("pmra", "pmdec", "phot_g_mean_mag", "ruwe"):
        values = _float_column(raw, column)
        if values is not None:
            out[column] = values

    # Astrometrically poorly behaved sources make bad reference points.
    if "ruwe" in out.colnames:
        good = ~np.isfinite(out["ruwe"]) | (out["ruwe"] < 1.4)
        out = out[good]

    out.meta["provider"] = "gaia_astrometry"
    return out


def fetch_panstarrs(ra: float, dec: float, radius_arcmin: float,
                    min_detections: int = 5, row_limit: int = 20000) -> Table:
    """Pan-STARRS1 DR2 mean-object catalogue through the MAST API."""
    import requests

    if dec < -31.0:
        raise CatalogError("Pan-STARRS1 does not cover Dec < -30 deg")

    bands = ["g", "r", "i", "z", "y"]
    columns = ["objID", "raMean", "decMean", "nDetections"]
    for b in bands:
        columns += [f"{b}MeanPSFMag", f"{b}MeanPSFMagErr", f"{b}MeanKronMag"]

    url = "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/mean.csv"
    params = {
        "ra": f"{ra:.7f}", "dec": f"{dec:.7f}", "radius": f"{radius_arcmin / 60.0:.7f}",
        "nDetections.gte": int(min_detections), "pagesize": int(row_limit),
        "columns": "[" + ",".join(columns) + "]",
    }
    log.info("querying Pan-STARRS1 DR2 around %.5f %+.5f (r=%.1f')", ra, dec, radius_arcmin)
    response = requests.get(url, params=params, timeout=120)
    response.raise_for_status()
    text = response.text.strip()
    if not text or "\n" not in text:
        raise CatalogError("Pan-STARRS1 returned no sources")

    raw = Table.read(text, format="ascii.csv")
    out = Table()
    out["ra"] = _float_column(raw, "raMean")
    out["dec"] = _float_column(raw, "decMean")
    out["objID"] = raw["objID"]
    out["nDetections"] = np.nan_to_num(_float_column(raw, "nDetections"), nan=0).astype(int)
    for b in bands:
        mag = _float_column(raw, f"{b}MeanPSFMag")
        err = _float_column(raw, f"{b}MeanPSFMagErr")
        kron = _float_column(raw, f"{b}MeanKronMag")
        mag = np.where(mag < -900, np.nan, mag)
        err = np.where(err < -900, np.nan, err)
        kron = np.where(kron < -900, np.nan, kron)
        out[b] = mag
        out[f"e_{b}"] = err
        out[f"k_{b}"] = kron          # PSF - Kron separates stars from galaxies

    # Star/galaxy separation: point sources have PSF ~= Kron magnitude. A
    # source with neither an r nor an i Kron magnitude cannot be classified at
    # all; giving it False dropped it from *every* band, g and z included, where
    # its own photometry was perfectly good. Unclassified now means "keep".
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # all-NaN rows are expected
        ref = np.nanmedian(np.column_stack([out[f"{b}"] - out[f"k_{b}"]
                                            for b in ("r", "i")]), axis=1)
    out["point_source"] = np.where(np.isfinite(ref), np.abs(ref) < 0.10, True)

    out.meta["provider"] = "panstarrs"
    out.meta["systems"] = {b: "ab" for b in bands}
    return out


def fetch_apass(ra: float, dec: float, radius_arcmin: float, row_limit: int = 20000) -> Table:
    """AAVSO APASS DR9 (VizieR II/336/apass9) -- B, V and Sloan g' r' i'."""
    from astroquery.vizier import Vizier

    viz = Vizier(columns=["RAJ2000", "DEJ2000", "nobs", "Bmag", "e_Bmag", "Vmag", "e_Vmag",
                          "g'mag", "e_g'mag", "r'mag", "e_r'mag", "i'mag", "e_i'mag"],
                 row_limit=row_limit)
    log.info("querying APASS DR9 around %.5f %+.5f (r=%.1f')", ra, dec, radius_arcmin)
    result = viz.query_region(SkyCoord(ra, dec, unit="deg"),
                              radius=radius_arcmin * u.arcmin, catalog="II/336/apass9")
    if result is None or len(result) == 0:
        raise CatalogError("APASS returned no sources")
    raw = result[0]

    out = Table()
    out["ra"] = _float_column(raw, "RAJ2000")
    out["dec"] = _float_column(raw, "DEJ2000")
    mapping = {"B": "Bmag", "V": "Vmag", "g": "g'mag", "r": "r'mag", "i": "i'mag"}
    for band, col in mapping.items():
        values = _float_column(raw, col)
        if values is None:
            continue
        out[band] = values
        errors = _float_column(raw, "e_" + col)
        out[f"e_{band}"] = errors if errors is not None else np.full(len(raw), np.nan)

    out.meta["provider"] = "apass"
    out.meta["systems"] = {"B": "vega", "V": "vega", "g": "ab", "r": "ab", "i": "ab"}
    return out


def load_local(path: Path) -> Table:
    """A user-supplied catalogue file (CSV/ECSV) with ra, dec, <band>, e_<band>."""
    path = Path(path)
    fmt = "ascii.ecsv" if path.suffix.lower() == ".ecsv" else "ascii.csv"
    table = Table.read(path, format=fmt)
    lower = {c.lower(): c for c in table.colnames}
    for want in ("ra", "dec"):
        if want not in lower:
            raise CatalogError(f"{path} must contain '{want}' column")
        table.rename_column(lower[want], want)
    table.meta["provider"] = "local"
    return table


_FETCHERS = {
    "gaia_astrometry": fetch_gaia_astrometry,
    "gaia_synth": fetch_gaia_synth,
    "panstarrs": fetch_panstarrs,
    "apass": fetch_apass,
}


def get_catalog(provider: str, ra: float, dec: float, radius_arcmin: float,
                cache_root: Path, refresh: bool = False,
                local_path: Optional[Path] = None,
                options: Optional[Mapping] = None) -> Table:
    """Fetch (or read from cache) a reference catalogue around a position.

    *options* are extra keyword arguments for the provider's fetcher, and any
    that differ from its default become part of the cache key -- so raising
    ``astrometry.max_mag`` fetches a deeper catalogue instead of quietly reusing
    the shallow cached one, while leaving it alone still hits the existing
    cache.
    """
    if provider == "local":
        if local_path is None:
            raise CatalogError("provider 'local' needs calibration.local_catalog to be set")
        return load_local(local_path)

    if provider not in _FETCHERS:
        raise CatalogError(f"unknown catalogue provider '{provider}'")

    fetcher = _FETCHERS[provider]
    options = _non_default_options(fetcher, options)
    variant = "_".join(f"{k}{v}" for k, v in sorted(options.items())) or None

    if not refresh:
        cached = load_cached(cache_root, provider, ra, dec, radius_arcmin, variant)
        if cached is not None:
            return cached

    # Query the grid point too, so the download matches what gets cached.
    table = fetcher(_snap(ra), _snap(dec), radius_arcmin, **options)
    save_cache(table, cache_root, provider, ra, dec, radius_arcmin, variant)
    log.info("%s: %d sources retrieved", provider, len(table))
    return table


def _non_default_options(fetcher, options: Optional[Mapping]) -> dict:
    """Keep only the options that actually change what the fetcher returns."""
    import inspect

    if not options:
        return {}
    defaults = inspect.signature(fetcher).parameters
    out = {}
    for key, value in options.items():
        if value is None or key not in defaults:
            continue
        default = defaults[key].default
        if default is inspect.Parameter.empty or float(value) != float(default):
            out[key] = value
    return out


def max_error_for(cfg, provider: str) -> float:
    """The catalogue-error cut, which is properly a per-survey quantity.

    Gaia synthetic photometry is precise to a few millimagnitudes; APASS is a
    small-telescope all-sky survey whose B errors sit around 0.05-0.15. One cut
    tight enough for Gaia throws almost all of APASS away -- on the SN2026acd
    field, 25 of its 27 B stars.
    """
    setting = cfg.get("calibration.max_cat_err", 0.05)
    if isinstance(setting, Mapping):
        for key, value in setting.items():
            if str(key).lower() == str(provider).lower():
                return float(value)
        return float(setting.get("default", 0.05))
    return float(setting)


def clean_catalog(table: Table, band: str, cfg) -> Table:
    """Rows usable to calibrate *band*: finite magnitude, small error, sane range."""
    if band not in table.colnames:
        return table[:0]

    mag = np.asarray(table[band], dtype=float)
    err_col = f"e_{band}"
    err = np.asarray(table[err_col], dtype=float) if err_col in table.colnames \
        else np.full(len(table), np.nan)

    lo, hi = cfg.get("calibration.mag_range", [10.0, 20.0])
    max_err = max_error_for(cfg, table.meta.get("provider", ""))
    provider = str(table.meta.get("provider", ""))

    keep = np.isfinite(mag) & (mag >= float(lo)) & (mag <= float(hi))

    # A missing error is not a small one. It is kept all the same, because
    # several providers quote none at all -- APASS through VizieR for some rows,
    # a user CSV -- and dropping them would empty the catalogue. The
    # parenthesisation is explicit because the precedence here is easy to
    # misread: `~a | b` is `(~a) | b`, so a NaN error passes.
    # `calibration.require_cat_err: true` drops them instead, which is the right
    # setting when the provider always quotes an error (Gaia GSPC, where a NaN
    # means the synthetic flux was non-positive).
    missing_err = ~np.isfinite(err)
    with np.errstate(invalid="ignore"):
        small_err = err <= max_err
    if bool(cfg.get("calibration.require_cat_err", False)):
        keep &= np.isfinite(err) & small_err
    else:
        keep &= missing_err | small_err
        n_missing = int((keep & missing_err).sum())
        if n_missing:
            log.debug("%s %s: %d entries kept with no quoted catalogue error",
                      provider or "catalogue", band, n_missing)

    # Gaia synthetic photometry carries a per-band quality flag; a non-zero
    # value means the Gaia team does not trust that synthetic magnitude.
    # Off by default because it costs stars on sparse fields.
    max_flag = cfg.get("calibration.max_gaia_synth_flag")
    flag_col = f"f_{band}"
    if max_flag is not None and flag_col in table.colnames:
        with np.errstate(invalid="ignore"):
            flags = np.asarray(table[flag_col], dtype=float)
        keep &= ~np.isfinite(flags) | (flags <= float(max_flag))

    if "point_source" in table.colnames:
        keep &= np.asarray(table["point_source"], dtype=bool)

    return table[keep]


def color_columns(table: Table, band: str) -> Optional[tuple[str, str]]:
    """Which two catalogue columns form the colour used for *band*."""
    pair = BAND_COLOR.get(band)
    if pair is None:
        return None
    b1, b2 = pair
    if b1 in table.colnames and b2 in table.colnames:
        return b1, b2
    # Fall back to any available adjacent pair.
    for cand in (("g", "r"), ("B", "V"), ("r", "i")):
        if cand[0] in table.colnames and cand[1] in table.colnames:
            return cand
    return None


# --------------------------------------------------------------------------
# known variable stars
# --------------------------------------------------------------------------
# SIMBAD object types that mean "this star does not keep the same brightness".
# Screening on the catalogue's own classification rather than on a measured
# amplitude is deliberate: with three to seven calibration stars on a sparse
# field, one Mira moves the zero point outright, and there is no way to catch it
# from a single epoch's data.
VARIABLE_OTYPES = {
    "V*", "V*?", "Pu*", "RR*", "Ce*", "cC*", "gD*", "SX*", "dS*", "RV*", "WV*",
    "bC*", "Mi*", "sr*", "LP*", "Ir*", "Or*", "RS*", "BY*", "Er*", "Fl*", "FU*",
    "RC*", "RC?", "Sy*", "CV*", "No*", "DN*", "AM*", "Ro*", "a2*", "Ae*", "be*",
    "EB*", "Al*", "bL*", "WU*", "El*", "Ep*", "SN*", "LPV*", "Pu?", "RR?", "Ce?",
}


def _variable_cache(cache_root: Path, ra: float, dec: float, radius: float) -> Path:
    return ensure_dir(Path(cache_root) / "catalogs") / (
        f"variables_{_snap(ra):.4f}{_snap(dec):+.4f}_r{radius:.1f}.ecsv")


def variable_stars(ra: float, dec: float, radius_arcmin: float, cache_root: Path,
                   refresh: bool = False) -> Optional[Table]:
    """Known variable stars in the field, from SIMBAD. Cached like a catalogue.

    Returns a table with ``ra``/``dec``/``otype``, an empty table when SIMBAD
    has nothing, or None when it could not be reached -- the caller must be able
    to tell "no variables here" from "I do not know", and must not silently
    calibrate on variables because a query failed. ``_drop_variables`` acts on
    that distinction and records it in the zero point's metadata, which reaches
    the output as the ``variables_unscreened`` flag.
    """
    path = _variable_cache(cache_root, ra, dec, radius_arcmin)
    if path.exists() and not refresh:
        try:
            table = Table.read(path)
            log.info("using cached variable-star list (%d entries)", len(table))
            return table
        except Exception as exc:
            log.warning("could not read %s: %s", path.name, exc)

    try:
        from astroquery.simbad import Simbad
        import astropy.units as u
        from astropy.coordinates import SkyCoord

        sim = Simbad()
        sim.TIMEOUT = 60
        sim.ROW_LIMIT = 2000
        for field in ("otype", "otypes"):
            try:
                sim.add_votable_fields(field)
                break
            except Exception:
                continue
        raw = sim.query_region(SkyCoord(ra * u.deg, dec * u.deg),
                               radius=float(radius_arcmin) * u.arcmin)
    except Exception as exc:
        log.warning("SIMBAD variable-star lookup failed (%s); calibration stars "
                    "are not screened for variability on this field", exc)
        return None
    if raw is None or len(raw) == 0:
        table = Table({"ra": [], "dec": [], "otype": []})
        table.write(path, overwrite=True)
        return table

    cols = {c.lower(): c for c in raw.colnames}
    ocol = cols.get("otype") or cols.get("otypes") or cols.get("main_type")
    rcol = cols.get("ra") or cols.get("ra_d")
    dcol = cols.get("dec") or cols.get("dec_d")
    if ocol is None or rcol is None or dcol is None:
        log.warning("SIMBAD returned columns %s, which do not include a type and a "
                    "position; skipping the variability screen", raw.colnames)
        return None

    otype = np.array([str(v).strip() for v in raw[ocol]])
    is_var = np.array([any(part.strip() in VARIABLE_OTYPES
                           for part in str(v).replace("|", ",").split(","))
                       for v in otype])
    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        coords = SkyCoord(raw[rcol], raw[dcol], unit=(u.deg, u.deg))
    except Exception:
        from astropy.coordinates import SkyCoord
        coords = SkyCoord(raw[rcol], raw[dcol], unit=("hourangle", "deg"))

    table = Table({"ra": coords.ra.deg[is_var], "dec": coords.dec.deg[is_var],
                   "otype": otype[is_var]})
    try:
        table.write(path, overwrite=True)
    except Exception as exc:
        log.warning("could not cache the variable-star list: %s", exc)
    log.info("SIMBAD: %d known variables among %d objects in the field",
             len(table), len(raw))
    return table
