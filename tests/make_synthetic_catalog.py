"""Build a synthetic reference catalogue from the images themselves.

This exists so the calibration, matching and light-curve machinery can be
exercised without any network access: stars are measured on one reference
epoch, given magnitudes by adding a chosen zero point per band, and written out
in the ``local`` catalogue format. A run calibrated against this file must
recover exactly those zero points.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tttphot.config import Config                      # noqa: E402
from tttphot.detection import detect_sources, select_psf_stars  # noqa: E402
from tttphot.image import ScienceImage                 # noqa: E402
from tttphot import photometry as ph                   # noqa: E402
from tttphot.pipeline import find_images, index_images, prepare_image  # noqa: E402

# Close to what the TTT3 observatory pipeline reports (g: ZP = 24.40 in ADU/s)
TRUE_ZP = {"u": 22.50, "B": 23.80, "V": 24.10, "g": 24.40, "r": 24.50, "i": 24.00, "z": 23.20}
TRUE_COLOR_TERM = {"u": 0.00, "B": 0.06, "V": -0.03, "g": 0.04, "r": -0.02, "i": 0.01, "z": 0.00}


def measure_band(entry, cfg, instrument) -> Table:
    image = prepare_image(entry.path, cfg, instrument)
    sources = detect_sources(image, cfg, fwhm_px=image.fwhm_px)
    if len(sources) == 0:
        return Table()
    fwhm = image.fwhm_px
    ann = cfg.get("photometry.annulus_scale", [5.0, 8.0])
    positions = np.column_stack([sources["xcentroid"], sources["ycentroid"]])
    phot = ph.measure_apertures(image, positions,
                                float(cfg.get("photometry.aperture_scale", 1.2)) * fwhm, cfg,
                                annulus_px=(ann[0] * fwhm, ann[1] * fwhm))
    psf_stars = select_psf_stars(image, sources, cfg, fwhm, max_stars=40,
                                 min_snr=float(cfg.get('calibration.min_snr', 20.0)))
    stars = ph.curve_of_growth(image, psf_stars, fwhm, cfg)
    flux = np.asarray(phot["flux"], dtype=float)
    err = np.asarray(phot["flux_err"], dtype=float)
    good = (np.asarray(phot["flags"]) == 0) & (flux > 0) & (err > 0) & (flux / err > 30)
    table = Table()
    table["ra"] = np.asarray(sources["ra"])[good]
    table["dec"] = np.asarray(sources["dec"])[good]
    table["inst"] = -2.5 * np.log10(flux[good] / image.meta.exptime) + stars.apcor_mag
    table["inst_err"] = 1.0857362 * err[good] / flux[good]
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", nargs="?", default="SN2026abgb")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", default="tests/synthetic_catalog.csv")
    parser.add_argument("--mjd", type=float, default=None,
                        help="reference night (defaults to the first)")
    args = parser.parse_args()

    cfg = Config.load(overrides={"paths": {"data_root": args.data_root}})
    instrument = cfg.get("instrument_config", {})
    index = index_images(find_images(cfg, args.target), instrument)

    per_band = {}
    for band in sorted({e.band for e in index}):
        entries = [e for e in index if e.band == band]
        if args.mjd is not None:
            entry = min(entries, key=lambda e: abs(e.mjd - args.mjd))
        else:
            entry = entries[0]
        table = measure_band(entry, cfg, instrument)
        if len(table):
            per_band[band] = table
            print(f"{band}: {len(table)} stars from {entry.path.name}")

    if "g" not in per_band and "V" not in per_band:
        print("need at least g or V to anchor the cross-match", file=sys.stderr)
        return 1

    anchor = "g" if "g" in per_band else "V"
    base = per_band[anchor]
    coords = SkyCoord(base["ra"] * u.deg, base["dec"] * u.deg)

    out = Table()
    out["ra"] = base["ra"]
    out["dec"] = base["dec"]
    for band, table in per_band.items():
        mag = np.full(len(base), np.nan)
        err = np.full(len(base), np.nan)
        other = SkyCoord(table["ra"] * u.deg, table["dec"] * u.deg)
        idx, sep, _ = coords.match_to_catalog_sky(other)
        hit = sep.arcsec < 1.0
        mag[hit] = np.asarray(table["inst"])[idx[hit]] + TRUE_ZP[band]
        err[hit] = np.asarray(table["inst_err"])[idx[hit]]
        out[band] = mag
        out[f"e_{band}"] = np.clip(err, 0.001, None)

    # Apply the intended colour terms so the fitted values can be checked too.
    if "g" in out.colnames and "r" in out.colnames:
        color = np.asarray(out["g"]) - np.asarray(out["r"])
        for band, term in TRUE_COLOR_TERM.items():
            if band in out.colnames:
                pivot = float(cfg.get(f"calibration.color_pivot.{band}", 0.45))
                out[band] = np.asarray(out[band]) + term * (np.nan_to_num(color) - pivot)

    keep = np.isfinite(out[anchor])
    out = out[keep]
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write(path, format="ascii.csv", overwrite=True)
    print(f"\nwrote {path} with {len(out)} stars")
    print("true zero points:", TRUE_ZP)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
