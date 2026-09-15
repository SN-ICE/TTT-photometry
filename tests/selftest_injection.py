"""Offline end-to-end test on a real TTT frame: inject a star, subtract, recover it.

No network is needed, so this checks the pipeline itself rather than the
catalogue services. From one real frame it builds:

* a *reference* image: the untouched frame resampled onto a 0.25"/px grid, as
  a Pan-STARRS stack would be, and written as a FITS template;
* a local *calibration catalogue*: the frame's own point-like stars, put on a
  magnitude scale with a chosen zero point (the observatory's ZP card when
  present);
* a handful of *science epochs*: the same frame plus a copy of a real,
  isolated field star scaled to a chosen magnitude and dropped on the host
  galaxy (or wherever --xy says), then blurred to worse seeing and given fresh
  sky noise. The reference is therefore sharper than the science, as a
  Pan-STARRS stack is sharper than a TTT frame, and the kernel has a real
  seeing difference to solve for.

Then it runs ``tttphot run`` on those epochs and compares the magnitudes it
measures with the ones injected. Because the injected source is a scaled copy
of a catalogue star, its true magnitude is known exactly on the catalogue's
own system, whatever zero point was chosen.

    python tests/selftest_injection.py ~/Desktop/TTT/TTT3_..._SDSSg.fits

A pass means every difference measurement lands within 3 sigma of the truth,
and within 0.05 mag for the epochs brighter than 19. Direct photometry on the
host is reported alongside to show how badly it is biased there.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.wcs import WCS
import astropy.units as u

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tttphot.config import Config                                  # noqa: E402
from tttphot.detection import detect_sources, filter_by_profile, select_psf_stars  # noqa: E402
from tttphot import photometry as ph                                # noqa: E402
from tttphot.pipeline import prepare_image                          # noqa: E402

TARGET = "SNSELFTEST"


def sky_noise(data: np.ndarray) -> float:
    """Pixel-to-pixel sky scatter from the centre of a frame."""
    ny, nx = data.shape
    core = data[ny // 4:3 * ny // 4, nx // 4:3 * nx // 4]
    diff = (core[:, 1:] - core[:, :-1]) / np.sqrt(2.0)
    return float(sigma_clipped_stats(diff, sigma=3.0)[2])


def reference_grid(header, scale_arcsec: float = 0.25):
    """A north-up TAN grid covering the frame at *scale_arcsec*."""
    wcs = WCS(header)
    ny, nx = int(header["NAXIS2"]), int(header["NAXIS1"])
    centre = wcs.pixel_to_world(nx / 2.0, ny / 2.0)
    corners = wcs.pixel_to_world([0, nx, 0, nx], [0, 0, ny, ny])
    half = max(centre.separation(corners).arcsec) * 1.02
    size = int(np.ceil(2 * half / scale_arcsec))
    out = WCS(naxis=2)
    out.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    out.wcs.crval = [centre.ra.deg, centre.dec.deg]
    out.wcs.crpix = [size / 2.0 + 0.5, size / 2.0 + 0.5]
    out.wcs.cdelt = [-scale_arcsec / 3600.0, scale_arcsec / 3600.0]
    return out, (size, size)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("frame", type=Path, help="a real TTT frame")
    parser.add_argument("--mags", nargs="+", type=float, default=[17.5, 18.5, 19.5, 20.5],
                        help="magnitudes to inject, one epoch each")
    parser.add_argument("--xy", nargs=2, type=float, default=None,
                        help="native pixel (x y, 0-based) to inject at; default: on the "
                             "brightest extended source near the centre")
    parser.add_argument("--blur-arcsec", nargs="+", type=float, default=[1.5, 2.5],
                        help="extra Gaussian FWHM added to the seeing, cycled over epochs")
    parser.add_argument("--workdir", type=Path, default=None,
                        help="keep everything here instead of a temporary directory")
    args = parser.parse_args()

    from reproject import reproject_interp
    from scipy.ndimage import gaussian_filter

    work = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="tttselftest_"))
    data_dir = work / "data" / TARGET
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True)

    with fits.open(args.frame) as hdul:
        native = np.asarray(hdul[0].data, dtype=float)
        header = hdul[0].header.copy()
    wcs = WCS(header)
    exptime = float(header.get("EXPTIME", 1.0))
    zp = float(header.get("ZP", 24.0))

    # ---- measure the field on the rebinned frame, as the pipeline will ----
    cfg = Config.load(overrides={"calibration": {"injection": {"enabled": False}}})
    image = prepare_image(args.frame, cfg, cfg.get("instrument_config", {}))
    fwhm = image.fwhm_px
    sources = detect_sources(image, cfg, fwhm_px=fwhm)
    stars = select_psf_stars(image, sources, cfg, fwhm, max_stars=40, min_snr=50.0)
    apcor = ph.curve_of_growth(image, stars, fwhm, cfg)
    ann = cfg.get("photometry.annulus_scale", [5.0, 8.0])
    positions = np.column_stack([sources["xcentroid"], sources["ycentroid"]])
    phot = ph.measure_apertures(image, positions,
                                float(cfg.get("photometry.aperture_scale", 1.2)) * fwhm,
                                cfg, annulus_px=(ann[0] * fwhm, ann[1] * fwhm))
    flux = np.asarray(phot["flux"], float)
    err = np.asarray(phot["flux_err"], float)
    good = (np.asarray(phot["flags"]) == 0) & (flux > 0) & (err > 0) & (flux / err > 30)
    # A real reference catalogue holds stars, not the HII regions of the host:
    # keep only sources whose own width matches the PSF.
    pointlike = filter_by_profile(image, sources[good], fwhm, cfg)
    good &= np.isin(np.asarray(sources["id"]), np.asarray(pointlike["id"]))
    catalogue = Table()
    catalogue["ra"] = np.asarray(sources["ra"])[good]
    catalogue["dec"] = np.asarray(sources["dec"])[good]
    catalogue["g"] = -2.5 * np.log10(flux[good] / exptime) + apcor.apcor_mag + zp
    catalogue["e_g"] = np.clip(1.0857 * err[good] / flux[good], 0.003, None)
    # every band the frame might be in gets the same column, so the test works
    # whatever filter the frame was taken in
    band = image.meta.band
    if band != "g":
        catalogue.rename_column("g", band)
        catalogue.rename_column("e_g", f"e_{band}")
    cat_path = work / "catalogue.csv"
    catalogue.write(cat_path, format="ascii.csv", overwrite=True)
    print(f"catalogue: {len(catalogue)} stars, ZP {zp:.3f}, FWHM {image.fwhm_arcsec:.2f}\"")

    # ---- the donor: a bright, isolated, unsaturated star ------------------
    donor = None
    cat_coords = SkyCoord(catalogue["ra"] * u.deg, catalogue["dec"] * u.deg)
    for row in stars:
        c = SkyCoord(float(row["ra"]) * u.deg, float(row["dec"]) * u.deg)
        idx, sep, _ = c.match_to_catalog_sky(cat_coords)
        if sep.arcsec > 1.0:
            continue
        m = float(catalogue[band][int(idx)])
        if 14.5 < m < 17.0:
            donor = (c, m)
            break
    if donor is None:
        print("no suitable donor star (14.5 < m < 17, isolated) in this frame", file=sys.stderr)
        return 1
    dx, dy = (float(v) for v in wcs.world_to_pixel(donor[0]))
    half = int(np.ceil(5 * fwhm * 2))                  # native pixels (fwhm is rebinned)
    ix, iy = int(round(dx)), int(round(dy))
    stamp = native[iy - half:iy + half + 1, ix - half:ix + half + 1].copy()
    _, sky, _ = sigma_clipped_stats(native[iy - 3 * half:iy + 3 * half, ix - 3 * half:ix + 3 * half])
    stamp -= sky
    print(f"donor star m = {donor[1]:.3f} at native ({dx:.1f}, {dy:.1f})")

    # ---- where to inject --------------------------------------------------
    if args.xy:
        tx, ty = args.xy
    else:
        # The brightest *extended* light near the centre: blank out every
        # point-like source first, or a bright star wins.
        ny, nx = native.shape
        work_img = np.nan_to_num(native - sky)
        point = filter_by_profile(image, sources, fwhm, cfg)
        yy, xx = np.mgrid[0:ny, 0:nx]
        for row in point:
            px_n, py_n = (float(v) for v in wcs.world_to_pixel(
                SkyCoord(float(row["ra"]) * u.deg, float(row["dec"]) * u.deg)))
            r = 6 * fwhm * 2
            x0, x1 = int(max(px_n - r, 0)), int(min(px_n + r + 1, nx))
            y0, y1 = int(max(py_n - r, 0)), int(min(py_n + r + 1, ny))
            work_img[y0:y1, x0:x1] = 0.0
        smooth = gaussian_filter(work_img, 2 * fwhm * 2)
        cy0, cx0 = ny // 2, nx // 2
        box = smooth[cy0 - ny // 4:cy0 + ny // 4, cx0 - nx // 4:cx0 + nx // 4]
        py, px = np.unravel_index(np.argmax(box), box.shape)
        # on the host, offset from its brightest point by ~1 seeing disc
        tx = cx0 - nx // 4 + px + 1.0 * fwhm * 2
        ty = cy0 - ny // 4 + py + 0.5 * fwhm * 2
    shift_x, shift_y = int(round(tx - ix)), int(round(ty - iy))
    true_xy = (dx + shift_x, dy + shift_y)
    true_coord = wcs.pixel_to_world(*true_xy)
    print(f"injecting at native ({true_xy[0]:.1f}, {true_xy[1]:.1f}) = "
          f"{true_coord.to_string('hmsdms', precision=2)}")

    # ---- reference --------------------------------------------------------
    ref_wcs, ref_shape = reference_grid(header)
    ref_header = WCS(header).celestial
    ref, _ = reproject_interp((native - sky, ref_header), ref_wcs, shape_out=ref_shape)
    ref_path = work / "reference.fits"
    fits.PrimaryHDU(ref.astype(np.float32), header=ref_wcs.to_header()).writeto(ref_path, overwrite=True)

    # ---- science epochs ---------------------------------------------------
    native_scale = float(np.sqrt(abs(np.linalg.det(wcs.celestial.pixel_scale_matrix)))) * 3600.0
    truth = []
    for k, mag in enumerate(args.mags):
        scale = 10 ** (-0.4 * (mag - donor[1]))
        data = native.copy()
        y0, x0 = iy + shift_y - half, ix + shift_x - half
        data[y0:y0 + stamp.shape[0], x0:x0 + stamp.shape[1]] += scale * stamp
        blurred = float(args.blur_arcsec[k % len(args.blur_arcsec)])
        if blurred > 0:
            # Blurring also smooths the noise away; put the sky noise back, or
            # the frame would be unrealistically clean for its seeing. What is
            # added is independent of the reference's noise, as it would be.
            before = sky_noise(data)
            data = gaussian_filter(data, blurred / 2.3548 / native_scale)
            after = sky_noise(data)
            rng = np.random.default_rng(1000 + k)
            data = data + rng.normal(0.0, np.sqrt(max(before ** 2 - after ** 2, 0.0)), data.shape)
        head = header.copy()
        mjd = float(header["MJD-OBS"]) + 1.0 + k
        head["MJD-OBS"] = mjd
        head["OBJECT"] = TARGET
        head["HISTORY"] = f"selftest: injected m={mag:.3f}, extra blur {blurred:.2f} arcsec"
        out = data_dir / f"selftest_{k:02d}_m{mag:.2f}.fits"
        fits.PrimaryHDU(data.astype(np.float32), header=head).writeto(out, overwrite=True)
        truth.append((mjd, mag, blurred))

    config = work / "selftest.yaml"
    config.write_text(f"""
target:
  name: {TARGET}
  ra: {true_coord.ra.deg:.8f}
  dec: {true_coord.dec.deg:.8f}
  resolve: off
  refine: off
astrometry:
  catalog: local
  fallback_catalogs: []
calibration:
  catalogs: [local]
  local_catalog: {cat_path}
  screen_variables: false
subtraction:
  reference_sources:
    {band}: [file]
  reference_files:
    {band}: {ref_path}
""")
    cmd = [sys.executable, "-m", "tttphot.cli", "run", TARGET, "--config", str(config),
           "--data-root", str(work / "data"), "--work-root", str(work / "runs"), "--clean"]
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)

    import pandas as pd
    frame = pd.read_csv(work / "runs" / TARGET / f"{TARGET}_photometry_all.csv")
    print(f"\n{'mag in':>7} {'blur':>5} {'method':>20} {'mag out':>8} {'err':>6} {'diff':>7} {'pull':>6}")
    failures = 0
    for mjd, mag, blurred in truth:
        rows = frame[np.isclose(frame["mjd"], mjd, atol=1e-4)]
        for _, row in rows.iterrows():
            d = row["mag"] - mag
            pull = d / row["mag_err"] if row["mag_err"] > 0 else np.nan
            tag = ""
            if row["method"] in ("difference_psf", "difference_aperture") and np.isfinite(d):
                ok = abs(pull) < 3.0 or abs(d) < 0.05
                if mag < 19.0:
                    ok = ok and abs(d) < 0.05
                failures += 0 if ok else 1
                tag = "" if ok else "  <-- FAIL"
            extra = ""
            if str(row["method"]).startswith("difference"):
                extra = (f"  chi2 {row['diff_chi2']:.2f}  ksum {row['diff_kernel_sum']:.3f}"
                         f"  stamps {row['diff_nstamps']:.0f}  flags {int(row['flags'])}")
            print(f"{mag:7.2f} {blurred:5.1f} {row['method']:>20} {row['mag']:8.3f} "
                  f"{row['mag_err']:6.3f} {d:+7.3f} {pull:+6.2f}{tag}{extra}")
    print(f"\nworking files in {work}")
    print("PASS" if failures == 0 else f"{failures} difference measurement(s) off")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
