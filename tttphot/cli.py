"""Command-line interface.

    tttphot run SN2026abgb --data-root ~/Desktop/TTT
    tttphot run SN2026abgb --bands g r --no-subtraction
    tttphot list SN2026abgb
    tttphot fetch-catalog SN2026abgb         # pre-download reference catalogues
    tttphot colorterms SN2026abgb            # instrument colour terms from many frames
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

from .config import Config, PIPELINE_ROOT
from .utils import get_logger, setup_logging

log = get_logger("tttphot.cli")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("target", help="target name, e.g. SN2026abgb")
    parser.add_argument("--config", type=Path, default=None, help="extra YAML config to merge")
    parser.add_argument("--data-root", type=Path, default=None, help="directory holding the images")
    parser.add_argument("--work-root", type=Path, default=None, help="where to write the outputs")
    parser.add_argument("--bands", nargs="+", default=None, help="restrict to these bands")
    parser.add_argument("--log-level", default="INFO")


def _build_config(args) -> Config:
    overrides: dict = {}
    if args.data_root:
        overrides.setdefault("paths", {})["data_root"] = str(Path(args.data_root).expanduser())
    if getattr(args, "work_root", None):
        overrides.setdefault("paths", {})["work_root"] = str(Path(args.work_root).expanduser())
    if getattr(args, "no_subtraction", False):
        overrides.setdefault("subtraction", {})["enabled"] = False
    if getattr(args, "no_astrometry", False):
        overrides.setdefault("astrometry", {})["mode"] = "off"
    if getattr(args, "no_pattern", False):
        overrides.setdefault("image", {}).setdefault("pattern", {})["enabled"] = False
    if getattr(args, "aperture_scale", None):
        overrides.setdefault("photometry", {})["aperture_scale"] = float(args.aperture_scale)
    if getattr(args, "catalogs", None):
        overrides.setdefault("calibration", {})["catalogs"] = list(args.catalogs)
    return Config.load(user_config=args.config, target=args.target, overrides=overrides)


def cmd_run(args) -> int:
    from . import pipeline

    # Logging first, so the configuration layering is reported; then again with
    # the log file, once the config says where the run directory is.
    setup_logging(args.log_level)
    cfg = _build_config(args)
    outdir = Path(cfg.work_root) / args.target
    if getattr(args, "clean", False) and outdir.is_dir():
        import shutil
        work_root = Path(cfg.work_root).resolve()
        if work_root in outdir.resolve().parents:
            shutil.rmtree(outdir, ignore_errors=True)
            print(f"removed the previous outputs in {outdir}")
        else:
            print(f"refusing to clean {outdir}: outside {work_root}", file=sys.stderr)

    setup_logging(args.log_level, logfile=outdir / "run.log")
    result = pipeline.run(cfg, args.target, bands=args.bands, max_images=args.max_images,
                          refresh=args.refresh, skip_subtraction=args.no_subtraction,
                          append=args.append, skip=args.skip)
    print(f"outputs in {result['outdir']}")
    return 0


def cmd_list(args) -> int:
    from .pipeline import find_images, index_images

    setup_logging(args.log_level)
    cfg = _build_config(args)
    paths = find_images(cfg, args.target)
    index = index_images(paths, cfg.get("instrument_config", {}), cfg=cfg)
    if args.bands:
        index = [e for e in index if e.band in args.bands]
    print(f"{len(index)} images for {args.target} under {cfg.data_root}")
    print(f"{'band':>5} {'N':>4} {'MJD first':>12} {'MJD last':>12}")
    for band in sorted({e.band for e in index}):
        group = [e for e in index if e.band == band]
        print(f"{band:>5} {len(group):>4} {min(e.mjd for e in group):>12.4f} "
              f"{max(e.mjd for e in group):>12.4f}")

    # A campaign can straddle a camera swap or a change of readout window.
    # That is not a detail: patterns, plate scale, gain and field size all
    # follow the detector, so say so plainly rather than leaving it to be
    # discovered in the light curve.
    detectors = sorted({e.detector for e in index if e.detector is not None},
                       key=lambda d: d.token)
    if len(detectors) > 1:
        print(f"\n{len(detectors)} detector configurations:")
        print(f"  {'detector':24} {'N':>4} {'MJD first':>12} {'MJD last':>12}  bands")
        for det in detectors:
            group = [e for e in index if e.detector == det]
            print(f"  {det.label:24} {len(group):>4} {min(e.mjd for e in group):>12.4f} "
                  f"{max(e.mjd for e in group):>12.4f}  "
                  f"{','.join(sorted({e.band for e in group}))}")
        calib_root = None
        try:
            from . import pipeline, patterns as pat
            calib_root = pipeline._calib_root(cfg)
            missing = [f"{b} on {d.label}"
                       for d in detectors for b in sorted({e.band for e in index if e.detector == d})
                       if pat.load_pattern(calib_root, b, d) is None]
        except Exception:
            missing = []
        if missing:
            print(f"\n  no detector pattern yet for: {', '.join(missing)}")
            print(f"  build them with:  tttphot build-pattern {args.target}")

    if args.verbose:
        print()
        for entry in index:
            label = entry.detector.label if entry.detector else "?"
            print(f"  {entry.mjd:12.5f} {entry.band:>3} {entry.exptime:6.0f}s  "
                  f"{label:20} {entry.path.name}")
    return 0


_COORD_IN_NAME = re.compile(r"(?<![0-9])(\d{2,3}\.\d{3,6})([+-]\d{1,2}\.\d{3,6})")


def _cached_for_target(directory: Path, target: str, coord, tol_arcmin: float = 8.0):
    """Cache files belonging to one target: by name, or by position in the name.

    Catalogue and survey-cutout caches are keyed by coordinates rather than by
    target, so matching on the numbers embedded in the filename is what ties
    them back to a target.
    """
    import numpy as np
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    if not directory.is_dir():
        return []
    hits = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        if path.name.startswith(f"{target}_"):
            hits.append(path)
            continue
        if coord is None:
            continue
        match = _COORD_IN_NAME.search(path.name)
        if not match:
            continue
        try:
            other = SkyCoord(float(match.group(1)) * u.deg, float(match.group(2)) * u.deg)
        except Exception:
            continue
        if coord.separation(other).arcmin < tol_arcmin:
            hits.append(path)
    return hits


def _describe_size(paths) -> str:
    total = sum(p.stat().st_size for p in paths if p.exists())
    return f"{total / 1e6:.1f} MB" if total else "0 MB"


def cmd_recalibrate(args) -> int:
    """Recover missing zero points in an existing run, without re-measuring."""
    import numpy as np
    import pandas as pd
    from . import lightcurve as lc
    from . import plots
    from . import zeropoints

    setup_logging(args.log_level)
    cfg = _build_config(args)
    outdir = Path(cfg.work_root) / args.target
    table = outdir / f"{args.target}_photometry_all.csv"
    if not table.exists():
        print(f"no photometry table at {table}", file=sys.stderr)
        return 1

    frame = pd.read_csv(table)
    before = int((~np.isfinite(frame["zp"])).sum())
    print(f"{len(frame)} rows, {before} without a zero point\n")

    offsets = zeropoints.band_offsets(frame,
                                      min_nights=int(cfg.get("calibration.transfer_min_nights", 3)))
    if offsets:
        print(f"{'pair':<8} {'offset':>8} {'scatter':>9} {'nights':>7}")
        for (a, b), (offset, scatter, n) in sorted(offsets.items()):
            if a < b:
                print(f"{a + '-' + b:<8} {offset:>8.3f} {scatter:>9.3f} {n:>7d}")
        print()

    frame, report = zeropoints.transfer(frame, cfg)
    if not report["recovered"]:
        print("nothing could be recovered")
        return 0

    print(f"{'image':<42} {'band':>4} {'from':>9} {'ZP':>8} {'+-':>6} {'spread':>7}")
    for item in report["recovered"]:
        print(f"{item['image'][:42]:<42} {item['band']:>4} {item['from']:>9} "
              f"{item['zp']:>8.3f} {item['zp_err']:>6.3f} "
              f"{item.get('partner_spread', float('nan')):>7.3f}")
    print("\n'spread' is how much the partner bands disagree: small means the "
          "night's\nextinction was grey and the transfer is solid, large means it "
          "was not.")
    if report["unrecovered"]:
        print(f"\n{len(report['unrecovered'])} frames still unrecovered "
              f"(no calibrated band that night)")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    # Keep the measured table before overwriting it: recalibration only fills
    # in what was missing, but it rewrites the file to do so.
    import shutil
    backup = table.with_suffix(".csv.bak")
    if not backup.exists():
        shutil.copy2(table, backup)
        print(f"\nkept the original as {backup.name}")

    frame = lc.reselect_primary(frame, str(cfg.get("output.primary_method", "auto")))
    frame.to_csv(table, index=False, float_format="%.6f")
    slim_path = outdir / f"{args.target}_lightcurve.csv"
    primary = frame[frame["primary"].astype(bool)]
    keep = [c for c in pd.read_csv(slim_path, nrows=0).columns if c in primary.columns]
    primary[keep].to_csv(slim_path, index=False, float_format="%.6f")
    plots.lightcurve(outdir / f"{args.target}_lightcurve.png", frame, args.target)
    plots.run_diagnostics(outdir / f"{args.target}_diagnostics.png", frame, args.target)
    print(f"\nrewrote {table.name}, {slim_path.name} and the plots")
    return 0


def cmd_clean(args) -> int:
    """Delete a target's outputs, and optionally its cached downloads."""
    import shutil
    from . import pipeline

    setup_logging(args.log_level)
    cfg = _build_config(args)

    work_root = Path(cfg.work_root).resolve()
    cache_root = Path(cfg.cache_root).resolve()
    outdir = (work_root / args.target).resolve()

    # Never delete outside the two roots the pipeline owns.
    if work_root not in outdir.parents:
        print(f"refusing to touch {outdir}: outside {work_root}", file=sys.stderr)
        return 1

    coord = None
    if args.templates or args.catalogs or args.all:
        try:
            index = pipeline.index_images(pipeline.find_images(cfg, args.target),
                                          cfg.get("instrument_config", {}), cfg=cfg)
            coord, _ = pipeline.resolve_target_coord(cfg, index, cfg.get("instrument_config", {}))
        except Exception as exc:
            log.warning("could not work out the field position (%s); cached downloads "
                        "will only be matched by target name", exc)

    plan = []
    if args.runs or args.all or not (args.templates or args.catalogs):
        if outdir.is_dir():
            files = [p for p in outdir.rglob("*") if p.is_file()]
            plan.append(("run outputs", outdir, files))
    if args.templates or args.all:
        hits = _cached_for_target(cache_root / "templates", args.target, coord)
        if hits:
            plan.append(("cached templates", cache_root / "templates", hits))
    if args.catalogs or args.all:
        hits = _cached_for_target(cache_root / "catalogs", args.target, coord)
        if hits:
            plan.append(("cached catalogues", cache_root / "catalogs", hits))

    if not plan:
        print(f"nothing to clean for {args.target}")
        return 0

    for label, where, files in plan:
        print(f"{label}: {len(files)} files, {_describe_size(files)}  in {where}")
        if args.verbose:
            for path in files[:40]:
                print(f"    {path.relative_to(where)}")
            if len(files) > 40:
                print(f"    ... and {len(files) - 40} more")

    if args.dry_run:
        print("\n--dry-run: nothing deleted")
        return 0

    for label, where, files in plan:
        if label == "run outputs":
            shutil.rmtree(where, ignore_errors=True)
        else:
            for path in files:
                try:
                    path.unlink()
                except OSError as exc:
                    log.warning("could not delete %s: %s", path, exc)
    print("\ndeleted")
    return 0


def cmd_resolve(args) -> int:
    """Look a target name up and optionally record it in the target file."""
    from . import nameresolve

    setup_logging(args.log_level)
    cfg = _build_config(args)

    sources = [args.source] if args.source else None
    found = nameresolve.resolve(args.target, cfg, sources=sources)
    if found is None:
        print(f"could not resolve {args.target}")
        print("TNS needs a bot key: set tns.api_key / tns.bot_id / tns.bot_name in a "
              "config file, or export TNS_API_KEY / TNS_BOT_ID / TNS_BOT_NAME.")
        return 1

    print(found.describe())
    current_ra, current_dec = cfg.get("target.ra"), cfg.get("target.dec")
    if current_ra is not None and current_dec is not None:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        try:
            if isinstance(current_ra, str) and ":" in current_ra:
                current = SkyCoord(str(current_ra), str(current_dec), unit=(u.hourangle, u.deg))
            else:
                current = SkyCoord(float(current_ra), float(current_dec), unit="deg")
            print(f"target file currently has {current.to_string('hmsdms', precision=3)}"
                  f"  ({current.separation(found.coord).arcsec:.2f}\" away)")
        except Exception:
            pass

    if args.write:
        path, previous = nameresolve.update_target_file(args.target, found)
        print(f"wrote {path}" + (f" (was ra: {previous})" if previous else ""))
    else:
        print("\nadd --write to record this in the target file")
    return 0


def cmd_fetch_catalog(args) -> int:
    """Download and cache the reference catalogues for a field, so that later
    runs work without internet access."""
    from astropy.io import fits
    from . import catalogs as cat
    from .pipeline import find_images, index_images, resolve_target_coord

    setup_logging(args.log_level)
    cfg = _build_config(args)
    paths = find_images(cfg, args.target)
    index = index_images(paths, cfg.get("instrument_config", {}), cfg=cfg)
    coord, _ = resolve_target_coord(cfg, index, cfg.get("instrument_config", {}))
    radius = _catalog_radius(cfg, index, args.radius)
    print(f"catalogue radius: {radius:.0f}'")

    providers = args.catalogs or cfg.get("calibration.catalogs", ["gaia_synth"])
    ok = 0
    for provider in providers:
        try:
            table = cat.get_catalog(provider, float(coord.ra.deg), float(coord.dec.deg),
                                    radius, Path(cfg.cache_root), refresh=args.refresh,
                                    local_path=cfg.resolve_path(
                                        cfg.get("calibration.local_catalog")))
            skip = {"ra", "dec", "objID", "source_id"}
            bands = [c for c in table.colnames
                     if len(c) <= 2 and c not in skip
                     and not c.startswith(("e_", "f_", "k_"))]
            print(f"{provider:>12}: {len(table)} sources, bands {','.join(bands)}")
            ok += 1
        except Exception as exc:
            print(f"{provider:>12}: FAILED -- {exc}")
    return 0 if ok else 1


def cmd_fetch_template(args) -> int:
    """Fetch the reference image for each band and report what came back.

    Worth running once on a new field: it exercises the download on its own,
    prints what the image looks like, and leaves a PNG you can eyeball before
    committing to a full run.
    """
    import numpy as np
    from . import pipeline
    from . import templates as tpl

    setup_logging(args.log_level)
    cfg = _build_config(args)
    instrument = cfg.get("instrument_config", {})
    cache_root = Path(cfg.cache_root)

    index = pipeline.index_images(pipeline.find_images(cfg, args.target), instrument, cfg=cfg)
    if not index:
        print(f"no images found for {args.target}", file=sys.stderr)
        return 1
    coord, _ = pipeline.resolve_target_coord(cfg, index, instrument)
    print(f"field centre: {coord.to_string('hmsdms', precision=2)}\n")

    bands = args.bands or sorted({e.band for e in index})
    outdir = Path(cfg.work_root) / args.target / "templates"
    patterns = pipeline.load_patterns(
        cfg, bands, sorted({e.detector for e in index if e.detector is not None},
                           key=lambda d: d.token))

    if args.compare:
        return _compare_templates(cfg, args, index, coord, bands, cache_root,
                                  patterns, outdir)

    ok = 0
    for band in bands:
        class _Stub:
            class meta:
                pass
        stub = _Stub()
        stub.meta.band = band
        stub.meta.mjd = float("nan")

        reference = tpl.get_reference(stub, cfg, cache_root, coord, [], args.target,
                                      refresh=args.refresh,
                                      stamp_arcmin=_stamp_for(cfg, index, band))
        if reference is None:
            print(f"{band:>3}: no reference available")
            continue

        data = reference.data
        finite = np.isfinite(data)
        print(f"{band:>3}: {reference.source:<10} {data.shape[1]}x{data.shape[0]} px, "
              f"{reference.pixel_scale:.3f}\"/px, "
              f"{100 * finite.mean():.0f}% valid, "
              f"median {np.nanmedian(data):.1f}, "
              f"range {np.nanpercentile(data, 1):.1f} to {np.nanpercentile(data, 99.9):.1f}")
        if reference.path:
            print(f"     cached at {reference.path}")

        science = _first_science(cfg, index, band, instrument, patterns)
        preview = _template_preview(reference, coord, outdir / f"template_{band}.png",
                                    science=science)
        if preview:
            print(f"     preview  {preview}")
        ok += 1

    if ok:
        print("\nTemplates are cached, so a run will not download them again.")
        print("Look at the previews: the supernova must NOT be there.")
    return 0 if ok else 1


def _north_up_wcs(coord, size_arcmin: float, pixel_scale: float):
    """A plain North-up, East-left TAN grid centred on *coord*."""
    import numpy as np
    from astropy.wcs import WCS

    npix = max(int(round(size_arcmin * 60.0 / pixel_scale)), 16)
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [float(coord.ra.deg), float(coord.dec.deg)]
    wcs.wcs.crpix = [npix / 2.0 + 0.5, npix / 2.0 + 0.5]
    # Negative on RA, positive on Dec: East to the left, North up.
    wcs.wcs.cdelt = [-pixel_scale / 3600.0, pixel_scale / 3600.0]
    return wcs, (npix, npix)


def _template_preview(reference, coord, path: Path, science=None,
                      size_arcmin: float = 2.5):
    """Reference and science frame, both resampled to the same North-up grid.

    Everything is put on one common North-up, East-left frame centred on the
    target, so the panels can be compared with each other and with any finder
    chart directly. A camera WCS can have both CD diagonal terms negative, which
    puts North *down* in the raw frames; drawing them as stored made the two
    panels mirror images of one another.
    """
    import numpy as np
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from astropy.visualization import ZScaleInterval
        from reproject import reproject_interp

        scale = min(float(reference.pixel_scale),
                    float(science.meta.pixel_scale) if science is not None else 9e9)
        scale = float(np.clip(scale, 0.15, 1.5))
        grid, shape = _north_up_wcs(coord, size_arcmin, scale)
        centre = (shape[1] / 2.0 - 0.5, shape[0] / 2.0 - 0.5)

        def _to_grid(data, wcs):
            try:
                out, _ = reproject_interp((np.asarray(data, dtype=float), wcs), grid,
                                          shape_out=shape)
                return out
            except Exception as exc:
                log.debug("preview reprojection failed: %s", exc)
                return None

        panels = [("reference: %s (%s)" % (reference.source,
                                           reference.source_band or reference.band),
                   _to_grid(reference.data, reference.wcs))]
        if science is not None:
            panels.append(("science: TTT %s, MJD %.3f"
                           % (science.meta.band, science.meta.mjd),
                           _to_grid(science.data_sub, science.wcs)))

        full = reference.data
        finite_full = full[np.isfinite(full)]
        ncols = len(panels) + 1
        fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 5.6))
        axes = np.atleast_1d(axes)

        vmin, vmax = ZScaleInterval().get_limits(finite_full) if finite_full.size \
            else (0.0, 1.0)
        axes[0].imshow(full, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        axes[0].set_title("%s reference, full cutout as stored\n%d x %d px, %.0f%% valid"
                          % (reference.band, full.shape[1], full.shape[0],
                             100 * np.isfinite(full).mean()), fontsize=10)

        for ax, (title, stamp) in zip(axes[1:], panels):
            if stamp is None:
                ax.text(0.5, 0.5, "could not resample", ha="center", va="center")
                continue
            finite = stamp[np.isfinite(stamp)]
            lo, hi = ZScaleInterval().get_limits(finite) if finite.size else (0.0, 1.0)
            ax.imshow(stamp, origin="lower", cmap="gray", vmin=lo, vmax=hi)
            ax.add_patch(plt.Circle(centre, max(shape) * 0.035, fill=False,
                                    color="#ff7f0e", lw=1.6))
            ax.axhline(centre[1], color="#ff7f0e", lw=0.5, alpha=0.3)
            ax.axvline(centre[0], color="#ff7f0e", lw=0.5, alpha=0.3)
            ax.set_title("%s\n%.1f' across, N up, E left" % (title, size_arcmin),
                         fontsize=10)
            _compass(ax, shape)

        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("%s   marker at %s"
                     % (reference.band, coord.to_string("hmsdms", precision=2)),
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return path
    except Exception as exc:
        log.warning("template preview failed: %s", exc)
        return None


def _compass(ax, shape, colour: str = "#4e9a06"):
    """N-up / E-left compass in the corner of a resampled panel."""
    ny, nx = shape
    x0, y0 = 0.90 * nx, 0.10 * ny
    arm = 0.07 * min(nx, ny)
    ax.annotate("", xy=(x0, y0 + arm), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=colour, lw=1.3))
    ax.annotate("", xy=(x0 - arm, y0), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=colour, lw=1.3))
    ax.text(x0, y0 + arm * 1.25, "N", color=colour, ha="center", va="bottom", fontsize=9)
    ax.text(x0 - arm * 1.3, y0, "E", color=colour, ha="right", va="center", fontsize=9)


def _widest(cfg, index):
    """The widest detector in an index, and its pixel scale."""
    from . import detector as det_mod
    dets = [e.detector for e in index if e.detector is not None]
    if not dets:
        return None, None
    widest = max(dets, key=lambda d: max(d.nx, d.ny))
    inst = cfg.instrument_for({"INSTRUME": widest.instrument}) \
        if hasattr(cfg, "instrument_for") else {}
    scale = float((inst.get("defaults", {}) or {}).get("pixel_scale", 0.4)) * widest.binning
    return widest, scale


def _catalog_radius(cfg, index, override=None) -> float:
    """The radius a run will use, so the pre-fetch caches under the same key."""
    from . import detector as det_mod
    if override:
        return float(override)
    widest, scale = _widest(cfg, index)
    return det_mod.catalog_radius_arcmin(cfg, detector=widest, pixel_scale=scale)


def _stamp_for(cfg, index, band: str) -> float:
    """Stamp size for a band, sized to the widest frame taken in it."""
    from . import detector as det_mod
    dets = [e.detector for e in index if e.band == band and e.detector is not None]
    if not dets:
        dets = [e.detector for e in index if e.detector is not None]
    if not dets:
        return det_mod.stamp_arcmin(cfg)
    widest = max(dets, key=lambda d: max(d.nx, d.ny))
    inst = cfg.instrument_for({"INSTRUME": widest.instrument}) \
        if hasattr(cfg, "instrument_for") else {}
    scale = float((inst.get("defaults", {}) or {}).get("pixel_scale", 0.4)) * widest.binning
    return det_mod.stamp_arcmin(cfg, detector=widest, pixel_scale=scale)


def _first_science(cfg, index, band, instrument, patterns):
    """Prepare one science frame of *band*, for side-by-side previews."""
    from . import pipeline
    for entry in index:
        if entry.band != band:
            continue
        try:
            return pipeline.prepare_image(
                entry.path, cfg, instrument,
                pattern=pipeline.pattern_for(patterns, band, entry.detector))
        except Exception:
            continue
    return None


def _compare_templates(cfg, args, index, coord, bands, cache_root, patterns, outdir) -> int:
    """Build a difference against each candidate template and report which wins.

    Choosing between a bandpass-matched photographic plate and a deeper survey
    in the wrong filter is not something to settle by argument: this measures
    both on one of your own frames.
    """
    import numpy as np
    from . import pipeline
    from . import subtraction as sub
    from . import templates as tpl

    instrument = cfg.get("instrument_config", {})
    candidates = args.compare if isinstance(args.compare, list) else []

    print(f"{'band':>5} {'template':<28} {'chi2':>6} {'resid rms':>10} "
          f"{'ref FWHM px':>12} {'valid':>6}")
    for band in bands:
        science = _first_science(cfg, index, band, instrument, patterns)
        if science is None:
            print(f"{band:>5} no usable science frame")
            continue

        options = []
        entry = cfg.get(f"subtraction.hips.{band}")
        if entry:
            survey = entry["survey"] if isinstance(entry, dict) else entry
            options.append(("hips", survey))
        for extra in candidates:
            options.append(("hips", extra))
        if band in ("g", "r", "i", "z", "y"):
            options.append(("panstarrs", band))
        borrowed = cfg.get(f"subtraction.reference_band.{band}")
        if borrowed:
            options.append(("panstarrs", borrowed))

        for source, which in options:
            try:
                if source == "hips":
                    reference = tpl.fetch_hips_cutout(
                        coord, band, which, _stamp_for(cfg, index, band) * 1.25,
                        cache_root, refresh=args.refresh)
                else:
                    reference = tpl.fetch_panstarrs_cutout(
                        coord, which, _stamp_for(cfg, index, band) * 1.25,
                        cache_root, refresh=args.refresh)
                    if reference is not None:
                        reference.source_band = which
                        reference.band = band
                if reference is None:
                    print(f"{band:>5} {str(which):<28} unavailable")
                    continue
                diff = sub.subtract(science, reference, cfg, coord,
                                    fwhm_px=science.fwhm_px)
                if not diff.ok:
                    print(f"{band:>5} {str(which):<28} failed: {diff.message}")
                    continue
                print(f"{band:>5} {str(which):<28} {diff.chi2:>6.2f} {diff.rms:>10.1f} "
                      f"{diff.meta['reference_fwhm_px']:>12.2f} "
                      f"{100 * diff.meta['coverage']:>5.0f}%")
            except Exception as exc:
                print(f"{band:>5} {str(which):<28} error: {exc}")

    print("\nLower chi2 and lower residual rms are better; chi2 near 1 means the")
    print("difference is consistent with pure photon noise. Set the winner in")
    print("subtraction.hips (or reference_band) in the target file.")
    return 0


def cmd_build_pattern(args) -> int:
    """Measure the fixed detector pattern for one or more bands."""
    from . import patterns as pat
    from . import pipeline

    setup_logging(args.log_level)
    cfg = _build_config(args)
    instrument = cfg.get("instrument_config", {})

    paths = list(pipeline.find_images(cfg, args.target))
    for extra in (args.extra_dirs or []):
        directory = Path(extra).expanduser()
        if directory.is_dir():
            for pattern_glob in ("*.fits", "*.fit", "*.fits.gz"):
                paths.extend(sorted(directory.rglob(pattern_glob)))
        else:
            paths.extend(sorted(Path().glob(str(extra))))
    # --extra-dirs commonly overlaps the target's own directory; the same frame
    # counted twice would be weighted twice in the median.
    seen, unique = set(), []
    for path in paths:
        key = Path(path).resolve()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    paths = unique
    if not paths:
        print("no images found", file=sys.stderr)
        return 1

    index = pipeline.index_images(paths, instrument, cfg=cfg)
    bands = args.bands or sorted({e.band for e in index})
    calib_root = pipeline._calib_root(cfg)

    # A pattern is fixed in detector coordinates, so one is needed per readout
    # configuration, not per band. Group first, build within each group.
    detectors = sorted({e.detector for e in index if e.detector is not None},
                       key=lambda d: d.token)
    if not detectors:
        print("no readable frames", file=sys.stderr)
        return 1
    if len(detectors) > 1:
        print(f"{len(detectors)} detector configurations found:")
        for det in detectors:
            print(f"   {det.token:22} {det.label:24} "
                  f"{sum(1 for e in index if e.detector == det):4} frames")
        print()
    if args.detectors:
        wanted = {str(d).strip().lower() for d in args.detectors}
        kept = [d for d in detectors if d.token in wanted]
        if not kept:
            print(f"none of {sorted(wanted)} matched; available: "
                  f"{', '.join(d.token for d in detectors)}", file=sys.stderr)
            return 1
        detectors = kept

    built = 0
    for det in detectors:
        if len(detectors) > 1:
            print(f"-- {det.label}")
        for band in bands:
            entries = [e for e in index if e.band == band and e.detector == det]
            if not entries:
                print(f"{band:>3}: no images")
                continue
            max_frames = int(cfg.get("image.pattern.max_frames", 60))
            chosen = pat.balance_fields(entries, max_frames)
            # A pattern built from one field absorbs that field's own sky and
            # then subtracts it back out of every frame of that field, which
            # looks like an improvement and is not one. Refuse rather than ship
            # it; --allow-single-field is there for when you have measured that
            # the field is empty enough.
            fields = {e.object_name for e in chosen if e.object_name}
            min_fields = int(cfg.get("image.pattern.min_fields", 2) or 0)
            if not args.allow_single_field and 0 < len(fields) < min_fields:
                print(f"{band:>3}: only {len(fields)} field "
                      f"({', '.join(sorted(fields))}) -- need {min_fields}. "
                      f"Add frames of other fields with --extra-dirs, or pass "
                      f"--allow-single-field if you know this field is empty enough.")
                continue
            pattern = pat.build_pattern(
                [e.path for e in chosen], band,
                block=int(cfg.get("image.pattern.block", 16)),
                smooth_blocks=float(cfg.get("image.pattern.smooth_blocks", 6.0)),
                min_frames=int(args.min_frames or cfg.get("image.pattern.min_frames", 8)),
                max_frames=max_frames,
                outlier_sigma=float(cfg.get("image.outlier_sigma", 30.0) or 0),
                max_amplitude=float(cfg.get("image.pattern.max_amplitude", 0.25)),
                instrument=cfg.instrument_for({"INSTRUME": det.instrument})
                if hasattr(cfg, "instrument_for") else instrument,
            )
            if pattern is None:
                print(f"{band:>3}: not enough usable frames ({len(entries)} found)")
                continue
            objects = {e.object_name for e in chosen if e.object_name}
            path = pat.save_pattern(pattern, calib_root)
            pat.preview(pattern, path.with_suffix(".png"))
            print(f"{band:>3}: {pattern.n_frames} frames -> {path.name}   "
                  f"illumination {100 * pattern.illumination_amplitude:5.2f}% pp, "
                  f"fringe {100 * pattern.fringe_amplitude:5.2f}% pp")
            built += 1

    if built:
        print(f"\npatterns written to {calib_root}")
        print("they are applied automatically on the next run "
              "(set image.pattern.enabled: false to skip)")
    return 0 if built else 1


def cmd_colorterms(args) -> int:
    """Fit the instrument colour term per band by combining many frames."""
    from . import pipeline
    from .calibration import calibrate_image
    from .detection import detect_sources

    setup_logging(args.log_level)
    cfg = _build_config(args)
    cfg.set("calibration.color_term", "fit")
    instrument = cfg.get("instrument_config", {})
    paths = pipeline.find_images(cfg, args.target)
    index = pipeline.index_images(paths, instrument, cfg=cfg)
    if args.bands:
        index = [e for e in index if e.band in args.bands]

    results: dict[str, list[tuple[float, float]]] = {}
    for entry in index[: args.max_images or len(index)]:
        try:
            image = pipeline.prepare_image(entry.path, cfg, instrument)
            sources = detect_sources(image, cfg, fwhm_px=image.fwhm_px)
            zp = calibrate_image(image, sources, image.fwhm_px, cfg, Path(cfg.cache_root))
            if zp.ok and np.isfinite(zp.color_term) and zp.color_term_err > 0:
                results.setdefault(zp.band, []).append((zp.color_term, zp.color_term_err))
        except Exception as exc:
            log.warning("%s: %s", entry.path.name, exc)

    summary = {}
    for band, values in sorted(results.items()):
        terms = np.array([v[0] for v in values])
        errs = np.array([v[1] for v in values])
        weights = 1.0 / np.maximum(errs, 1e-3) ** 2
        summary[band] = (float(np.sum(weights * terms) / np.sum(weights)),
                         float(np.sqrt(1.0 / np.sum(weights))), len(terms))

    if not summary:
        print("\nno colour terms could be fitted")
        return 1

    print(f"\n{'band':>5} {'N':>4} {'colour term':>12} {'error':>8}")
    for band, (mean, err, n) in summary.items():
        print(f"{band:>5} {n:>4} {mean:>12.4f} {err:>8.4f}")

    print("\nPaste into your config to freeze these:\n")
    print("calibration:")
    print("  color_term: fixed")
    print("  color_term_fixed:")
    for band, (mean, _, _) in summary.items():
        print(f"    {band}: {mean:.4f}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="tttphot",
        description="Photometry pipeline for TTT transient imaging.")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="measure a light curve")
    _add_common(run_parser)
    run_parser.add_argument("--clean", action="store_true",
                            help="delete this target's previous outputs before starting")
    run_parser.add_argument("--max-images", type=int, default=None)
    run_parser.add_argument("--skip", type=int, default=0,
                            help="skip the first N epochs (resume a long run with --append)")
    run_parser.add_argument("--no-subtraction", action="store_true",
                            help="skip difference imaging")
    run_parser.add_argument("--no-astrometry", action="store_true",
                            help="keep the header WCS instead of refining it")
    run_parser.add_argument("--no-pattern", action="store_true",
                            help="do not apply the stored detector patterns")
    run_parser.add_argument("--refresh", action="store_true",
                            help="ignore cached catalogues and templates")
    run_parser.add_argument("--append", action="store_true",
                            help="merge into an existing table instead of replacing it "
                                 "(use when re-running one band)")
    run_parser.add_argument("--aperture-scale", type=float, default=None)
    run_parser.add_argument("--catalogs", nargs="+", default=None)
    run_parser.set_defaults(func=cmd_run)

    list_parser = sub.add_parser("list", help="show the epochs found for a target")
    _add_common(list_parser)
    list_parser.add_argument("-v", "--verbose", action="store_true")
    list_parser.set_defaults(func=cmd_list)

    rc_parser = sub.add_parser(
        "recalibrate",
        help="recover missing zero points in a finished run from another band")
    _add_common(rc_parser)
    rc_parser.add_argument("-n", "--dry-run", action="store_true",
                           help="report what would be recovered, write nothing")
    rc_parser.set_defaults(func=cmd_recalibrate)

    cl_parser = sub.add_parser(
        "clean", help="delete a target's outputs, and optionally its cached downloads")
    _add_common(cl_parser)
    cl_parser.add_argument("--runs", action="store_true",
                           help="run outputs only (the default when nothing is named)")
    cl_parser.add_argument("--templates", action="store_true",
                           help="also the cached reference images for this field")
    cl_parser.add_argument("--catalogs", action="store_true",
                           help="also the cached reference catalogues for this field")
    cl_parser.add_argument("--all", action="store_true", help="outputs, templates and catalogues")
    cl_parser.add_argument("-n", "--dry-run", action="store_true",
                           help="list what would go, delete nothing")
    cl_parser.add_argument("-v", "--verbose", action="store_true")
    cl_parser.set_defaults(func=cmd_clean)

    rs_parser = sub.add_parser(
        "resolve", help="look the target name up on TNS (or SIMBAD) for its position")
    _add_common(rs_parser)
    rs_parser.add_argument("--write", action="store_true",
                           help="record the position in targets/<TARGET>.yaml")
    rs_parser.add_argument("--source", choices=["tns", "simbad"], default=None,
                           help="use only this resolver")
    rs_parser.set_defaults(func=cmd_resolve)

    fetch_parser = sub.add_parser("fetch-catalog",
                                  help="download and cache the reference catalogues")
    _add_common(fetch_parser)
    fetch_parser.add_argument("--radius", type=float, default=None, help="arcmin")
    fetch_parser.add_argument("--catalogs", nargs="+", default=None)
    fetch_parser.add_argument("--refresh", action="store_true")
    fetch_parser.set_defaults(func=cmd_fetch_catalog)

    ft_parser = sub.add_parser(
        "fetch-template",
        help="download and inspect the reference image for each band")
    _add_common(ft_parser)
    ft_parser.add_argument("--refresh", action="store_true",
                           help="re-download even if it is already cached")
    ft_parser.add_argument("--compare", nargs="*", default=None, metavar="SURVEY",
                           help="build a difference against each candidate template "
                                "and report which subtracts best; extra HiPS ids or "
                                "shorthands may be listed")
    ft_parser.set_defaults(func=cmd_fetch_template)

    bp_parser = sub.add_parser(
        "build-pattern",
        help="measure the fixed detector pattern (illumination + fringes) per band")
    _add_common(bp_parser)
    bp_parser.add_argument("--extra-dirs", nargs="+", default=None,
                           help="additional directories of frames to include; other "
                                "fields make a cleaner pattern")
    bp_parser.add_argument("--min-frames", type=int, default=None)
    bp_parser.add_argument("--allow-single-field", action="store_true",
                           help="build even when only one field is available; the "
                                "pattern will contain that field's own sky structure")
    bp_parser.add_argument("--detectors", nargs="+", default=None, metavar="TOKEN",
                           help="only build for these detector configurations "
                                "(tokens as printed by 'tttphot list -v', e.g. "
                                "qhy600-ttt3_1595x1064_b6); default is all of them")
    bp_parser.set_defaults(func=cmd_build_pattern)

    ct_parser = sub.add_parser("colorterms", help="fit the instrument colour terms")
    _add_common(ct_parser)
    ct_parser.add_argument("--max-images", type=int, default=None)
    ct_parser.add_argument("--catalogs", nargs="+", default=None)
    ct_parser.set_defaults(func=cmd_colorterms)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except (RuntimeError, FileNotFoundError, KeyError) as exc:
        # Expected: the data or the configuration is not what was asked for.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception:
        # Exit 2 means the tool itself broke, as distinct from exit 1, which
        # means it ran but the data or configuration was not there. full-run.sh
        # treats them differently: a bug should stop the run, not be reported
        # as "no catalogue reached".
        import traceback
        traceback.print_exc()
        print("\nThis is a bug in tttphot, not a problem with the data.",
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
