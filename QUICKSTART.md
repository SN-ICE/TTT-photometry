# Running tttphot

## One-time setup

```bash
cd ~/Desktop/TTT/pipeline
pip install -e .
```

Use the same conda environment as tjophot; the dependencies are identical
(numpy, scipy, astropy, photutils, astroquery, reproject, matplotlib, pandas,
PyYAML, astroscrappy). `-e` installs in place, so edits to `config/` and
`targets/` take effect immediately. Without installing, use
`python -m tttphot.cli ...` from inside this directory.

`astroscrappy` (cosmic-ray rejection) is a dependency; without it the run
works but warns and leaves cosmic rays in. `pip install -e .` installs it.

photutils must be older than 3.0: version 3 renamed the source-finder columns
(`xcentroid` became `x_centroid`), which breaks this code and tjophot alike.
`pip install -e .` enforces that.

`config/tns.yaml` (TNS bot credentials, gitignored) was copied from the TJO
pipeline, so `tttphot resolve` works out of the box.

## A new target, start to finish

`snname` stands for the target's name; `SN2026abgb`, `AT2026abgb` and
`2026abgb` all find the same frames and the same `targets/` file. TNS
registers a transient as AT and renames it SN once it is classified, so either
prefix works for any object.

### 1. Put the frames where the pipeline looks

Anywhere under `~/Desktop/TTT`. Either drop them in as they come (the
`OBJECT` header identifies the target), or keep one folder per target:

```
~/Desktop/TTT/SN2026abgb/TTT3_QHY600-3_..._2026abgb_SDSSg.fits
```

Check what it found:

```bash
tttphot list snname -v
```

### 2. Get the position

TTT headers carry no target coordinates (`RA`/`DEC` is the centre of the
field), so this step matters more than on TJO:

```bash
tttphot resolve snname --write
```

writes `targets/snname.yaml` with the TNS position. Without TNS it tries
SIMBAD, which is often too slow to list a new transient; then write the file by
hand from `targets/TEMPLATE.yaml`:

```yaml
target:
  name: SN2026abgb
  ra: 351.9...      # decimal degrees, or "23:27:..." sexagesimal hours
  dec: 23.5...
```

As a last resort the run starts from the field centre and tries to centroid
the transient within `target.refine_radius_arcsec` (3"). That only works if the
target really is at the centre, so do not rely on it.

### 3. Catalogues and templates (internet, once per field)

```bash
tttphot fetch-catalog snname
tttphot fetch-template snname
```

Look at the source counts `fetch-catalog` reports. A TTT3 frame is only
10.2' x 6.8', so a high-latitude field can have few Gaia synthetic-photometry
stars (they stop at G = 17.65); Pan-STARRS is the automatic fallback for griz.
`fetch-template` writes `runs/snname/templates/template_<band>.png`: open them
and check the reference covers the field and has no transient in it.

Both are downloaded once per field. Catalogues go to `cache/catalogs/` and the
template FITS files to `cache/templates/` in the pipeline folder, not under
`runs/`, and every later `run`, `fetch-template` or `./full-run.sh` reuses
them. That holds even if the target position moves by up to ~40". Neither
`run --clean` nor `tttphot clean snname` touches them; `tttphot clean snname
--templates` (or `--catalogs`, `--all`) deletes them, and `--refresh`
downloads them again. To keep them somewhere else, e.g. next to the frames,
put this in `config/local.yaml`:

```yaml
paths:
  cache_root: ~/Desktop/TTT/cache
```

### 4. A short trial, then the full run

```bash
tttphot run snname --max-images 3
tttphot run snname --clean
```

or all of steps 1-4 in one go: `./full-run.sh snname`.

### 5. First things to check in the output

In `runs/snname/snname_photometry_all.csv`:

- `sky_noise_ratio` near 1. Far from 1 means the gain model does not fit that
  frame (a stacked frame, a different binning, a different camera).
- `zp` against `obs_zp` (the observatory's own zero point): the offset should
  be about the same for every frame in a band.
- `fwhm_arcsec` against `obs_seeing_arcsec`: tttphot's number is typically
  ~15 per cent larger (different estimator); a frame far off that trend has a
  problem.
- `diff_chi2`, `flags_str`, and the `qa/` panels, exactly as for TJO.
- `diff_host_residual`: how badly the host is subtracted around the transient,
  as a fraction of its flux. Above ~0.1 the frame's value leans on the error
  bar; with several frames that night, trust the good-seeing ones.

## Frames from another TTT telescope or camera

A frame whose `INSTRUME` is not in the `instruments:` map of
`config/default.yaml` still runs, with the TTT3 numbers and a warning. To do it
properly, copy `config/instruments/ttt3_qhy600.yaml`, fix the numbers against
one of its headers and add the mapping. `sky_noise_ratio` on its first frames
tells you whether the gain and binning mode are right.

## Offline self-test

```bash
python tests/selftest_injection.py ~/Desktop/TTT/<any frame>.fits
```

Injects copies of a real field star onto the host galaxy of that frame at
g = 17.5-20.5, builds a sharper reference and a local catalogue from the frame
itself, runs the whole pipeline and prints injected against recovered
magnitudes. Needs no network. Takes about two minutes.

## If something goes wrong

- `no FITS files found`: check `paths.data_root` (`--data-root` overrides it),
  and that `OBJECT` really is the designation you are asking for.
- `filter 'XYZ' not in the instrument map`: add it under `filters:` in the
  instrument file.
- `the target position falls outside this frame`: the position is wrong, or
  that frame is of another field with the same `OBJECT`.
- Every row `no_zeropoint`: no catalogue reached; run `fetch-catalog` and read
  its messages.
