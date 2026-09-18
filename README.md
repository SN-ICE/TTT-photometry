# tttphot

[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)

A supernova photometry pipeline for the TTT telescopes at Teide Observatory,
built on astropy and photutils. It takes reduced TTT frames and produces a
calibrated, multi-band light curve with per-epoch diagnostics.

It is a port of **tjophot**, the TJO pipeline (`~/Desktop/TJO/pipeline`), and
keeps its whole method -- aperture and empirical-PSF photometry, Alard & Lupton
difference imaging, Gaia DR3 synthetic-photometry zero points, empirical depth
and errors, the same quality flags and the same output tables. What changed is
everything that knows about the telescope. Configured for the three cameras
the observatory has used on 2026abgb: **TTT3 + QHY600M**, **TTT3 + iKon-L 936**
and **TTT4 + QHY600M**, all 2 m; another camera is one YAML file.

`snname` below stands for the target's name. TTT writes the bare TNS
designation into `OBJECT` (`2026abgb`); the pipeline accepts `SN2026abgb`,
`AT2026abgb` or `2026abgb` interchangeably.

---

## Quick start

```bash
cd ~/Desktop/TTT/pipeline
pip install -e .                  # -e matters: it keeps config/ and targets/ live

tttphot list SN2026abgb -v        # what frames it finds
./full-run.sh SN2026abgb          # clean, resolve, fetch catalogues + templates, measure
```

`QUICKSTART.md` walks through a new target step by step.

Outputs land in `runs/snname/` -- exactly as in tjophot:

| file | contents |
|---|---|
| `snname_lightcurve.csv` | one row per epoch: the adopted measurement |
| `snname_photometry_all.csv` | every method for every epoch |
| `snname_lightcurve.png`, `snname_diagnostics.png` | light curve; seeing, ZP and depth over the campaign |
| `qa/<image>.png` | per-epoch panel: stamps, curve of growth, zero-point fit, kernel |
| `run_summary.json`, `config_used.yaml`, `run.log` | exactly what was run |

---

## Where it looks for frames

`paths.data_root` defaults to `~/Desktop/TTT`. For a target it uses, in order:

1. a folder named after the target -- `~/Desktop/TTT/SN2026abgb/` or
   `~/Desktop/TTT/2026abgb/` -- and every FITS file inside it;
2. otherwise **every FITS file anywhere under `~/Desktop/TTT`** whose `OBJECT`
   header names the target. So frames can simply be dropped into the TTT folder
   as they arrive, in any sub-folder layout, and several targets can share one.

The pipeline's own directory (its cached templates are FITS too) and hidden
folders are always skipped. Per-target folders are tidier once there are many
frames, and faster, since no headers need reading.

---

## What is different from tjophot

| | TJO (tjophot) | TTT (tttphot) |
|---|---|---|
| instrument files | `tjo_meia2/3.yaml` | `ttt3_qhy600.yaml` (TTT3 and TTT4 QHY600), `ttt_ikon936.yaml` |
| filters | `SDSS g`, `B`, ... | `SDSSg`, `SDSSr`, ... (Johnson spellings mapped too) |
| gain, read noise | per binned pixel in the header | header quotes the **unbinned** sensor; converted, see below |
| target position | `OBJRA/OBJDEC` | none in the header (`RA/DEC` is the frame centre) -- TNS/SIMBAD, or `targets/snname.yaml` |
| sampling | ~5 px FWHM | 13 px FWHM at 0.19"/px -- **rebinned 2x2 on load** |
| astrometry refit | `linear` (TAN) | `shift`, which keeps the header's SIP distortion |
| border trim, detector patterns | MEIA3 trim; patterns in `calib/` | none needed so far; `build-pattern` still available |
| template stack of own epochs | `tjo_stack` | `ttt_stack` |
| extra columns | -- | `rebin`, `gain_eff`, `sky_noise_ratio`, `obs_zp`, `obs_zp_err`, `obs_maglim`, `obs_seeing_arcsec` |

### Gain and read noise

TTT3 frames are 3x3 binned, and the `GAIN = 0.44` / `RDNOISE = 4.74` cards are
for the unbinned chip. The binned pixels are **averages**, not sums. That is
measured, not assumed: on 2026abgb (g, 120 s) the sky is 85 ADU with a
pixel-to-pixel scatter of 5.8 ADU; averaging predicts 5.9, summing would
predict 35. So per binned pixel the gain is 0.44 x 9 = **3.96 e-/ADU** and the
read noise 4.74 x 3 = **14.2 e-**. `binning_mode: average` in the instrument
file does this.

Every frame re-checks it: `sky_noise_ratio` is the measured sky scatter over
the scatter that gain and read noise predict. On the test frame it is **1.02**.
If a frame is far from 1 the log warns -- the first place that will matter is a
stacked frame (`TOTIMA > 1`), whose convention (`stack_mode`) has not been seen
in the archive yet.

### Rebinning

At 0.192"/px a 2.5" seeing disc is 13 px across, which buys nothing and makes
every step (cosmic rays, PSF model, kernel solve) about four times slower than
necessary. With `rebin: auto` the frame is **summed** 2x2 on load, to
0.385"/px -- the sampling tjophot was tuned for, so its pixel-based defaults
apply unchanged. Summing keeps fluxes in the camera's own ADU, so zero points
stay directly comparable with un-rebinned runs and with the observatory's `ZP`
card. The WCS (SIP terms included) is rescaled exactly (checked to 1e-10"),
the saturation level and read noise are scaled, and a block containing a
saturated pixel is kept flagged. On the test frame the FWHM comes out 3.01"
either way, and one frame takes 4 s to prepare instead of 20. `image.rebin: 1`
switches it off. If the seeing is bad enough that the FWHM exceeds ~5 px anyway,
the subtraction kernel and its stamps now grow in proportion
(`subtraction.kernel.scale_with_fwhm`).

### The observatory's own numbers

TTT's pipeline writes a zero point (`ZP`, against PS1), a limiting magnitude
and a seeing into each header. They are **never used**, only carried into the
output as `obs_zp`, `obs_maglim`, `obs_seeing_arcsec`, so every epoch has an
independent cross-check. Expect `zp` - `obs_zp` to be a small constant for a
band, not zero: the definitions differ (tttphot's ZP is for aperture-corrected
counts against Gaia synthetic photometry). A frame where that difference jumps
is worth a look. Their FWHM estimator also runs ~15 per cent below tttphot's
Gaussian-fit FWHM (2.5" against 3.0" on the test frame).

### Mixed cameras

The observatory takes a QHY600 set and an iKon936 set of the same target a few
minutes apart, and TTT4 adds a second QHY600. The pipeline measures every frame
on its own and never assumes two cameras agree, but four places compared
frames with each other and had to become camera-aware. The first one is why a
first run of 2026abgb showed the QHY600 and iKon936 epochs of the same night
1.5 mag apart:

- **Kernel-sum check.** A difference whose kernel sum is far from the band's
  median is flagged as a failed subtraction. The raw sum scales with each
  camera's ADU per photon and pixel size (0.12-0.17 for the QHY600 in g,
  0.78-0.92 for the iKon936), so every frame of the minority camera was
  flagged, and those epochs fell back to direct photometry -- on the host,
  1.5 mag too bright. The sums are now divided by exposure time,
  10^(0.4 ZP) and pixel area first; after that the three cameras' medians
  agree to 3 per cent and single epochs to 10 (a `convolve_science` kernel is
  inverted before comparing).
- **Zero-point transfer** between bands of the same night is done per camera
  (their zero points differ by 1.4 mag in the same band).
- **`ttt_stack` templates** use one detector configuration only.
- **Instrument files** are chosen per frame from `INSTRUME`.

Two more changes came out of the same run, and apply to every camera:

- **Template registration.** The Pan-STARRS stack cutout sat 0.26" off Gaia
  (the PS1 *catalogue* agrees with Gaia to 20 mas), while every TTT frame is
  re-centred on Gaia. The reference WCS is now registered on the same
  catalogue before differencing (`subtraction.register_reference`).
- **Host guard.** If every difference measurement of an epoch is flagged and
  the fallback would be direct photometry that is more than 0.2 mag brighter,
  the flagged difference is kept instead (`output.host_guard_mag`).

And two that only showed on the TTT3 QHY600 in good seeing:

- **Elongated stars.** On 2026-09-12 (seeing ~1") its stars are about twice as
  wide along the detector columns as along the rows (2.6 against 5.5 px FWHM);
  the iKon936 on the same telescope and the TTT4 QHY600 are round. With the
  TJO roundness limits only 7 of 78 Gaia stars were detected and the frame lost
  its astrometry and calibration; `detection.roundlo/roundhi` are now +-1.5.
  Worth mentioning to the observatory.
- **Seeing measurement** kept the last of its iterations even when that pass
  was based on a handful of blends (5.3 px instead of 3.8 px), which then made
  the PSF-star profile test reject real stars. It now keeps the pass with the
  most stars.

### Errors on a bright host

The error of a difference measurement used to be photon noise only, which says
nothing about how well the galaxy under the transient was removed. On
2026abgb (12" from the nucleus of NGC 7673) in 2.0-2.6" seeing the host inside
the aperture is 2-4 times the supernova, and four r frames of one night
scattered by 0.06 mag against quoted errors of 0.013-0.021 (reduced chi2 10).
Two changes:

- **Host-residual error.** The science aperture is laid on sites near the
  transient where the matched reference is about as bright as at the
  transient; the robust scatter of the difference flux there, less the sky
  noise, is `diff_host_residual` (as a fraction of the transient's flux). Half
  of it (`subtraction.host_residual.error_scale`) is added to `mag_err`. It is
  1-3 per cent in ~1" seeing and 4-30 per cent in 2-3". It widens the error
  bar only: `flux_err` and `snr` stay statistical, so a clear detection is
  never turned into a limit by it, and the systematic part is in
  `flux_err_sys`.
- **The transient is kept out of the kernel fit.** It was often picked as a
  kernel stamp. Sigma clipping removed most of it, but injected sources came
  back 1-3 per cent faint; its pixels within 2 FWHM
  (`subtraction.kernel.exclude_target_fwhm`) are now excluded. Only those
  pixels: rejecting every stamp near it also threw out the host nucleus, and
  the kernel sums then scattered by 5 per cent instead of 1-2.

### Bad pixels

The QHY600 writes a GPS time stamp into the first pixels of row 0 (values like
9003 and 93792 ADU). They are set to NaN on load (`bad_regions`), and would be
inside the edge margin anyway.

---

## What it does, step by step

Unchanged from tjophot; in brief:

1. **Ingest** through the instrument file; mask bad, saturated and edge pixels;
   reject cosmic rays (L.A.Cosmic); 2D background and per-pixel error map.
2. **Seeing**: median Gaussian FWHM of isolated stars. Every aperture and box is
   in units of it.
3. **Astrometry**: re-centre the header WCS on Gaia DR3; only kept if it helps.
4. **Photometry**: 1.2 x FWHM aperture with curve-of-growth correction, and
   empirical-PSF forced photometry on the same system.
5. **Difference imaging**: reference resampled and PSF-matched (Alard & Lupton,
   numpy/scipy only); if the reference is the blurrier one, the science frame
   is convolved instead. Quality checks: noise-model chi2, kernel-sum
   consistency across epochs, dipole residuals.
6. **Calibration**: per-frame zero point, optional colour term, against Gaia DR3
   synthetic photometry, then Pan-STARRS DR2, then APASS. Cached to disk.
7. **Depth and errors** from blank apertures and from injecting the frame's own
   PSF around the target.

`README.md` and `QUICKSTART.md` in the TJO pipeline explain each of these at
length, and everything there about method, flags, templates and catalogues
applies here word for word.

---

## Commands

```bash
tttphot list snname -v                 # frames, bands, MJD coverage
tttphot resolve snname --write         # position from TNS into targets/snname.yaml
tttphot fetch-catalog snname           # download and cache the catalogues
tttphot fetch-template snname          # fetch and inspect the reference images
tttphot run snname                     # the pipeline
tttphot run snname --clean             # same, removing the previous outputs first
tttphot run snname --bands g r         # a subset of bands
tttphot run snname --no-subtraction    # direct photometry only
tttphot run snname --skip 40 --append  # resume a long run
tttphot colorterms snname              # fit the instrument colour terms
tttphot recalibrate snname             # recover failed zero points from other bands
tttphot clean snname                   # remove outputs (--templates, --catalogs, --all)
tttphot build-pattern snname           # detector patterns, once there are enough frames
```

Without installing, every command works as `python -m tttphot.cli ...` from
inside the pipeline directory.

---

## Configuration

Layers, each overriding the last: `config/default.yaml` (everything,
documented) -> `config/local.yaml` and `config/tns.yaml` if present
(gitignored) -> `targets/snname.yaml` -> `--config my.yaml` -> command-line flags.

`config/tns.yaml` holds the TNS bot credentials, copied from the TJO pipeline.

---

## Adding another TTT camera

Copy `config/instruments/ttt3_qhy600.yaml`, check against one of its headers
the keyword lists, `filters`, whether `GAIN`/`RDNOISE` are unbinned, and the
binning mode (the sky-noise check will tell you), then map its `INSTRUME` value
under `instruments:` in `config/default.yaml`. A frame whose `INSTRUME` is not
mapped still runs, with the TTT3 numbers and a warning.

---

## Validation

See `docs/VALIDATION.md`. In short: offline, on the real 2026abgb frame, stars
injected on the host galaxy and recovered through the full `tttphot run`
(template, kernel, calibration, both difference methods) come back within
0.03 mag at g = 17.5-18.5 and within 1.3 sigma at 19.5-20.5, while direct
photometry at the same spot is 2.5-5.7 mag too bright from the host.

On the real 2026abgb g frames from all three cameras (with the Gaia catalogues
and PS1 template cached by a live run), same-night epochs from different
cameras agree to 0.02-0.03 mag, and `zp - obs_zp` is +0.05 +- 0.02 mag for
every frame of every camera. With the host-residual term, same-night r and g
frames in 1-3" seeing agree to within their errors (reduced chi2 0.2-1.1,
from 1.7-15 before).

```bash
python tests/selftest_injection.py ~/Desktop/TTT/<a frame>.fits   # offline, ~2 min
```

---

## Licence and citation

BSD 3-Clause, see `LICENSE`. Please also cite astropy and photutils.
