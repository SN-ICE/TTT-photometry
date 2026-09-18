# tttphot — code review

Review of the pipeline as it stands at `79d454a` plus the uncommitted working tree
(17 modified files, two untracked). Scope: all of `tttphot/`, `config/default.yaml`,
`tests/`, and the CLI. Focus, as requested: scientific correctness, bugs and
robustness, architecture, performance.

## Overall

This is a well-built pipeline. The structure is clean (ingest → detect → astrometry →
photometry → subtraction → calibration → assembly), each module has one job, and the
docstrings are the best part: almost every non-obvious choice records the failure that
motivated it, with the numbers from the frame where it showed up. That is rare and it
made this review far faster than it would otherwise have been. The physics is sound —
the gain/binning reasoning, the kernel-sum normalisation across cameras, the
host-residual error term and the dipole test are all careful, and they are validated
against real data rather than asserted.

What follows is what I found. Most of it is small. Three items (H1–H3) are real bugs
that silently change results or waste work, and the absence of any automated test suite
(R1) is the single biggest structural risk, because nearly everything below could have
been caught by one.

Severity: **H** = changes results or wastes work today · **M** = a scientific or
robustness question worth a decision · **L** = hygiene.

---

## H. Bugs

### H1 — The `ttt_stack` template cache never hits, and grows without bound
`tttphot/templates.py:137`

```python
digest = abs(hash((target, band, tuple(mjds)))) % (10 ** 10)
```

Python randomises `hash()` for `str` per process (PEP 456, on by default since 3.3).
Verified on your machine: the same tuple gives `444056015`, `8747087037`,
`9853775576` on three consecutive interpreters. So the cache filename changes every
run: the `cache_key.exists()` branch above can never fire across processes, every run
re-stacks and re-reprojects the epochs, and every run writes a new multi-megabyte FITS
into `cache/templates/` that nothing will ever read again.

Nothing in the current `runs/` output depends on `ttt_stack` (the 2026abgb bands all
resolve to PS1), so this is latent — but it will bite the moment you set
`ttt_stack.mjd_min` for a faded target, which is exactly the case the README points
users to.

Fix: use a stable digest, as the catalogue cache already does.

```python
key = f"{target}|{band}|" + ",".join(f"{m:.5f}" for m in mjds)
digest = hashlib.md5(key.encode()).hexdigest()[:10]
```

### H2 — `satlevel` is passed to astroscrappy in ADU; it wants electrons
`tttphot/cosmics.py:109`

```python
crmask, cleaned = astroscrappy.detect_cosmics(
    work.astype(np.float32), inmask=keep,
    gain=float(gain) or 1.0, readnoise=float(readnoise) or 0.0,
    satlevel=float(saturation),      # <- ADU
```

Confirmed against astroscrappy 1.3.0 source: `astroscrappy.pyx:200` does
`cleanarr = indat * gain`, line 227 passes that electron-scale array to `update_mask`,
and line 461 tests `data >= satlevel`. The docstring says so explicitly — *"satlevel:
Saturation of level of the image (electrons)"*. The returned `cleanarr` is divided back
by gain at line 426, so the cleaned frame's units are fine; only the saturation
threshold is wrong.

With the TTT3 effective gain of 3.96 e⁻/ADU the threshold is a factor of ~4 too low, so
every pixel above roughly a quarter of full well is treated by L.A.Cosmic as saturated
and added to its internal mask, along with a grown region around it. The failure is
conservative — cosmic rays landing on or near moderately bright stars are simply not
cleaned — but it quietly reduces the cleaning on precisely the stars the PSF and the
aperture correction are built from.

Fix: `satlevel=float(saturation) * (float(gain) or 1.0)`.

### H3 — PS1 skycell mosaics are linearised with the wrong asinh parameters
`tttphot/templates.py:554-555`

```python
data, header = _combine_skycells(pieces)
data = linearise_ps1(data, header)
```

`_combine_skycells` fills the blank part of the first cutout with pixels from the
neighbouring skycells *while they are still in asinh units*, and `header` is
`pieces[0]`'s header. `linearise_ps1` then applies `pieces[0]`'s `BSOFTEN`/`BOFFSET` to
the whole array. PS1 stack skycells do not share those constants, so every pixel that
came from a neighbour gets an incorrect flux.

This only matters for a field near a tile edge — which is exactly the case
`mosaic: true` exists to handle, and it is silent when it happens. The reference is
wrong over part of the stamp, the kernel fits a compromise scaling, and the residual
looks like a subtraction that "just isn't very good" rather than a bug.

Fix: linearise each piece with its own header inside `_combine_skycells`, before
resampling and filling.

### H4 — `image.edge_margin_arcsec` is ignored by three of its four consumers
`tttphot/detection.py:89`, `tttphot/photometry.py:302`, `tttphot/pipeline.py:1042`

`config/default.yaml` sets only `edge_margin_arcsec: 9.0`; `edge_margin_px` is never
defined. `ScienceImage._edge_margin` reads the arcsec form correctly, but these three
sites do

```python
int(cfg.get("image.edge_margin_px", 24))
```

and so always get the literal 24. At the default `rebin: auto` (0.385″/px) the arcsec
setting means 23 px, so the two agree by coincidence. With `image.rebin: 1` — which the
README documents as a supported switch — the mask margin becomes 47 px while PSF-star
selection, `empirical_limit` and the "is the target on this frame?" test still use 24.
The comment in `_edge_margin` explaining *why* the margin is in arcsec applies equally
to all four.

Fix: give `ScienceImage` a public `edge_margin_px` property (or a module-level
`edge_margin(cfg, pixel_scale)` helper) and route all four through it.

### H5 — `photutils>=1.10` is too loose: `xy_bounds` will raise
`pyproject.toml:15`, `requirements.txt:4`, used at `tttphot/photometry.py:246`

`PSFPhotometry(..., xy_bounds=...)` did not exist in photutils 1.10–1.12. The
constructor sits *outside* the `try` at line 248, so on such an install every call
raises `TypeError` and takes the whole epoch down through `process_image`'s caller —
not a degraded PSF row, a lost frame.

Fix: raise the floor to the version that introduced `xy_bounds` (`photutils>=2.0,<3`
is the safe choice given the `<3` ceiling is already there for the column rename).

### H6 — Catalogue rows with no quoted error survive the error cut
`tttphot/catalogs.py:401`

```python
keep &= ~np.isfinite(err) | (err <= max_err)
```

`~` binds tighter than `|`, so this reads "keep if the error is missing **or** small".
Verified: `err = [0.01, 0.5, nan]` with `max_err = 0.05` gives `[True, False, True]`.
`_float_column` turns every masked/absent value into NaN, and `fetch_gaia_synth`
produces NaN errors wherever the synthetic flux is non-positive, so entries whose
photometry the survey itself could not quantify go straight into the zero-point fit
with the default 0.02 mag weight assigned at `calibration.py:302`.

It may be deliberate (keeping numbers on a sparse field), but it is not what the
docstring says and the parenthesisation reads like an accident. If it is deliberate,
write it as `(~np.isfinite(err)) | (err <= max_err)` with a comment; if not, drop them.

### H7 — Pan-STARRS star/galaxy cut drops sources that have no *r* or *i* Kron magnitude
`tttphot/catalogs.py:285-286`

```python
ref = np.nanmedian(np.column_stack([out[b] - out[f"k_{b}"] for b in ("r", "i")]), axis=1)
out["point_source"] = np.abs(ref) < 0.10
```

A source with neither an *r* nor an *i* Kron magnitude gives `ref = nan`, and
`abs(nan) < 0.10` is `False` — so `clean_catalog` discards it, in **every** band,
including *g* and *z* where its own photometry is perfectly good. The same expression
emits an `All-NaN slice encountered` RuntimeWarning per affected row. Verified both.

Fix: default the verdict to `True` when the classifier has no information
(`np.where(np.isfinite(ref), np.abs(ref) < 0.10, True)`), and suppress the warning with
an explicit `np.errstate`/`warnings` guard.

### H8 — "SIMBAD could not be reached" is silently treated as "no variables here"
`tttphot/catalogs.py:444-451` vs `tttphot/calibration.py:186-188`

`variable_stars`'s own docstring is unambiguous:

> Returns … an empty table when SIMBAD has nothing, or **None when it could not be
> reached** — the caller must be able to tell "no variables here" from "I do not know",
> and must not silently calibrate on variables because a query failed.

The only caller does:

```python
known = cat.variable_stars(...)
if known is None or len(known) == 0:
    return table
```

which collapses exactly the distinction the function went to the trouble of making. On
a sparse field with three to seven calibration stars — the scenario the `VARIABLE_OTYPES`
comment describes — a transient network hiccup silently reintroduces the Mira the screen
exists to remove, with nothing in the output to say so.

Fix: on `None`, log at WARNING and set a flag on the `ZeroPoint` (or a new
`FLAG_VARIABLES_UNSCREENED` bit) so the epoch carries the fact.

### H9 — Gaia GSPC quality flags are fetched, stored, and never used
`tttphot/catalogs.py:145, 184-186` and `clean_catalog`

The ADQL pulls `s.<band>_flag` for every band and writes it as `f_<band>`, and
`astrometric_excess_noise` / `phot_bp_rp_excess_factor` are selected but never even
copied into the output table. Nothing downstream reads any of them. GSPC flags mark
synthetic photometry that the Gaia team considers unreliable (extrapolated bandpass,
poor BP/RP sampling), and `gaia_synth` is the *primary* calibrator for every band.

Fix: either filter on `f_<band> == 0` in `clean_catalog` (behind a config key, since it
will cost stars on some fields), or drop the unused columns from the query.

### H10 — `FLAG_MASKED_PIXEL` and `FLAG_OFF_IMAGE` bias the flux low and demote nothing
`tttphot/photometry.py:63-74`, `tttphot/lightcurve.py:93`

`aperture_photometry(..., mask=image.mask)` *excludes* masked pixels from the sum, and
`area_overlap(..., mask=...)` correctly reduces the area used for the background
subtraction — but the source flux is never rescaled for the missing pixels. So an
aperture clipped by a bad column, a cosmic ray or the frame edge returns a
systematically faint flux. Both cases are flagged (bits 1 and 4), which is the right
design, except that `_suspect` only looks at

```python
flags & (FLAG_BAD_SUBTRACTION | FLAG_SATURATED | FLAG_IN_TEMPLATE)
```

so a row measured through a clipped aperture can still be chosen as primary over a
clean one of lower method priority.

Fix: add `FLAG_MASKED_PIXEL | FLAG_OFF_IMAGE` to `_suspect` (they are already
"only used if nothing cleaner exists" material), and optionally rescale the flux by
`total_area / area_overlap` when the missing fraction is small.

### H11 — `FLAG_SATURATED` can never fire on a difference row
`tttphot/pipeline.py:704, 733`

Difference rows take `flags=int(tab["flags"][0])` from `measure_apertures` run on
`_DifferenceImage`, whose `meta` is the *science* frame's. The saturation test is
`sub >= image.meta.saturation` applied to difference pixels, which scatter around zero,
so it is never true. A saturated transient therefore produces an unflagged
`difference_psf` row, and `_suspect` — which does check `FLAG_SATURATED` — has nothing
to act on.

Fix: carry the science-frame aperture flags (the `tab` from `process_image:524`)
into the difference rows with `|`, as is already done for `bad_subtraction`.

---

## M. Scientific questions worth a decision

### M1 — The kernel solve is unweighted
`tttphot/subtraction.py:234`

```python
coefficients, *_ = np.linalg.lstsq(design, science[keep], rcond=None)
```

Alard & Lupton fit the kernel weighted by the per-pixel noise. Here every pixel counts
equally, so the solution is set almost entirely by the brightest few star cores in the
stamps — the pixels most affected by detector non-linearity and by the `peak_limit`
heuristic in M2. The sigma clipping at line 241 uses `np.std(residual)` over the same
pixels, which those cores also dominate, so it rejects very little.

This may well be a deliberate and defensible choice (bright cores *do* carry the PSF
information), and the validation numbers suggest it works. But it is the one major
methodological departure from the reference implementation that the docstrings do not
record. Either weight by `1/variance` and compare, or add the paragraph explaining why
not — this module's other choices all have one.

### M2 — `peak_limit` is a percentile of the stamp, not a saturation level
`tttphot/subtraction.py:113`, and the same idiom at `:486`

```python
peak_limit = np.nanpercentile(science, 99.999)
...
if float(row["peak"]) > 0.9 * peak_limit: continue
```

The 99.999th percentile is "the N-th brightest pixel" where N depends on the stamp
area: ~11 pixels on a 1091² stamp, ~120 on a 3437² one (`rebin: 1`). So which stars are
rejected as "too bright to trust" changes with the stamp size and with the host
galaxy's brightness rather than with anything about the detector. On a stamp dominated
by a bright nucleus, the cut can exclude the best kernel stars.

`image.meta.saturation` (scaled for rebinning, already correct) is available and is the
physical quantity meant here.

### M3 — `area_ratio` assumes flux-conserving resampling; `reproject_interp` is not
`tttphot/subtraction.py:324-325, 384`

The comment says "flux-conserving resampling changes surface brightness with pixel
area", but `reproject_interp` interpolates values — it preserves per-pixel value, not
flux. Two consequences: the `area_ratio` factor applied to `ref_var` is a heuristic
rather than a derivation, and interpolation correlates neighbouring reference pixels,
so `fftconvolve(ref_var, kernel**2)` underestimates the propagated noise regardless.

In practice `blank_aperture_noise` and `combine_errors` take the max against an
empirical scatter, which covers this — that is a good design. Worth saying so in the
docstring so nobody later "fixes" the empirical term away.

### M4 — `build_ttt_stack` median-combines epochs without putting them on a common scale
`tttphot/templates.py:156-174`

Each plane is background-subtracted (good) but not divided by exposure time or scaled
to a common transparency before `np.nanmedian`. Frames of different depth therefore
contribute on different flux scales, and the median of a mixed set is not a clean
estimator of any of them. The kernel fits a single scaling afterwards, so the *mean*
level is absorbed; the per-pixel noise and any zero-point drift across the stack are
not. The stack also carries no `variance`, so `subtract` falls back to a constant
`ref_rms²`.

The cheapest improvement: scale each plane by `exptime` and by `10**(0.4 * zp)` where
a zero point is available, and build a `counts`-based variance plane alongside `stack`.

### M5 — Injected sources do not update the error map, which biases `error_scale`
`tttphot/injection.py:139-151`

`recovered_fraction` adds the model to `image.data` and calls `measure_apertures`, but
`image.error` is untouched, so the quoted `flux_err` omits the injected source's own
Poisson noise. The `z = (measured - true) / quoted` diagnostic at line 203 therefore
has a systematically small denominator, and `error_scale` comes out high — the more so
for the bright bracketing passes. Related: `truth` accumulates across *every* call to
`recovered_fraction`, so `error_scale` is an MAD over a sample spanning the whole
bracket-and-bisect magnitude range, not over sources near the limit.

At the 50% completeness point the source contributes little, so the limit itself is
fine; it is the error-honesty diagnostic that is skewed. Either add
`model / gain` to a local copy of the variance, or restrict `truth` to the bisection
passes and say in the docstring what the number means.

### M6 — `refine_wcs` compares residuals measured on different star sets
`tttphot/astrometry.py:111, 121, 171`

`rms_before` is computed on iteration 0 with `match_radius_arcsec` (3″);
`rms_after` after the last iteration with `final_match_radius_arcsec` (1″), on whatever
subset survived. With the default `iterations: 2` the acceptance test
`rms_after > rms_before → reject` therefore compares a tight-matched subset against a
loose-matched superset, which biases towards accepting the refit. Recomputing
`rms_before` on the *final* matched set before the comparison would make the test mean
what its message says.

Separately, `_fit_shift` (line 189) moves `CRPIX`, which is also the origin of the SIP
polynomials, so a shift technically re-anchors the distortion. For the sub-arcsecond
shifts seen here it is negligible; the README's "keeps the header's SIP distortion" is
worth qualifying to "keeps it, re-anchored by the shift".

### M7 — Transferred zero points get a different limiting magnitude and a hardcoded SNR
`tttphot/zeropoints.py:262, 278-281` and `tttphot/pipeline.py:809`

When a zero point is recovered from another band, `limit_mag` is recomputed with the
analytic `sky_rms * sqrt(pi r²)` formula, while every other row in the table carries
the injected or blank-aperture limit. Two definitions of depth in one column, with
nothing to distinguish them.

Both files also hardcode `snr >= 3.0` as the detection/limit boundary, duplicated and
unconnected to `calibration.limiting_sigma: 3.0`. If the threshold is ever changed in
one place the table becomes internally inconsistent.

Fix: one `output.detection_snr` key, read in both places; and either recompute the
empirical limit on transfer or leave `limit_mag` as it was and scale it by `Δzp`.

### M8 — Calibration matching allows many-to-one
`tttphot/calibration.py:291`

`det.match_to_catalog_sky(ref)` is taken at face value; two detections can claim the
same catalogue star. `astrometry._match` (line 47) implements exactly the
mutual-uniqueness filter that is missing here. On a blended pair this puts the same
catalogue magnitude against two different instrumental magnitudes, which pulls the zero
point and inflates `rms`. Reusing `_match` here would cost nothing.

### M9 — `convolve_science` mode: effective gain and host-residual reference
`tttphot/pipeline.py:687, 723, 689`

`combine_errors(..., gain=image.meta.gain * scale)` treats the convolution as a simple
flux multiplication by `sum(kernel)`; the variance actually scales as `sum(kernel²)`, so
the source shot-noise term is approximate. And `host_residual` is handed
`diff.reference_stamp`, which in this mode is the convolved *science* frame — it
contains the transient, so "surface brightness of the host" includes it. The
`avoid_fwhm: 3.0` exclusion covers the immediate neighbourhood, but the docstring's
assumption (matched *reference*) no longer holds.

Both are minor and this mode "rarely fires" per the config comment — worth a note in
the code so it is not rediscovered later as a mystery.

---

## R. Robustness and architecture

### R1 — There is no test suite
`tests/` contains `make_synthetic_catalog.py` and `selftest_injection.py` — two scripts
that need a real FITS frame on disk and network access. There is no `pytest` suite, no
CI, and no way to check a refactor without a ~2-minute end-to-end run against data.

For 7,000 lines of numerical code where most of the subtlety is in pure functions, this
is the highest-leverage thing to add, and most of it needs no data at all:

- `utils`: `normalise_name`, `band_token`, `find_by_designation`, `weighted_mean`,
  `flux_to_mag`, `robust_sigma` — pure, trivially testable.
- `detector`: `_rebin_array` (sum, NaN handling, saturated-block promotion),
  `_rebin_header` (WCS round-trip to the 1e-10″ the README claims), `trim_slices`,
  `resolve_size`, `identify`.
- `image.effective_gain`: the whole binning/stacking truth table.
- `calibration._sigma_clipped_fit`: recover a known slope and intercept from
  synthetic data with a planted outlier.
- `zeropoints`: `observing_night` boundaries, `band_offsets`, `_combine`, and a
  `transfer` round-trip on a small hand-built DataFrame.
- `lightcurve`: `choose_primary` and the host guard — a table of
  (flags, method, mag) → expected winner would have pinned down the 0.75 → 0.2 mag
  change you describe in the docstring.
- `catalogs.clean_catalog`: H6 and H7 above are both one-line tests.
- `subtraction.solve_kernel`: convolve a synthetic star field with a known Gaussian,
  check the recovered kernel and `kernel_sum`.

Half a day of this would have caught H1, H6, H7 and H10 outright.

### R2 — The top-level handler reports internal bugs as user errors
`tttphot/cli.py:971-976`

```python
except (RuntimeError, FileNotFoundError, KeyError) as exc:
    print(f"error: {exc}", file=sys.stderr); return 1
```

The comment says exit 1 means "the data or the configuration is not what was asked
for" and exit 2 means "the tool itself broke". But `RuntimeError` is also what
`ScienceImage.build_error_map` raises for a programming-order mistake
(`image.py:421`), what `data_sub` raises (`:432`), and what `solve_kernel` raises for a
degenerate fit (`subtraction.py:227`). Those are bugs, and they exit 1 with no
traceback.

Fix: define a `TttphotUserError(Exception)` (or reuse `CatalogError`) for the
genuinely-expected cases — no files found, no epochs, missing config — and catch only
that.

### R3 — `_Relaxed` is a duck-typed config view with two methods
`tttphot/calibration.py:95-108`

It forwards `get` and `resolve_path` only, and is passed as `cfg` into
`_calibrate_once`, which in turn passes it to `detector.catalog_radius_arcmin` and
`cat.clean_catalog`. Today none of those need anything else. The moment one of them
calls `cfg.require`, `cfg.cache_root`, `cfg.as_dict` or `cfg["x"]` — all of which exist
on the real `Config` — the relaxation path breaks and the strict path does not, which
is a nasty way to find out.

Fix: make it a real subclass of `Config` built by `deep_merge`-ing the overrides, or
give `Config` an `overridden(**kw)` method that returns a proper `Config`.

### R4 — `to_frame` silently drops unknown columns
`tttphot/lightcurve.py:210`

```python
frame = frame[COLUMNS]
```

Any key a row carries that is not in `COLUMNS` disappears without a word. Given that
`common` is assembled across three files and `diff_meta` keeps growing, a new
diagnostic added in `pipeline.py` and forgotten in `COLUMNS` vanishes from the output
and nothing says so. One `set(frame.columns) - set(COLUMNS)` check with a
`log.debug`/`log.warning` would close it.

### R5 — `Detector` ignores the y binning
`tttphot/detector.py:284`

```python
bx, _ = camera_binning(header, instrument)
```

`camera_binning` returns both, `effective_gain` uses both, and `identify` throws `by`
away. Non-square binning (rare but legal on both these cameras) would produce a
detector token that mislabels the readout and could silently merge two configurations
in `group_by_detector` and in the kernel-sum grouping.

### R6 — Fragile string coupling for the camera key
`tttphot/zeropoints.py:154`

```python
return frame["detector"].fillna("").astype(str).str.split().str[0]
```

This reconstructs the camera by splitting `Detector.label`, whose format is defined in
a different module. Changing `label` to, say, `"QHY600-TTT3 (1595×1064)"` silently
breaks per-camera zero-point transfer with no error — and the docstring explains that
mixing cameras scattered g−i by 0.62 mag. Write the camera into its own column in
`process_image` (`camera=identify(image.header).instrument`) and read that.

### R7 — Smaller robustness notes

`detector.resolve_size:337` raises `ValueError` on any non-numeric string that is not
`auto`/`frame`, rather than falling back to `default` with a warning — a typo in
`cutout_arcmin` becomes a stack trace.

`templates.register_reference:650` sets `ref.registration` on a dataclass that has no
such field; it works, but it should be declared on `ReferenceImage`.

`subtraction.solve_kernel` returns `background`, and `subtract:365` unpacks it and never
uses it — either drop it from the signature or fold it into `info`.

`catalogs.fetch_gaia_*` mutates `Gaia.ROW_LIMIT`, a class attribute on an imported
module, as a side effect of a function call.

`plots.py:11` calls `matplotlib.use("Agg")` at import, which `pipeline` imports — so
importing `tttphot.pipeline` in a notebook silently kills interactive plotting for the
whole session. Move it inside the plotting functions, or guard it on
`matplotlib.get_backend()`.

The working tree has 17 modified files and 2 untracked ones on a repository with a
single commit. Everything above was reviewed against the working tree; it would be
worth committing before changing anything, so the review has a baseline to diff against.

---

## P. Performance and memory

### P1 — `solve_kernel` holds ~50 full-stamp float64 arrays
`tttphot/subtraction.py:176-184, 240, 246`

With the default basis (`sigmas [0.7, 1.5, 3.0]`, `degrees [4, 2, 0]`) there are 22
basis functions. `convolved` holds 22 full-stamp arrays; `columns` then holds 22 more
(because `conv * poly` copies even when `spatial_order: 0` makes `poly` all-ones) plus
3 background terms; `model_full` and `model` add two more, each iteration.

At the default `rebin: auto` the stamp is ~1091² → 9.5 MB per array → roughly **450 MB**
peak. With `image.rebin: 1` the stamp is ~3437² → 94 MB per array → roughly **4.4 GB**,
which will not survive on most laptops. This is the single largest memory consumer in
the pipeline.

Two cheap wins: skip the `conv * poly` copy when `spatial_order == 0` (reuse
`convolved` directly), and build the normal equations incrementally
(`A.T @ A` accumulated column-pair by column-pair over `keep`) instead of
materialising the full design matrix and every `columns` entry.

### P2 — `curve_of_growth` recomputes the same annulus ten times
`tttphot/photometry.py:136-139`

`measure_apertures` is called once per radius, and each call runs `ApertureStats` with
`SigmaClip(maxiters=10)` on an annulus that is *identical* across radii — the annulus
depends only on the FWHM. With the default `curve_of_growth_scales` that is ~11
identical sigma-clipped annulus computations over up to 40 stars, per frame.

Fix: compute the local background once and pass it in, or add a `bkg_per_pix=`
parameter to `measure_apertures`.

### P3 — `empirical_limit` places four times the apertures it advertises
`tttphot/photometry.py:305-307`

`n_apertures=300` but `size=n_apertures * 4`, and all 1200 are used — the `* 4` reads
like an over-sample-then-select that was never finished. Either select down to
`n_apertures` after the source rejection, or drop the factor and say 1200.

### P4 — Injection is ~800 single-position `measure_apertures` calls per frame
`tttphot/injection.py:141`

40 sites × (1 start + up to 14 bracket + 6 bisect) magnitudes, each call building a
`CircularAnnulus`, an `ApertureStats` with sigma clipping, and an astropy `Table` for
one position. Batching all 40 sites into one `measure_apertures` call per magnitude
(inject all of them at once into a scratch copy, measure, restore) would cut this by a
factor of ~40 at the cost of one full-frame copy.

### P5 — Reference images are held as float64 for the whole run
`tttphot/templates.py:77, 344` (`np.asarray(hdu.data, dtype=float)`)

A 3200² PS1 cutout is 82 MB as float64, 41 MB as float32, and `build_references` keeps
one per band alive for the entire run. Three bands is ~250 MB resident that never gets
used at more than float32 precision (the cached FITS is written as float32 anyway).
Same for `ScienceImage.data`.

### P6 — `build_references` loads up to eight sibling frames simultaneously
`tttphot/pipeline.py:417-424`

Each `prepare_image` produces `data`, `mask`, `bkg_map`, `bkg_rms` and `error` at full
frame size, and `siblings` holds them all before the stack is built. Eight 1595×1064
rebinned frames is ~270 MB; unrebinned it is four times that. Reprojecting each plane
into the target grid as it is loaded, rather than keeping the `ScienceImage` objects,
would cap this at one frame plus the cube.

### P7 — `zeropoints._transfer_one` walks rows with `.loc`
`tttphot/zeropoints.py:243-281`

`for row_index in index: row = frame.loc[row_index]` materialises a mixed-dtype Series
per row and then does several `frame.loc[row_index, col] = ...` assignments. It is
correct but O(n) per access and it fights pandas' dtype inference. The block is
vectorisable over `index` in one pass.

---

## C. Config hygiene and dead code

These config keys are documented in `config/default.yaml` and read by nothing:
`photometry.local_bkg` (`measure_apertures` takes a Python default instead),
`output.write_diagnostics`, `output.write_stamps`, `output.overwrite`, and
`astrometry.max_mag` (`fetch_gaia_astrometry` has its own 20.0 default and
`get_catalog` never forwards the setting — they happen to agree today).

`config/default.yaml:239` has a merge artifact: the `transfer_error_floor` line carries
a trailing `# above this the frame is flagged poor_calibration`, which belongs to
`max_rms` twelve lines above.

Dead code: `zeropoints._partner` (defined, never called — its own docstring says "kept
for the single-partner case", which `_combine` now handles), and
`astrometry.internal_consistency` (never called from anywhere).

`detection.detect_sources:49` overwrites photutils' 1-based `id` column with a 0-based
`np.arange`, which is fine but surprising if the table is ever compared against a raw
finder result.

---

## Suggested order of work

1. **H1, H2, H5** — three one-line fixes, all verified, all affecting results or cost today.
2. **H3, H7, H6** — catalogue and template correctness; each is a handful of lines.
3. **R1** — start the pytest suite with the pure functions listed. This is what makes
   everything after it safe.
4. **H4, H10, H11, M7** — consistency of margins, flags and thresholds across modules.
5. **P1, P2, P4** — the three performance items with the best ratio of speedup to risk.
6. **M1, M3, M4** — the scientific questions, once there are tests to measure the answer
   against.

Everything in **M** is a judgement call rather than a defect; the pipeline's existing
validation (`docs/VALIDATION.md`, the 0.02–0.03 mag cross-camera agreement) suggests
the current choices work. The value in addressing them is mostly that the *reasons*
join the rest of the docstrings, which are otherwise the most reliable documentation in
the repository.
