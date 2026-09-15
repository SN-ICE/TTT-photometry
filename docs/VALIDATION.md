# Validation

State on 2026-09-15, with one real frame:
`TTT3_QHY600-3_2026-09-11-22-15-41-277664_2026abgb_SDSSg.fits` (TTT3, SDSSg,
120 s, 3x3 binned, 3191 x 2129 px at 0.192"/px).

Everything below was done **offline**: the build machine could not reach Gaia,
Pan-STARRS, SIMBAD or TNS. So the photometric method is checked end to end, but
**not** a live calibration against real catalogues or a real PS1 template. That
is the next check, and the observatory's own `ZP` card (carried as `obs_zp`) is
the natural first comparison.

## 1. Detector model

| check | result |
|---|---|
| sky level / pixel-to-pixel scatter | 85.3 ADU / 5.8 ADU |
| scatter predicted, binned pixels *averaged* (gain 3.96, RON 14.2 e-) | 5.9 ADU |
| scatter predicted, binned pixels *summed* (gain 0.44, RON 14.2 e-) | 35 ADU |
| `sky_noise_ratio` on load, rebinned 2x2 | **1.02** |
| `sky_noise_ratio` on load, not rebinned | **1.00** |

So the camera averages when it bins, and the header `GAIN`/`RDNOISE` are
per unbinned pixel, as their comments say.

Row 0, columns 0-12 hold a GPS time stamp (9003, 93792 ADU); masked. No
vignetting or shadowed border: row and column medians stay within 85 +- 2 ADU
to the edges, so no trim.

## 2. Rebinning

| check | result |
|---|---|
| WCS after 2x2 rebin vs native, 2000 random positions (SIP included) | max 3e-10" forward, 4e-8 px inverse |
| FWHM, rebinned / native | 3.014" / 3.018" |
| time to prepare one frame, rebinned / native | 3.8 s / 20.1 s |

## 3. Injection and recovery through the full pipeline

`tests/selftest_injection.py`, which anyone can re-run. A local catalogue of
the frame's 34 point-like stars is put on the observatory's zero point
(24.405); the reference is the frame itself resampled to a PS1-like 0.25"/px
grid; each science epoch is the frame plus a scaled copy of a real g = 14.55
field star placed ~1 FWHM from the brightest part of the host galaxy, then
convolved with a Gaussian of FWHM 1.5" or 2.5" (seeing 3.0" -> 3.4" or 4.0"),
with fresh sky noise added back. Then `tttphot run` as normal.

Recovered zero points: 24.397-24.406 (truth 24.405).

| injected g | extra blur | difference PSF | difference aperture | direct aperture (host in it) |
|---|---|---|---|---|
| 17.50 | 1.5" | +0.011 +- 0.012 | +0.014 +- 0.013 | -2.46 |
| 18.50 | 2.5" | +0.027 +- 0.022 | +0.017 +- 0.025 | -3.72 |
| 19.50 | 1.5" | -0.033 +- 0.035 | -0.044 +- 0.048 | -4.37 |
| 20.50 | 2.5" | +0.181 +- 0.141 | +0.096 +- 0.152 | -5.69 |

(recovered - injected, mag). All difference measurements within 1.3 sigma;
chi2 0.90-0.95, kernel sums 4.000-4.005 (the 0.25" reference against summed
0.385" pixels), no flags. Every epoch was PSF-matched in `convolve_reference`
mode, the realistic case against a PS1 template.

The same injection on empty sky recovers all four methods to <= 0.035 mag, so
the chain itself (injection, rebinning, apertures, PSF, calibration) carries no
bias at that level.

### Caveat found along the way

A first version of the test placed the source by mistake a few arcsec from a
bright field star rather than on the host. There the bright epochs came out
4-8 per cent faint in difference PSF photometry and 9-14 per cent faint in
difference aperture photometry (3.5-7.6 sigma), growing with the seeing
mismatch. Not investigated further; a transient that
close to a bright star deserves a look at its QA panel.

## 4. Failure paths

With no network: the run completes, the epoch is reported with
`no_zeropoint`, and `list`, `clean`, `fetch-template`, `colorterms`,
`recalibrate` and `build-pattern` all fail gracefully with a message.
