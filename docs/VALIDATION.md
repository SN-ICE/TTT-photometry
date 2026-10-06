# Validation

This document records historical validation runs. References below to the
host guard and automatic fallback after subtraction-quality failures describe
the selection policy at the time. Since 2026-10-05, usable difference
measurements are retained with their flags and drawn as open symbols; see
the README for the current selection and color-pairing policy.

Sections 1-4: state on 2026-09-15, with one real frame:
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

## 5. Three cameras on SN 2026abgb (2026-09-16)

The first live run had 47 frames: 18 TTT3/QHY600, 18 TTT3/iKon936, 11
TTT4/QHY600. Same-night epochs of the two TTT3 cameras came out up to 1.6 mag
apart. Re-running the g band (12 frames, all three cameras) with the Gaia
catalogues and PS1 template cached from that run:

| MJD | camera | before: adopted | before: method | after: adopted | after: method |
|---|---|---|---|---|---|
| 61294.928 | TTT3 QHY600 | 16.88 | psf (direct) | 18.56 | difference_psf |
| 61294.932 | TTT3 QHY600 | 16.96 | psf (direct) | 18.52 | difference_psf (flagged, host guard) |
| 61294.938 | TTT3 iKon936 | 18.54 | difference_psf | 18.51 | difference_psf |
| 61294.943 | TTT3 iKon936 | 18.55 | difference_psf | 18.55 | difference_psf |
| 61295.159 | TTT3 QHY600 | 16.95 | aperture (direct) | 18.48 | difference_psf |
| 61295.163 | TTT3 QHY600 | 17.48 | psf (direct) | 18.45 | difference_psf |
| 61297.071 | TTT3 iKon936 | 18.00 | difference_psf | 18.01 | difference_psf |
| 61297.075 | TTT3 iKon936 | 17.99 | difference_psf | 17.98 | difference_psf |
| 61297.959 | TTT4 QHY600 | 17.23 | psf (direct) | 17.82 | difference_psf |
| 61297.963 | TTT4 QHY600 | 17.32 | psf (direct) | 17.82 | difference_psf |
| 61297.972 | TTT3 iKon936 | 17.83 | difference_psf | 17.80 | difference_psf |
| 61297.977 | TTT3 iKon936 | 17.80 | difference_psf | 17.80 | difference_psf |

("before" is the live run; the cloud re-run of the old code reproduced it,
including "11 of 23 difference epochs have a kernel sum far from the band
median".) `zp - obs_zp` after: +0.05 to +0.10 on all 12 frames.

What was wrong, in order of impact:

1. Kernel sums compared across cameras (0.12-0.17 QHY600, 0.78-0.92 iKon936
   in g). Normalised by exposure time, 10^(0.4 ZP) and pixel area, the
   per-camera medians agree to 3 per cent in g, r and i and every epoch lies
   within 10 per cent of them (checked on the full live-run table), including
   the one `convolve_science` epoch once inverted. The tolerance is x3.
2. PS1 template WCS 0.75 px (0.26") off Gaia in both axes; TTT frames within
   0.04" of Gaia for all three cameras. Now registered (moved 0.26", residual
   0.48" -> 0.09").
3. TTT3 QHY600 stars elongated along detector columns in good seeing
   (2026-09-12: 2.6 vs 5.5 px FWHM; median DAOStarFinder roundness2 0.93).
   Detection limits widened to +-1.5.
4. The seeing iteration adopting a 10-source pass (5.3 px) over a 27-star
   one (3.8 px); the profile test then kept 2 PSF stars, the aperture
   correction came from one star and the epoch was 0.4 mag off.

Not changed: the dipole test still calls some frames of every camera
"downgrade" or "fail" even after registration, with no effect on registration
or kernel spatial order. Their difference photometry agrees with the other
camera's at the same epoch, so it stays an annotation (a "fail" is now kept by
the host guard rather than replaced with direct photometry).

Frames that fail for reasons outside the pipeline, now reported as such:
TTT3/QHY600 2026-09-14 03:27-03:37 UT (6 frames, no WCS in the header; the
one inspected has a sky of ~4 ADU and no stars) and TTT4/QHY600 2026-09-13
22:09-22:20 UT (5 frames: four carry a good solution, 27-38 catalogue stars,
6.3 deg away on the sky at RA 345.05 -- the telescope was on another field --
and one has none).

## 6. New epoch, two r outliers (2026-09-16)

After adding the 2026-09-15/16 frames (65 in total), two r epochs sat
0.5-0.7 mag above the rest of the light curve: TTT3/QHY600 at MJD 61298.918
and 61298.922, adopted at 16.82 and 16.68 by direct PSF photometry.

Nothing was wrong with those frames' measurement. Zero points were normal
(`zp - obs_zp` -0.02, 40 and 26 Gaia stars), the subtractions solved (kernel
sums 0.076/0.077, chi2 1.01/1.03), and their difference photometry, 17.39 and
17.36, matches the iKon936 r frames of the same night (17.34, 17.34). The
dipole test called both subtractions "fail" (41 and 31 per cent aligned
dipoles), which demotes them, and the host guard that should have kept them
did not fire: the direct values were brighter by 0.57 and 0.69 mag, under the
0.75 mag threshold.

The threshold is now 0.2 mag. Re-selecting the adopted rows of the full table
with 0.1, 0.2, 0.3 and 0.5 changes exactly these two epochs and nothing else;
the smallest difference-minus-direct gap of any epoch on this target is
0.32 mag (the SN sits on NGC 7673), so there is no epoch between the old and
new thresholds except these two.

## 7. Scatter within the first r night (2026-09-16)

The four r frames of 2026-09-11 (MJD 61294.93-61294.94, two TTT3/QHY600, two
TTT3/iKon936, all 22:17-22:40 UT) came out at 18.32, 18.19, 18.20 and
18.25 with errors of 0.013-0.021: reduced chi2 10 about their mean. The g
frames of the same night agreed to 0.02.

Ruled out: registration (science on Gaia and on the template within 0.06" in
every frame), PSF asymmetry (flux centroid minus Gaussian centroid <= 0.04"),
the annulus background, and the position (the SN sits 0.14" from the TNS
coordinates, E +0.09", N -0.11", the same in all cameras; worth ~0.01 mag).

What it is: that night the seeing was 2.0-2.6" (the other r nights 0.9-1.5").
The SN is 12" from the nucleus of NGC 7673 and the host inside the aperture is
2-4 times the SN flux, so a 1 per cent error in the kernel's flux scale moves
the SN by 3-4 per cent. Laying the aperture on other sites of similar host
brightness within 15 FWHM, the difference image holds 4-30 per cent of the SN
flux in 2-3" seeing and 1-3 per cent in ~1". No kernel variant helped
(spatial order 1 and 2, a 4' cutout, background order 2, a larger kernel):
none brought the residuals below ~15 per cent, and each moved the SN by
0.1-0.28 mag.

So the error is added, not removed (see README, "Errors on a bright host").
Difference PSF magnitudes, all 20 staged frames (primary rows):

| | r ep. 1 chi2/dof | r same-camera pairs, rms (sigma) | g ep. 1 chi2/dof | g pairs, rms (sigma) |
|---|---|---|---|---|
| photon errors only (target already out of the kernel fit) | 15 | 3.3 | 1.7 | 1.3 |
| + host residual x 1.0 | 0.11 | 0.74 | 0.06 | 0.37 |
| + host residual x 0.5 (default) | 0.43 | 1.1 | 0.22 | 0.64 |
| + host residual x 0.25 | 1.5 | 1.5 | 0.60 | 0.93 |

Same-camera pairs share seeing and host residual pattern, so they understate a
correlated error; the four-frame, two-camera night is the better test. The
r frame at 22:17 UT (2.63", flagged `bad_subtraction|dipole_residual`) is the
one furthest off, +0.10 mag, and has the largest residual, 31 per cent, so an
error of 0.17. The per-frame estimate is noisy (10-30 sites), which is why the
scale is below 1; it is fitted on these 20 frames only.

Along the way: the SN was among the kernel stamps in every frame (nearest
stamp 0.4-1.1 px from it). Sigma clipping hid most of it. Keeping its pixels
within 2 FWHM out of the fit changes the real frames by <= 0.02 mag, and the
injection test of section 3 goes from +0.012/+0.033/-0.028/+0.202 to
+0.002/+0.006/-0.066/+0.021 (difference PSF, g = 17.5-20.5). A first version
rejected every stamp overlapping the SN; that also removed the host nucleus,
shifted the SN 0.05-0.2 mag brighter and tripled the kernel-sum scatter about
the zero-point prediction (1.1 -> 4.6 per cent in g), so it was dropped.

Not yet looked into: the injection depth (`inject_limit_mag`) comes out NaN on
TTT frames with 40 sites.
