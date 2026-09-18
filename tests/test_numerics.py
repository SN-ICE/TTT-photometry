"""Tests for the numerical machinery: the kernel solve, aperture photometry,
the difference-quality tests, and the cosmic-ray call.

These need numpy, scipy and photutils but still no FITS file and no network.
Where a test exists because a bug was really there, it says so.
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import tttphot
from tttphot import catalogs, cosmics, diffquality, photometry as ph, subtraction as sub


class FakeConfig:
    def __init__(self, **values):
        self._values = values

    def get(self, dotted, default=None):
        return self._values.get(dotted, default)


class FakeMeta:
    def __init__(self, saturation=60000.0, gain=3.96, exptime=120.0, pixel_scale=0.385):
        self.saturation = saturation
        self.gain = gain
        self.exptime = exptime
        self.pixel_scale = pixel_scale
        self.path = Path("synthetic.fits")


class FakeImage:
    """The bit of the ScienceImage interface that measure_apertures touches."""

    def __init__(self, data, sky=100.0, rms=3.0):
        self.data = np.asarray(data, dtype=float)
        self.mask = np.zeros(self.data.shape, dtype=bool)
        self.bkg_map = np.full(self.data.shape, float(sky))
        self.bkg_rms = np.full(self.data.shape, float(rms))
        self.error = np.full(self.data.shape, float(rms))
        self.meta = FakeMeta()
        self.qa = {"sky_adu": float(sky)}

    @property
    def data_sub(self):
        return self.data - self.bkg_map

    def build_error_map(self):
        return self.error


def gaussian_field(n=240, n_stars=30, fwhm=3.0, seed=5, amplitude=(2.0, 4.0)):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    field = np.zeros((n, n))
    sigma = fwhm / 2.3548200
    for _ in range(n_stars):
        x0, y0 = rng.uniform(30, n - 30, 2)
        peak = 10.0 ** rng.uniform(*amplitude)
        field += peak * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2))
    return field


# ---------------------------------------------------------------------------
# subtraction: the kernel solve
# ---------------------------------------------------------------------------
def _kernel_case(ref_fwhm=3.0, sci_fwhm=4.6, scale=0.63, sky=21.0, noise=1.0, seed=5):
    from scipy.ndimage import gaussian_filter

    reference = gaussian_field(fwhm=ref_fwhm, seed=seed)
    extra = np.sqrt(max(sci_fwhm ** 2 - ref_fwhm ** 2, 1e-6)) / 2.3548200
    science = scale * gaussian_filter(reference, extra) + sky
    science += np.random.default_rng(seed + 1).normal(0, noise, science.shape)
    return reference, science


def test_solve_kernel_recovers_the_flux_scale():
    """The kernel sum is the flux scaling from reference to science, so a
    reference multiplied by 0.63 and blurred must come back at 0.63."""
    reference, science = _kernel_case(scale=0.63)
    cfg = FakeConfig(**{"subtraction.kernel.scale_with_fwhm": False})
    mask = np.zeros(science.shape, dtype=bool)
    _model, kernel, info = sub.solve_kernel(science, reference, mask, 4.6, cfg)
    assert info["kernel_sum"] == pytest.approx(0.63, rel=0.03)
    assert kernel.sum() == pytest.approx(0.63, rel=0.03)


def test_solve_kernel_matches_the_science_frame():
    """What is left after subtracting the matched reference should be the noise
    that was put in, not a residual of the stars."""
    reference, science = _kernel_case(noise=1.0)
    cfg = FakeConfig(**{"subtraction.kernel.scale_with_fwhm": False})
    mask = np.zeros(science.shape, dtype=bool)
    model, _kernel, info = sub.solve_kernel(science, reference, mask, 4.6, cfg)
    inner = (slice(20, -20), slice(20, -20))
    residual = (science - model)[inner]
    assert np.std(residual) < 3.0            # against an input sigma of 1.0
    assert info["resid_rms"] < 3.0
    assert info["n_pixels"] > 10 * info["n_columns"]


def test_solve_kernel_finds_a_centred_kernel():
    """A kernel centroid far from zero means the solve tilted, which is what
    letting the transient into the fit used to do."""
    reference, science = _kernel_case()
    cfg = FakeConfig(**{"subtraction.kernel.scale_with_fwhm": False})
    mask = np.zeros(science.shape, dtype=bool)
    _model, _kernel, info = sub.solve_kernel(science, reference, mask, 4.6, cfg)
    assert abs(info["kernel_dx"]) < 0.25
    assert abs(info["kernel_dy"]) < 0.25


def test_solve_kernel_absorbs_a_differential_background():
    reference, science = _kernel_case(sky=0.0)
    yy, xx = np.mgrid[0:science.shape[0], 0:science.shape[1]].astype(float)
    science = science + 40.0 + 0.05 * xx - 0.03 * yy
    cfg = FakeConfig(**{"subtraction.kernel.scale_with_fwhm": False,
                        "subtraction.kernel.background_order": 1})
    mask = np.zeros(science.shape, dtype=bool)
    model, _kernel, info = sub.solve_kernel(science, reference, mask, 4.6, cfg)
    inner = (slice(20, -20), slice(20, -20))
    assert abs(np.median((science - model)[inner])) < 1.0
    assert info["kernel_sum"] == pytest.approx(0.63, rel=0.05)


def test_solve_kernel_excludes_the_target_pixels():
    """The transient is in the science frame and not in the reference, so a
    stamp on it asks the kernel to conjure its flux out of the host."""
    reference, science = _kernel_case()
    cfg = FakeConfig(**{"subtraction.kernel.scale_with_fwhm": False,
                        "subtraction.kernel.exclude_target_fwhm": 2.0})
    mask = np.zeros(science.shape, dtype=bool)
    _model, _kernel, info = sub.solve_kernel(science, reference, mask, 4.6, cfg,
                                             exclude_xy=(120.0, 120.0))
    assert info["target_excluded"] is True
    _model, _kernel, plain = sub.solve_kernel(science, reference, mask, 4.6, cfg)
    assert plain["target_excluded"] is False
    assert info["n_pixels"] <= plain["n_pixels"]


def test_solve_kernel_refuses_a_stamp_with_nothing_to_fit():
    reference, science = _kernel_case()
    cfg = FakeConfig(**{"subtraction.kernel.scale_with_fwhm": False})
    mask = np.ones(science.shape, dtype=bool)          # everything masked
    with pytest.raises(RuntimeError, match="too few pixels"):
        sub.solve_kernel(science, reference, mask, 4.6, cfg)


def test_gaussian_basis_is_normalised_and_mean_subtracted():
    """Only the first term carries flux, so sum(kernel) is set by its
    coefficient alone -- which is what makes the kernel sum meaningful."""
    basis = sub.gaussian_basis(9, [0.7, 1.5, 3.0], [4, 2, 0])
    assert len(basis) == 15 + 6 + 1
    assert basis[0].sum() == pytest.approx(1.0, abs=1e-6)
    for term in basis[1:]:
        assert abs(term.sum()) < 1e-9


def test_kernel_grows_with_a_coarse_psf():
    """The pixel sizes were tuned at a 5 px FWHM; on a wider PSF everything
    scales in proportion rather than trying to carry the seeing in 9 pixels."""
    reference, science = _kernel_case(ref_fwhm=6.0, sci_fwhm=11.0)
    mask = np.zeros(science.shape, dtype=bool)
    scaled = FakeConfig(**{"subtraction.kernel.scale_with_fwhm": True,
                           "subtraction.kernel.tuned_fwhm_px": 5.0})
    fixed = FakeConfig(**{"subtraction.kernel.scale_with_fwhm": False})
    _m, big, _i = sub.solve_kernel(science, reference, mask, 11.0, scaled)
    _m, small, _i = sub.solve_kernel(science, reference, mask, 11.0, fixed)
    assert big.shape[0] > small.shape[0]


# ---------------------------------------------------------------------------
# photometry
# ---------------------------------------------------------------------------
def _star_image(peak=5000.0, fwhm=4.0, n=120, sky=100.0):
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    sigma = fwhm / 2.3548200
    star = peak * np.exp(-((xx - n / 2) ** 2 + (yy - n / 2) ** 2) / (2 * sigma ** 2))
    return FakeImage(star + sky, sky=sky)


def test_measure_apertures_recovers_a_known_flux():
    """A Gaussian of peak A and width sigma holds 2*pi*A*sigma**2 in total; a
    1.2 FWHM aperture holds a well-defined fraction of it."""
    fwhm, peak = 4.0, 5000.0
    image = _star_image(peak=peak, fwhm=fwhm)
    sigma = fwhm / 2.3548200
    radius = 1.2 * fwhm
    expected = 2 * np.pi * peak * sigma ** 2 * (1 - np.exp(-radius ** 2 / (2 * sigma ** 2)))
    table = ph.measure_apertures(image, [(60.0, 60.0)], radius, FakeConfig(),
                                 annulus_px=(5 * fwhm, 8 * fwhm))
    assert float(table["flux"][0]) == pytest.approx(expected, rel=0.02)
    assert int(table["flags"][0]) == 0


def test_a_precomputed_annulus_gives_the_same_answer():
    """The curve of growth measures eleven radii on the same positions with the
    same annulus; it now pays for the sigma clipping once."""
    image = _star_image()
    annulus = (20.0, 32.0)
    positions = [(60.0, 60.0), (40.0, 45.0)]
    background = ph.annulus_background(image, positions, annulus)
    for radius in (3.0, 4.8, 9.0):
        fresh = ph.measure_apertures(image, positions, radius, FakeConfig(),
                                     annulus_px=annulus)
        reused = ph.measure_apertures(image, positions, radius, FakeConfig(),
                                      annulus_px=annulus, background=background)
        assert np.allclose(np.asarray(fresh["flux"]), np.asarray(reused["flux"]))
        assert np.allclose(np.asarray(fresh["flux_err"]), np.asarray(reused["flux_err"]))


def test_measure_apertures_flags_a_masked_pixel_and_an_off_frame_aperture():
    image = _star_image()
    image.mask[58:62, 58:62] = True
    table = ph.measure_apertures(image, [(60.0, 60.0), (2.0, 2.0)], 5.0, FakeConfig(),
                                 annulus_px=(20.0, 32.0))
    from tttphot.lightcurve import FLAG_MASKED_PIXEL, FLAG_OFF_IMAGE

    assert int(table["flags"][0]) & FLAG_MASKED_PIXEL
    assert int(table["flags"][1]) & FLAG_OFF_IMAGE


def test_measure_apertures_flags_saturation():
    image = _star_image(peak=90000.0)
    table = ph.measure_apertures(image, [(60.0, 60.0)], 6.0, FakeConfig(),
                                 annulus_px=(20.0, 32.0))
    from tttphot.lightcurve import FLAG_SATURATED

    assert int(table["flags"][0]) & FLAG_SATURATED


def test_local_bkg_global_uses_the_background_model():
    """photometry.local_bkg was documented and never read."""
    image = _star_image()
    annulus = (20.0, 32.0)
    local = ph.measure_apertures(image, [(60.0, 60.0)], 5.0, FakeConfig(), annulus_px=annulus)
    glob = ph.measure_apertures(image, [(60.0, 60.0)], 5.0,
                                FakeConfig(**{"photometry.local_bkg": "global"}),
                                annulus_px=annulus)
    assert float(local["local_bkg"][0]) != 0.0
    assert float(glob["local_bkg"][0]) == 0.0
    # Both remove the same sky, so the fluxes still agree.
    assert float(glob["flux"][0]) == pytest.approx(float(local["flux"][0]), rel=0.02)


def test_combine_errors_takes_the_larger_of_the_two_estimates():
    assert ph.combine_errors(100.0, 5.0, 12.0, 4.0) == pytest.approx(
        np.hypot(12.0, np.sqrt(100.0 / 4.0)))
    # A propagated error larger than the empirical one wins instead.
    assert ph.combine_errors(100.0, 50.0, 1.0, 4.0) == pytest.approx(50.0)
    # No empirical estimate: fall back on what was propagated.
    assert ph.combine_errors(100.0, 5.0, np.nan, 4.0) == pytest.approx(5.0)


def test_curve_of_growth_correction_is_negative_and_small():
    """The correction goes from a 1.2 FWHM aperture to 'total', so it makes a
    star brighter -- a negative number added to the magnitude."""
    from astropy.table import Table

    fwhm = 4.0
    n = 400
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    sigma = fwhm / 2.3548200
    data = np.full((n, n), 100.0)
    positions = [(80, 80), (200, 120), (300, 300), (120, 300), (250, 220), (330, 90)]
    for x0, y0 in positions:
        data += 4000.0 * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2))
    image = FakeImage(data, sky=100.0, rms=2.0)
    stars = Table({"xcentroid": [float(p[0]) for p in positions],
                   "ycentroid": [float(p[1]) for p in positions]})
    result = ph.curve_of_growth(image, stars, fwhm, FakeConfig())
    assert result.n_stars >= 5
    assert -0.6 < result.apcor_mag < -0.005
    # The growth curve is monotonic and reaches 1 at the 'total' radius.
    growth = result.growth[np.isfinite(result.growth)]
    assert np.all(np.diff(growth) >= -1e-6)


# ---------------------------------------------------------------------------
# diffquality
# ---------------------------------------------------------------------------
def _dipole_stamp(n=15, amplitude=50.0):
    stamp = np.zeros((n, n))
    c = n // 2
    stamp[c, c - 2] = amplitude
    stamp[c, c + 2] = -amplitude
    return stamp


def test_dipole_is_recognised():
    got, angle = diffquality._dipole(_dipole_stamp(), sigma=1.0, nsigma=5.0,
                                     antisymmetry=0.3)
    assert got is True
    assert np.isfinite(angle)


def test_a_one_sided_residual_is_not_a_dipole():
    """A flux mismatch leaves a blob of one sign; only a registration error
    leaves lobes of opposite sign on opposite sides."""
    stamp = np.zeros((15, 15))
    stamp[7, 5] = 50.0
    assert diffquality._dipole(stamp, 1.0, 5.0, 0.3)[0] is False


def test_a_lopsided_pair_is_not_a_dipole():
    stamp = _dipole_stamp()
    stamp[7, 9] = -1.0          # the negative lobe is 2 per cent of the positive
    assert diffquality._dipole(stamp, 1.0, 5.0, 0.3)[0] is False


def test_noise_below_the_threshold_is_not_a_dipole():
    assert diffquality._dipole(_dipole_stamp(amplitude=2.0), 1.0, 5.0, 0.3)[0] is False


def test_alignment_separates_a_shift_from_noise():
    """A shift displaces every source the same way, so real dipoles align;
    residuals thrown up by photon noise point wherever they like."""
    parallel = [0.3, 0.32, 0.28, 0.31, 0.29]
    assert diffquality._alignment(parallel) > 0.95
    scattered = np.linspace(-np.pi / 2, np.pi / 2, 12)
    assert diffquality._alignment(scattered) < 0.4
    # An axis has no head or tail, so opposite directions are the same axis.
    assert diffquality._alignment([0.1, 0.1 + np.pi, 0.1, 0.1 + np.pi]) > 0.95


def test_alignment_needs_a_few_angles():
    assert np.isnan(diffquality._alignment([0.1, 0.2]))


# ---------------------------------------------------------------------------
# cosmics  (review item H2)
# ---------------------------------------------------------------------------
def test_satlevel_is_passed_to_astroscrappy_in_electrons(monkeypatch):
    """detect_cosmics multiplies the frame by the gain and works in electrons
    throughout, and compares satlevel against that array. Passing the ADU level
    made the threshold a factor of gain too low -- 3.96 on TTT3 -- so every
    pixel above a quarter of full well was shielded from cleaning.
    """
    seen = {}

    def fake_detect_cosmics(data, **kwargs):
        seen.update(kwargs)
        return np.zeros(data.shape, dtype=bool), np.asarray(data, dtype=np.float32)

    stub = types.ModuleType("astroscrappy")
    stub.detect_cosmics = fake_detect_cosmics
    monkeypatch.setitem(sys.modules, "astroscrappy", stub)

    data = np.full((80, 80), 100.0)
    cosmics.remove(data, FakeConfig(), gain=3.96, readnoise=14.2,
                   saturation=240000.0, fwhm_px=4.0)
    assert seen["gain"] == pytest.approx(3.96)
    assert seen["satlevel"] == pytest.approx(240000.0 * 3.96)


def test_cosmic_rejection_keeps_the_frame_when_it_misfires(monkeypatch):
    """Flagging this much of a frame is not a cosmic-ray rate, it is undersampled
    stars being shredded."""
    def fake_detect_cosmics(data, **kwargs):
        return np.ones(data.shape, dtype=bool), np.zeros(data.shape, dtype=np.float32)

    stub = types.ModuleType("astroscrappy")
    stub.detect_cosmics = fake_detect_cosmics
    monkeypatch.setitem(sys.modules, "astroscrappy", stub)

    data = np.full((60, 60), 100.0)
    out, crmask, info = cosmics.remove(data, FakeConfig(), gain=2.0, readnoise=5.0,
                                       saturation=60000.0, fwhm_px=4.0)
    assert info["applied"] is False
    assert not crmask.any()
    assert np.array_equal(out, data)


def test_cosmic_rejection_is_a_no_op_without_astroscrappy(monkeypatch):
    """A missing cosmic ray costs one bad pixel; a broken frame costs the epoch."""
    import builtins

    real_import = builtins.__import__

    def no_astroscrappy(name, *args, **kwargs):
        if name == "astroscrappy":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "astroscrappy", raising=False)
    monkeypatch.setattr(builtins, "__import__", no_astroscrappy)
    data = np.full((40, 40), 5.0)
    out, crmask, info = cosmics.remove(data, FakeConfig(), gain=2.0, readnoise=5.0,
                                       saturation=60000.0, fwhm_px=4.0)
    assert info["applied"] is False
    assert np.array_equal(out, data)
    assert not crmask.any()


def test_protect_mask_shields_stars_but_not_lone_spikes():
    data = np.full((60, 60), 10.0)
    data[30:34, 30:34] = 5000.0       # a star: sixteen connected bright pixels
    data[10, 10] = 5000.0             # a spike: one pixel on its own
    keep = cosmics.protect_mask(data, saturation=60000.0, nsigma=5.0, min_star_pixels=4)
    assert keep[31, 31]
    assert not keep[10, 10]


# ---------------------------------------------------------------------------
# catalogue caching  (review items H1, C)
# ---------------------------------------------------------------------------
def test_no_cache_key_is_built_from_the_builtin_hash():
    """Python randomises hash() of a str per process (PEP 456).

    The ttt_stack template cache key was built that way, so it changed on every
    run: the cache could never be read back and each run left another
    multi-megabyte stack behind. Nothing that names a file may use it.
    """
    offenders = []
    for path in sorted(Path(tttphot.__file__).parent.glob("*.py")):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            if re.search(r"\bhash\(", code) and "hashlib" not in code:
                offenders.append(f"{path.name}:{number}: {line.strip()}")
    assert not offenders, "use hashlib for anything that keys a cache:\n" + "\n".join(offenders)


def test_cache_path_is_stable_across_calls(tmp_path):
    first = catalogs._cache_path(tmp_path, "gaia_synth", 351.9271, 23.5881, 7.0)
    second = catalogs._cache_path(tmp_path, "gaia_synth", 351.9289, 23.5893, 7.0)
    assert first == second        # snapped to the cache grid, so dithers share one file


def test_a_non_default_option_gets_its_own_cache_entry(tmp_path):
    """Raising astrometry.max_mag must fetch a deeper catalogue, not silently
    reuse the shallow cached one."""
    plain = catalogs._cache_path(tmp_path, "gaia_astrometry", 351.925, 23.588, 7.0)
    deeper = catalogs._cache_path(tmp_path, "gaia_astrometry", 351.925, 23.588, 7.0,
                                  variant="max_mag21.0")
    assert plain != deeper
    assert "max_mag21.0" in deeper.name


def test_only_a_changed_option_changes_the_cache_key():
    """Leaving max_mag at its default must still hit the caches on disk."""
    kept = catalogs._non_default_options(catalogs.fetch_gaia_astrometry, {"max_mag": 20.0})
    assert kept == {}
    changed = catalogs._non_default_options(catalogs.fetch_gaia_astrometry, {"max_mag": 21.0})
    assert changed == {"max_mag": 21.0}
    assert catalogs._non_default_options(catalogs.fetch_gaia_astrometry,
                                         {"max_mag": None}) == {}
