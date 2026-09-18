"""Unit tests for the pure parts of tttphot.

Nothing here needs a FITS file, a network connection or a telescope. Everything
that can be checked from first principles is checked here, so that a refactor
can be judged in a second rather than by a two-minute end-to-end run against
real data.

    pytest tests/ -q

A handful of these exist because the bug they describe was actually present,
and each of those says so.
"""

from __future__ import annotations

from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.table import Table

from tttphot import detector, lightcurve as lc, utils, zeropoints
from tttphot.catalogs import clean_catalog
from tttphot.calibration import _sigma_clipped_fit
from tttphot.config import Config
from tttphot.image import effective_gain


class FakeConfig:
    """The dotted-key interface, for the functions that only need that."""

    def __init__(self, **values):
        self._values = values

    def get(self, dotted, default=None):
        return self._values.get(dotted, default)


# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,expected", [
    ("SN2026abgb", "2026abgb"),
    ("SN 2026abgb", "2026abgb"),
    ("AT2026abgb", "2026abgb"),
    ("AT 2026abgb", "2026abgb"),
    ("2026abgb", "2026abgb"),
    ("sn_2026abgb", "2026abgb"),
    ("SN-2026abgb", "2026abgb"),
    ("TDE2026abgb", "2026abgb"),
    ("NGC 7673", "ngc7673"),
    ("", ""),
    (None, ""),
])
def test_normalise_name(name, expected):
    assert utils.normalise_name(name) == expected


def test_normalise_name_does_not_strip_a_non_designation():
    # 'snake' starts with 'sn' but is not <prefix><year><letters>.
    assert utils.normalise_name("snake") == "snake"


def test_band_token_separates_case_on_a_case_insensitive_filesystem():
    # The whole point: Johnson I and SDSS i must not collide in a filename.
    assert utils.band_token("I") != utils.band_token("i")
    assert utils.band_token("R") != utils.band_token("r")
    assert utils.band_token("Halpha") != utils.band_token("HALPHA")


def test_find_by_designation_matches_across_prefixes(tmp_path):
    (tmp_path / "AT2026abwp.yaml").write_text("target: {}\n")
    assert utils.find_by_designation(tmp_path, "SN2026abwp").name == "AT2026abwp.yaml"
    assert utils.find_by_designation(tmp_path, "2026abwp").name == "AT2026abwp.yaml"
    assert utils.find_by_designation(tmp_path, "SN2026zzzz") is None


def test_find_by_designation_prefers_the_exact_name(tmp_path):
    (tmp_path / "AT2026abwp.yaml").write_text("target: {}\n")
    (tmp_path / "SN2026abwp.yaml").write_text("target: {}\n")
    assert utils.find_by_designation(tmp_path, "SN2026abwp").name == "SN2026abwp.yaml"


def test_weighted_mean_weights_by_inverse_variance():
    mean, err = utils.weighted_mean([10.0, 10.2], [0.1, 0.2])
    # The tighter measurement dominates: w = 100 and 25.
    assert mean == pytest.approx((100 * 10.0 + 25 * 10.2) / 125)
    assert err == pytest.approx(np.sqrt(1 / 125))


def test_weighted_mean_ignores_bad_entries():
    mean, err = utils.weighted_mean([10.0, np.nan, 5.0], [0.1, 0.1, 0.0])
    assert mean == pytest.approx(10.0)
    assert np.isfinite(err)
    assert all(np.isnan(v) for v in utils.weighted_mean([np.nan], [np.nan]))


def test_flux_to_mag_round_trip_and_zero_point():
    mag, err = utils.flux_to_mag(100.0, 1.0, zp=25.0)
    assert mag == pytest.approx(25.0 - 5.0)
    assert err == pytest.approx(1.0857362 * 0.01)
    # A 1 per cent flux error is 0.01086 mag, the standard conversion.
    assert np.isnan(utils.flux_to_mag(-5.0, 1.0, zp=25.0)[0])


def test_flux_to_mag_folds_the_zero_point_error_in_quadrature():
    _, err = utils.flux_to_mag(100.0, 1.0, zp=25.0, zp_err=0.03)
    assert err == pytest.approx(np.hypot(1.0857362 * 0.01, 0.03))


def test_header_mjd_prefers_mjd_obs_then_jd_then_date_obs():
    """Two TTT3 frames of 2026-09-14 carry only DATE-OBS, and came through with
    a NaN MJD: they sorted to the end of the campaign, their observing night was
    meaningless, and the epoch reached the light curve with no date on it. The
    instrument files already mapped the card as `dateobs`; nothing read it."""
    hmap = {"mjd": ["MJD-OBS"], "jd": ["JD-OBS", "JD"], "dateobs": ["DATE-OBS"]}
    assert utils.header_mjd({"MJD-OBS": 61297.5}, hmap) == pytest.approx(61297.5)
    assert utils.header_mjd({"JD-OBS": 2461298.0}, hmap) == pytest.approx(61297.5)
    only_date = {"DATE-OBS": "2026-09-14T03:27:11.235346"}
    assert utils.header_mjd(only_date, hmap) == pytest.approx(61297.1438800, abs=1e-5)
    assert not np.isfinite(utils.header_mjd({"EXPTIME": 120.0}, hmap))


def test_header_mjd_survives_an_unreadable_date():
    assert not np.isfinite(utils.header_mjd({"DATE-OBS": "not a date"}, {}))


def test_robust_sigma_is_unmoved_by_an_outlier():
    values = np.concatenate([np.full(99, 5.0), [1e6]])
    assert utils.robust_sigma(values) == pytest.approx(0.0)
    assert utils.robust_sigma(np.array([])) != utils.robust_sigma(np.array([]))  # NaN


def test_odd_rounds_up_to_an_odd_number_of_at_least_three():
    assert [utils.odd(n) for n in (0, 1, 2, 4, 5, 6.4)] == [3, 3, 3, 5, 5, 7]


# ---------------------------------------------------------------------------
# utils.edge_margin_px  (review item H4)
# ---------------------------------------------------------------------------
def test_edge_margin_follows_the_pixel_scale():
    """9 arcsec is a different number of pixels at each binning.

    detection, photometry and pipeline each carried their own hard-coded 24 px
    and ignored image.edge_margin_arcsec entirely. That agreed with the
    configured 9" only at the default rebinning; with image.rebin: 1 the frame
    was masked 47 px deep while three other tests still used 24.
    """
    cfg = FakeConfig(**{"image.edge_margin_arcsec": 9.0})
    assert utils.edge_margin_px(cfg, 0.385) == 23     # rebinned 2x2
    assert utils.edge_margin_px(cfg, 0.1925) == 47    # image.rebin: 1


def test_edge_margin_px_overrides_the_arcsec_form():
    cfg = FakeConfig(**{"image.edge_margin_px": 30})
    assert utils.edge_margin_px(cfg, 0.385) == 30


def test_edge_margin_is_zero_when_nothing_is_configured():
    assert utils.edge_margin_px(FakeConfig(), 0.385) == 0


# ---------------------------------------------------------------------------
# image.effective_gain
# ---------------------------------------------------------------------------
def _hmap():
    return {"gain": ["GAIN"], "readnoise": ["RDNOISE"], "n_subframes": ["TOTIMA"]}


def test_effective_gain_leaves_an_unbinned_header_alone():
    header = {"GAIN": 1.5, "RDNOISE": 6.0}
    gain, rn, _ = effective_gain(header, _hmap(), {}, (1, 1), 1)
    assert (gain, rn) == pytest.approx((1.5, 6.0))


def test_effective_gain_for_averaged_camera_binning():
    """TTT3 bins 3x3 and averages, and quotes the unbinned sensor.

    An average of n pixels carries n times the electrons per ADU, and its read
    noise in those electrons is sqrt(n) times the native value: 0.44 -> 3.96
    e-/ADU and 4.74 -> 14.2 e-, which is what the README measured from the sky.
    """
    header = {"GAIN": 0.44, "RDNOISE": 4.74}
    defaults = {"gain_is_unbinned": True, "binning_mode": "average"}
    gain, rn, info = effective_gain(header, _hmap(), defaults, (3, 3), 1)
    assert gain == pytest.approx(0.44 * 9)
    assert rn == pytest.approx(4.74 * 3)
    assert info["gain_multiplier"] == 9


def test_effective_gain_for_summed_camera_binning():
    """Summing leaves e-/ADU alone and grows the read noise by sqrt(n)."""
    header = {"GAIN": 0.44, "RDNOISE": 4.74}
    defaults = {"gain_is_unbinned": True, "binning_mode": "sum"}
    gain, rn, _ = effective_gain(header, _hmap(), defaults, (3, 3), 1)
    assert gain == pytest.approx(0.44)
    assert rn == pytest.approx(4.74 * 3)


def test_software_rebinning_grows_the_read_noise_by_the_factor():
    """Our own 2x2 rebinning sums, so four read noises add in quadrature."""
    header = {"GAIN": 2.0, "RDNOISE": 5.0}
    gain, rn, _ = effective_gain(header, _hmap(), {}, (1, 1), 2)
    assert gain == pytest.approx(2.0)
    assert rn == pytest.approx(5.0 * 2)


def test_effective_gain_counts_averaged_sub_exposures():
    header = {"GAIN": 1.0, "RDNOISE": 4.0, "TOTIMA": 4}
    gain, rn, info = effective_gain(header, _hmap(), {"stack_mode": "average"}, (1, 1), 1)
    assert gain == pytest.approx(4.0)
    assert rn == pytest.approx(4.0 * 2)
    assert info["n_subframes"] == 4


# ---------------------------------------------------------------------------
# detector: rebinning
# ---------------------------------------------------------------------------
def test_rebin_array_sums_and_keeps_the_total():
    data = np.arange(16, dtype=float).reshape(4, 4)
    out = detector._rebin_array(data, 2)
    assert out.shape == (2, 2)
    assert out.sum() == pytest.approx(data.sum())
    assert out[0, 0] == pytest.approx(0 + 1 + 4 + 5)


def test_rebin_array_drops_the_remainder_rows():
    assert detector._rebin_array(np.ones((5, 7)), 2).shape == (2, 3)


def test_rebin_array_scales_a_partly_masked_block_up():
    block = np.array([[1.0, 1.0], [1.0, np.nan]])
    # Three finite pixels summing to 3, scaled to the four-pixel equivalent.
    assert detector._rebin_array(block, 2)[0, 0] == pytest.approx(4.0)


def test_rebin_array_blanks_a_mostly_masked_block():
    block = np.array([[1.0, np.nan], [np.nan, np.nan]])
    assert np.isnan(detector._rebin_array(block, 2)[0, 0])


def test_rebin_array_keeps_a_saturated_block_saturated():
    """A saturated core diluted by its neighbours must still trip the mask.

    The saturation threshold downstream is scaled by factor**2, so a block
    containing a saturated pixel is promoted to at least that level.
    """
    block = np.array([[60000.0, 10.0], [10.0, 10.0]])
    out = detector._rebin_array(block, 2, saturation=60000.0)
    assert out[0, 0] >= 60000.0 * 4


def test_rebin_header_rescales_the_wcs_exactly():
    """The README claims the WCS survives rebinning to 1e-10 arcsec."""
    pytest.importorskip("astropy.wcs")
    from astropy.io import fits
    from astropy.wcs import WCS

    header = fits.Header({
        "NAXIS": 2, "NAXIS1": 400, "NAXIS2": 400,
        "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
        "CRPIX1": 200.5, "CRPIX2": 200.5, "CRVAL1": 351.925, "CRVAL2": 23.588,
        "CD1_1": -5.3e-5, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": 5.3e-5,
    })
    rebinned = detector._rebin_header(header, 2, (200, 200))
    before, after = WCS(header), WCS(rebinned)
    # The same patch of sky: pixel p in the rebinned frame is 2p + 0.5 before.
    for x, y in [(0, 0), (50, 120), (199, 199)]:
        a = before.pixel_to_world(2 * x + 0.5, 2 * y + 0.5)
        b = after.pixel_to_world(x, y)
        assert a.separation(b).arcsec < 1e-9


def test_rebin_factor_is_one_for_an_already_rebinned_frame():
    assert detector.rebin_factor({"TTTREBIN": 2}, {"defaults": {"rebin": "auto"}}) == 1


def test_rebin_factor_auto_targets_the_configured_sampling():
    header = {"CD1_1": -0.1925 / 3600, "CD1_2": 0, "CD2_1": 0, "CD2_2": 0.1925 / 3600}
    instrument = {"defaults": {"rebin": "auto", "rebin_target_arcsec": 0.385}}
    assert detector.rebin_factor(header, instrument) == 2


# ---------------------------------------------------------------------------
# detector: sizes
# ---------------------------------------------------------------------------
def test_resolve_size_passes_an_explicit_number_through():
    assert detector.resolve_size(7.5, frame_arcmin=10.0) == pytest.approx(7.5)


def test_resolve_size_auto_scales_with_the_frame_and_snaps():
    # 0.75 x 10.2' = 7.65', floored to a whole arcmin so the cache key is stable.
    assert detector.resolve_size("auto", 10.2, fraction=0.75, rounding="down") == 7.0


def test_resolve_size_clamps():
    assert detector.resolve_size("auto", 100.0, fraction=1.0, hi=12.0) == 12.0
    assert detector.resolve_size("auto", 1.0, fraction=1.0, lo=4.0) == 4.0


def test_resolve_size_falls_back_on_a_typo_instead_of_raising(caplog):
    """A mistyped size setting used to come back as a ValueError traceback."""
    assert detector.resolve_size("atuo", 10.0, default=9.0) == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# calibration: the zero-point fit
# ---------------------------------------------------------------------------
def test_sigma_clipped_fit_recovers_a_known_zero_point():
    rng = np.random.default_rng(0)
    colour = rng.uniform(-0.3, 0.9, 60)
    delta = 24.5 + 0.08 * colour + rng.normal(0, 0.01, 60)
    weights = np.full(60, 1 / 0.01 ** 2)
    zp, slope, zp_err, _, rms, keep = _sigma_clipped_fit(colour, delta, weights, True)
    assert zp == pytest.approx(24.5, abs=0.01)
    assert slope == pytest.approx(0.08, abs=0.02)
    assert rms == pytest.approx(0.01, abs=0.005)
    assert keep.sum() > 50


def test_sigma_clipped_fit_rejects_a_planted_outlier():
    rng = np.random.default_rng(1)
    colour = rng.uniform(-0.3, 0.9, 40)
    delta = 24.5 + rng.normal(0, 0.01, 40)
    delta[7] += 3.0                       # one star that is not a star
    weights = np.full(40, 1 / 0.01 ** 2)
    zp, _, _, _, _, keep = _sigma_clipped_fit(colour, delta, weights, False)
    assert not keep[7]
    assert zp == pytest.approx(24.5, abs=0.02)


def test_sigma_clipped_fit_gives_up_gracefully_on_too_few_points():
    zp, slope, *_ = _sigma_clipped_fit(np.array([0.0]), np.array([1.0]),
                                       np.array([1.0]), True)
    assert np.isnan(zp)


# ---------------------------------------------------------------------------
# catalogs.clean_catalog  (review items H6, H7)
# ---------------------------------------------------------------------------
def _catalog(**columns):
    table = Table(columns)
    table.meta["provider"] = columns.pop("_provider", "gaia_synth")
    return table


def test_clean_catalog_applies_the_magnitude_range_and_error_cut():
    table = _catalog(ra=[1.0, 2.0, 3.0, 4.0], dec=[0.0] * 4,
                     g=[9.0, 15.0, 15.0, 25.0], e_g=[0.01, 0.01, 0.5, 0.01])
    cfg = FakeConfig(**{"calibration.mag_range": [11.0, 19.0],
                        "calibration.max_cat_err": 0.05})
    kept = clean_catalog(table, "g", cfg)
    assert list(kept["ra"]) == [2.0]      # too bright, too noisy and too faint all go


def test_clean_catalog_keeps_entries_with_no_quoted_error_by_default():
    """APASS through VizieR quotes no error for some rows; dropping them empties
    the catalogue. The precedence here is easy to misread -- `~a | b` is
    `(~a) | b` -- so the behaviour is pinned down rather than left to the eye."""
    table = _catalog(ra=[1.0, 2.0], dec=[0.0, 0.0], g=[15.0, 15.0],
                     e_g=[np.nan, 0.5])
    cfg = FakeConfig(**{"calibration.mag_range": [11.0, 19.0],
                        "calibration.max_cat_err": 0.05})
    assert list(clean_catalog(table, "g", cfg)["ra"]) == [1.0]


def test_clean_catalog_can_be_told_to_require_an_error():
    table = _catalog(ra=[1.0, 2.0], dec=[0.0, 0.0], g=[15.0, 15.0],
                     e_g=[np.nan, 0.01])
    cfg = FakeConfig(**{"calibration.mag_range": [11.0, 19.0],
                        "calibration.max_cat_err": 0.05,
                        "calibration.require_cat_err": True})
    assert list(clean_catalog(table, "g", cfg)["ra"]) == [2.0]


def test_clean_catalog_keeps_an_unclassifiable_pan_starrs_source():
    """A PS1 source with neither an r nor an i Kron magnitude scored
    point_source = False, because abs(nan) < 0.10 is False, and was dropped in
    every band -- g and z included, where its own photometry was fine."""
    table = _catalog(ra=[1.0, 2.0, 3.0], dec=[0.0] * 3, g=[15.0] * 3,
                     e_g=[0.01] * 3, point_source=[True, False, True])
    cfg = FakeConfig(**{"calibration.mag_range": [11.0, 19.0],
                        "calibration.max_cat_err": 0.05})
    assert list(clean_catalog(table, "g", cfg)["ra"]) == [1.0, 3.0]


def test_clean_catalog_can_cut_on_the_gaia_quality_flag():
    table = _catalog(ra=[1.0, 2.0], dec=[0.0, 0.0], g=[15.0, 15.0],
                     e_g=[0.01, 0.01], f_g=[0, 3])
    cfg = FakeConfig(**{"calibration.mag_range": [11.0, 19.0],
                        "calibration.max_cat_err": 0.05,
                        "calibration.max_gaia_synth_flag": 0})
    assert list(clean_catalog(table, "g", cfg)["ra"]) == [1.0]
    # ...and leaves the flags alone when the key is unset.
    cfg = FakeConfig(**{"calibration.mag_range": [11.0, 19.0],
                        "calibration.max_cat_err": 0.05})
    assert len(clean_catalog(table, "g", cfg)) == 2


def test_clean_catalog_returns_nothing_for_a_band_the_provider_lacks():
    table = _catalog(ra=[1.0], dec=[0.0], g=[15.0], e_g=[0.01])
    assert len(clean_catalog(table, "z", FakeConfig())) == 0


# ---------------------------------------------------------------------------
# zeropoints
# ---------------------------------------------------------------------------
def test_observing_night_keeps_one_night_together():
    """Teide: an evening and the small hours after it are one night.

    MJD .0 is midnight UT, so the boundary is put at .3 (07:12 UT) -- well after
    dawn and well before dusk.
    """
    evening, small_hours = 60900.85, 60901.15
    assert zeropoints.observing_night([evening])[0] == zeropoints.observing_night([small_hours])[0]
    assert zeropoints.observing_night([60901.85])[0] != zeropoints.observing_night([evening])[0]


def test_band_offsets_measures_a_same_night_difference():
    frame = pd.DataFrame({
        "mjd": [60900.9, 60900.9, 60901.9, 60901.9, 60902.9, 60902.9],
        "band": ["g", "r"] * 3,
        "zp": [24.0, 24.5, 23.2, 23.7, 24.4, 24.9],
    })
    offsets = zeropoints.band_offsets(frame, min_nights=3)
    assert offsets[("g", "r")][0] == pytest.approx(-0.5)
    assert offsets[("r", "g")][0] == pytest.approx(0.5)
    assert offsets[("g", "r")][2] == 3


def test_band_offsets_needs_enough_shared_nights():
    frame = pd.DataFrame({"mjd": [60900.9, 60900.9], "band": ["g", "r"], "zp": [24.0, 24.5]})
    assert zeropoints.band_offsets(frame, min_nights=3) == {}


def test_combine_widens_the_error_when_partners_disagree():
    """Partners that agree mean the night's extinction was grey; partners that
    disagree are the best available evidence that it was not."""
    agree = zeropoints._combine([{"band": "r", "zp": 24.0, "scatter": 0.05, "n": 5},
                                 {"band": "i", "zp": 24.01, "scatter": 0.05, "n": 5}])
    disagree = zeropoints._combine([{"band": "r", "zp": 24.0, "scatter": 0.05, "n": 5},
                                    {"band": "i", "zp": 24.4, "scatter": 0.05, "n": 5}])
    assert disagree["err"] > 3 * agree["err"]
    assert disagree["n_partners"] == 2


def test_combine_returns_nothing_without_a_partner():
    assert zeropoints._combine([]) is None


def test_camera_of_prefers_the_camera_column():
    frame = pd.DataFrame({"camera": ["QHY600-TTT3", "iKon936-TTT3"],
                          "detector": ["nonsense 1x1", "nonsense 1x1"]})
    assert list(zeropoints.camera_of(frame)) == ["QHY600-TTT3", "iKon936-TTT3"]


def test_camera_of_falls_back_to_the_detector_label_for_an_older_table():
    frame = pd.DataFrame({"detector": ["QHY600-TTT3 1595x1064", "iKon936-TTT3 1024x1024"]})
    assert list(zeropoints.camera_of(frame)) == ["QHY600-TTT3", "iKon936-TTT3"]


def test_transfer_recovers_a_missing_zero_point_from_the_same_night():
    nights = [60900.9, 60901.9, 60902.9, 60903.9]
    rows = []
    for night in nights:
        rows.append({"image": f"g_{night}", "mjd": night, "band": "g", "method": "aperture",
                     "zp": np.nan if night == nights[-1] else 24.0, "zp_err": 0.02,
                     "flux": 100.0, "flux_err": 2.0, "flags": 0, "mag_offset": -0.05,
                     "exptime": 120.0, "camera": "QHY600-TTT3"})
        rows.append({"image": f"r_{night}", "mjd": night, "band": "r", "method": "aperture",
                     "zp": 24.5, "zp_err": 0.02, "flux": 100.0, "flux_err": 2.0,
                     "flags": 0, "mag_offset": -0.05, "exptime": 120.0,
                     "camera": "QHY600-TTT3"})
    frame = pd.DataFrame(rows)

    out, report = zeropoints.transfer(frame, cfg=None, min_nights=3)
    recovered = out[out["image"] == f"g_{nights[-1]}"].iloc[0]
    assert recovered["zp"] == pytest.approx(24.0, abs=1e-6)     # 24.5 + (24.0 - 24.5)
    assert int(recovered["flags"]) & lc.FLAG_ZP_TRANSFERRED
    assert not int(recovered["flags"]) & lc.FLAG_NO_ZEROPOINT
    assert recovered["mag"] == pytest.approx(-2.5 * np.log10(100.0) - 0.05 + 24.0)
    assert len(report["recovered"]) == 1


def test_transfer_keeps_the_cameras_apart():
    """The iKon936 zero points sit 1.4 mag above the QHY600 ones in the same
    band on the same night; one per-night median across both is noise."""
    nights = [60900.9, 60901.9, 60902.9, 60903.9]
    rows = []
    for night in nights:
        for camera, base in (("QHY600-TTT3", 24.0), ("iKon936-TTT3", 25.4)):
            missing = camera == "QHY600-TTT3" and night == nights[-1]
            rows.append({"image": f"g_{camera}_{night}", "mjd": night, "band": "g",
                         "method": "aperture", "zp": np.nan if missing else base,
                         "zp_err": 0.02, "flux": 100.0, "flux_err": 2.0, "flags": 0,
                         "mag_offset": 0.0, "exptime": 120.0, "camera": camera})
            rows.append({"image": f"r_{camera}_{night}", "mjd": night, "band": "r",
                         "method": "aperture", "zp": base + 0.5, "zp_err": 0.02,
                         "flux": 100.0, "flux_err": 2.0, "flags": 0, "mag_offset": 0.0,
                         "exptime": 120.0, "camera": camera})
    out, _ = zeropoints.transfer(pd.DataFrame(rows), cfg=None, min_nights=3)
    recovered = out[out["image"] == f"g_QHY600-TTT3_{nights[-1]}"].iloc[0]
    # The QHY600 answer, not something averaged with the iKon936's 25.4.
    assert recovered["zp"] == pytest.approx(24.0, abs=1e-6)


# ---------------------------------------------------------------------------
# the kernel-sum spread report  (SN2026abwp, 2026-09-18)
# ---------------------------------------------------------------------------
def _kernel_frame(kernel_sums, zps, band="r"):
    n = len(kernel_sums)
    return pd.DataFrame({
        "image": [f"{band}{i}.fits" for i in range(n)],
        "method": ["difference_psf"] * n, "band": [band] * n,
        "diff_kernel_sum": list(kernel_sums), "zp": list(zps),
        "exptime": [120.0] * n, "pixel_scale": [0.385] * n,
        "diff_mode": ["convolve_reference"] * n, "ref_source": ["panstarrs"] * n,
        "detector": ["QHY600-TTT3 1595x1064"] * n, "flags": [0] * n,
        "mjd": [61300.0 + 0.01 * i for i in range(n)],
    })


def _warnings_from(frame, caplog):
    import logging
    from tttphot import pipeline as pl
    from tttphot.config import Config, DEFAULT_CONFIG, load_yaml

    cfg = Config(load_yaml(DEFAULT_CONFIG))
    with caplog.at_level(logging.WARNING, logger="tttphot.pipeline"):
        pl._flag_collapsed_kernels(frame.copy(), cfg)
    return [r for r in caplog.records if "scatters by" in r.getMessage()]


def test_kernel_sum_spread_is_reported_when_the_flux_scale_wanders(caplog):
    """The real SN2026abwp r band, as it was before the stamp fix.

    Four frames whose normalised kernel sums scatter by 8.6 per cent. Two of
    them, nine minutes apart, came out 0.21 mag (5.4 sigma) apart because the
    host inside the aperture is 2.6-4.5 times the transient, so the kernel's
    flux scale is multiplied by that before it reaches the magnitude. The
    factor-of-three tolerance cannot see a wobble this size, and the group was
    too small for the consistency check to run at all.
    """
    frame = _kernel_frame([0.1128, 0.1174, 0.1078, 0.1205],
                          [23.489, 23.544, 23.414, 23.424])
    assert _warnings_from(frame, caplog), "the 8.6% case must be reported"


def test_kernel_sum_spread_is_quiet_on_healthy_data(caplog):
    """The same band once the kernel had its full complement of stamps."""
    frame = _kernel_frame([0.1127, 0.1227, 0.1058, 0.1084],
                          [23.489, 23.544, 23.414, 23.424])
    assert not _warnings_from(frame, caplog), "healthy data must not warn"


def test_kernel_sum_spread_does_not_grow_with_the_number_of_frames(caplog):
    """Peak-to-peak would: it widens with n on its own, so a long healthy
    campaign read 5-8 per cent of it and would have warned for nothing. The
    16-84 percentile range is the same quantity whatever n."""
    rng = np.random.default_rng(0)
    for n in (4, 12, 28, 60):
        sums = 0.112 * (1 + rng.normal(0, 0.012, n))      # 1.2 per cent, healthy
        frame = _kernel_frame(sums, np.full(n, 23.45))
        assert not _warnings_from(frame, caplog), f"warned on {n} healthy frames"


# ---------------------------------------------------------------------------
# lightcurve
# ---------------------------------------------------------------------------
def _row(method, mag=18.0, flags=0, **kw):
    row = {"method": method, "mag": mag, "is_limit": False, "diff_chi2": 1.0,
           "flags": flags}
    row.update(kw)
    return row


def test_choose_primary_follows_the_method_priority():
    rows = [_row("aperture"), _row("psf"), _row("difference_aperture"),
            _row("difference_psf")]
    assert lc.choose_primary(rows)["method"] == "difference_psf"


def test_choose_primary_avoids_a_hard_flag_when_it_can():
    rows = [_row("difference_psf", flags=lc.FLAG_BAD_SUBTRACTION), _row("psf")]
    assert lc.choose_primary(rows)["method"] == "psf"


def test_choose_primary_uses_a_flagged_row_when_there_is_nothing_else():
    rows = [_row("difference_psf", flags=lc.FLAG_BAD_SUBTRACTION)]
    assert lc.choose_primary(rows)["method"] == "difference_psf"


def test_a_clipped_aperture_loses_to_a_clean_measurement():
    """aperture_photometry drops masked pixels from the sum and nothing puts
    their flux back, so those rows are biased faint. They used to rank purely on
    method and could beat a clean row of lower priority."""
    rows = [_row("difference_psf", flags=lc.FLAG_MASKED_PIXEL),
            _row("difference_aperture")]
    assert lc.choose_primary(rows)["method"] == "difference_aperture"
    assert lc._suspect(_row("aperture", flags=lc.FLAG_OFF_IMAGE))


def test_host_guard_keeps_a_flagged_difference_over_bright_direct_photometry():
    """On SN 2026abgb this is the difference between 18.5 and 16.9 mag for the
    same g epoch: the direct measurement is mostly galaxy."""
    rows = [_row("difference_psf", mag=18.5, flags=lc.FLAG_DIPOLE_RESIDUAL
                 | lc.FLAG_BAD_SUBTRACTION),
            _row("psf", mag=16.9)]
    assert lc.choose_primary(rows, host_guard_mag=0.2)["method"] == "difference_psf"


def test_host_guard_does_not_fire_when_the_host_is_faint():
    rows = [_row("difference_psf", mag=18.55, flags=lc.FLAG_DIPOLE_RESIDUAL
                 | lc.FLAG_BAD_SUBTRACTION),
            _row("psf", mag=18.5)]
    assert lc.choose_primary(rows, host_guard_mag=0.2)["method"] == "psf"


def test_host_guard_refuses_a_difference_whose_kernel_collapsed():
    rows = [_row("difference_psf", mag=18.5, flags=lc.FLAG_KERNEL_SUM
                 | lc.FLAG_BAD_SUBTRACTION),
            _row("psf", mag=16.9)]
    assert lc.choose_primary(rows, host_guard_mag=0.2)["method"] == "psf"


def test_host_guard_can_be_switched_off():
    rows = [_row("difference_psf", mag=18.5, flags=lc.FLAG_DIPOLE_RESIDUAL
                 | lc.FLAG_BAD_SUBTRACTION),
            _row("psf", mag=16.9)]
    assert lc.choose_primary(rows, host_guard_mag=None)["method"] == "psf"


def test_describe_flags_names_every_bit():
    value = lc.FLAG_SATURATED | lc.FLAG_ZP_TRANSFERRED
    assert set(lc.describe_flags(value).split("|")) == {"saturated", "zp_transferred"}
    assert lc.describe_flags(0) == ""
    assert lc.describe_flags(np.nan) == ""


def test_every_flag_has_a_name():
    bits = {v for k, v in vars(lc).items()
            if k.startswith("FLAG_") and isinstance(v, int)}
    assert bits == set(lc.FLAG_NAMES)
    # ...and every bit is distinct, so two flags cannot alias each other.
    assert len(bits) == len(lc.FLAG_NAMES)


def test_to_frame_warns_about_a_column_it_is_about_to_drop(caplog):
    """A new diagnostic added upstream and forgotten in COLUMNS used to vanish
    from the output without a word."""
    import logging
    with caplog.at_level(logging.WARNING, logger="tttphot.lightcurve"):
        frame = lc.to_frame([lc.new_row(mjd=60900.0, band="g", method="aperture",
                                        some_new_diagnostic=1.0)])
    assert "some_new_diagnostic" in caplog.text
    assert list(frame.columns) == lc.COLUMNS


def test_merge_lets_the_newer_measurement_win():
    old = lc.to_frame([lc.new_row(image="a.fits", method="aperture", band="g",
                                  mjd=60900.0, mag=18.0, primary=True)])
    new = lc.to_frame([lc.new_row(image="a.fits", method="aperture", band="g",
                                  mjd=60900.0, mag=17.5, primary=True)])
    merged = lc.merge(old, new)
    assert len(merged) == 1
    assert merged.iloc[0]["mag"] == pytest.approx(17.5)
    assert merged["primary"].sum() == 1


def test_reselect_primary_marks_exactly_one_row_per_image():
    frame = lc.to_frame([
        lc.new_row(image="a.fits", method="aperture", band="g", mjd=60900.0, mag=18.0),
        lc.new_row(image="a.fits", method="difference_psf", band="g", mjd=60900.0, mag=18.4),
        lc.new_row(image="b.fits", method="aperture", band="r", mjd=60900.1, mag=17.0),
    ])
    out = lc.reselect_primary(frame)
    assert out.groupby("image")["primary"].sum().tolist() == [1, 1]
    chosen = out[out["primary"] & (out["image"] == "a.fits")].iloc[0]
    assert chosen["method"] == "difference_psf"


# ---------------------------------------------------------------------------
# resuming a run
# ---------------------------------------------------------------------------
def _epoch(path, band="g", mjd=61300.0):
    from tttphot.pipeline import EpochIndex

    return EpochIndex(path=Path(path), mjd=mjd, band=band, object_name="SN2026abwp",
                      exptime=120.0)


def _write_table(outdir, target, names):
    frame = lc.to_frame([lc.new_row(image=n, method="aperture", band="g",
                                    mjd=61300.0 + i, mag=19.0)
                         for i, n in enumerate(names)])
    path = Path(outdir) / f"{target}_photometry_all.csv"
    frame.to_csv(path, index=False)
    return path


def test_previous_run_reads_back_what_was_measured(tmp_path):
    from tttphot.pipeline import previous_run

    _write_table(tmp_path, "SN2026abwp", ["a.fits", "b.fits"])
    table, measured, written_at = previous_run(tmp_path, "SN2026abwp")
    assert measured == {"a.fits", "b.fits"}
    assert len(table) == 2
    assert written_at > 0


def test_previous_run_on_a_fresh_target(tmp_path):
    from tttphot.pipeline import previous_run

    assert previous_run(tmp_path, "SN2026abwp") == (None, set(), 0.0)


def test_previous_run_survives_a_truncated_table(tmp_path):
    """A half-written table must mean "measure everything", not crash."""
    from tttphot.pipeline import previous_run

    (tmp_path / "SN2026abwp_photometry_all.csv").write_text("")
    assert previous_run(tmp_path, "SN2026abwp")[1] == set()


def test_previous_run_ignores_a_table_with_no_image_column(tmp_path):
    from tttphot.pipeline import previous_run

    (tmp_path / "SN2026abwp_photometry_all.csv").write_text("mjd,mag\n61300.0,19.0\n")
    assert previous_run(tmp_path, "SN2026abwp") == (None, set(), 0.0)


def test_pending_epochs_skips_what_is_already_done(tmp_path):
    from tttphot.pipeline import pending_epochs

    for name in ("a.fits", "b.fits", "c.fits"):
        (tmp_path / name).write_text("x")
    written_at = max((tmp_path / n).stat().st_mtime for n in ("a.fits", "b.fits")) + 10
    index = [_epoch(tmp_path / n) for n in ("a.fits", "b.fits", "c.fits")]
    todo, skipped, changed = pending_epochs(index, {"a.fits", "b.fits"}, written_at)
    assert [e.path.name for e in todo] == ["c.fits"]
    assert skipped == 2
    assert changed == []


def test_pending_epochs_remeasures_a_frame_the_observatory_re_reduced(tmp_path):
    """A frame that changed on disk since the table was written is not done.

    Otherwise a re-reduction would silently keep the numbers measured from the
    version that no longer exists.
    """
    import os
    from tttphot.pipeline import pending_epochs

    old = tmp_path / "old.fits"
    fresh = tmp_path / "fresh.fits"
    old.write_text("x")
    fresh.write_text("x")
    written_at = old.stat().st_mtime + 10
    os.utime(fresh, (written_at + 100, written_at + 100))   # re-reduced since

    index = [_epoch(old), _epoch(fresh)]
    todo, skipped, changed = pending_epochs(index, {"old.fits", "fresh.fits"}, written_at)
    assert [e.path.name for e in todo] == ["fresh.fits"]
    assert skipped == 1
    assert changed == ["fresh.fits"]


def test_pending_epochs_does_everything_when_there_is_no_table(tmp_path):
    from tttphot.pipeline import pending_epochs

    index = [_epoch(tmp_path / "a.fits"), _epoch(tmp_path / "b.fits")]
    todo, skipped, changed = pending_epochs(index, set(), 0.0)
    assert len(todo) == 2 and skipped == 0 and changed == []


def test_previous_coord_is_read_back_so_a_resumed_run_keeps_the_position(tmp_path):
    """The centroid refinement samples whatever epochs exist, so re-deriving the
    position on a resumed run would put the new rows a fraction of an arcsecond
    from the old ones in the same table."""
    import json
    from astropy.coordinates import SkyCoord
    from tttphot.pipeline import previous_coord

    original = SkyCoord(61.8617875 * u.deg, 1.7614889 * u.deg)
    (tmp_path / "run_summary.json").write_text(json.dumps(
        {"coord": original.to_string("hmsdms", precision=3)}))
    back = previous_coord(tmp_path, "SN2026abwp")
    assert back is not None
    assert original.separation(back).arcsec < 0.01


def test_previous_coord_is_none_without_a_summary(tmp_path):
    from tttphot.pipeline import previous_coord

    assert previous_coord(tmp_path, "SN2026abwp") is None
    (tmp_path / "run_summary.json").write_text("{}")
    assert previous_coord(tmp_path, "SN2026abwp") is None


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def test_deep_merge_does_not_modify_its_inputs():
    from tttphot.config import deep_merge

    base = {"a": {"b": 1, "c": 2}}
    merged = deep_merge(base, {"a": {"c": 3, "d": 4}})
    assert merged == {"a": {"b": 1, "c": 3, "d": 4}}
    assert base == {"a": {"b": 1, "c": 2}}


def test_config_dotted_access():
    cfg = Config({"a": {"b": {"c": 5}}})
    assert cfg.get("a.b.c") == 5
    assert cfg.get("a.b.missing", "fallback") == "fallback"
    assert cfg.get("nothing.at.all") is None


def test_config_overridden_is_a_real_config_and_leaves_the_original_alone():
    """The relaxation ladder used to pass a two-method stand-in object. The
    first callee to reach for require, cache_root or cfg[...] would have broken
    the relaxed path while the strict one kept working."""
    cfg = Config({"calibration": {"min_snr": 20.0, "min_stars": 5}})
    relaxed = cfg.overridden({"calibration.min_snr": 8.0})
    assert isinstance(relaxed, Config)
    assert relaxed.get("calibration.min_snr") == 8.0
    assert relaxed.get("calibration.min_stars") == 5
    assert cfg.get("calibration.min_snr") == 20.0
    assert relaxed.require("calibration.min_stars") == 5      # the full interface
    assert relaxed.as_dict()["calibration"]["min_snr"] == 8.0


def test_config_require_raises_a_user_error():
    with pytest.raises(utils.UserError):
        Config({}).require("missing.key")


def test_the_shipped_default_config_parses_and_has_what_the_code_reads():
    """Guards against a key being renamed in one place and not the other."""
    from tttphot.config import DEFAULT_CONFIG, load_yaml

    cfg = Config(load_yaml(DEFAULT_CONFIG))
    for key in ("paths.data_root", "image.edge_margin_arcsec", "detection.threshold_sigma",
                "photometry.aperture_scale", "photometry.annulus_scale",
                "calibration.catalogs", "calibration.min_stars",
                "subtraction.kernel.half_size_px", "output.primary_method",
                "output.detection_snr"):
        assert cfg.get(key) is not None, f"{key} is missing from config/default.yaml"
