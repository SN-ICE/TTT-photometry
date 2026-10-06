"""Retained difference photometry, nearest-epoch colors, and their presentation."""

import numpy as np
import pandas as pd
import pytest

from tttphot import lightcurve as lc, plots


def detection(band, minute, mag=18., **kwargs):
    values = dict(target="test", band=band, mjd=61000 + minute / 1440.,
                  mag=mag, mag_err=.03, flags=0, method="difference_psf",
                  system="AB", primary=True, image=f"{band}_{minute}.fits")
    values.update(kwargs)
    return lc.new_row(**values)


@pytest.mark.parametrize("diff_mag,direct_mag,chi2,flags", [
    (16.348193, 16.040258, 6.99741, lc.FLAG_BAD_SUBTRACTION),
    (16.305343, 16.128398, .96202, lc.FLAG_BAD_SUBTRACTION | lc.FLAG_DIPOLE_RESIDUAL),
])
def test_abgb_flagged_epochs_keep_measured_difference_after_csv(tmp_path, diff_mag, direct_mag, chi2, flags):
    difference = detection("r", 0, diff_mag, flags=flags, diff_chi2=chi2)
    direct = detection("r", 0, direct_mag, method="psf")
    frame = lc.reselect_primary(lc.to_frame([difference, direct]))
    paths = lc.write(frame, tmp_path, "test")
    primary = pd.read_csv(paths["lightcurve"])
    assert primary.method.tolist() == ["difference_psf"]
    assert primary.mag.tolist() == [diff_mag]
    assert primary["flags"].tolist() == [flags]


@pytest.mark.parametrize("flags", [lc.FLAG_BAD_SUBTRACTION, lc.FLAG_DIPOLE_RESIDUAL,
                                   lc.FLAG_KERNEL_SUM])
def test_flagged_difference_limits_still_beat_host_detections(flags):
    difference = detection("g", 0, np.nan, flags=flags, is_limit=True, limit_mag=21.)
    direct = detection("g", 0, 17., method="psf")
    assert lc.choose_primary([direct, difference]) is difference


def test_colors_use_closest_pairs_once_and_propagate_values():
    # The first g epoch must yield the r observation to the closer second g.
    frame = lc.to_frame([detection("g", 0, 19.), detection("g", 9, 18.),
                         detection("r", 10, 17., mag_err=.04, flags=lc.FLAG_BAD_SUBTRACTION),
                         detection("i", 11, 16.), detection("g", 90, 20.)])
    colors = lc.color_curve(frame, 30).set_index("color")
    assert set(colors.index) == {"g-r", "r-i", "g-i"}
    gr = colors.loc["g-r"]
    assert gr.image_a == "g_9.fits"
    assert gr.image_b == "r_10.fits"
    assert gr.color_mag == 1.
    assert gr.color_err == pytest.approx(.05)
    assert gr.mjd == pytest.approx(61000 + 9.5 / 1440.)
    assert gr.separation_minutes == pytest.approx(1.)
    assert gr.flagged
    assert colors.loc["r-i", "flagged"]
    assert not colors.loc["g-i", "flagged"]


def test_color_matching_is_deterministic_without_reuse():
    frame = lc.to_frame([detection("g", 0), detection("g", 10),
                         detection("r", 1), detection("r", 11)])
    colors = lc.color_curve(frame)
    assert len(colors) == 2
    assert colors.image_a.is_unique and colors.image_b.is_unique
    pd.testing.assert_frame_equal(colors, lc.color_curve(frame.sample(frac=1, random_state=1)))


@pytest.mark.parametrize("change", [
    {"is_limit": True, "limit_mag": 20.}, {"primary": False}, {"mag": np.nan},
    {"mag_err": np.nan}, {"mag_err": -1.}, {"mjd": np.nan},
    {"system": "Vega"}, {"system": np.nan}, {"method": "psf"}, {"target": "other"},
])
def test_colors_exclude_invalid_detections_and_incompatible_pairs(change):
    frame = lc.to_frame([detection("g", 0), detection("r", 1, **change)])
    assert lc.color_curve(frame).empty


def test_color_window_boundary_and_empty_results():
    frame = lc.to_frame([detection("g", 0), detection("r", 30)])
    assert len(lc.color_curve(frame, 30)) == 1
    assert lc.color_curve(frame, 29.99).empty
    assert list(lc.color_curve(frame, 0).columns) == lc.COLOR_COLUMNS
    same_time = lc.to_frame([detection("g", 0), detection("r", 0)])
    assert len(lc.color_curve(same_time, 0)) == 1
    for window in (-1, np.nan, np.inf):
        with pytest.raises(ValueError):
            lc.color_curve(frame, window)


def test_colors_csv_respects_configured_window(tmp_path):
    frame = lc.to_frame([detection("g", 0), detection("r", 40)])
    paths = lc.write(frame, tmp_path, "test", color_max_separation_minutes=60)
    assert len(pd.read_csv(paths["colors"])) == 1
    lc.write(frame, tmp_path, "test", color_max_separation_minutes=30)
    assert pd.read_csv(paths["colors"]).empty


def test_figure_has_equal_panels_and_open_flagged_points(tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    captured = []
    save = Figure.savefig

    def inspect(fig, *args, **kwargs):
        captured.append(fig)
        return save(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect)
    frame = lc.to_frame([detection("g", 0, 19., flags=lc.FLAG_BAD_SUBTRACTION),
                         detection("r", 1, 18.), detection("i", 2, 17.)])
    path = tmp_path / "lc.png"
    assert plots.lightcurve(path, frame, "test") == path
    assert path.stat().st_size > 0
    left, right = captured[0].axes
    assert left.yaxis_inverted() and not right.yaxis_inverted()
    assert left.get_position().width == pytest.approx(right.get_position().width)
    # Shifted g=20 on the left; unshifted g-r=1 on the right.
    assert any(line.get_markerfacecolor() == "white" and list(line.get_ydata()) == [20.]
               for line in left.lines)
    assert any(line.get_markerfacecolor() == "white" and list(line.get_ydata()) == [1.]
               for line in right.lines)


def test_single_band_figure_explains_absent_colors(tmp_path):
    frame = lc.to_frame([detection("g", 0)])
    assert plots.lightcurve(tmp_path / "single.png", frame, "test") is not None
