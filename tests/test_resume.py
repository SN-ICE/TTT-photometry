"""Run orchestration regressions, with real CSV I/O and offline science stubs."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tttphot import lightcurve as lc, pipeline
from tttphot.config import Config


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    cfg = Config({
        "paths": {"data_root": str(tmp_path), "work_root": str(tmp_path / "runs"),
                  "cache_root": str(tmp_path / "cache")},
        "target": {"ra": 1.0, "dec": 1.0, "resolve": "off", "refine": "off"},
        "subtraction": {"enabled": False, "check_kernel_sum": False},
        "calibration": {"transfer_zeropoints": False},
        "output": {"write_qa_plots": False, "write_diagnostics": False},
    })
    epochs = []
    for band in ("g", "r"):
        path = tmp_path / f"{band}.fits"
        path.write_bytes(b"original input")
        epochs.append(pipeline.EpochIndex(path, 60000., band, "SN2026test", 120.))
    state = SimpleNamespace(cfg=cfg, epochs=epochs, calls=[], fail=set(),
                            methods=["aperture", "difference_psf"], mag=18.,
                            mutate=False, target="SN2026test")
    state.outdir = Path(cfg.work_root) / state.target

    monkeypatch.setattr(pipeline, "find_images", lambda *a: [e.path for e in epochs])
    monkeypatch.setattr(pipeline, "index_images", lambda *a, **k: epochs)
    monkeypatch.setattr(pipeline, "load_patterns", lambda *a: {})
    monkeypatch.setattr(pipeline, "load_astrometry_reference", lambda *a, **k: None)

    def prepare(path, *args, **kwargs):
        state.calls.append(path.name)
        if path.name in state.fail:
            raise RuntimeError("simulated measurement failure")
        return SimpleNamespace(
            path=path, wcs=True, contains=lambda *a, **k: True,
            meta=SimpleNamespace(pixel_scale=1.),
        )

    def process(image, *args, **kwargs):
        if state.mutate:
            image.path.write_bytes(b"input changed while being measured")
        epoch = next(e for e in epochs if e.path == image.path)
        return [lc.new_row(target=state.target, image=image.path.name, method=method,
                           band=epoch.band, mjd=epoch.mjd, mag=state.mag, flags=0,
                           is_limit=False, diff_chi2=1.) for method in state.methods]

    monkeypatch.setattr(pipeline, "prepare_image", prepare)
    monkeypatch.setattr(pipeline, "process_image", process)
    state.run = lambda **kwargs: pipeline.run(cfg, state.target, **kwargs)
    return state


@pytest.mark.parametrize("selection,selected", [
    ({"bands": ["g"]}, "g.fits"),
    ({"max_images": 1}, "g.fits"),
    ({"skip": 1}, "r.fits"),
])
@pytest.mark.parametrize("append", [False, True])
def test_no_resume_preserves_unselected_epochs(campaign, selection, selected, append):
    campaign.run()
    campaign.calls.clear()
    campaign.mag = 19.
    result = campaign.run(resume=False, append=append, **selection)
    assert campaign.calls == [selected]
    frame = result["frame"]
    assert set(frame.image) == {"g.fits", "r.fits"}
    assert (frame.loc[frame.image == selected, "mag"] == 19.).all()
    assert (frame.loc[frame.image != selected, "mag"] == 18.).all()
    written = pd.read_csv(result["all"])
    assert set(written.image) == {"g.fits", "r.fits"}
    assert written.groupby("image").primary.sum().eq(1).all()


def test_rerun_removes_methods_that_were_not_reproduced(campaign):
    campaign.run()
    campaign.methods = ["aperture"]
    campaign.mag = 19.
    result = campaign.run(resume=False, bands=["g"])
    current = result["frame"].query("image == 'g.fits'")
    assert current.method.tolist() == ["aperture"]
    assert current.mag.tolist() == [19.]
    assert current.primary.tolist() == [True]
    assert set(result["frame"].query("image == 'r.fits'").method) == {
        "aperture", "difference_psf"}


def test_partial_run_does_not_hide_changes_in_another_band(campaign):
    campaign.run()
    campaign.epochs[0].path.write_bytes(b"new reduction")
    campaign.calls.clear()
    campaign.run(bands=["r"])
    assert campaign.calls == []
    campaign.run()
    assert campaign.calls == ["g.fits"]
    campaign.calls.clear()
    campaign.run()
    assert campaign.calls == []


def test_failed_remeasurement_retains_old_rows_and_is_retried(campaign):
    original = campaign.run()["frame"]
    campaign.epochs[0].path.write_bytes(b"new reduction")
    campaign.fail.add("g.fits")
    campaign.calls.clear()
    failed = campaign.run()
    assert failed["summary"]["n_failures"] == 1
    assert campaign.calls == ["g.fits"]
    assert failed["frame"].input_fingerprint.tolist() == original.input_fingerprint.tolist()
    campaign.fail.clear()
    campaign.calls.clear()
    campaign.run()
    assert campaign.calls == ["g.fits"]


def test_legacy_rows_are_remeasured_once_even_after_a_failed_attempt(campaign):
    result = campaign.run()
    table = pd.read_csv(result["all"]).drop(columns="input_fingerprint")
    table.to_csv(result["all"], index=False)
    campaign.fail.add("g.fits")
    campaign.calls.clear()
    campaign.run()
    assert campaign.calls == ["g.fits", "r.fits"]
    campaign.fail.clear()
    campaign.calls.clear()
    campaign.run()
    assert campaign.calls == ["g.fits"]
    campaign.calls.clear()
    campaign.run()
    assert campaign.calls == []


def test_changed_during_measurement_does_not_record_a_success(campaign):
    original = campaign.run()["frame"]
    campaign.mutate = True
    failed = campaign.run(resume=False, bands=["g"])
    assert failed["summary"]["n_failures"] == 1
    assert "input changed" in failed["summary"]["failures"][0]["error"]
    assert failed["frame"].input_fingerprint.tolist() == original.input_fingerprint.tolist()
    campaign.mutate = False
    campaign.calls.clear()
    campaign.run()
    assert campaign.calls == ["g.fits"]


def test_config_can_disable_resume(campaign):
    campaign.run()
    campaign.cfg.set("output.resume", False)
    campaign.calls.clear()
    campaign.run()
    assert campaign.calls == ["g.fits", "r.fits"]


def test_cli_no_resume_replaces_selected_methods_and_preserves_other_bands(campaign, monkeypatch):
    from tttphot import cli

    campaign.run()
    campaign.methods = ["aperture"]
    campaign.mag = 19.
    campaign.calls.clear()
    monkeypatch.setattr(cli, "_build_config", lambda args: campaign.cfg)
    monkeypatch.setattr(cli, "setup_logging", lambda *a, **k: None)
    assert cli.main(["run", campaign.target, "--no-resume", "--bands", "g"]) == 0
    assert campaign.calls == ["g.fits"]
    saved = pd.read_csv(campaign.outdir / f"{campaign.target}_photometry_all.csv")
    assert saved.query("image == 'g.fits'").method.tolist() == ["aperture"]
    assert saved.query("image == 'g.fits'").mag.tolist() == [19.]
    assert len(saved.query("image == 'r.fits'")) == 2


def test_clean_difference_limit_remains_primary_after_merge_and_csv(tmp_path):
    common = dict(image="g.fits", target="test", band="g", mjd=60000., flags=0)
    old = lc.to_frame([lc.new_row(**common, method="difference_psf", mag=18.)])
    new = lc.to_frame([
        lc.new_row(**common, method="difference_psf", mag=np.nan, is_limit=True, limit_mag=21.),
        lc.new_row(**common, method="aperture", mag=17.),
    ])
    result = lc.merge(old, new)
    paths = lc.write(result, tmp_path, "test")
    primary = pd.read_csv(paths["lightcurve"])
    assert primary.method.tolist() == ["difference_psf"]
    assert primary.is_limit.tolist() == [True]
    assert primary.limit_mag.tolist() == [21.]


@pytest.mark.parametrize("method", ["difference_psf", "difference_aperture"])
@pytest.mark.parametrize("direct", ["psf", "aperture"])
def test_clean_difference_limit_beats_direct_detection(method, direct):
    limit = lc.new_row(method=method, mag=np.nan, is_limit=True, limit_mag=21., flags=0)
    host = lc.new_row(method=direct, mag=17., is_limit=False, flags=0)
    assert lc.choose_primary([host, limit]) is limit
    assert lc.choose_primary([host, limit], preference=direct) is host


@pytest.mark.parametrize("flags,depth", [
    (lc.FLAG_BAD_PSF, 21.), (lc.FLAG_MASKED_PIXEL, 21.),
    (lc.FLAG_IN_TEMPLATE, 21.), (0, np.nan),
])
def test_unusable_difference_limit_does_not_displace_clean_detection(flags, depth):
    limit = lc.new_row(method="difference_psf", mag=np.nan, is_limit=True,
                       limit_mag=depth, flags=flags)
    direct = lc.new_row(method="psf", mag=17., is_limit=False, flags=0)
    assert lc.choose_primary([limit, direct]) is direct


def test_difference_aperture_detection_beats_difference_psf_limit():
    limit = lc.new_row(method="difference_psf", mag=np.nan, is_limit=True,
                       limit_mag=21., flags=0)
    detected = lc.new_row(method="difference_aperture", mag=20., flags=0)
    assert lc.choose_primary([limit, detected]) is detected
