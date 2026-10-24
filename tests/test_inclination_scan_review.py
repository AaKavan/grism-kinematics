"""An optional inclination diagnostic cannot replace or silently refit a source."""
import hashlib
import json
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from dingo import inclination_scan as scans
from dingo.three_file_plot import physical_scale


def hashes(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob("*") if p.is_file()}


@pytest.fixture
def saved_scan(tmp_path):
    results = tmp_path / "results"
    parent = results / "three_file_sources" / "ID15665" / "broad_test"
    path = parent.with_name(parent.name + "_inclination_scan")
    parent.mkdir(parents=True)
    path.mkdir()
    prior = dict(type="gaussian", mu=float(np.deg2rad(32.)),
                 sigma=float(np.deg2rad(10.)), min=float(np.deg2rad(5.)),
                 max=float(np.deg2rad(85.)))
    search = dict(source_id=15665, search_settings={"method": "broad"},
                  inclination_prior=prior)
    (parent / "start_search.json").write_text(json.dumps(search))
    # Treat every parent-run file as untouchable when reviewing or scanning.
    (parent / "samples.npz").write_bytes(b"existing posterior")
    (parent / "run.log").write_text("existing posterior log")
    angles = np.array([5., 15., 30.])
    names = ["image.R.dx", "velocity.inc_v", "velocity.R_v", "velocity.V_sini",
             "velocity.theta_v", "velocity.x0_v", "velocity.y0_v", "image.R.dy",
             "noise.ln_f"]
    points = np.array([[.1, np.deg2rad(angle), rt, amp, 2., 3., 3., -.2, .1]
                       for angle, rt, amp in zip(angles, [2., 3., 4.], [200., 190., 180.])])
    likelihood = np.array([100., 98., 90.])
    inclination_prior = -.5 * ((np.deg2rad(angles) - prior["mu"]) / prior["sigma"])**2
    log_prior = inclination_prior - 2.
    r = np.stack([np.arange(30.).reshape(5, 6) * scale for scale in (1., 2., 4.)])
    residual = np.zeros_like(r)
    residual[:, 2, 2] = [.2, -1., 5.]
    mask = np.ones((5, 6), bool)
    mask[0, 0] = False
    arrays = dict(inclinations_deg=angles, points=points, param_names=np.asarray(names),
                  log_likelihood=likelihood, log_prior=log_prior,
                  log_posterior=likelihood + log_prior,
                  log_inclination_prior=inclination_prior, candidate_ids=np.arange(3),
                  rectified_R=r, rectified_C=r - residual, difference=residual,
                  mask=mask, observed_R=r[0], observed_C=r[0] + .1, image=r[0])
    rows = [dict(candidate_id=i, full_params=dict(zip(names, point)),
                 log_likelihood=float(likelihood[i]), log_prior=float(log_prior[i]),
                 log_posterior=float(likelihood[i] + log_prior[i]),
                 log_inclination_prior=float(inclination_prior[i]),
                 kind="fixed inclination diagnostic")
            for i, point in enumerate(points)]
    report = dict(schema=scans.SCHEMA, source_id=15665, status="complete",
                  redshift=1.1, pixel_scale_arcsec=.063, inclination_prior=prior,
                  settings={"iterations": 80, "inclinations_deg": angles.tolist()},
                  trials=rows, finished_utc="2026-10-20T18:00:00+00:00",
                  parent_search_sha256=hashlib.sha256(
                      (parent / "start_search.json").read_bytes()).hexdigest())

    def write():
        np.savez(path / "inclination_scan.npz", **arrays)
        report["scan_sha256"] = hashlib.sha256(
            (path / "inclination_scan.npz").read_bytes()).hexdigest()
        (path / "report.json").write_text(json.dumps(report))

    write()
    return SimpleNamespace(path=path, parent=parent, results=results, arrays=arrays,
                           report=report, write=write, working=tmp_path / "working")


def test_saved_only_review_never_launches_even_with_override(saved_scan, monkeypatch):
    before = hashes(saved_scan.results)
    monkeypatch.setattr(scans, "run_command",
                        lambda *a, **k: pytest.fail("Review must never run a worker"))
    monkeypatch.setattr(scans.ScanResults, "display", lambda *a, **k: None)
    result = scans.review(15665, run_name="broad_test", results=saved_scan.results,
                          run=False, override=True, working_dir=None)
    assert isinstance(result, scans.ScanResults)
    assert hashes(saved_scan.results) == before


def test_no_saved_scan_is_a_small_noop_and_does_not_create_outputs(saved_scan, monkeypatch, capsys):
    before = hashes(saved_scan.results)
    monkeypatch.setattr(scans, "run_command",
                        lambda *a, **k: pytest.fail("Off means no worker"))
    assert scans.review(6170, run_name="broad_test", results=saved_scan.results,
                        run=False, override=True, working_dir=saved_scan.working) is None
    assert "RUN_INCLINATION_SCAN" in capsys.readouterr().out
    assert not saved_scan.working.exists()
    assert hashes(saved_scan.results) == before


@pytest.mark.parametrize("mutation", ["source", "archive", "parent", "missing_parent_hash"])
def test_wrong_source_changed_archive_or_replaced_parent_rejected(saved_scan, mutation):
    if mutation == "source":
        saved_scan.report["source_id"] = 6170
        saved_scan.write()
    elif mutation == "archive":
        with (saved_scan.path / "inclination_scan.npz").open("ab") as stream:
            stream.write(b"changed")
    elif mutation == "parent":
        (saved_scan.parent / "start_search.json").write_text('{"new_search": true}')
    else:
        saved_scan.report.pop("parent_search_sha256")
        saved_scan.write()
    with pytest.raises(ValueError):
        scans.load(15665, run_name="broad_test", results=saved_scan.results)


def test_rotation_values_use_same_radius_and_actual_inclination(saved_scan):
    result = scans.load(15665, run_name="broad_test", results=saved_scan.results)
    pc_per_pixel = physical_scale(saved_scan.report)["pc_per_pixel"]
    radius = 3 * pc_per_pixel
    table = result.table(radius_pc=radius)
    expected = (2 / np.pi) * np.array([200., 190., 180.]) * np.arctan(3 / np.array([2., 3., 4.]))
    np.testing.assert_allclose(table["Vrot_sin_i_kms"], expected)
    np.testing.assert_allclose(table["Vrot_kms"], expected / np.sin(np.deg2rad([5., 15., 30.])))
    np.testing.assert_allclose(table["rturn_pc"], np.array([2., 3., 4.]) * pc_per_pixel)
    assert (table["radius_pc"] == radius).all()


def test_every_residual_selector_uses_same_image_and_residual_scales(saved_scan):
    result = scans.load(15665, run_name="broad_test", results=saved_scan.results)
    limits = []
    for angle in result.angles:
        fig = result.residuals(angle)
        try:
            limits.append([ax.images[0].get_clim() for ax in fig.axes[:3]])
            residual = fig.axes[2].images[0].get_array()
            assert np.ma.getmaskarray(residual)[0, 0]
            assert "fixed inclination" in fig._suptitle.get_text()
        finally:
            plt.close(fig)
    np.testing.assert_array_equal(limits[0], limits[1])
    np.testing.assert_array_equal(limits[1], limits[2])
    with pytest.raises(ValueError, match="saved scan"):
        result.residuals(17.)


def test_score_curves_are_separately_zeroed_and_not_probabilities(saved_scan):
    result = scans.load(15665, run_name="broad_test", results=saved_scan.results)
    fig = result.figure()
    try:
        for key, line in [("log_posterior", fig.axes[0].lines[0]),
                          ("log_likelihood", fig.axes[0].lines[1]),
                          ("log_inclination_prior", fig.axes[1].lines[0])]:
            scores = saved_scan.arrays[key]
            np.testing.assert_allclose(line.get_ydata(), 2 * (scores.max() - scores))
        assert "not marginalized probabilities" in fig._supxlabel.get_text()
    finally:
        plt.close(fig)


@pytest.mark.parametrize("mutation", ["scores", "wrong_inclination", "wrong_residual"])
def test_internally_inconsistent_diagnostic_is_rejected(saved_scan, mutation):
    if mutation == "scores":
        saved_scan.arrays["log_posterior"][1] += 1.
    elif mutation == "wrong_inclination":
        saved_scan.report["trials"][1]["full_params"]["velocity.inc_v"] += .1
    else:
        saved_scan.arrays["difference"] *= -1
    saved_scan.write()
    with pytest.raises(ValueError):
        scans.ScanResults(saved_scan.path, 15665)


@pytest.mark.parametrize("mutation", ["points", "names", "trial", "duplicate_names"])
def test_saved_parameter_vectors_and_displayed_parameters_must_agree(saved_scan, mutation):
    if mutation == "points":
        saved_scan.arrays["points"][1, 3] += 50.
    elif mutation == "names":
        saved_scan.arrays["param_names"][[0, 3]] = saved_scan.arrays["param_names"][[3, 0]]
    elif mutation == "trial":
        saved_scan.report["trials"][1]["full_params"]["velocity.V_sini"] += 50.
    else:
        saved_scan.arrays["param_names"][0] = saved_scan.arrays["param_names"][1]
    saved_scan.write()
    with pytest.raises(ValueError):
        scans.ScanResults(saved_scan.path, 15665)


@pytest.mark.parametrize("kind", ["uniform", "log_uniform"])
def test_prior_panel_uses_the_actual_saved_prior_family(saved_scan, kind):
    spec = saved_scan.report["inclination_prior"]
    spec["type"] = kind
    spec.pop("mu")
    spec.pop("sigma")
    radians = np.deg2rad(saved_scan.arrays["inclinations_deg"])
    values = np.zeros_like(radians) if kind == "uniform" else -np.log(radians)
    saved_scan.arrays["log_inclination_prior"] = values
    saved_scan.arrays["log_prior"] = values - 2.
    saved_scan.arrays["log_posterior"] = saved_scan.arrays["log_likelihood"] + values - 2.
    for i, row in enumerate(saved_scan.report["trials"]):
        for key in ("log_inclination_prior", "log_prior", "log_posterior"):
            row[key] = float(saved_scan.arrays[key][i])
    saved_scan.write()
    result = scans.ScanResults(saved_scan.path, 15665)
    fig = result.figure()
    try:
        np.testing.assert_allclose(fig.axes[1].lines[0].get_ydata(),
                                   2 * (values.max() - values))
        continuous = fig.axes[1].lines[1]
        x = np.deg2rad(continuous.get_xdata())
        expected = np.zeros_like(x) if kind == "uniform" else -np.log(x)
        np.testing.assert_allclose(continuous.get_ydata(), 2 * (values.max() - expected))
    finally:
        plt.close(fig)


def test_scan_dispatch_is_separate_sorted_and_never_sampling(saved_scan, monkeypatch):
    before = hashes(saved_scan.parent)
    dispatched = []
    monkeypatch.setattr(scans, "run_command",
                        lambda args, log, **kwargs: dispatched.append((args, log, kwargs)))
    result = scans.run_scan(15665, run_name="broad_test", results=saved_scan.results,
                       inclinations_deg=[30., 5., 15.], device="cpu", iterations=17)
    assert isinstance(result, scans.ScanResults)
    assert len(dispatched) == 1
    args, log, kwargs = dispatched[0]
    assert args[:2] == ["-m", "dingo.inclination_scan_fit"]
    assert args[args.index("--output") + 1] == str(saved_scan.path)
    assert args[args.index("--run-dir") + 1] == str(saved_scan.parent)
    assert args[args.index("--iterations") + 1] == "17"
    assert args[args.index("--inclinations") + 1:] == ["5.0", "15.0", "30.0"]
    assert not {"--candidate", "--draws", "--warmup"}.intersection(args)
    assert log == saved_scan.path / "run.log"
    assert kwargs == {"device": "cpu"}
    assert hashes(saved_scan.parent) == before


def test_override_cannot_delete_main_run_with_colliding_directory_name(saved_scan, monkeypatch):
    # A user can legitimately name a main run "<parent>_inclination_scan".
    # It is not disposable just because its directory matches our default.
    (saved_scan.path / "report.json").write_text(json.dumps(
        dict(schema="dingo-three-file-v1", source_id=15665, status="complete")))
    (saved_scan.path / "search_identity.json").write_text(json.dumps(
        dict(schema="dingo-three-file-search-v1", source_id=15665)))
    (saved_scan.path / "samples.npz").write_bytes(b"other main run posterior")
    before = hashes(saved_scan.results)
    monkeypatch.setattr(scans, "run_command",
                        lambda *a, **k: pytest.fail("Foreign output must not run"))
    with pytest.raises(ValueError):
        scans.run_scan(15665, run_name="broad_test", results=saved_scan.results,
                       inclinations_deg=[5., 15., 30.], override=True)
    assert hashes(saved_scan.results) == before


def test_override_replaces_only_identified_diagnostic(saved_scan, monkeypatch):
    (saved_scan.path / "identity.json").write_text(json.dumps(
        dict(schema=scans.SCHEMA, source_id=15665)))
    (saved_scan.path / "old_checkpoint.npz").write_bytes(b"diagnostic checkpoint")
    before = hashes(saved_scan.parent)
    called = []

    def worker(args, log, **kwargs):
        called.append(args)
        assert not saved_scan.path.exists()
        saved_scan.path.mkdir()
        saved_scan.write()

    monkeypatch.setattr(scans, "run_command", worker)
    result = scans.run_scan(15665, run_name="broad_test", results=saved_scan.results,
                           inclinations_deg=[5., 15., 30.], override=True)
    assert isinstance(result, scans.ScanResults)
    assert len(called) == 1
    assert not (saved_scan.path / "old_checkpoint.npz").exists()
    assert hashes(saved_scan.parent) == before


@pytest.mark.parametrize("angles", [[0., 10.], [5., 90.], [10., 10.], [np.nan], []])
def test_invalid_scan_settings_do_not_dispatch_or_change_outputs(saved_scan, monkeypatch, angles):
    before = hashes(saved_scan.results)
    monkeypatch.setattr(scans, "run_command",
                        lambda *a, **k: pytest.fail("Bad settings must not run"))
    with pytest.raises(ValueError):
        scans.run_scan(15665, run_name="broad_test", results=saved_scan.results,
                  inclinations_deg=angles, override=True)
    assert hashes(saved_scan.results) == before


def test_export_contains_review_data_and_preserves_final_posterior(saved_scan):
    final = saved_scan.working / "ID15665"
    final.mkdir(parents=True)
    (final / "results.npz").write_bytes(b"posterior")
    (final / "corner.png").write_bytes(b"posterior corner")
    before_final, before_cache = hashes(final), hashes(saved_scan.results)
    result = scans.load(15665, run_name="broad_test", results=saved_scan.results)
    output = result.export(saved_scan.working, run_name="broad_test")
    assert output == saved_scan.working / "inclination_scans" / "ID15665" / "broad_test"
    assert {p.name for p in output.iterdir()} == {
        "inclination_scan.npz", "inclination_scan.png", "run_info.txt",
        "residual_5deg.png", "residual_15deg.png", "residual_30deg.png"}
    assert hashes(final) == before_final
    assert hashes(saved_scan.results) == before_cache
    with np.load(output / "inclination_scan.npz", allow_pickle=False) as archive:
        for key, value in saved_scan.arrays.items():
            np.testing.assert_array_equal(archive[key], value)
        metadata = json.loads(archive["metadata_json"].item())
        assert metadata["report"] == saved_scan.report
        assert metadata["schema"] == scans.EXPORT_SCHEMA
    text = (output / "run_info.txt").read_text()
    assert "No MCMC." in text
    assert "2026-10-20T18:00:00" in text
    assert "80" in text
    # A second named diagnostic must leave the first untouched.
    before = hashes(output)
    other = result.export(saved_scan.working, run_name="another_review")
    assert other != output and hashes(output) == before


def test_failed_reexport_preserves_previous_diagnostic(saved_scan, monkeypatch):
    result = scans.load(15665, run_name="broad_test", results=saved_scan.results)
    output = result.export(saved_scan.working, run_name="broad_test")
    before = hashes(output)
    monkeypatch.setattr(result, "figure",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("render failed")))
    with pytest.raises(RuntimeError, match="render failed"):
        result.export(saved_scan.working, run_name="broad_test", radius_pc=1000.)
    assert hashes(output) == before
    assert not list(output.parent.glob(".inclination_export_*"))


def test_changed_archive_after_loading_cannot_be_reexported(saved_scan):
    result = scans.load(15665, run_name="broad_test", results=saved_scan.results)
    output = result.export(saved_scan.working, run_name="broad_test")
    before = hashes(output)
    with (saved_scan.path / "inclination_scan.npz").open("ab") as stream:
        stream.write(b"changed after load")
    with pytest.raises(ValueError, match="changed"):
        result.export(saved_scan.working, run_name="broad_test")
    assert hashes(output) == before


def test_unrelated_files_in_scan_export_are_preserved(saved_scan):
    result = scans.load(15665, run_name="broad_test", results=saved_scan.results)
    output = result.export(saved_scan.working, run_name="broad_test")
    (output / "my_notes.txt").write_text("keep")
    before = hashes(output)
    with pytest.raises(ValueError, match="Unrelated"):
        result.export(saved_scan.working, run_name="broad_test", radius_pc=1000.)
    assert hashes(output) == before
