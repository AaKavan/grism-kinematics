"""Saved-only plotting and fit dispatch for the main and archived notebooks."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def native_notebook():
    path = (Path(__file__).resolve().parents[1] /
            "notebooks/archive/rotation_only_native_2026-10-19.ipynb")
    if not path.exists():
        pytest.skip("Optional local historical notebook; its saved outputs are not versioned")
    return json.loads(path.read_text())


@pytest.fixture
def imaging_cell(native_notebook):
    cell = next(cell for cell in native_notebook["cells"] if cell.get("id") == "imaging-preparation")
    return compile("".join(cell["source"]), "rotation_only:imaging-preparation", "exec")


@pytest.fixture
def grism_cell(native_notebook):
    cell = next(cell for cell in native_notebook["cells"] if cell.get("id") == "ecc461e4")
    return compile("".join(cell["source"]), "rotation_only:grism", "exec")


def namespace(events, *, run_mcmc=True, run_imaging=True, inclination="imaging", ready=True,
              override=True):
    path = Path("/results/ID15665/imaging_runs/pysersic_v1")

    def require_ready():
        events.append("checked readiness")
        if not ready:
            raise RuntimeError("Imaging sampling checks failed")
        return path

    result = SimpleNamespace(
        path=path,
        summary=lambda: "summary table",
        figure=lambda: "image panels",
        require_ready=require_ready,
    )

    def prepare(source_id, **kwargs):
        events.append(("prepare", source_id, kwargs))
        return result

    return dict(
        SOURCE_ID=15665, RUN_MCMC=run_mcmc, RUN_IMAGING=run_imaging,
        OVERRIDE=override,
        INCLINATION=inclination, IMAGING=None, IMAGING_RUN_NAME="pysersic_v1",
        BUNDLE=None, RESULTS=None, DEVICE="cuda", IMAGING_WARMUP=1000,
        IMAGING_DRAWS=2000, prepare_imaging=prepare,
        display=lambda obj: events.append(obj),
        plt=SimpleNamespace(close=lambda obj: events.append("closed figure")),
    )


@pytest.mark.parametrize("run_imaging", [True, False])
@pytest.mark.parametrize("override", [True, False])
def test_inspect_saved_grism_does_not_launch_or_require_imaging(imaging_cell, run_imaging, override):
    events = []
    scope = namespace(events, run_mcmc=False, run_imaging=run_imaging, ready=False,
                      override=override)
    exec(imaging_cell, scope)
    assert events == []
    assert scope["IMAGING"] is None


def test_explicit_isotropic_prior_skips_imaging(imaging_cell):
    events = []
    exec(imaging_cell, namespace(events, inclination="isotropic"))
    assert events == []


@pytest.mark.parametrize("run_imaging", [True, False])
@pytest.mark.parametrize("override", [True, False])
def test_imaging_path_passed_to_grism_only_after_review(imaging_cell, run_imaging, override):
    events = []
    scope = namespace(events, run_imaging=run_imaging, override=override)
    exec(imaging_cell, scope)
    assert events[0][0:2] == ("prepare", 15665)
    assert events[0][2]["run"] is run_imaging
    assert events[0][2]["override"] is override
    assert events[0][2]["warmup"] == 1000
    assert events[0][2]["samples"] == 2000
    assert events[1:] == ["summary table", "image panels", "closed figure", "checked readiness"]
    assert scope["IMAGING"] == scope["imaging_result"].path


def test_failed_imaging_still_shows_summary_and_images(imaging_cell):
    events = []
    scope = namespace(events, ready=False)
    with pytest.raises(RuntimeError, match="sampling checks failed"):
        exec(imaging_cell, scope)
    assert events[1:] == ["summary table", "image panels", "closed figure", "checked readiness"]
    assert scope["IMAGING"] is None


@pytest.mark.parametrize("run_mcmc", [True, False])
@pytest.mark.parametrize("override", [True, False])
def test_grism_override_only_reaches_fitting_path(grism_cell, run_mcmc, override):
    events = []
    result = SimpleNamespace(status="test result")

    def record(name):
        def call(source_id, **kwargs):
            events.append((name, source_id, kwargs))
            return result
        return call

    scope = namespace(events, run_mcmc=run_mcmc, override=override)
    scope.update(RUN_NAME="rotation_v1", WARMUP=1000, DRAWS=2000, CHAINS=4,
                 fit=record("fit"), load=record("load"))
    exec(grism_cell, scope)
    assert scope["result"] is result
    assert len(events) == 1
    name, source_id, kwargs = events[0]
    assert source_id == 15665
    if run_mcmc:
        assert name == "fit"
        assert kwargs["override"] is override
        assert kwargs["resume"] is True
    else:
        assert name == "load"
        assert kwargs == {"run_name": "rotation_v1", "results": None}


@pytest.mark.parametrize("run_mcmc", [True, False])
def test_failed_grism_rerun_clears_previous_result(grism_cell, run_mcmc):
    def fail(*args, **kwargs):
        raise RuntimeError("This run failed")

    scope = namespace([], run_mcmc=run_mcmc)
    scope.update(RUN_NAME="rotation_v1", WARMUP=1000, DRAWS=2000, CHAINS=4,
                 fit=fail, load=fail, result=SimpleNamespace(status="older source result"))
    with pytest.raises(RuntimeError, match="This run failed"):
        exec(grism_cell, scope)
    assert scope["result"] is None


@pytest.fixture
def main_cells():
    notebook = json.loads(
        (Path(__file__).resolve().parents[1] / "notebooks" / "legacy" /
         "rotation_only.ipynb").read_text()
    )
    return {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]
            if cell["cell_type"] == "code"}


def stripped_imports(source):
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    return compile(tree, "rotation_only:mocked-imports", "exec")


def selection_code(main_cells, candidate_id):
    """Run the actual cell as if the user had edited its candidate setting."""
    tree = ast.parse(main_cells["candidate-selection"])
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "CANDIDATE_ID"
                for target in node.targets):
            node.value = ast.Constant(candidate_id)
    return compile(ast.fix_missing_locations(tree), "rotation_only:candidate-selection", "exec")


def three_file_scope(events, *, run=True, override=False):
    saved_prior = dict(inclination_deg=42., inclination_sigma_deg=14.,
                       inclination_metadata=dict(source_id=6170, note="saved search prior"))

    def export(working_dir, **kwargs):
        events.append(("export", working_dir, kwargs))
        return working_dir / "ID6170"

    result = SimpleNamespace(status="three-file result", report={"status": "complete"},
                             inclination_settings=lambda: saved_prior, export=export)

    def record(name):
        def call(source_id, **kwargs):
            events.append((name, source_id, kwargs))
            return result
        return call

    return dict(SOURCE_ID=6170, RUN_NAME="test_simple_v1", RUN_SEARCH=run, RUN_MCMC=run,
                OVERRIDE=override, WARMUP=1000, DRAWS=4000, WALKERS=64, VELOCITY_PARAMETER="V_rot",
                imaging_prior_record=dict(source_id=6170, inclination_deg=35.,
                    inclination_sigma_deg=12., kind="imaging_gaussian_approximation"),
                VELOCITY_RADIUS_PC=2000., WORKING_DIR=Path("/test/working"),
                three_file=SimpleNamespace(fit=record("fit"), load=record("load"),
                    search=record("search"), load_candidates=record("load_candidates"),
                    source_number=lambda value: int(str(value).removeprefix("ID"))))


@pytest.mark.parametrize("run", [True, False])
@pytest.mark.parametrize("override", [True, False])
def test_search_dispatch_and_read_only_loading(main_cells, run, override):
    events = []
    scope = three_file_scope(events, run=run, override=override)
    scope["result"] = "stale posterior"
    exec(main_cells["candidate-search"], scope)
    assert scope["result"] is None
    assert len(events) == 1
    name, source_id, kwargs = events[0]
    assert source_id == 6170
    if run:
        assert name == "search"
        assert kwargs == dict(run_name="test_simple_v1", override=override,
                              inclination_deg=35., inclination_sigma_deg=12.,
                              inclination_metadata=scope["imaging_prior_record"],
                              search_method="broad", n_candidates=8, profile_radius_pc=2000.,
                              velocity_parameter="V_rot")
    else:
        assert name == "load_candidates"
        assert kwargs == dict(run_name="test_simple_v1")


@pytest.mark.parametrize("prior", [None, {"source_id": 15665}])
def test_new_search_requires_this_sources_reviewed_imaging(main_cells, prior):
    events = []
    scope = three_file_scope(events)
    scope.update(imaging_prior_record=prior, result="stale", candidates="stale")
    with pytest.raises(RuntimeError, match="PySersic block for this source"):
        exec(main_cells["candidate-search"], scope)
    assert events == []
    assert scope["result"] is None
    assert scope["candidates"] is None


def test_saved_search_load_ignores_missing_current_imaging(main_cells):
    events = []
    scope = three_file_scope(events, run=False, override=True)
    scope.pop("imaging_prior_record")
    exec(main_cells["candidate-search"], scope)
    assert events == [("load_candidates", 6170, {"run_name": "test_simple_v1"})]


@pytest.mark.parametrize("run", [True, False])
@pytest.mark.parametrize("override", [True, False])
def test_selected_mcmc_reuses_saved_prior_without_overriding_search(main_cells, run, override):
    events = []
    scope = three_file_scope(events, run=run, override=override)
    exec(selection_code(main_cells, 1), scope)
    if run:
        assert events[0] == ("load_candidates", 6170, {"run_name": "test_simple_v1"})
        assert events[1] == ("fit", 6170, dict(run_name="test_simple_v1", candidate_id=1,
            warmup=1000, samples=4000, walkers=64, override=False, velocity_parameter="V_rot",
            **scope["saved_prior"]))
        assert scope["saved_prior"]["inclination_deg"] == 42.
    else:
        assert events[0] == ("load", 6170, dict(run_name="test_simple_v1", candidate_id=1))
    assert events[-1] == ("export", scope["WORKING_DIR"], {"radius_pc": 2000.})
    assert len(events) == (3 if run else 2)
    assert scope["final_directory"] == scope["WORKING_DIR"] / "ID6170"


@pytest.mark.parametrize("run", [True, False])
def test_unselected_candidate_never_runs_or_loads_mcmc(main_cells, run):
    events = []
    scope = three_file_scope(events, run=run, override=True)
    scope["result"] = "stale posterior"
    exec(selection_code(main_cells, None), scope)
    assert events == []
    assert scope["result"] is None


def test_interrupted_mcmc_does_not_replace_final_export(main_cells):
    scope = three_file_scope([])
    scope["three_file"].fit = lambda *args, **kwargs: SimpleNamespace(
        status="interrupted", report={"status": "interrupted"},
        export=lambda *a, **k: pytest.fail("Interrupted run must not replace final results"))
    exec(selection_code(main_cells, 1), scope)
    assert "final_directory" not in scope


@pytest.mark.parametrize("run", [True, False])
def test_three_file_failed_dispatch_clears_stale_result(main_cells, run):
    def fail(*args, **kwargs):
        raise RuntimeError("Unavailable three-file run")

    scope = three_file_scope([], run=run)
    scope.update(three_file=SimpleNamespace(fit=fail, load=fail, load_candidates=fail), result="stale")
    with pytest.raises(RuntimeError, match="Unavailable three-file run"):
        exec(selection_code(main_cells, 1), scope)
    assert scope["result"] is None


def test_loader_uses_current_source_instead_of_stale_candidate_object(main_cells):
    events = []
    scope = three_file_scope(events, run=False)
    exec(selection_code(main_cells, 1), scope)
    events.clear()
    scope.update(SOURCE_ID=15665, RUN_NAME="changed_source", CANDIDATE_ID=3,
                 candidates="stale source", result="stale posterior")
    scope["load_selected_posterior"]()
    assert events == [("load", 15665, dict(run_name="changed_source", candidate_id=3))]


def test_candidate_display_includes_every_saved_id_and_both_velocities(main_cells):
    import numpy as np
    import pandas as pd
    events = []
    table = pd.DataFrame(dict(Vrot_at_radius_kms=[200., 350., 600.],
                              Vsin_i_at_radius_kms=[100., 350.*np.sin(np.deg2rad(45.)),
                                                   600.*np.sin(np.deg2rad(60.))],
                              inclination_deg=[30., 45., 60.], radius_pc=[2000.]*3),
                         index=pd.Index([0, 4, 9], name="candidate_id"))

    def figure(**kwargs):
        events.append(("candidate", kwargs))
        return "candidate figure"

    candidates = SimpleNamespace(candidate_ids=[0, 4, 9], velocity_parameter="V_sini",
        input_figure=lambda: "input figure", table=lambda **kwargs: table,
        figure=figure, search_notes=lambda: "saved search notes")

    def load(source_id, **kwargs):
        events.append(("load", source_id, kwargs))
        return candidates

    scope = three_file_scope(events, run=True, override=True)
    scope.update(np=np, pd=pd, display=lambda value: events.append(("display", value)),
                 plt=SimpleNamespace(close=lambda value: None),
                 three_file=SimpleNamespace(load_candidates=load,
                     search=lambda *a, **k: pytest.fail("Display must never search"),
                     fit=lambda *a, **k: pytest.fail("Display must never sample")))
    exec(main_cells["candidate-residuals"], scope)
    assert events[0] == ("load", 6170, {"run_name": "test_simple_v1"})
    assert [event[1]["candidate_ids"] for event in events if event[0] == "candidate"] == [[0], [4], [9]]
    shown = next(event[1] for event in events if event[0] == "display" and isinstance(event[1], pd.DataFrame))
    assert list(shown.index) == [0, 4, 9]
    assert shown.columns[0] == "Vsin_i_at_radius_kms"
    assert "Vrot_at_radius_kms" not in shown
    np.testing.assert_allclose(shown.Vsin_i_at_radius_kms,
                               [100., 350.*np.sin(np.deg2rad(45.)), 600.*np.sin(np.deg2rad(60.))])


def test_mcmc_in_fresh_kernel_explains_missing_settings_without_installing_broken_loader(main_cells):
    scope = {}
    with pytest.raises(RuntimeError, match="Run the settings cell.*Results cell directly"):
        exec(selection_code(main_cells, 6), scope)
    assert scope["result"] is None
    # Results can still recover the saved selection, rather than calling a
    # closure whose required SOURCE_ID/RUN_NAME globals do not exist.
    assert "load_selected_posterior" not in scope


@pytest.mark.parametrize("run_search", [True, False])
def test_imaging_review_only_starts_fits_for_a_new_search(main_cells, run_search):
    events = []
    result = SimpleNamespace(source_id=6170)
    record = dict(source_id=6170, inclination_deg=43.2, inclination_sigma_deg=5.)

    def review(source_id, **kwargs):
        events.append(("review", source_id, kwargs))
        return result

    def prior(value, **kwargs):
        events.append(("prior", value, kwargs))
        return record

    scope = dict(SOURCE_ID=6170, RUN_SEARCH=run_search, OVERRIDE=True,
                 INCLINATION_SIGMA_DEG=5.,
                 imaging_result="stale", imaging_prior_record="stale",
                 review_imaging=review, inclination_prior=prior)
    exec(stripped_imports(main_cells["pysersic-review"]), scope)
    assert events[0] == ("review", 6170, dict(run=run_search, override=True))
    if run_search:
        assert events[1] == ("prior", result, {"total_sigma_deg": 5.})
        assert scope["imaging_prior_record"] is record
    else:
        assert len(events) == 1
        assert scope["imaging_prior_record"] is None


def test_saved_review_without_imaging_does_not_block_rc(main_cells, capsys):
    def missing(*args, **kwargs):
        raise FileNotFoundError("No saved imaging fit")
    scope = dict(SOURCE_ID=6170, RUN_SEARCH=False, OVERRIDE=True,
                 imaging_result="stale", imaging_prior_record="stale", review_imaging=missing)
    exec(stripped_imports(main_cells["pysersic-review"]), scope)
    assert scope["imaging_result"] is None
    assert scope["imaging_prior_record"] is None
    assert "saved search keeps its original prior" in capsys.readouterr().out


@pytest.mark.parametrize("missing", [True, False])
def test_new_search_imaging_failure_does_not_leave_a_stale_prior(main_cells, missing):
    def review(*args, **kwargs):
        if missing:
            raise FileNotFoundError("Missing fit")
        return SimpleNamespace(source_id=6170)

    def prior(*args, **kwargs):
        raise RuntimeError("Imaging fit not ready")

    scope = dict(SOURCE_ID=6170, RUN_SEARCH=True, OVERRIDE=False,
                 INCLINATION_SIGMA_DEG=5.,
                 imaging_result="stale", imaging_prior_record="stale",
                 review_imaging=review, inclination_prior=prior)
    with pytest.raises((FileNotFoundError, RuntimeError)):
        exec(stripped_imports(main_cells["pysersic-review"]), scope)
    assert scope["imaging_prior_record"] is None


@pytest.mark.parametrize("cell_id,methods", [
    ("fit-results", ("traces", "corner")),
    ("rotation-curve", ("rotation_curve",)),
])
@pytest.mark.parametrize("saved", [True, False])
def test_every_review_cell_recovers_after_restart_without_fitting(
        main_cells, monkeypatch, tmp_path, cell_id, methods, saved):
    from dingo import inclination_posterior, notebook_results, three_file, three_file_center_plot
    import importlib
    import IPython.display
    import matplotlib.pyplot as plt

    # Keep the loader/fit sentinels below installed while exercising the cell.
    monkeypatch.setattr(importlib, "reload", lambda module: module)

    notebook_path = tmp_path / "rotation_only.ipynb"
    notebook_path.write_text(json.dumps({"cells": [
        {"id": "three-file-settings", "source": [
            "SOURCE_ID = 6170\nRUN_NAME = 'saved_run'\nVELOCITY_RADIUS_PC = 2000.0\n"
            "RUN_MCMC = True\nOVERRIDE = True\n"]},
        {"id": "candidate-selection", "source": ["CANDIDATE_ID = 9\n"]},
    ]}))
    actual_load = notebook_results.load_for_review
    monkeypatch.setattr(notebook_results, "load_for_review",
                        lambda scope: actual_load(scope, notebook_path=notebook_path))
    events = []

    def forbidden(*args, **kwargs):
        pytest.fail("Review cells must not fit, delete, or choose a different posterior")

    def plot(name):
        def call(**kwargs):
            events.append(("plot", name, kwargs))
            return name
        return call

    result = SimpleNamespace(status="saved result", report={"status": "complete"}, velocity_parameter="V_rot",
        diagnostics=lambda: "diagnostics", velocity_summary=lambda **kwargs: "velocities",
        **{method: plot(method) for method in methods})

    def load(source_id, **kwargs):
        events.append(("load", source_id, kwargs))
        if not saved:
            raise FileNotFoundError("No saved posterior")
        return result

    def center(value, **kwargs):
        assert value is result
        events.append(("plot", "center", kwargs))
        return "center"

    monkeypatch.setattr(three_file, "load", load)
    monkeypatch.setattr(three_file, "fit", forbidden)
    monkeypatch.setattr(three_file, "search", forbidden)
    monkeypatch.setattr(inclination_posterior, "load", forbidden)
    monkeypatch.setattr(inclination_posterior, "fit", forbidden)
    monkeypatch.setattr(three_file_center_plot, "review_center", center)
    monkeypatch.setattr(IPython.display, "display", lambda value: events.append(("display", value)))
    monkeypatch.setattr(plt, "close", lambda value: events.append(("close", value)))
    scope = {}
    exec(main_cells[cell_id], scope)
    assert events[0] == ("load", 6170, {"run_name": "saved_run", "candidate_id": 9, "results": None})
    assert scope["result"] is (result if saved else None)
    if saved:
        for method in methods:
            assert ("plot", method, {}) in events
        if cell_id == "fit-results":
            assert ("plot", "center", {"radius_pc": 2000., "half_width": 15.}) in events
        assert events[-2:] == [("display", methods[-1]), ("close", methods[-1])]
    else:
        assert len(events) == 1


@pytest.mark.parametrize("cell_id", ["fit-results", "rotation-curve"])
def test_review_never_displays_stale_result_when_selection_is_missing(main_cells, cell_id):
    from dingo.notebook_results import load_for_review
    scope = dict(load_for_review=load_for_review, result="stale",
                 load_selected_posterior=lambda: None,
                 reload=lambda module: module, three_file=SimpleNamespace(),
                 three_file_plot=SimpleNamespace(), three_file_center_plot=SimpleNamespace(),
                 display=lambda value: pytest.fail("Stale result must not display"))
    exec(stripped_imports(main_cells[cell_id]), scope)
    assert scope["result"] is None


def test_saved_posterior_mismatch_is_not_silenced(main_cells):
    from dingo.notebook_results import load_for_review
    def mismatch():
        raise ValueError("Selected candidate does not match saved posterior")
    scope = dict(load_for_review=load_for_review, load_selected_posterior=mismatch, result="stale")
    with pytest.raises(ValueError, match="Selected candidate"):
        exec(stripped_imports(main_cells["rotation-curve"]), scope)
    assert scope["result"] is None


def test_notebook_selects_cpu_for_display_before_importing_dingo(main_cells, monkeypatch):
    import os
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    prefix = main_cells["three-file-settings"].split("import numpy", 1)[0]
    assert "from dingo" not in prefix
    exec("\n".join(line for line in prefix.splitlines() if not line.startswith("%")), {})
    assert os.environ["JAX_PLATFORMS"] == "cpu"
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
