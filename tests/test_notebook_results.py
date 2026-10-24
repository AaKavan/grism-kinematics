"""Cold-kernel result recovery reads settings, never executes notebook code."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dingo import notebook_results as review


@pytest.fixture
def notebook(tmp_path):
    path = tmp_path / "review.ipynb"
    cells = [dict(id="three-file-settings", source=[
        "%matplotlib inline\n", "SOURCE_ID = 15665\n", "RUN_NAME = 'broad_test'\n",
        "RESULTS = None\n", "VELOCITY_RADIUS_PC = 3000.\n",
        "raise AssertionError('Must never execute any setup cell')\n"]),
        dict(id="candidate-selection", source=["CANDIDATE_ID = 9\n",
        "MCMC_INCLINATION_DEG = 30.\n", "MCMC_INCLINATION_SIGMA_DEG = 10.\n",
        "launch_unwanted_mcmc()\n"])]
    path.write_text(json.dumps(dict(cells=cells)))
    return path


def test_cold_kernel_uses_exact_saved_gaussian_selection_without_executing_cells(notebook, monkeypatch):
    from dingo import inclination_posterior, three_file
    calls = []
    result = object()
    def load(source, **kwargs):
        calls.append((source, kwargs))
        return result
    monkeypatch.setattr(inclination_posterior, "load", load)
    monkeypatch.setattr(inclination_posterior, "fit", lambda *a, **k: pytest.fail("Must not fit"))
    monkeypatch.setattr(three_file, "load", lambda *a, **k: pytest.fail("Wrong posterior route"))
    assert review.load_for_review({}, notebook_path=notebook) == (result, 3000.)
    assert calls == [(15665, dict(run_name="broad_test", inclination_deg=30.,
        inclination_sigma_deg=10., results=None))]


def test_runtime_selection_takes_precedence_and_preserves_candidate_guard(notebook, monkeypatch):
    from dingo import three_file
    calls = []
    def load(source, **kwargs):
        calls.append((source, kwargs))
        return "saved"
    monkeypatch.setattr(three_file, "load", load)
    state = dict(SOURCE_ID=6170, RUN_NAME="other", CANDIDATE_ID=2,
                 MCMC_INCLINATION_DEG=None, VELOCITY_RADIUS_PC=2000.)
    assert review.load_for_review(state, notebook_path=notebook) == ("saved", 2000.)
    assert calls == [(6170, dict(run_name="other", candidate_id=2, results=None))]


def test_existing_loader_does_not_need_the_notebook_file():
    result = SimpleNamespace()
    state = dict(load_selected_posterior=lambda: result, VELOCITY_RADIUS_PC=1500.)
    assert review.load_for_review(state, notebook_path="/missing/notebook") == (result, 1500.)


@pytest.mark.parametrize("exception", [ValueError("wrong prior"), RuntimeError("corrupt samples")])
def test_mismatched_or_invalid_saved_result_is_not_silenced(exception):
    def load():
        raise exception
    with pytest.raises(type(exception), match=str(exception)):
        review.load_for_review(dict(load_selected_posterior=load))


def test_missing_result_is_a_clear_noop(capsys):
    def load():
        raise FileNotFoundError("not ready")
    assert review.load_for_review(dict(load_selected_posterior=load)) == (None, None)
    assert "does not start a fit" in capsys.readouterr().out


def test_expressions_cannot_execute_during_recovery(notebook, tmp_path):
    marker = tmp_path / "must_not_exist"
    data = json.loads(notebook.read_text())
    data["cells"][0]["source"][1] = f"SOURCE_ID = open({str(marker)!r}, 'w')\n"
    notebook.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="SOURCE_ID is an expression"):
        review.load_for_review({}, notebook_path=notebook)
    assert not marker.exists()


def test_path_setting_is_read_as_data(notebook):
    data = json.loads(notebook.read_text())
    data["cells"][0]["source"][3] = "RESULTS = Path('/tmp/saved_data')\n"
    notebook.write_text(json.dumps(data))
    assert review.saved_settings(notebook)["RESULTS"] == Path("/tmp/saved_data")


def test_minimal_notebook_recovers_candidate_without_legacy_scan_settings(tmp_path, monkeypatch):
    from dingo import three_file, inclination_posterior
    path = tmp_path / "minimal.ipynb"
    path.write_text(json.dumps(dict(cells=[
        dict(id="three-file-settings", source=[
            "SOURCE_ID = 15665\nRUN_NAME = 'rotation_simple_v1'\n"]),
        dict(id="candidate-selection", source=[
            "CANDIDATE_ID = 3\nraise AssertionError('must not execute')\n"]),
    ])))
    calls = []
    def load(source_id, **kwargs):
        calls.append((source_id, kwargs))
        return "correct saved candidate"
    monkeypatch.setattr(three_file, "load", load)
    monkeypatch.setattr(three_file, "fit", lambda *a, **k: pytest.fail("Must not fit"))
    monkeypatch.setattr(inclination_posterior, "load", lambda *a, **k: pytest.fail("Wrong route"))
    assert review.load_for_review({}, notebook_path=path) == ("correct saved candidate", None)
    assert calls == [(15665, dict(run_name="rotation_simple_v1", candidate_id=3, results=None))]


def test_minimal_recovery_does_not_guess_an_unspecified_source(tmp_path):
    path = tmp_path / "incomplete.ipynb"
    path.write_text(json.dumps(dict(cells=[
        dict(id="candidate-selection", source=["CANDIDATE_ID = 3\n"]),
    ])))
    with pytest.raises(ValueError, match="saved settings missing"):
        review.saved_settings(path)
