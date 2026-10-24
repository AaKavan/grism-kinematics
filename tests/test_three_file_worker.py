"""The simple worker keeps input files/configs and output contracts isolated."""
from pathlib import Path

import numpy as np
import pytest
import yaml
from astropy.io import fits

from dingo.three_file_fit import build_config, _json, _choose_candidate, _guard_identity


@pytest.fixture
def source(tmp_path):
    data = tmp_path / "data"
    folder = data / "ID123"
    folder.mkdir(parents=True)
    for name in ("image", "R", "C"):
        fits.writeto(folder / f"{name}.fits", np.ones((21, 21)))
    original = {
        "summary": {"name": "ID123", "z": 1.2, "ra": 64.0, "dec": -24.0,
                    "filter": "F444W", "lambda_rest": 1.875, "r_fit": 10},
        "image": {"R": {"module": "A", "path": ["wrong/R.fits", 0]},
                  "C": {"module": "A", "path": ["wrong/C.fits", 0]}},
        "velocity": {"V_rot": 500},
        "mcmc": {"checkpoint": "old_native_checkpoint.npz"},
    }
    config = folder / "config_kinematics.yaml"
    config.write_text(yaml.safe_dump(original))
    return data, config, tmp_path / "isolated_run"


def test_worker_normalizes_and_never_edits_input_config(source):
    data, path, output = source
    before = path.read_bytes()
    cfg = build_config(123, data, path, output)
    assert path.read_bytes() == before
    assert "V_rot" not in cfg["velocity"]
    assert cfg["velocity"]["V_sini"] == 200.0  # placeholder; automatic scan replaces it
    assert cfg["mcmc"]["checkpoint"] == str(output / "checkpoint.npz")
    assert cfg["mcmc"]["priors"]["velocity.inc_v"]["type"] == "sin_i"
    for pupil in ("R", "C"):
        assert Path(cfg["image"][pupil]["path"][0]).is_absolute()
    assert cfg["mcmc"]["noise"]["fit_ln_f"] is True
    assert all(s["override"]["image.C.dx, image.C.dy"]["fit"] is False
               for s in cfg["fitting"])
    assert not output.exists()


def test_deprojected_velocity_and_inclination_are_distinct_sampled_coordinates(source):
    data, path, output = source
    cfg = build_config(123, data, path, output, velocity_parameter="V_rot",
                       inclination_deg=32.1, inclination_sigma_deg=5.)
    assert cfg["velocity"]["V_rot"] == 200.
    assert "V_sini" not in cfg["velocity"]
    assert "velocity.V_sini" not in cfg["mcmc"]["priors"]
    assert cfg["mcmc"]["priors"]["velocity.V_rot"] == dict(type="uniform", min=0., max=1500.)
    assert cfg["mcmc"]["init_scatter"]["velocity.V_rot"] == 2.
    for stage in cfg["fitting"]:
        assert "velocity.V_rot" in stage["override"]
        assert "velocity.V_sini" not in stage["override"]
    inc = cfg["mcmc"]["priors"]["velocity.inc_v"]
    assert inc["type"] == "gaussian"
    assert inc["sigma"] == pytest.approx(np.deg2rad(5.))
    assert inc["mu"] == pytest.approx(np.deg2rad(32.1))
    assert inc["min"] == pytest.approx(np.deg2rad(5.))
    assert inc["max"] == pytest.approx(np.deg2rad(85.))


def test_invalid_velocity_coordinate_fails_before_reading_inputs(source):
    data, path, output = source
    with pytest.raises(ValueError, match="velocity_parameter"):
        build_config(123, data, path, output, velocity_parameter="V")


def test_imaging_prior_is_optional_and_soft(source):
    data, path, output = source
    cfg = build_config(123, data, path, output, inclination_deg=40,
                       inclination_sigma_deg=10, device="cpu")
    prior = cfg["mcmc"]["priors"]["velocity.inc_v"]
    assert prior["type"] == "gaussian"
    assert prior["mu"] == pytest.approx(np.deg2rad(40))
    assert prior["min"] == pytest.approx(np.deg2rad(5))
    assert prior["max"] == pytest.approx(np.deg2rad(85))
    assert cfg["mcmc"]["vectorize"] is False
    assert cfg["fitting"][0]["override"]["velocity.inc_v"]["min"] == pytest.approx(np.deg2rad(20))


def test_inclination_prior_requires_both_parts(source):
    data, path, output = source
    with pytest.raises(ValueError, match="positive sigma"):
        build_config(123, data, path, output, inclination_deg=40)
    with pytest.raises(ValueError, match="needs an inclination mean"):
        build_config(123, data, path, output, inclination_sigma_deg=10)


def test_wrong_source_or_nonfinite_inputs_fail_before_fit(source):
    data, path, output = source
    with pytest.raises(ValueError, match="Source ID"):
        build_config(124, data, path, output)
    invalid = np.ones((21, 21)); invalid[3, 4] = np.nan
    fits.writeto(data / "ID123" / "R.fits", invalid, overwrite=True)
    with pytest.raises(ValueError, match="finite 2D"):
        build_config(123, data, path, output)


def test_json_export_is_valid_even_for_short_chain_diagnostics(tmp_path):
    path = tmp_path / "report.json"
    _json(path, {"tau": np.float64(np.nan), "array": np.array([1, 2]),
                 "pass": np.bool_(False)})
    import json
    assert json.loads(path.read_text()) == {"tau": None, "array": [1, 2], "pass": False}
    assert not path.with_suffix(".json.tmp").exists()


def test_candidate_selection_preserves_original_ids_and_allows_high_velocity():
    search = {"selected": 2, "trials": [
        {"candidate_id": 0, "loss": 10, "params": {"V_sini": 1000}},
        {"candidate_id": 1, "loss": None},
        {"candidate_id": 2, "loss": 5, "params": {"V_sini": 100}},
    ]}
    assert _choose_candidate(search)[0] == 2
    assert _choose_candidate(search, 0)[1]["params"]["V_sini"] == 1000
    with pytest.raises(ValueError, match="finite"):
        _choose_candidate(search, 1)
    with pytest.raises(ValueError, match="not in"):
        _choose_candidate(search, 3)
    with pytest.raises(ValueError, match="integer"):
        _choose_candidate(search, True)


def test_sampling_identity_rejects_candidate_change_without_overwriting(tmp_path):
    path = tmp_path / "identity.json"
    _guard_identity(path, {"chosen_candidate_id": 0}, description="sampling")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="OVERRIDE"):
        _guard_identity(path, {"chosen_candidate_id": 1}, description="sampling")
    assert path.read_bytes() == before


@pytest.mark.parametrize("with_imaging", [False, True])
@pytest.mark.parametrize("amplitude_name", ["V_sini", "V_rot"])
def test_search_exports_every_finite_candidate_without_sampling_and_reuses_for_changed_budget(source, monkeypatch, with_imaging, amplitude_name):
    """Exercise search -> user choice -> sampling handoff without doing a fit."""
    import json
    from dingo import fitting, startpoint, grism
    from dingo.three_file_fit import parser, run, _sha

    data, config_path, output = source
    calls = {"search": 0, "sample": []}

    def trial(amplitude, loss):
        params = {amplitude_name: amplitude, "inc_v": np.pi/6, "R_v": 2.0}
        return {"start": [amplitude, 0.0], "loss": loss, "params": params,
                "full_params": {f"velocity.{key}": value for key, value in params.items()}}

    trials = [trial(1000, 10), trial(200, np.nan), trial(100, 5)]

    def search(*args, **kwargs):
        calls["search"] += 1
        return trials[2]["params"], 5, trials

    class StopBeforeSampling(Exception):
        pass

    class ReviewOnlyFitter:
        def __init__(self, path, device=None):
            self.config = yaml.safe_load(Path(path).read_text())
            self.params = {}

        def set_params_dict(self, params):
            self.params = params

        def _reset_state(self):
            pass

        def loss(self):
            return 1.0

        def get_true_images(self):
            return np.ones((21, 21)), np.full((21, 21), 2.)

        def get_fitting_results(self):
            r = np.full((21, 21), self.params[f"velocity.{amplitude_name}"] / 100.)
            c = np.ones((21, 21))
            return r, c, np.zeros_like(r), np.zeros_like(c)

        def fit_MCMC(self):
            calls["sample"].append((self.params.copy(), self.config["mcmc"]))
            raise StopBeforeSampling

    monkeypatch.setattr(fitting, "KinematicsFitter", ReviewOnlyFitter)
    monkeypatch.setattr(startpoint, "find_start", search)
    monkeypatch.setattr(grism, "get_grism_cal_dir", lambda: str(data))
    base = ["--source", "123", "--data-dir", str(data), "--config", str(config_path),
            "--output", str(output), "--device", "cpu", "--velocity-parameter", amplitude_name]
    metadata = None
    if with_imaging:
        metadata = dict(source_id=123, inclination_deg=43.25, inclination_sigma_deg=11.75,
                        q0_range=[.05, .3], alignment_scatter_deg=10.,
                        imaging_directory="/source123/imaging", samples_sha256={"full_pixel": "original"})
        base += ["--inclination-deg", "43.25", "--inclination-sigma-deg", "11.75",
                 "--inclination-metadata-json", json.dumps(metadata)]
    first = parser().parse_args(base + ["--search-only"])
    saved = run(first)
    assert calls == {"search": 1, "sample": []}
    assert not (output / "identity.json").exists()
    assert not (output / "checkpoint.npz").exists()
    assert not (output / "report.json").exists()
    assert saved["schema"] == "dingo-three-file-candidates-v1"
    assert saved["selected"] == 2
    assert [t["candidate_id"] for t in saved["trials"]] == [0, 1, 2]
    assert saved["velocity_parameter"] == amplitude_name
    assert saved["trials"][0]["velocity_summary"]["V_inf_kms"] == pytest.approx(
        2000 if amplitude_name == "V_sini" else 1000)
    assert saved["candidates_sha256"] == _sha(output / "candidates.npz")
    expected_prior = yaml.safe_load((output/"search_config.yaml").read_text())["mcmc"]["priors"]["velocity.inc_v"]
    assert saved["inclination_prior"] == expected_prior
    assert saved["inclination_metadata"] == metadata
    assert saved["inclination_settings"] == dict(
        inclination_deg=43.25 if with_imaging else None,
        inclination_sigma_deg=11.75 if with_imaging else 10.)
    search_identity = json.loads((output/"search_identity.json").read_text())
    assert search_identity["inclination_prior"] == expected_prior
    assert search_identity["inclination_metadata"] == metadata
    with np.load(output / "candidates.npz", allow_pickle=False) as archive:
        assert archive["candidate_ids"].tolist() == [0, 2]
        assert archive["rectified_R"].shape == (2, 21, 21)
        assert np.all(archive["difference"][0] == 9)
        assert np.all(archive["difference"][1] == 0)
        assert archive["display_mask"].all()
    original_candidates = (output / "candidates.npz").read_bytes()
    new_budget = ["--warmup", "5", "--draws", "10", "--walkers", "20"]
    run(parser().parse_args(base + new_budget + ["--search-only"]))
    assert calls["search"] == 1
    assert original_candidates == (output / "candidates.npz").read_bytes()
    with pytest.raises(StopBeforeSampling):
        run(parser().parse_args(base + new_budget + ["--candidate", "0"]))
    assert calls["search"] == 1
    assert len(calls["sample"]) == 1
    params, settings = calls["sample"][0]
    assert params[f"velocity.{amplitude_name}"] == 1000
    assert (settings["nburn"], settings["nsteps"], settings["nwalkers"]) == (5, 10, 20)
    sampling_identity = json.loads((output / "identity.json").read_text())
    assert sampling_identity["chosen_candidate_id"] == 0
    assert sampling_identity["search_identity"]["inclination_prior"] == expected_prior
    assert sampling_identity["search_identity"]["inclination_metadata"] == metadata
    assert sampling_identity["config"]["mcmc"]["priors"]["velocity.inc_v"] == expected_prior
    assert settings["priors"]["velocity.inc_v"] == expected_prior
    before_config = (output / "run_config.yaml").read_bytes()
    with pytest.raises(ValueError, match="chosen candidate"):
        run(parser().parse_args(base + new_budget + ["--candidate", "2"]))
    assert len(calls["sample"]) == 1
    assert before_config == (output / "run_config.yaml").read_bytes()
    if with_imaging:
        # Even an unchanged Gaussian cannot silently switch its imaging source
        # samples: source provenance is part of the search/sampling identity.
        changed = dict(metadata, samples_sha256={"full_pixel": "different"})
        alternative = base.copy()
        alternative[alternative.index("--inclination-metadata-json")+1] = json.dumps(changed)
        saved_files = {name: (output/name).read_bytes() for name in
                       ("search_identity.json", "identity.json", "start_search.json", "search_config.yaml")}
        with pytest.raises(ValueError, match="search has different inputs"):
            run(parser().parse_args(alternative + new_budget + ["--candidate", "0"]))
        assert len(calls["sample"]) == 1
        assert saved_files == {name: (output/name).read_bytes() for name in saved_files}


@pytest.mark.parametrize("change", [dict(source_id=999), dict(inclination_deg=50.),
                                     dict(inclination_sigma_deg=3.)])
def test_worker_rejects_mismatched_imaging_metadata_before_creating_output(source, monkeypatch, change):
    import json
    from dingo import fitting, startpoint
    from dingo.three_file_fit import parser, run
    data, config_path, output = source
    metadata = dict(source_id=123, inclination_deg=40., inclination_sigma_deg=10.) | change
    monkeypatch.setattr(fitting, "KinematicsFitter", lambda *a, **k: pytest.fail("No fitter may start"))
    monkeypatch.setattr(startpoint, "find_start", lambda *a, **k: pytest.fail("No search may start"))
    args = parser().parse_args(["--source", "123", "--data-dir", str(data),
        "--config", str(config_path), "--output", str(output), "--device", "cpu",
        "--inclination-deg", "40", "--inclination-sigma-deg", "10",
        "--inclination-metadata-json", json.dumps(metadata), "--search-only"])
    with pytest.raises(ValueError, match="different source|disagrees"):
        run(args)
    assert not output.exists()
