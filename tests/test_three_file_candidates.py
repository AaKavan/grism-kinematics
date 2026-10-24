"""Saved visual choices stay attached to their pixels, IDs and posterior draws."""

import hashlib
import json

import numpy as np
import pytest
import yaml

from dingo import three_file
from dingo.three_file_plot import physical_scale


@pytest.fixture
def candidate_run(tmp_path, monkeypatch):
    repo = tmp_path/"repo"
    source = repo/"data"/"ID6170"
    source.mkdir(parents=True)
    for name in ("image.fits", "R.fits", "C.fits", "config_kinematics.yaml"):
        (source/name).write_bytes(b"API fixture; no fitting is permitted")
    monkeypatch.setattr(three_file, "REPO", repo)
    results = tmp_path/"results"
    path = results/"three_file_sources"/"ID6170"/"visual_test"
    path.mkdir(parents=True)
    # The finite candidates are saved in a deliberately nonsequential order.
    # Candidate 1 failed; visible row zero must still refer to original ID 2.
    r = np.arange(60., dtype=float).reshape(2, 5, 6)
    c = r.copy()
    c[0, 1, 1] += 2
    c[1, 2, 2] -= 10
    arrays = dict(candidate_ids=np.array([2, 0]), rectified_R=r,
                  rectified_C=c, difference=r-c, display_mask=np.ones((5, 6), bool))
    report = dict(schema="dingo-three-file-candidates-v1", source_id=6170,
        redshift=1.1147, pixel_scale_arcsec=.063, r_fit_pixels=40, selected=2,
        inclination_prior=dict(type="sin_i", min=float(np.deg2rad(5.)),
                               max=float(np.deg2rad(85.))),
        search_settings=dict(amp_step=100., n_keep=7, n_theta=10,
                             optimizer_steps=[41, 23], max_velocity=1500.),
        trials=[dict(loss=10., params=dict(V_sini=400., inc_v=np.pi/6, R_v=4.)),
                dict(loss=None, params={}),
                dict(loss=3., params=dict(V_sini=150., inc_v=np.pi/6, R_v=2.))])

    def write():
        np.savez(path/"candidates.npz", **arrays)
        report["candidates_sha256"] = hashlib.sha256((path/"candidates.npz").read_bytes()).hexdigest()
        (path/"start_search.json").write_text(json.dumps(report))

    write()
    return path, results, arrays, report, write


def test_candidate_ids_and_velocities_belong_to_actual_saved_rows(candidate_run):
    path, _, _, report, _ = candidate_run
    result = three_file.CandidateResults(path, 6170)
    assert result.candidate_ids == [2, 0]
    radius = 2*physical_scale(report)["pc_per_pixel"]
    table = result.table(radius_pc=radius)
    assert table.index.tolist() == [2, 0]
    assert table.loc[2, "Vinf_kms"] == pytest.approx(300.)
    assert table.loc[2, "Vinf_sin_i_kms"] == 150.
    assert table.loc[2, "Vrot_at_radius_kms"] == pytest.approx(150.)
    assert table.loc[2, "Vsin_i_at_radius_kms"] == pytest.approx(75.)
    assert table.loc[0, "Vinf_kms"] == pytest.approx(800.)
    assert table.loc[0, "Vrot_at_radius_kms"] == pytest.approx((2/np.pi)*800*np.arctan(.5))
    assert table.loc[0, "Vsin_i_at_radius_kms"] == pytest.approx((2/np.pi)*400*np.arctan(.5))
    assert table.loc[0, "residual_sum_squares"] == 10.
    assert table.loc[2, "radius_pc"] == radius


def test_candidate_source_and_hash_checked_before_display(candidate_run):
    path, _, _, _, _ = candidate_run
    with pytest.raises(ValueError, match="source"):
        three_file.CandidateResults(path, 15665)
    with (path/"candidates.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        three_file.CandidateResults(path, 6170)


@pytest.mark.parametrize("ids", [np.array([2, 2]), np.array([2, 3]),
                                     np.array([2, -1]), np.array([2., 0.])])
def test_candidate_ids_cannot_be_duplicated_missing_or_retyped(candidate_run, ids):
    path, _, arrays, _, write = candidate_run
    arrays["candidate_ids"] = ids
    write()
    with pytest.raises(ValueError, match="candidate IDs"):
        three_file.CandidateResults(path, 6170)


def test_hash_valid_but_wrong_residual_is_rejected(candidate_run):
    path, _, arrays, _, write = candidate_run
    arrays["difference"] *= -1
    write()
    with pytest.raises(ValueError, match="arrays disagree"):
        three_file.CandidateResults(path, 6170)


def test_search_dispatches_search_only_and_returns_review_without_sampler(candidate_run, monkeypatch):
    _, results, _, _, _ = candidate_run
    dispatched = []

    def worker(arguments, log, *, device):
        dispatched.append((arguments, log, device))

    monkeypatch.setattr(three_file, "run_command", worker)
    result = three_file.search(6170, run_name="visual_test", results=results,
        device="cpu", velocity_step=100., n_candidates=7, n_angles=10,
        optimizer_steps=(41, 23))
    assert isinstance(result, three_file.CandidateResults)
    assert len(dispatched) == 1
    args, log, device = dispatched[0]
    assert "--search-only" in args
    assert "--candidate" not in args
    assert args[args.index("--n-keep")+1] == "7"
    assert args[args.index("--amp-step")+1] == "100.0"
    assert device == "cpu" and log.name == "run.log"


def _write_minimal_posterior_report(path, candidate_id=0):
    report = dict(schema="dingo-three-file-v1", source_id=6170,
                  intrinsic_dispersion_kms=0., chosen_candidate_id=candidate_id)
    (path/"report.json").write_text(json.dumps(report))


def test_selected_fit_reuses_search_settings_and_preserves_selection_text(candidate_run, monkeypatch):
    path, results, _, _, _ = candidate_run
    dispatched = []
    _write_minimal_posterior_report(path)
    monkeypatch.setattr(three_file, "run_command",
        lambda arguments, log, **kwargs: dispatched.append(arguments))
    notes = "Residual structure is weak; inspect the projected velocity too."
    result = three_file.fit(6170, run_name="visual_test", results=results,
        candidate_id=0, selection_notes=notes, device="cpu", warmup=10,
        samples=20, walkers=24, velocity_step=25., n_candidates=4,
        n_angles=16, optimizer_steps=(2500, 1500))
    assert isinstance(result, three_file.ThreeFileResults)
    assert len(dispatched) == 1
    args = dispatched[0]
    for flag, expected in (("--candidate", "0"), ("--selection-notes", notes),
            ("--amp-step", "100.0"), ("--n-keep", "7"), ("--n-theta", "10"),
            ("--warmup", "10"), ("--draws", "20")):
        assert args[args.index(flag)+1] == expected
    start = args.index("--optimizer-steps")+1
    assert args[start:start+2] == ["41", "23"]
    assert "--search-only" not in args


def test_cannot_start_v_sampling_from_a_vsini_search(candidate_run, monkeypatch):
    _, results, _, _, _ = candidate_run
    monkeypatch.setattr(three_file, "run_command", lambda *a, **k: pytest.fail("Wrong coordinates must not sample"))
    with pytest.raises(ValueError, match="different velocity parameter"):
        three_file.fit(6170, run_name="visual_test", results=results,
                       candidate_id=0, velocity_parameter="V_rot", device="cpu")


def test_v_search_handoff_uses_v_and_preserves_inclination_prior(candidate_run, monkeypatch):
    path, results, _, report, write = candidate_run
    for trial in report["trials"]:
        if "V_sini" in trial["params"]:
            trial["params"]["V_rot"] = trial["params"].pop("V_sini")/np.sin(trial["params"]["inc_v"])
    report["velocity_parameter"] = "V_rot"
    _saved_imaging_prior(report, mean=32.1, sigma=5.)
    write()
    _write_minimal_posterior_report(path)
    dispatched = []
    monkeypatch.setattr(three_file, "run_command", lambda args, *a, **k: dispatched.append(args))
    three_file.fit(6170, run_name="visual_test", results=results, candidate_id=0, device="cpu")
    args = dispatched[0]
    assert args[args.index("--velocity-parameter")+1] == "V_rot"
    assert float(args[args.index("--inclination-sigma-deg")+1]) == 5.


def _saved_imaging_prior(report, mean=42.1234567890123, sigma=10.2345678901234):
    metadata = dict(source_id=6170, inclination_deg=mean, inclination_sigma_deg=sigma,
                    imaging_directory="/saved/source-specific/imaging",
                    samples_sha256={"full_pixel": "samples-digest"},
                    q0_range=[.05, .3], alignment_scatter_deg=10.)
    report.update(inclination_prior=dict(type="gaussian", mu=float(np.deg2rad(mean)),
                    sigma=float(np.deg2rad(sigma)), min=float(np.deg2rad(5.)), max=float(np.deg2rad(85.))),
                  inclination_settings=dict(inclination_deg=mean, inclination_sigma_deg=sigma),
                  inclination_metadata=metadata)
    return metadata


@pytest.mark.parametrize("changes", [dict(source_id=15665), dict(inclination_deg=55.),
                                      dict(inclination_sigma_deg=1.)])
def test_search_rejects_wrong_imaging_source_or_prior_values_before_worker(candidate_run, monkeypatch, changes):
    path, results, _, _, _ = candidate_run
    metadata = dict(source_id=6170, inclination_deg=40., inclination_sigma_deg=10.) | changes
    before = {p.name: p.read_bytes() for p in path.iterdir() if p.is_file()}
    monkeypatch.setattr(three_file, "run_command",
                        lambda *args, **kwargs: pytest.fail("Invalid prior must not launch a worker"))
    with pytest.raises(ValueError, match="different source|disagrees"):
        three_file.search(6170, run_name="visual_test", results=results, device="cpu",
                          inclination_deg=40., inclination_sigma_deg=10., inclination_metadata=metadata)
    assert before == {p.name: p.read_bytes() for p in path.iterdir() if p.is_file()}


def test_selected_fit_inherits_exact_search_prior_and_provenance(candidate_run, monkeypatch):
    path, results, _, report, write = candidate_run
    metadata = _saved_imaging_prior(report)
    write()
    _write_minimal_posterior_report(path)
    dispatched = []
    monkeypatch.setattr(three_file, "run_command", lambda args, *a, **k: dispatched.append(args))
    # Local notebook defaults can differ after the search. Omitted mean means
    # reuse; even a changed default sigma must not redefine this selected fit.
    three_file.fit(6170, run_name="visual_test", results=results, candidate_id=0,
                   inclination_sigma_deg=99., device="cpu")
    args = dispatched[0]
    assert float(args[args.index("--inclination-deg")+1]) == metadata["inclination_deg"]
    assert float(args[args.index("--inclination-sigma-deg")+1]) == metadata["inclination_sigma_deg"]
    assert json.loads(args[args.index("--inclination-metadata-json")+1]) == metadata


@pytest.mark.parametrize("change", ["mean", "sigma", "provenance"])
def test_selected_fit_cannot_redefine_saved_imaging_prior(candidate_run, monkeypatch, change):
    _, results, _, report, write = candidate_run
    metadata = _saved_imaging_prior(report)
    write()
    kwargs = dict(inclination_deg=metadata["inclination_deg"],
                  inclination_sigma_deg=metadata["inclination_sigma_deg"], inclination_metadata=metadata.copy())
    if change == "provenance":
        kwargs["inclination_metadata"]["imaging_directory"] = "/a/different/imaging/run"
    else:
        kwargs["inclination_deg" if change == "mean" else "inclination_sigma_deg"] += 1.
    monkeypatch.setattr(three_file, "run_command",
                        lambda *args, **kwargs: pytest.fail("Changed prior must not launch sampling"))
    with pytest.raises(ValueError, match="saved inclination prior|provenance differs"):
        three_file.fit(6170, run_name="visual_test", results=results,
                       candidate_id=0, device="cpu", **kwargs)


@pytest.mark.parametrize("prior_type", ["sin_i", "gaussian"])
def test_old_search_recovers_prior_from_worker_config_without_writing(candidate_run, prior_type):
    path, _, _, report, write = candidate_run
    if prior_type == "gaussian":
        _saved_imaging_prior(report, mean=44., sigma=12.)
    prior = report.pop("inclination_prior")
    report.pop("inclination_settings", None)
    report.pop("inclination_metadata", None)
    write()
    config = path/"search_config.yaml"
    config.write_text(yaml.safe_dump({"mcmc": {"priors": {"velocity.inc_v": prior}}}))
    before = config.read_bytes()
    recovered = three_file.CandidateResults(path, 6170).inclination_settings()
    assert recovered["inclination_metadata"] is None
    if prior_type == "gaussian":
        assert recovered["inclination_deg"] == pytest.approx(44.)
        assert recovered["inclination_sigma_deg"] == pytest.approx(12.)
    else:
        assert recovered["inclination_deg"] is None
    assert config.read_bytes() == before


def test_selected_fit_requires_recorded_prior_instead_of_guessing(candidate_run, monkeypatch):
    path, results, _, report, write = candidate_run
    report.pop("inclination_prior")
    write()
    monkeypatch.setattr(three_file, "run_command",
                        lambda *args, **kwargs: pytest.fail("Unknown prior must not launch sampling"))
    with pytest.raises(ValueError, match="no recorded inclination prior"):
        three_file.fit(6170, run_name="visual_test", results=results, candidate_id=0, device="cpu")
    assert not (path/"run_config.yaml").exists()


@pytest.mark.parametrize("change", ["support", "settings", "source"])
def test_saved_prior_requires_consistent_support_settings_and_source(candidate_run, change):
    path, _, _, report, write = candidate_run
    _saved_imaging_prior(report)
    if change == "support":
        report["inclination_prior"]["min"] = 0.
    elif change == "settings":
        report["inclination_settings"]["inclination_deg"] += 2.
    else:
        report["inclination_metadata"]["source_id"] = 15665
    write()
    with pytest.raises(ValueError, match="support differs|settings disagree|different source"):
        three_file.CandidateResults(path, 6170).inclination_settings()


def test_selected_fit_cannot_delete_the_candidates_being_selected(candidate_run, monkeypatch):
    path, results, _, _, _ = candidate_run
    before = (path/"start_search.json").read_bytes()
    monkeypatch.setattr(three_file, "run_command",
                        lambda *args, **kwargs: pytest.fail("No worker should run"))
    with pytest.raises(ValueError, match="override=False"):
        three_file.fit(6170, run_name="visual_test", results=results,
                       candidate_id=0, override=True, device="cpu")
    assert (path/"start_search.json").read_bytes() == before


def test_load_rejects_a_different_posterior_candidate_without_running(candidate_run, monkeypatch):
    path, results, _, _, _ = candidate_run
    _write_minimal_posterior_report(path, candidate_id=2)
    monkeypatch.setattr(three_file, "run_command",
                        lambda *args, **kwargs: pytest.fail("Loading must not run a worker"))
    result = three_file.load(6170, run_name="visual_test", results=results, candidate_id=2)
    assert result.report["chosen_candidate_id"] == 2
    with pytest.raises(ValueError, match="different candidate"):
        three_file.load(6170, run_name="visual_test", results=results, candidate_id=0)


def test_posterior_velocity_summary_preserves_draw_pairing(candidate_run):
    path, _, _, candidate_report, _ = candidate_run
    names = ["velocity.V_sini", "velocity.R_v", "velocity.inc_v",
             "velocity.theta_v", "velocity.x0_v", "velocity.y0_v"]
    values = np.array([
        [80., 1., .3, .6, 39., 41.], [100., 2., .4, .7, 40., 42.],
        [140., 4., .5, .8, 41., 43.], [180., 6., .6, .9, 42., 44.],
        [230., 9., .7, 1., 43., 45.], [300., 12., .8, 1.1, 44., 46.],
        [400., 18., .9, 1.2, 45., 47.], [500., 25., 1., 1.3, 46., 48.],
    ]).reshape(4, 2, len(names))
    np.savez(path/"samples.npz", chain=values, param_names=np.asarray(names), warmup=1000)
    report = dict(schema="dingo-three-file-v1", source_id=6170,
        intrinsic_dispersion_kms=0., chosen_candidate_id=2,
        redshift=candidate_report["redshift"], pixel_scale_arcsec=.063,
        r_fit_pixels=40, walkers=2, draws=4, warmup=1000, param_names=names,
        fixed_params={}, samples_sha256=hashlib.sha256((path/"samples.npz").read_bytes()).hexdigest())
    (path/"report.json").write_text(json.dumps(report))
    result = three_file.ThreeFileResults(path, 6170)
    radius = 1500.
    table = result.velocity_summary(radius_pc=radius, include_deprojected=True)
    rturn = values[..., 1]*physical_scale(report)["pc_per_pixel"]
    vinf = values[..., 0]/np.sin(values[..., 2])
    shape = (2/np.pi)*np.arctan(radius/rturn)
    expected = {"Vrot(1500 pc) sin(i) [km/s]": values[..., 0]*shape,
                "Vinf sin(i) [km/s]": values[..., 0],
                "Inclination [deg]": np.rad2deg(values[..., 2]),
                "Vrot(1500 pc) [km/s]": vinf*shape,
                "Vinf [km/s]": vinf}
    assert set(table.index) == set(expected)
    for label, row in table.iterrows():
        draws = expected[label]
        low, median, high = np.quantile(draws, [.16, .5, .84])
        np.testing.assert_allclose(row[["median", "minus_1sigma", "plus_1sigma"]],
                                   [median, median-low, high-median])
    shortcut = (2/np.pi)*np.median(vinf)*np.arctan(radius/np.median(rturn))
    assert table.loc["Vrot(1500 pc) [km/s]", "median"] != pytest.approx(shortcut, rel=1e-3)
    assert "1500 pc" in table.index[0]


def test_broad_sampling_dispatch_reuses_saved_search_settings(candidate_run, monkeypatch):
    _, results, _, report, write = candidate_run
    report['search_settings'].update(method='broad', refine_iterations=30, profile_iterations=20,
        profile_step=50., profile_fine_step=0., profile_max=800., profile_radius_pc=3000.)
    write()
    dispatched=[]
    class StopBeforeWorker(Exception): pass
    def worker(arguments, log, *, device):
        dispatched.append(arguments)
        raise StopBeforeWorker
    monkeypatch.setattr(three_file,'run_command',worker)
    with pytest.raises(StopBeforeWorker):
        three_file.fit(6170, run_name='visual_test', candidate_id=2, results=results, device='cpu')
    args=dispatched[0]
    assert args[1] == 'dingo.three_file_search'
    for key,value in {'--refine-iterations':'30','--profile-iterations':'20',
                       '--profile-step':'50.0','--profile-fine-step':'0.0',
                       '--profile-max':'800.0','--profile-radius-pc':'3000.0'}.items():
        assert args[args.index(key)+1] == value


def test_velocity_profile_is_saved_only_and_labels_score_not_errors(candidate_run, monkeypatch):
    import matplotlib.pyplot as plt
    path,_,_,report,write=candidate_run
    for t,score in zip(report['trials'],[10.,5.,12.]): t['log_posterior']=score
    np.savez(path/'velocity_profile.npz',velocity_kms=np.array([100.,200.,300.]),
             log_posterior=np.array([8.,10.,11.]),radius_pc=np.array(3000.))
    report['velocity_profile_sha256']=hashlib.sha256((path/'velocity_profile.npz').read_bytes()).hexdigest()
    write()
    def forbidden(*a,**k): raise AssertionError('Review must not fit')
    monkeypatch.setattr(three_file,'run_command',forbidden)
    r=three_file.CandidateResults(path,6170)
    f=r.velocity_profile(save=True)
    np.testing.assert_array_equal(f.axes[0].lines[0].get_ydata(),[8.,4.,2.])
    assert f.axes[1].get_ylim() == (0.,50.)
    assert '3000 pc' in f.axes[1].get_xlabel()
    assert 'not a probability' in f._supxlabel.get_text()
    assert (path/'figures/velocity_profile.png').exists()
    plt.close(f)
    with (path/'velocity_profile.npz').open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError,match='changed'): r.velocity_profile()
