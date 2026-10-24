"""Notebook isolation, exact center geometry and portable export guarantees."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from dingo import holdout_workflow as workflow
from diagnostics.native_holdout_centers import project_centers


def test_read_only_import_does_not_initialize_gpu_libraries():
    check = subprocess.run([sys.executable, "-c",
        "import sys; from dingo.holdout_workflow import load_results; "
        "assert 'jax' not in sys.modules; assert 'torch' not in sys.modules"],
        cwd=workflow.REPO, capture_output=True, text=True)
    assert check.returncode == 0, check.stderr


def test_cpu_kernel_environment_does_not_disable_gpu_worker(monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JAX_PLATFORM_NAME", "cpu")
    gpu = workflow._worker_environment(True)
    cpu = workflow._worker_environment(False)
    assert "JAX_PLATFORMS" not in gpu and "JAX_PLATFORM_NAME" not in gpu
    assert cpu["JAX_PLATFORMS"] == "cpu" and cpu["CUDA_VISIBLE_DEVICES"] == ""
    import os
    assert os.environ["JAX_PLATFORMS"] == "cpu"


def test_center_projection_uses_sky_center_systemic_wavelength_and_registration():
    class Exposure:
        calibration = None
        origin_xy = (0., 0.)

        def source_detector_xy(self, east, north):
            return np.column_stack((2*east+11, 3*north+13))

    def coefficients(calibration, xy, wavelength, origin):
        result = np.zeros((1, 2, 7))
        result[0, :, 0] = xy[0]
        result[0, :, 1] = [7., 9.]
        result[0, :, 2] = [2., -1.]
        return result

    target = SimpleNamespace(wavelength_um=4.2, exposures=[Exposure(), Exposure()])
    nmap = 8
    p = np.zeros(11+nmap+4*len(target.exposures))
    p[4] = 2.3  # v_sys=230 km/s
    p[6:8] = [.4, -.7]
    p[11+nmap:11+nmap+4] = [2., -1., 3., 4.]
    actual = project_centers(target, p, nmap=nmap, registration_sigma=.25,
                             projector=coefficients)
    wave = 4.2*230/299792.458
    expected = np.array([11+.08, 13-.21])+wave*np.array([7., 9.])+wave**2*np.array([2., -1.])
    np.testing.assert_allclose(actual, expected+np.array([[.5, -.25], [.75, 1.]]))
    with pytest.raises(ValueError, match="parameter layout"):
        project_centers(target, p, nmap=nmap+1, registration_sigma=.25, projector=coefficients)


@pytest.fixture
def completed_archive(tmp_path):
    cache = tmp_path/"cache"; cache.mkdir()
    source = tmp_path/"source"; source.mkdir()
    origin = tmp_path/"origin"; origin.mkdir()
    baseline = tmp_path/"baseline"; baseline.mkdir()
    grid = tmp_path/"grid"; grid.mkdir()
    write = lambda p, x: p.write_text(json.dumps(x))
    write(source/"dispersion.json", dict(settings=dict(run=str(origin))))
    write(origin/"dispersion.json", dict(settings=dict(baseline=str(baseline))))
    write(baseline/"coupled.json", dict(settings=dict(grid=str(grid))))
    write(grid/"grid.json", dict(source_summary=dict(z=1.1906)))
    held, train = np.array([1, 3]), np.array([0, 2])
    shape = (4, 3, 3)
    science = np.arange(36).reshape(shape)*.1
    good = np.ones(shape, dtype=bool); good[1, 0, 0] = False
    obs = dict(science=science, error=np.ones(shape)*.2, valid=good,
        pupils=np.array(['R', 'R', 'C', 'C']), line_centers_xy=np.ones((4, 2)),
        held_indices=held, train_indices=train)
    np.savez(cache/"observations.npz", **obs)
    names = ['saved_lower', 'saved_higher', 'smooth_200', 'smooth_800']
    fits, centers = [], {}
    for k, name in enumerate(names):
        line = np.ones(shape)*(k+1)
        common = dict(parameters=np.arange(20)+k, initial_parameters=np.arange(20),
            source_east=np.array([-1., 1.]), source_north=np.array([-1., 1.]),
            source_flux=np.array([.4, .6]), source_velocity=np.array([-12., 12.]),
            line=line, total=line+.2, continuum=np.ones(shape)*.15,
            background=np.ones(4)*.05, residual=(science-line-.2)/.2)
        np.savez(cache/f"{name}.npz", **common)
        prediction = dict(held_indices=held, line=line[held], total=line[held]+.3,
            residual_raw_ERR=(science[held]-line[held]-.3)/.2,
            projected_kinematic_centers_xy=np.ones((2, 2))*(k+.5),
            sideband_prediction=np.ones((2, 3, 3))*.3,
            sideband_prediction_variance=np.ones((2, 3, 3))*.01,
            line_window=good[held], core=good[held])
        np.savez(cache/f"heldout_{name}.npz", **prediction)
        fits.append(dict(start=name, artifact=f'{name}.npz',
            artifact_sha256=workflow._sha(cache/f'{name}.npz'),
            heldout_artifact=f'heldout_{name}.npz',
            heldout_artifact_sha256=workflow._sha(cache/f'heldout_{name}.npz'),
            vrot_0p5arcsec_kms=200.+k, vproj_0p5arcsec_kms=180.+k,
            inclination_deg=60., turnover_arcsec=.02, objective=100.+k,
            stationarity_passed=True, boundary_parameters=[]))
        centers[name] = dict(centers_xy=np.full((4, 2), k+.5).tolist())
    report = dict(source_id=21636, completed=True, fits=fits,
        settings=dict(starts=names), source_report=str(source/'dispersion.json'),
        observation_sha256=workflow._sha(cache/'observations.npz'),
        exposures=[dict(path=f'/inputs/exposure_{i}.fits') for i in range(4)],
        started_utc='2026-10-21T00:00:00+00:00', split_policy='Two held exposures',
        holdout_policy='Frozen galaxies', limitations=['One split'])
    write(cache/'holdout.json', report)
    write(cache/'projected_centers.json', centers)
    write(cache/'covariance_assessment.json', dict(totals={}, interpretation='Conditional', limitations=[]))
    for name in ('saved_lower', 'saved_higher'):
        for suffix in ('RC', 'all'):
            (cache/f'heldout_{name}_{suffix}.png').write_bytes(b'plot fixture')
    return cache, obs


def test_export_preserves_train_holdout_mapping_and_never_overwrites(completed_archive, tmp_path):
    cache, obs = completed_archive
    before = {p.name: workflow._sha(p) for p in cache.iterdir() if p.is_file()}
    path = workflow.export_results(cache, tmp_path/'final')
    actual = workflow.load_results(path)
    np.testing.assert_array_equal(actual['science'], obs['science'][obs['held_indices']])
    np.testing.assert_array_equal(actual['all_science'], obs['science'])
    np.testing.assert_array_equal(actual['valid'], obs['valid'][obs['held_indices']])
    np.testing.assert_array_equal(actual['all_pupils'], obs['pupils'])
    assert actual['exposure_names'].tolist() == ['exposure_1.fits', 'exposure_3.fits']
    assert actual['predicted_total'].shape == (4, 2, 3, 3)
    assert actual['training_predicted_total'].shape == (4, 4, 3, 3)
    assert set(p.name for p in path.parent.iterdir()) == {
        'results.npz', 'run_info.txt', 'lower_velocity_RC.png', 'higher_velocity_RC.png',
        'lower_velocity_all_exposures.png', 'higher_velocity_all_exposures.png'}
    with pytest.raises(FileExistsError):
        workflow.export_results(cache, path.parent)
    assert before == {p.name: workflow._sha(p) for p in cache.iterdir() if p.is_file()}


def test_export_rejects_changed_prediction(completed_archive, tmp_path):
    cache, _ = completed_archive
    (cache/'heldout_saved_lower.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='artifact changed'):
        workflow.export_results(cache, tmp_path/'must_not_exist')
    assert not (tmp_path/'must_not_exist').exists()


def test_worker_streams_error_and_preserves_log(tmp_path, capsys):
    log = tmp_path/'run.log'
    with pytest.raises(RuntimeError, match='exit 7'):
        workflow._run_worker([sys.executable, '-u', '-c',
            "import sys; print('progress marker', flush=True); print('failure reason',file=sys.stderr); sys.exit(7)"],
            gpu=False, log_path=log, stage='test', heartbeat_seconds=.05)
    assert 'progress marker' in capsys.readouterr().out
    assert 'failure reason' in log.read_text()


def test_stopping_worker_only_terminates_its_own_process_group():
    own = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
    other = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
    try:
        workflow._stop_child(own)
        assert own.poll() is not None
        assert other.poll() is None
    finally:
        workflow._stop_child(own)
        workflow._stop_child(other)
