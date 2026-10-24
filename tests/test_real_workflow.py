"""Identity, provenance and stale-cache regressions for the real notebook."""
import json
from pathlib import Path
import numpy as np
import pytest
import yaml

from dingo.real_workflow import RealSource, write_json, read_json, record_review
from dingo.real_review import RealResults


def config(tmp_path, source_id=8910):
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump({'summary': {'name': f'ID{source_id}', 'ra': 63.94, 'dec': -24.16,
        'z': 1.4233, 'lambda_rest': 1.875, 'filter': 'F444W'}}))
    return path


def test_source_identity_and_missing_native_inputs_are_not_silently_substituted(tmp_path):
    path = config(tmp_path)
    with pytest.raises(ValueError, match='different source'):
        RealSource(3700, tmp_path, tmp_path, path)
    source = RealSource(8910, tmp_path, tmp_path, path)
    status = source.availability()
    assert not status['ready']
    assert len(status['missing']) == 2
    assert all('ID8910' in p for p in status['missing'])
    assert source.wavelength == pytest.approx(1.875*(1+1.4233))
    with pytest.raises(FileNotFoundError, match='ID8910'):
        source.require_data()


def saved_run(tmp_path):
    """Minimal saved-file schema to exercise metadata/cache handling, no fit."""
    vector = np.arange(12, dtype=float)
    np.savez(tmp_path / 'fit_00000.npz', vector=vector)
    fit = {'vrot_0p5arcsec_kms': 300., 'objective': -1., 'artifact': 'fit_00000.npz'}
    report = {'source_id': 8910, 'real_observations': True, 'completed': True, 'best_by_velocity': [fit]}
    write_json(tmp_path / 'grid.json', report)
    metadata = {'source_id': 8910, 'summary': {'z': 1.4233}, 'exposures': []}
    np.savez(tmp_path / 'native_data.npz', metadata_json=json.dumps(metadata),
             science=np.empty(0), error=np.empty(0), valid=np.empty(0), emline=np.empty(0))
    return vector


def test_viewer_rejects_another_sources_run_and_pixels(tmp_path):
    saved_run(tmp_path)
    with pytest.raises(ValueError, match='different source'):
        RealResults(tmp_path, source_id=3700)
    p = tmp_path / 'grid.json'; r = read_json(p); r['source_id'] = 3700; write_json(p, r)
    with pytest.raises(ValueError, match='pixels belong'):
        RealResults(tmp_path, source_id=3700)


def test_candidate_cache_does_not_return_an_old_fit_after_resume(tmp_path):
    x = saved_run(tmp_path)
    np.savez(tmp_path / 'candidate_300.npz', vector=x+1)
    review = RealResults(tmp_path, 8910)
    with pytest.raises(FileNotFoundError, match='not cached'):
        review.candidate(300)
    np.savez(tmp_path / 'candidate_300.npz', vector=x)
    np.testing.assert_array_equal(review.candidate(300)['vector'], x)
    with pytest.raises(ValueError, match='fitted velocity'):
        review.candidate(301)


def test_visual_review_preserves_numeric_results_and_requires_a_reason(tmp_path):
    saved_run(tmp_path)
    before = (tmp_path / 'grid.json').read_bytes()
    with pytest.raises(ValueError):
        record_review(tmp_path, 300, 'plausible', '  ')
    with pytest.raises(ValueError, match='saved fitted trial'):
        record_review(tmp_path, 301, 'plausible', 'Reason')
    p = record_review(tmp_path, 300, 'needs investigation', 'Residual clump in exposure 2.')
    row = json.loads(p.read_text())
    assert row['source_id'] == 8910 and row['objective'] == -1.
    assert len(row['fit_sha256']) == 64
    assert (tmp_path / 'grid.json').read_bytes() == before


def test_imaging_prior_rejects_other_object_even_if_chains_converged(tmp_path):
    from diagnostics.native_source_grid import inclination_prior
    path = tmp_path / 'full_pixel'; path.mkdir()
    write_json(path / 'imaging.json', {'source_id': 3700, 'completed': True,
        'diagnostics': {'convergence_gate_passed': True}, 'orientation_parameterization_ok': True})
    with pytest.raises(ValueError, match='different source'):
        inclination_prior(8910, tmp_path, 'imaging')


def test_imaging_prior_handles_edge_on_q_thickness_pairs(tmp_path):
    from diagnostics.native_source_grid import inclination_prior
    for name in ('full_pixel', 'core_pixel_clean'):
        path = tmp_path / name; path.mkdir()
        write_json(path / 'imaging.json', {'source_id': 8910, 'completed': True,
            'diagnostics': {'convergence_gate_passed': True}, 'orientation_parameterization_ok': True,
            'input_sha256': {}})
        np.savez(path / 'samples.npz', q=np.full((4, 1000), .15))
    grid, logp, report = inclination_prior(8910, tmp_path, 'imaging')
    assert np.isfinite(logp).all()
    assert np.trapezoid(np.exp(logp), grid) == pytest.approx(1.)
    assert all(.4 < x < .8 for x in report['invalid_q_q0_pair_fraction'])


def test_process_interruption_terminates_active_worker(tmp_path, monkeypatch):
    from dingo import real_workflow
    class Process:
        terminated = False
        waited = False
        @property
        def stdout(self):
            raise KeyboardInterrupt()
        def poll(self): return None
        def terminate(self): self.terminated = True
        def wait(self, timeout=None): self.waited = True
    process = Process()
    monkeypatch.setattr(real_workflow.subprocess, 'Popen', lambda *a, **k: process)
    with pytest.raises(KeyboardInterrupt):
        real_workflow.run_command(['-m', 'unused'], tmp_path / 'run.log')
    assert process.terminated and process.waited


def test_real_posterior_discovery_and_runner_keep_source_identity(tmp_path):
    source = RealSource(8910, tmp_path, tmp_path, config(tmp_path))
    for name, sid, real in [('posterior_real', 8910, True),
                            ('posterior_wrong_source', 3700, True),
                            ('posterior_mock', 8910, False)]:
        path = source.directory / name
        path.mkdir(parents=True)
        write_json(path / 'posterior.json', dict(source_id=sid, real_observations=real,
            completed=True, retained_draws_per_chain=100,
            diagnostics={'convergence_gate_passed': False}))
    found = source.posterior_runs()
    assert len(found) == 1 and found[0]['path'].endswith('posterior_real')
    assert not found[0]['sampling_checks_passed']
    grid = tmp_path / 'grid'; grid.mkdir()
    write_json(grid / 'grid.json', dict(source_id=3700, real_observations=True))
    with pytest.raises(ValueError, match='own real-data grid'):
        source.sample_posterior(grid)
