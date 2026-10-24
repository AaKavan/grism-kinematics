"""Fresh runs clear exactly their output folders, never inputs or active jobs."""
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dingo import rotation, rotation_imaging
from dingo.real_workflow import RealSource, sha256
from dingo.run_output import output_run


@pytest.fixture
def source(tmp_path):
    config = tmp_path/'config.yaml'
    config.write_text('summary:\n  name: ID15665\n  ra: 64.0\n  dec: -24.0\n'
                      '  z: 1.16\n  lambda_rest: 1.875\n  filter: F444W\n')
    return RealSource(15665, bundle=tmp_path/'raw', results=tmp_path/'results', config=config)


def put(path, text='keep'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def options(source):
    return dict(bundle=source.bundle, results=source.results, config=source.config)


def accept(path, source_id):
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path/'samples.npz', q=np.full((4, 100), .7))
    put(path/'imaging.json', json.dumps(dict(source_id=source_id, completed=True,
        orientation_parameterization_ok=True, diagnostics=dict(convergence_gate_passed=True),
        input_sha256={}, samples_sha256=sha256(path/'samples.npz'))))


def test_rotation_override_removes_old_checkpoint_and_log_before_fitting(source, monkeypatch):
    output = source.directory/'rotation_v1'
    for name in ('rotation.json', 'checkpoint.json', 'initial_values.pkl',
                 'jax_cache/stale', 'figures/corner.png', 'samples.npz', 'run.log'):
        put(output/name, 'stale run')
    preserved = [put(source.directory/'other_run/samples.npz'),
                 put(source.results/'real_sources/ID3700/rotation_v1/samples.npz'),
                 put(source.bundle/'observations.fits'), source.config]
    before = {p: p.read_bytes() for p in preserved}
    monkeypatch.setattr(RealSource, 'require_data', lambda self: None)
    monkeypatch.setattr(rotation, 'RotationResults', lambda path, sid: SimpleNamespace(path=path))
    calls = []

    def worker(args, log, **kwargs):
        calls.append(args)
        assert not output.exists()
        assert '--resume' not in args
        put(log, 'new run\n')
        put(output/'rotation.json', 'new result')

    monkeypatch.setattr(rotation, 'run_command', worker)
    result = rotation.fit(15665, override=True, resume=True, inclination='isotropic', **options(source))
    assert len(calls) == 1 and result.path == output
    assert (output/'run.log').read_text() == 'new run\n'
    assert not (output/'checkpoint.json').exists()
    assert {p: p.read_bytes() for p in preserved} == before


def test_rotation_without_override_keeps_state_and_appends_log(source, monkeypatch):
    output = source.directory/'rotation_v1'
    checkpoint = put(output/'checkpoint.json', 'previous state')
    put(output/'run.log', 'previous run\n')
    monkeypatch.setattr(RealSource, 'require_data', lambda self: None)
    monkeypatch.setattr(rotation, 'RotationResults', lambda path, sid: None)

    def worker(args, log, **kwargs):
        assert '--resume' in args
        assert checkpoint.read_text() == 'previous state'
        with log.open('a') as stream:
            stream.write('resumed run\n')

    monkeypatch.setattr(rotation, 'run_command', worker)
    rotation.fit(15665, override=False, inclination='isotropic', **options(source))
    assert (output/'run.log').read_text() == 'previous run\nresumed run\n'


def test_imaging_override_refits_both_variants_and_ignores_explicit_old_prior(source, monkeypatch):
    output = source.directory/'imaging_runs/pysersic_v1'
    old = source.directory/'imaging'
    for name, _ in rotation_imaging.VARIANTS:
        accept(old/name, 15665)
        accept(output/name, 15665)
        put(output/name/'checkpoint.pkl', 'stale state')
        put(output/f'{name}.log', 'stale log')
    sentinel = put(source.directory/'imaging_runs/other_run/keep')
    old_hash = sha256(old/'full_pixel/imaging.json')
    calls = []

    def prepare(source, path, **kwargs):
        assert path == output and not path.exists()
        put(path/'input_provenance.json', json.dumps(dict(source_id=source.source_id)))

    def worker(args, log, **kwargs):
        path = Path(args[args.index('--output')+1])
        calls.append(path.name)
        assert not path.exists() and not log.exists()
        put(log, 'new imaging run')
        accept(path, 15665)

    monkeypatch.setattr(rotation_imaging, 'prepare_inputs', prepare)
    monkeypatch.setattr(rotation_imaging, 'run_command', worker)
    result = rotation_imaging.prepare_imaging(15665, override=True, imaging=old, **options(source))
    assert calls == ['full_pixel', 'core_pixel_clean']
    assert result.ready and result.path == output
    assert sentinel.read_text() == 'keep' and sha256(old/'full_pixel/imaging.json') == old_hash


def test_read_only_imaging_ignores_override(source, monkeypatch):
    output = source.directory/'imaging_runs/pysersic_v1'
    for name, _ in rotation_imaging.VARIANTS:
        accept(output/name, 15665)
    before = {p: p.read_bytes() for p in output.rglob('*') if p.is_file()}
    monkeypatch.setattr(rotation_imaging, 'prepare_inputs', lambda *a, **k: pytest.fail('Unexpected preparation'))
    monkeypatch.setattr(rotation_imaging, 'run_command', lambda *a, **k: pytest.fail('Unexpected worker'))
    result = rotation_imaging.prepare_imaging(15665, run=False, override=True, **options(source))
    assert result.ready and result.path == output
    assert {p: p.read_bytes() for p in output.rglob('*') if p.is_file()} == before
    assert not (source.directory/'.run_locks').exists()


@pytest.mark.parametrize('kind,relative,lock_name', [
    ('rotation', 'rotation_v1', 'worker.lock'),
    ('imaging', 'imaging_runs/pysersic_v1', 'full_pixel/.fit.lock'),
    ('imaging', 'imaging_runs/pysersic_v1', 'core_pixel_clean/.fit.lock'),
])
def test_active_worker_is_never_deleted(source, kind, relative, lock_name):
    output = source.directory/relative
    marker = put(output/'existing_result', 'running')
    lock_path = put(output/lock_name)
    with lock_path.open('r+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='still running'):
            with output_run(output, source, kind=kind, override=True):
                pytest.fail('Should not enter active run')
        assert marker.read_text() == 'running'
    # An unlocked leftover lock file is normal and must not block a fresh run.
    with output_run(output, source, kind=kind, override=True):
        assert not output.exists()


def test_two_notebook_resets_cannot_race(source):
    output = source.directory/'rotation_v1'
    with output_run(output, source, kind='rotation', override=True):
        marker = put(output/'new_state')
        with pytest.raises(RuntimeError, match='still running'):
            with output_run(output, source, kind='rotation', override=True):
                pytest.fail('Second run should not enter')
        assert marker.exists()


def test_imaging_override_preserves_inputs_of_running_grism(source):
    output = source.directory/'imaging_runs/pysersic_v1'
    marker = put(output/'samples.npz', 'used by grism')
    lock_path = put(source.directory/'rotation_v1/worker.lock')
    with lock_path.open('r+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='still running'):
            with output_run(output, source, kind='imaging', override=True):
                pytest.fail('Would replace active grism inputs')
        assert marker.read_text() == 'used by grism'


@pytest.mark.parametrize('target', ['source', 'sibling_source', 'imaging', 'imaging_runs', 'raw'])
def test_override_rejects_broad_or_outside_paths(source, target):
    paths = dict(source=source.directory, sibling_source=source.results/'real_sources/ID3700/rotation_v1',
                 imaging=source.directory/'imaging', imaging_runs=source.directory/'imaging_runs', raw=source.bundle)
    output = paths[target]
    marker = put(output/'keep')
    with pytest.raises(ValueError, match='named output run'):
        with output_run(output, source, kind='rotation', override=True):
            pytest.fail('Unsafe target')
    assert marker.exists()


@pytest.mark.parametrize('linked', ['output', 'source'])
def test_override_does_not_follow_directory_symlinks(source, linked):
    raw = put(source.bundle/'keep').parent
    output = source.directory/'rotation_v1'
    link = output if linked == 'output' else source.directory
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(raw, target_is_directory=True)
    with pytest.raises(ValueError, match='symlinked'):
        with output_run(output, source, kind='rotation', override=True):
            pytest.fail('Must not follow symlink')
    assert (raw/'keep').read_text() == 'keep'
