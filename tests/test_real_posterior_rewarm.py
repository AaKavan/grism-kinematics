"""Starting-position fidelity and notebook routing without launching sampling."""
from pathlib import Path
import numpy as np
import pytest

from dingo.real_workflow import RealSource, write_json
from tests.test_real_workflow import config


def test_initialization_preserves_each_chain_position_and_all_nuisance_parameters():
    from diagnostics.native_real_posterior_rewarm import SITES, initial_positions
    arrays = {k: np.arange(12.).reshape(2, 6)+i for i, k in enumerate(SITES)}
    arrays['center'] = np.arange(24.).reshape(2, 6, 2)
    arrays['map_z'] = np.arange(60.).reshape(2, 6, 5)
    arrays['registration_z'] = np.arange(72.).reshape(2, 6, 3, 2)
    arrays['external_inclination'] = np.full((2, 6), -123.)
    initial = initial_positions(arrays, 2)
    assert set(initial) == set(SITES)
    for k, v in initial.items():
        np.testing.assert_array_equal(v, arrays[k][:, -1])
        assert not np.shares_memory(v, arrays[k])
    with pytest.raises(ValueError, match='chain count'):
        initial_positions(arrays, 4)
    arrays['map_z'][0, -1, 0] = np.nan
    with pytest.raises(ValueError, match='map_z'):
        initial_positions(arrays, 2)


def setup_source(tmp_path):
    source = RealSource(8910, tmp_path, tmp_path, config(tmp_path))
    grid = tmp_path / 'grid'; grid.mkdir()
    write_json(grid/'grid.json', dict(source_id=8910, real_observations=True))
    return source, grid


def parent_run(source, grid, name, **overrides):
    path = source.directory / name; path.mkdir(parents=True)
    report = dict(source_id=source.source_id, real_observations=True, completed=True,
                  warmup_completed=True, retained_draws_per_chain=100,
                  settings=dict(grid=str(grid), chains=4))
    report.update(overrides)
    write_json(path/'posterior.json', report)
    return path


def test_notebook_selects_matching_parent_and_fresh_warmup_then_pins_it_on_resume(tmp_path, monkeypatch):
    from dingo import real_workflow
    source, grid = setup_source(tmp_path)
    parent = parent_run(source, grid, 'posterior_old')
    parent_run(source, grid, 'posterior_unfinished', completed=False)
    parent_run(source, tmp_path/'other_grid', 'posterior_wrong_grid')
    calls = []
    monkeypatch.setattr(real_workflow, 'run_command', lambda args, *rest: calls.append(args))
    output = source.sample_posterior(grid, run_name='posterior_long', initial_posterior='latest',
                                    warmup=1500, samples=2000, max_tree_depth=10)
    args = calls[-1]
    assert args[1] == 'diagnostics.native_real_posterior_rewarm'
    assert args[args.index('--initial-posterior')+1] == str(parent)
    assert args[args.index('--warmup')+1] == '1500'
    assert args[args.index('--samples')+1] == '2000'
    assert args[args.index('--max-tree-depth')+1] == '10'
    assert '--resume' not in args
    output.mkdir()
    write_json(output/'posterior.json', dict(source_id=8910, real_observations=True,
        completed=False, initialized_from=dict(directory=str(parent))))
    parent_run(source, grid, 'posterior_newer')
    source.sample_posterior(grid, run_name='posterior_long', initial_posterior='latest')
    assert calls[-1][calls[-1].index('--initial-posterior')+1] == str(parent)
    assert '--resume' in calls[-1]


def test_initialization_rejects_wrong_identity_grid_or_chain_count_before_worker(tmp_path, monkeypatch):
    from dingo import real_workflow
    source, grid = setup_source(tmp_path)
    def no_worker(*args, **kwargs):
        pytest.fail('Invalid initialization must not start a worker.')
    monkeypatch.setattr(real_workflow, 'run_command', no_worker)
    wrong_source = parent_run(source, grid, 'posterior_other_source', source_id=3700)
    wrong_grid = parent_run(source, tmp_path/'wrong_grid', 'posterior_other_grid')
    valid = parent_run(source, grid, 'posterior_valid')
    for path, message in [(wrong_source, 'own real-data'), (wrong_grid, 'selected grid')]:
        with pytest.raises(ValueError, match=message):
            source.sample_posterior(grid, initial_posterior=path)
    with pytest.raises(ValueError, match='chain count'):
        source.sample_posterior(grid, initial_posterior=valid, chains=6)
    with pytest.raises(ValueError, match='not both'):
        source.sample_posterior(grid, initial_posterior=valid, from_posterior=valid)
    with pytest.raises(ValueError, match='fresh run tag'):
        source.sample_posterior(grid, run_name=valid.name, initial_posterior=valid)


def test_no_saved_posterior_uses_competing_grid_starts(tmp_path, monkeypatch):
    from dingo import real_workflow
    source, grid = setup_source(tmp_path)
    calls = []
    monkeypatch.setattr(real_workflow, 'run_command', lambda args, *rest: calls.append(args))
    source.sample_posterior(grid, initial_posterior='latest', warmup=1500, samples=2000)
    assert calls[0][1] == 'diagnostics.native_real_posterior'
    assert calls[0][calls[0].index('--warmup')+1] == '1500'
