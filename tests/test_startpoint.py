'''The starting point must not depend on which tooth the optimiser was dropped in.'''
import os

import numpy as np
import pytest

from dingo import startpoint


def test_amplitude_name_reads_whichever_the_config_defines():
    assert startpoint.amplitude_name({'velocity': {'V_rot': 300.0}}) == 'V_rot'
    assert startpoint.amplitude_name({'velocity': {'V_sini': 272.5}}) == 'V_sini'


def test_amplitude_name_refuses_both_or_neither():
    # a velocity block defines exactly one parameterisation; both is ambiguous
    with pytest.raises(ValueError):
        startpoint.amplitude_name({'velocity': {'V_rot': 1.0, 'V_sini': 1.0}})
    with pytest.raises(ValueError):
        startpoint.amplitude_name({'velocity': {'R_v': 2.0}})


def _grid(amps, thetas, fn):
    return np.array([[fn(v, t) for t in thetas] for v in amps])


def test_degenerate_zero_amplitude_nodes_collapse_to_one_seed():
    '''At V = 0 the field is zero everywhere, so every theta gives the same loss.

    Counting those as distinct minima spends the whole refinement budget
    re-fitting a single point, which is what happened on the first ID16887 run:
    six seeds, six identical fits.
    '''
    amps = np.arange(0.0, 300.0, 25.0)
    thetas = np.arange(0.0, 2*np.pi, 2*np.pi/16)
    loss = _grid(amps, thetas, lambda v, t: 0.1 if v == 0 else 0.2 + 0.001*v)
    keep = startpoint.distinct_minima(loss, amps, thetas, n_keep=6, amp_tol=12.5)
    assert sum(1 for _, v, _ in keep if v == 0.0) == 1


def test_teeth_at_different_amplitudes_are_all_kept():
    amps = np.arange(0.0, 400.0, 25.0)
    thetas = np.arange(0.0, 2*np.pi, 2*np.pi/16)
    # three teeth, 100 apart, same theta -- the ID16887 comb
    loss = _grid(amps, thetas,
                 lambda v, t: 0.2 - 0.05*np.exp(-((v - 100)/10.0)**2)
                                  - 0.04*np.exp(-((v - 200)/10.0)**2)
                                  - 0.03*np.exp(-((v - 300)/10.0)**2))
    keep = startpoint.distinct_minima(loss, amps, thetas, n_keep=6, amp_tol=12.5)
    found = sorted({round(v) for _, v, _ in keep})
    for tooth in (100, 200, 300):
        assert tooth in found, f'tooth at {tooth} was merged away: {found}'


def test_theta_neighbourhood_wraps():
    '''theta_v is an angle: a minimum at theta=0 must see theta=2pi-step as adjacent.'''
    amps = np.array([100.0, 125.0])
    thetas = np.arange(0.0, 2*np.pi, 2*np.pi/8)
    loss = np.full((2, 8), 0.2)
    loss[0, 0] = 0.1          # the minimum
    loss[0, -1] = 0.15        # its wrapped neighbour -- must NOT also count as a minimum
    keep = startpoint.distinct_minima(loss, amps, thetas, n_keep=6, amp_tol=12.5)
    assert len([k for k in keep if k[0] == 0.15]) == 0


def test_fixing_the_inclination_moves_the_checkpoint_aside():
    '''A fixed-inclination run samples one parameter fewer than a free one.

    Sharing a checkpoint between them fails at resume with "existing checkpoint
    parameter names do not match this configuration", which is what stopped the
    first end-to-end run of kinematics.ipynb. The path must be namespaced
    automatically rather than left to the caller to remember.
    '''
    from dingo.fitting import fixed_inc_path

    assert (fixed_inc_path('../results/ID16887/ID16887_chain_checkpoint.npz')
            == '../results/ID16887/ID16887_chain_fixedinc_checkpoint.npz')
    # matches what the earlier notebook already wrote, so an existing
    # fixed-inclination checkpoint is still found instead of being resampled
    assert os.path.basename(fixed_inc_path('x/ID3700_chain_checkpoint.npz')) \
        == 'ID3700_chain_fixedinc_checkpoint.npz'


def test_fixed_inc_path_is_idempotent():
    '''fix_inclination may run twice on one config; the path must not grow.'''
    from dingo.fitting import fixed_inc_path

    once = fixed_inc_path('r/ID16887_chain_checkpoint.npz')
    assert fixed_inc_path(once) == once


@pytest.mark.data
def test_scan_winner_restores_registration_and_its_reported_loss(fitter, monkeypatch):
    """A new fitter must reproduce the winner before taking another step."""
    import torch

    def build(_):
        f = fitter('ID3700')
        for stage in f.config['fitting']:
            stage['maxiter'] = 2
        return f

    # Select one seed cheaply; refinement and restoration use actual R/C data.
    monkeypatch.setattr(startpoint, 'scan_grid',
                        lambda *args: np.array([[3., 4.], [1., 2.]]))
    _, loss, trials = startpoint.find_start(
        'unused', amp_max=400., amp_step=200., n_theta=2, n_keep=1,
        make_fitter=build)
    state = trials[0]['full_params']
    assert abs(state['image.R.dx']) + abs(state['image.R.dy']) > 0
    restored = build('unused')
    restored.set_params_dict(state)
    restored._reset_state()
    with torch.no_grad():
        assert float(restored.loss()) == pytest.approx(loss, abs=1e-9)
