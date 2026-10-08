'''The likelihood, pinned at a known point so no fit is needed.

Every test here sets theta explicitly to the MAP of the chain on disk and then
freezes the noise model there, rather than running fit_all(). That makes the
frozen state a deterministic function of the literals below instead of of 4000
Adam iterations, so the goldens stay valid across torch versions and thread
counts, and the suite costs seconds rather than minutes.
'''
import numpy as np
import pytest
import torch

# MAP of the production chain for each source, as float64 literals. Changing one
# invalidates the goldens that follow it.
MAP = {
    'ID21636': {
        'velocity.V_rot': 222.13283567919183,
        'velocity.R_v': 2.255910698588148,
        'velocity.x0_v': 41.469841034394975,
        'velocity.y0_v': 40.34306325460271,
        'velocity.theta_v': 1.3942651983904015,
        'velocity.inc_v': 1.0965711225352208,
        'image.R.dx': 0.969505682757662,
        'image.R.dy': -0.10151444013581051,
        'noise.ln_f': 0.1410891192809925,
    },
    'ID15665': {
        'velocity.V_sini': 274.5469250048075,
        'velocity.R_v': 4.591174060148847,
        'velocity.x0_v': 43.0088832061675,
        'velocity.y0_v': 40.01836813456412,
        'velocity.theta_v': 2.3263943345575404,
        'velocity.inc_v': 0.38455815308544267,
        'image.R.dx': 1.8914907511218368,
        'image.R.dy': -0.9774941812594207,
        'noise.ln_f': 0.24870126221166736,
    },
}

# Frozen-state fingerprints at that MAP. A refactor can leave lnP coincidentally
# right while the noise model underneath it is wrong, so these are asserted too.
# n_zero_var is the number of zero-variance pixels: ID21636 has 2 and ID15665 has
# 0, which is the entire reason both sources are tested. An earlier
# log_likelihood_batch divided by var_map before applying the mask -- 1/0 = inf,
# inf*0 = NaN -- so every ID21636 walker came back -inf and the chain silently
# froze, while ID15665 passed regardless.
FROZEN = {
    # lnP is the end-to-end golden: priors, likelihood, rectification and noise
    # model all feed it. Captured on CPU with one thread.
    'ID21636': {'n_pix': 6266, 'n_zero_var': 2, 'lnP': 9197.511687},
    'ID15665': {'n_pix': 6226, 'n_zero_var': 0, 'lnP': 9420.840339},
}

SOURCES = sorted(MAP)


@pytest.fixture(scope='module')
def prepared(request, grism_cal):
    '''Fitter at the pinned MAP with the noise model frozen there. No fit.'''
    import os
    from dingo import fitting
    sid = request.param
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cwd = os.getcwd()
    os.chdir(os.path.join(repo, 'data'))
    try:
        f = fitting.KinematicsFitter(config_path=f'{sid}/config_kinematics.yaml')
        f.set_params_dict({k: v for k, v in MAP[sid].items() if not k.startswith('noise.')})
        setup = f.prepare_mcmc()
        theta = np.array([MAP[sid][n] for n in setup['param_names']], dtype=float)
        yield sid, f, setup, theta
    finally:
        os.chdir(cwd)


@pytest.mark.data
@pytest.mark.parametrize('prepared', SOURCES, indirect=True)
def test_frozen_noise_model_fingerprint(prepared):
    sid, f, setup, _ = prepared
    exp = FROZEN[sid]
    assert int(f.mcmc_mask.sum()) == f.n_mcmc_pix
    assert f.n_mcmc_pix == exp['n_pix']
    assert int((f.mcmc_var_map <= 0).sum()) == exp['n_zero_var']
    assert setup['param_names'] == list(MAP[sid])


@pytest.mark.data
@pytest.mark.parametrize('prepared', SOURCES, indirect=True)
def test_batched_log_probability_matches_scalar(prepared):
    '''The vectorised path must agree with the scalar one it replaced, including
    on which points are -inf.'''
    sid, f, setup, theta = prepared
    rng = np.random.default_rng(0)
    thetas = np.vstack([theta] + [theta*(1 + 1e-3*rng.standard_normal(len(theta)))
                                  for _ in range(7)])
    scalar = np.array([f.log_probability(t) for t in thetas])
    batch = np.asarray(f.log_probability_batch(thetas))
    assert np.array_equal(np.isfinite(scalar), np.isfinite(batch)), 'support differs'
    ok = np.isfinite(scalar)
    assert ok.all(), 'the MAP and its neighbourhood must all be in support'
    rel = np.abs(batch[ok] - scalar[ok])/np.maximum(np.abs(scalar[ok]), 1e-12)
    assert rel.max() < 1e-6, rel.max()


@pytest.mark.data
@pytest.mark.parametrize('prepared', SOURCES, indirect=True)
def test_log_probability_is_finite_and_reproducible(prepared):
    sid, f, setup, theta = prepared
    a = f.log_probability(theta)
    b = f.log_probability(theta)
    assert np.isfinite(a)
    assert a == b, 'the likelihood must be a deterministic function of theta'


@pytest.mark.data
@pytest.mark.parametrize('prepared', SOURCES, indirect=True)
def test_log_probability_matches_the_golden(prepared):
    '''The end-to-end regression net. A refactor is a claim that the organisation
    changed and the arithmetic did not; this is how that claim gets tested.'''
    sid, f, setup, theta = prepared
    assert f.log_probability(theta) == pytest.approx(FROZEN[sid]['lnP'], abs=1e-6)


@pytest.mark.data
@pytest.mark.parametrize('prepared', SOURCES, indirect=True)
def test_out_of_prior_support_is_minus_inf(prepared):
    '''R_v has a log-uniform prior with a hard lower bound; below it the posterior
    must be -inf rather than a finite number from an unguarded likelihood.'''
    sid, f, setup, theta = prepared
    bad = theta.copy()
    bad[setup['param_names'].index('velocity.R_v')] = -1.0
    assert f.log_probability(bad) == -np.inf


@pytest.mark.data
@pytest.mark.parametrize('prepared', SOURCES, indirect=True)
def test_prepare_mcmc_is_idempotent(prepared):
    '''The five external call sites all assume this: preparing twice must not move
    the state they then read.'''
    sid, f, setup, theta = prepared
    before = (list(setup['param_names']), list(setup['priors']),
              setup['theta0'].copy(), int(f.mcmc_mask.sum()),
              f.n_mcmc_pix, f.mcmc_n_eff, f.log_probability(theta))
    again = f.prepare_mcmc()
    after = (list(again['param_names']), list(again['priors']),
             again['theta0'].copy(), int(f.mcmc_mask.sum()),
             f.n_mcmc_pix, f.mcmc_n_eff, f.log_probability(theta))
    assert before[0] == after[0]
    assert before[1] == after[1]
    assert np.array_equal(before[2], after[2])
    assert before[3:] == after[3:]


@pytest.mark.data
@pytest.mark.parametrize('prepared', SOURCES, indirect=True)
def test_prepare_mcmc_does_not_need_emcee(prepared, monkeypatch):
    '''A script that only scores a saved chain should not need the sampler
    installed. That is the point of keeping _import_emcee out of prepare_mcmc.'''
    from dingo import fitting
    sid, f, setup, theta = prepared
    monkeypatch.setattr(fitting, '_import_emcee',
                        lambda: (_ for _ in ()).throw(ImportError('emcee missing')))
    f.prepare_mcmc()
    assert np.isfinite(f.log_probability(theta))
