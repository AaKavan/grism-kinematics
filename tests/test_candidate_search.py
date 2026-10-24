"""Search mechanics and exact score agreement, without running science chains."""
import numpy as np
import pytest
from dingo.candidate_search import PosteriorSearch, profile, diverse_indices


class QuadraticFitter:
    names = ['velocity.V_sini', 'velocity.R_v', 'velocity.inc_v', 'velocity.theta_v',
             'velocity.x0_v', 'velocity.y0_v', 'image.R.dx', 'image.R.dy', 'noise.ln_f']
    mcmc_temper = .5
    n_mcmc_pix = 100
    _mcmc_lnvar_const = 7.

    def __init__(self, amplitude="V_sini"):
        self.names = [f'velocity.{amplitude}', *type(self).names[1:]]
        self.target = np.array([120., 3., .6, 1., 40., 40., 1., -1., 0.])
        bounds = [(0,1500),(.5,40),(.1,1.4),(0,2*np.pi),(10,70),(10,70),(-10,10),(-10,10),(-2,2)]
        self._mcmc_priors = [(n,dict(type='uniform',min=b[0],max=b[1])) for n,b in zip(self.names,bounds)]
        self._mcmc_priors[2][1].update(type='gaussian',mu=.6,sigma=.2)

    def prepare_mcmc(self):
        return dict(param_names=self.names, theta0=self.target.copy())

    def log_prior(self, p):
        if any(not s['min'] <= v <= s['max'] for (_,s),v in zip(self._mcmc_priors,p)):
            return -np.inf
        return -.5*((p[2]-.6)/.2)**2

    def _chi2_to_lnL(self, chi2, ln_f):
        return self.mcmc_temper*(-.5*chi2*np.exp(-2*ln_f)-self.n_mcmc_pix*ln_f-.5*self._mcmc_lnvar_const)

    def log_likelihood_batch(self, p):
        scales = np.array([25.,.5,.1,.1,.5,.5,.25,.25,1.])
        chi2 = 100+np.sum(((p[:,:-1]-self.target[:-1])/scales[:-1])**2,axis=1)
        return self._chi2_to_lnL(chi2,p[:,-1])


def test_profiled_noise_matches_actual_posterior_and_bounds():
    f = QuadraticFitter(); s = PosteriorSearch(f, batch_size=2,radius_pixels=8.)
    p = np.tile(f.target,(4,1)); p[:,0] = [120,250,10000,-1]
    values, score = s.evaluate(p)
    actual = f.log_likelihood_batch(values[:2])+[f.log_prior(t) for t in values[:2]]
    np.testing.assert_allclose(score[:2],actual)
    assert np.isneginf(score[2:]).all()
    assert values[1,-1] > 0


def test_multistart_refinement_moves_alignment_and_velocity():
    f = QuadraticFitter(); s = PosteriorSearch(f,radius_pixels=8.)
    p = np.tile(f.target,(2,1)); p[:,0] = [50,350]; p[:,6:8] = 0
    refined,score = s.optimize(p,iterations=100)
    assert np.max(abs(refined[:,0]-120)) < 2
    np.testing.assert_allclose(refined[:,6:8],np.tile([1,-1],(2,1)),atol=.02)
    assert np.isfinite(score).all()


@pytest.mark.parametrize("amplitude", ["V_sini", "V_rot"])
def test_profile_holds_physical_velocity_and_refits_other_parameters(amplitude):
    f = QuadraticFitter(amplitude); s = PosteriorSearch(f,radius_pixels=8.)
    starts = np.tile(f.target,(2,1)); starts[:,6:8] = 0
    v = np.array([80.,120.,200.])
    points, scores = profile(s,starts,v,iterations=100)
    np.testing.assert_allclose(s.velocity(points),v,atol=1e-10)
    assert np.max(abs(points[:,6]-1)) < .05
    assert np.isfinite(scores).all()


def test_distinct_solutions_are_not_merged_by_velocity_alone():
    f = QuadraticFitter(); s = PosteriorSearch(f,radius_pixels=8.)
    p = np.tile(f.target,(3,1)); p[1,6] += 1; p[2,3] += 2*np.pi
    ids = diverse_indices(p,np.array([3.,2.,1.]),s,3)
    assert ids == [0,1]


def test_profiled_scale_respects_prior_limits():
    f = QuadraticFitter(); s = PosteriorSearch(f,radius_pixels=8.)
    p = f.target.copy(); p[0] = 1499; p[1] = 39
    t,score = s.evaluate([p])
    assert t[0,-1] == 2.
    assert score[0] == pytest.approx(f.log_likelihood_batch(t)[0]+f.log_prior(t[0]))


@pytest.mark.parametrize("amplitude", ["V_sini", "V_rot"])
def test_shared_comparison_is_identical_for_different_starts(tmp_path, amplitude):
    """Actual R/C model: candidate -> independent MCMC preparation, no draws."""
    import os
    from pathlib import Path
    import torch
    import yaml
    from dingo.three_file_fit import build_config, REPO
    from dingo.three_file_search import SharedComparisonFitter, save_comparison
    os.environ.setdefault('GRISM_CAL_DIR',str(REPO.parent/'grism_cal'))
    config = build_config(15665, REPO/'data', REPO/'data/ID15665/config_kinematics.yaml', tmp_path,
                         inclination_deg=32.1, inclination_sigma_deg=5., device='cpu',
                         velocity_parameter=amplitude)
    config['shared_comparison'] = str(tmp_path/'comparison.npz')
    path = tmp_path/'config.yaml'; path.write_text(yaml.safe_dump(config))
    torch.set_num_threads(2)
    save_comparison(path,tmp_path/'comparison.npz',torch.device('cpu'))
    a = SharedComparisonFitter(str(path),device=torch.device('cpu'))
    s = PosteriorSearch(a,radius_pixels=8.)
    p = np.tile(s.theta0,(2,1))
    p[:,s.amp] = [100,250]; p[:,s.index['image.R.dx']] = [0,1]
    values,scores = s.evaluate(p)
    assert np.isfinite(scores).all()
    for value,score in zip(values,scores):
        b = SharedComparisonFitter(str(path),device=torch.device('cpu'))
        b.set_params_dict(dict(zip(s.names,value)))
        b.prepare_mcmc()
        np.testing.assert_array_equal(a.mcmc_mask,b.mcmc_mask)
        np.testing.assert_array_equal(a.mcmc_var_map,b.mcmc_var_map)
        assert b.log_probability(value) == pytest.approx(score,abs=.05)
        assert b.mcmc_fingerprint()['shared_comparison_sha256'] == a.mcmc_fingerprint()['shared_comparison_sha256']


@pytest.mark.parametrize("amplitude", ["V_sini", "V_rot"])
def test_selected_broad_candidate_handoff_uses_profiled_start_and_shared_model(tmp_path, monkeypatch, amplitude):
    import os, time
    from types import SimpleNamespace
    import torch, yaml
    from dingo.three_file_fit import build_config, REPO
    from dingo.three_file_search import SharedComparisonFitter, save_comparison, sample
    os.environ.setdefault('GRISM_CAL_DIR',str(REPO.parent/'grism_cal'))
    config = build_config(15665, REPO/'data', REPO/'data/ID15665/config_kinematics.yaml',tmp_path,
                         inclination_deg=32.1,inclination_sigma_deg=5.,device='cpu',
                         velocity_parameter=amplitude)
    config['shared_comparison'] = str(tmp_path/'comparison.npz')
    path=tmp_path/'search_config.yaml'; path.write_text(yaml.safe_dump(config))
    save_comparison(path,tmp_path/'comparison.npz',torch.device('cpu'))
    f=SharedComparisonFitter(str(path),device=torch.device('cpu'))
    s=PosteriorSearch(f,radius_pixels=8.)
    theta,score=s.evaluate([s.theta0]); theta=theta[0]; score=score[0]
    full=dict(zip(s.names,map(float,theta)))
    params={k.split('.')[1]:v for k,v in full.items() if k.startswith('velocity.')}
    trial=dict(candidate_id=0,loss=1.,full_params=full,params=params,
               velocity_summary={'V_inf_kms':(params['V_sini']/np.sin(params['inc_v'])
                                             if amplitude == 'V_sini' else params['V_rot'])})
    class StopBeforeSampling(Exception): pass
    def stop(self):
        info=self.prepare_mcmc()
        expected = np.array([full[n] for n in info['param_names']])
        np.testing.assert_allclose(info['theta0'],expected,rtol=1e-6)
        assert self.log_probability(info['theta0']) == pytest.approx(score,abs=.05)
        np.testing.assert_array_equal(self.mcmc_var_map,f.mcmc_var_map)
        raise StopBeforeSampling
    monkeypatch.setattr(SharedComparisonFitter,'fit_MCMC',stop)
    args=SimpleNamespace(candidate=0,draws=4000,device='cpu')
    with pytest.raises(StopBeforeSampling):
        sample(args,config,tmp_path,REPO/'data/ID15665',{},None,
               {'selected':0,'trials':[trial]}, {'test':True},time.monotonic())
    assert not (tmp_path/'samples.npz').exists()


def test_profile_cross_check_keeps_better_existing_points():
    from dingo.candidate_search import polish_profile
    f=QuadraticFitter(); s=PosteriorSearch(f,radius_pixels=8.)
    velocities=np.array([80.,120.,200.])
    points,scores=profile(s,[f.target],velocities,iterations=10)
    new,ns,gain=polish_profile(s,points,scores,velocities,np.array([f.target]),iterations=20)
    assert np.all(ns >= scores)
    np.testing.assert_allclose(s.velocity(new),velocities,atol=1e-10)
    assert gain >= 0


def test_invalid_turnover_profile_poll_is_rejected_without_runtime_warning():
    import warnings
    f=QuadraticFitter(); s=PosteriorSearch(f,radius_pixels=8.)
    p=np.tile(f.target,(2,1)); p[:,s.rt]=[0.,-.5]
    with warnings.catch_warnings():
        warnings.simplefilter('error',RuntimeWarning)
        _,scores=s.evaluate(p,velocities=[100.,100.])
    assert np.isneginf(scores).all()


def test_deprojected_and_projected_coordinates_render_identical_actual_data(tmp_path):
    """Changing coordinates changes the prior measure, not the physical field."""
    import os
    import torch, yaml
    from dingo.three_file_fit import build_config, REPO
    from dingo.three_file_search import SharedComparisonFitter, save_comparison
    os.environ.setdefault('GRISM_CAL_DIR', str(REPO.parent/'grism_cal'))
    torch.set_num_threads(2)
    rendered, loglike, masks = {}, {}, {}
    for amplitude in ('V_sini', 'V_rot'):
        config = build_config(15665, REPO/'data', REPO/'data/ID15665/config_kinematics.yaml',
            tmp_path, inclination_deg=32.1, inclination_sigma_deg=5.,
            device='cpu', velocity_parameter=amplitude)
        config['shared_comparison'] = str(tmp_path/f'{amplitude}_comparison.npz')
        path = tmp_path/f'{amplitude}.yaml'; path.write_text(yaml.safe_dump(config))
        save_comparison(path, tmp_path/f'{amplitude}_comparison.npz', torch.device('cpu'))
        f = SharedComparisonFitter(str(path), device=torch.device('cpu'))
        search = PosteriorSearch(f, radius_pixels=8.)
        assert search.amplitude_name == f'velocity.{amplitude}'
        p = np.tile(search.theta0, (3, 1))
        p[:, search.inc] = np.deg2rad([27.1, 32.1, 37.1])
        p[:, search.amp] = [300., 350., 400.]
        if amplitude == 'V_sini':
            p[:, search.amp] = (torch.as_tensor(p[:, search.amp], dtype=torch.float32)
                * torch.sin(torch.as_tensor(p[:, search.inc], dtype=torch.float32))).numpy()
        p[:, search.angle] = 2.2
        p[:, search.index['image.R.dx']] = 1.8
        p[:, search.index['image.R.dy']] = -1.
        batch = f.log_probability_batch(p)
        scalar = np.array([f.log_probability(row) for row in p])
        np.testing.assert_allclose(batch, scalar, atol=.05, rtol=1e-6)
        assert np.isfinite(scalar).all()
        rendered[amplitude] = np.array(f.get_fitting_results()[:2])
        loglike[amplitude] = np.array(f.log_likelihood_batch(p))
        masks[amplitude] = f.mcmc_mask.cpu().numpy().copy()
        # A fixed deprojected velocity constrains V_rot without multiplying sin(i).
        constrained = search.constrain(p, velocities=[100., 200., 300.])
        np.testing.assert_allclose(search.velocity(constrained), [100., 200., 300.])
    np.testing.assert_array_equal(masks['V_sini'], masks['V_rot'])
    np.testing.assert_allclose(rendered['V_sini'], rendered['V_rot'], atol=2e-7, rtol=1e-4)
    np.testing.assert_allclose(loglike['V_sini'], loglike['V_rot'], atol=.05, rtol=1e-6)
