'''fit_report(), on a toy fitter: no data, no calibration, milliseconds.

A one-parameter quadratic whose minimum lies outside the clamp is enough to
exercise the railing detector, which is the check that would have caught the
inc_v collapse. The real sources are too slow to use as a unit test and would
not isolate the statistic anyway.
'''
import textwrap

import pytest
import torch
import yaml

from dingo import fitting

CONFIG = textwrap.dedent('''
    summary: {name: toy, ra: 0.0, dec: 0.0, z: 1.0, lambda_rest: 1.0,
              filter: F444W, mode: kinematics}
    fitting:
      - method: Adam
        maxiter: 400
        scheduler: StepLR
        default: {lr: 0.05, min: -1e10, max: 1e10, fit: true}
        override:
          toy.p: {min: MIN_, max: MAX_}
''')


class _Toy(fitting.BaseFitter):
    '''Minimises (p - target)^2. The target is settable so a test can place it
    inside or outside the clamp.'''
    target = 5.0

    def _setup_data(self):
        cfgs = fitting.build_param_config_dict_with_alias(
            {'p': 0.0},
            [st['default'] for st in self.config['fitting']],
            [dict(st.get('override') or {}) for st in self.config['fitting']],
            prefix='toy', allowed_keys_extra={'toy': {'p'}})
        self.toy_cfg_list = cfgs
        self.param_config_lists = {'toy': cfgs}

    def loss(self):
        p = self._get_model_params('toy')['p']
        return (p - self.target)**2


def _fitter(tmp_path, lo, hi, target):
    txt = CONFIG.replace('MIN_', repr(lo)).replace('MAX_', repr(hi))
    path = tmp_path/'toy.yaml'
    path.write_text(txt)
    f = _Toy(config_path=str(path), device=torch.device('cpu'))
    f.target = target
    return f


def test_railed_parameter_is_reported(tmp_path):
    '''The minimum sits at 5.0 but the clamp stops at 1.0, so the optimiser pushes
    outward on every step. Clamp ACTIVITY catches that; distance-from-bound over
    recent movement cannot, because a railed parameter does not move at all.'''
    f = _fitter(tmp_path, -1.0, 1.0, target=5.0)
    f.fit_all()
    rep = f.fit_report()
    assert not rep['ok']
    rail = rep['stages'][0]['railed']
    assert 'toy.p' in rail
    assert rail['toy.p']['rail_frac'] == pytest.approx(1.0)
    assert any('railed' in p for p in rep['problems'])


def test_unrailed_parameter_is_not_reported(tmp_path):
    '''Same fit with the minimum inside the clamp: the clamp must never fire.'''
    f = _fitter(tmp_path, -10.0, 10.0, target=5.0)
    f.fit_all()
    rep = f.fit_report()
    assert rep['stages'][0]['railed'] == {}
    assert not any('railed' in p for p in rep['problems'])


def test_strict_raises_on_a_railed_fit(tmp_path):
    f = _fitter(tmp_path, -1.0, 1.0, target=5.0)
    f.fit_all()
    with pytest.raises(RuntimeError, match='railed'):
        f.fit_report(strict=True)


def test_report_before_fit_is_an_error(tmp_path):
    f = _fitter(tmp_path, -1.0, 1.0, target=5.0)
    with pytest.raises(RuntimeError, match='fit_all'):
        f.fit_report()


def test_non_finite_loss_is_caught(tmp_path):
    '''There is no NaN guard in the gradient path: a NaN loss decays the learning
    rate to min_lr and the fit grinds out every remaining step producing NaNs.
    At minimum it must not be reported as a healthy fit.'''
    f = _fitter(tmp_path, -10.0, 10.0, target=float('nan'))
    f.fit_all()
    rep = f.fit_report()
    assert rep['stages'][0]['finite'] is False
    assert any('not finite' in p for p in rep['problems'])
