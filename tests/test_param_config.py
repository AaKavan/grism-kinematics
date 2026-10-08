'''The config -> free-parameter machinery, with no data and no fit.

These are the cheap tests that guard the things most likely to be re-broken:
a bound that silently vanishes between stages, an override key that names
nothing, and per-stage dicts that turn out to be one shared dict.
'''
import glob
import os

import pytest
import yaml

from dingo import fitting

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _build(stages, raw=None, prefix='velocity'):
    '''Run the real cfg builder over a hand-written stage list.'''
    raw = raw or {'V_rot': 300.0, 'R_v': 2.0, 'inc_v': 1.2}
    defaults = [st['default'] for st in stages]
    overrides = [dict(st.get('override') or {}) for st in stages]
    return fitting.build_param_config_dict_with_alias(
        raw, defaults, overrides, prefix=prefix)


DEFAULT = {'lr': 0.05, 'min': -1e10, 'max': 1e10, 'fit': True}


def test_per_stage_cfg_dicts_are_distinct_objects():
    '''`all_cfgs = [{}]*n` made every stage share one alias namespace.

    It could not leak with the current callers, because each stage's own cfgs
    overwrite the shared dict before aliases resolve -- but that is a property
    of the callers, not of the function, and this pins the function.
    '''
    cfgs = _build([{'default': dict(DEFAULT)}, {'default': dict(DEFAULT)}])
    assert len(cfgs) == 2
    assert cfgs[0] is not cfgs[1]
    for key in cfgs[0]:
        assert cfgs[0][key] is not cfgs[1][key], key


def test_override_applies_to_its_own_stage():
    cfgs = _build([
        {'default': dict(DEFAULT), 'override': {'velocity.inc_v': {'min': 0.0, 'max': 1.57}}},
        {'default': dict(DEFAULT)},
    ])
    assert float(cfgs[0]['velocity.inc_v'].min) == pytest.approx(0.0)
    assert float(cfgs[0]['velocity.inc_v'].max) == pytest.approx(1.57)


def test_grouped_override_keys_expand():
    cfgs = _build([{'default': dict(DEFAULT),
                    'override': {'velocity.R_v, velocity.inc_v': {'fit': False}}}])
    assert cfgs[0]['velocity.R_v'].fit is False
    assert cfgs[0]['velocity.inc_v'].fit is False
    assert cfgs[0]['velocity.V_rot'].fit is True


def test_unused_override_key_raises():
    '''`velocity.inc` instead of `velocity.inc_v` used to do nothing, silently,
    leaving the parameter on the stage default of [-1e10, 1e10].'''
    overrides = [{'velocity.inc': {'min': 0.0}}]
    fitting.build_param_config_dict_with_alias(
        {'V_rot': 300.0, 'inc_v': 1.2}, [dict(DEFAULT)], overrides, prefix='velocity')
    with pytest.raises(ValueError, match='match no parameter'):
        fitting.check_unused_overrides(overrides, ('velocity',), 'test')


def test_unused_override_message_names_the_valid_keys():
    overrides = [{'velocity.V_sini': {'min': 0.0}}]
    with pytest.raises(ValueError) as e:
        fitting.check_unused_overrides(overrides, ('velocity',), 'test')
    msg = str(e.value)
    assert 'velocity.V_sini' in msg and 'stage 1' in msg
    assert 'velocity.inc_v' in msg          # quotes back what WOULD be valid
    assert 'V_rot or V_sini' in msg         # names the most likely real cause


def test_no_leftover_keys_means_no_raise():
    fitting.check_unused_overrides([{}, {}], ('velocity',), 'test')


CONFIGS = sorted(glob.glob(os.path.join(REPO, 'data', '*', 'config_kinematics*.yaml')))


@pytest.mark.parametrize('cfg_path', CONFIGS, ids=lambda p: os.path.basename(os.path.dirname(p)))
def test_shipped_config_consumes_every_override_key(cfg_path):
    '''Every `override:` key in every shipped config must name a real parameter.

    Pure: reads the YAML and replays the build, so it needs no FITS and no
    calibration. This is the test that catches a config typo before a fit.
    '''
    cfg = yaml.safe_load(open(cfg_path))
    defaults = [st['default'] for st in cfg['fitting']]
    overrides = [dict(st.get('override') or {}) for st in cfg['fitting']]
    for raw, prefix in ((cfg['velocity'], 'velocity'),
                        ({k: v for k, v in cfg['image']['R'].items() if k in ('dx', 'dy')}, 'image.R'),
                        ({k: v for k, v in cfg['image']['C'].items() if k in ('dx', 'dy')}, 'image.C')):
        fitting.build_param_config_dict_with_alias(raw, defaults, overrides, prefix=prefix)
    leftovers = {i: sorted(k) for i, k in enumerate(overrides) if k}
    assert not leftovers, f'{os.path.basename(cfg_path)} has unused override keys: {leftovers}'


def test_yaml_scientific_notation_is_a_string_not_a_float():
    '''PyYAML 1.1 does not resolve unquoted 1e10 as a float.

    FitParamConfig rescues it via ensure_float_like, so it works today -- but any
    future code that compares or does arithmetic on a raw bound straight from the
    YAML will be comparing against a str. Pinned so that is a known fact.
    '''
    assert yaml.safe_load('a: -1e10')['a'] == '-1e10'
    assert isinstance(yaml.safe_load('a: -1e10')['a'], str)
    assert fitting.ensure_float_like('-1e10') == pytest.approx(-1e10)


@pytest.mark.parametrize('cfg_path', CONFIGS, ids=lambda p: os.path.basename(os.path.dirname(p)))
def test_v_sys_is_registered_but_frozen_in_every_shipped_config(cfg_path):
    '''v_sys must exist everywhere and be free nowhere.

    Every shipped config has `fit: true` as its stage default, so if v_sys were
    injected through the default block it would become free in all of them and
    silently change every result. It is built with fit=False explicitly instead.
    '''
    cfg = yaml.safe_load(open(cfg_path))
    defaults = [st['default'] for st in cfg['fitting']]
    overrides = [dict(st.get('override') or {}) for st in cfg['fitting']]
    raw = dict(cfg['velocity'])
    raw.setdefault('v_sys', 0.0)
    for ov in overrides:
        ov.setdefault('velocity.v_sys', {'fit': False})
    stages = fitting.build_param_config_dict_with_alias(
        raw, defaults, overrides, prefix='velocity')
    for i, st in enumerate(stages):
        assert 'velocity.v_sys' in st, f'stage {i+1} has no v_sys'
        assert st['velocity.v_sys'].fit is False, f'stage {i+1} has v_sys FREE'
        assert float(st['velocity.v_sys'].tensor) == 0.0


def test_bounds_are_inherited_by_later_stages():
    '''A bound set in stage 1 must still hold in stage 2.

    This is the bug that let inc_v rail to 0 and then 0.0098: stage 2 rebuilt
    every parameter from its own override plus the stage default, so a bound set
    only in stage 1 reverted to [-1e10, 1e10] -- and stage 2 runs last, so those
    are the values MCMC initialises from.
    '''
    cfgs = _build([
        {'default': dict(DEFAULT), 'override': {'velocity.inc_v': {'min': 0.0, 'max': 1.57}}},
        {'default': dict(DEFAULT)},
    ])
    for st in (0, 1):
        assert float(cfgs[st]['velocity.inc_v'].min) == pytest.approx(0.0), f'stage {st+1}'
        assert float(cfgs[st]['velocity.inc_v'].max) == pytest.approx(1.57), f'stage {st+1}'


def test_a_later_stage_can_still_widen_a_bound():
    '''Inheritance must not make a deliberately temporary bound inexpressible.'''
    cfgs = _build([
        {'default': dict(DEFAULT), 'override': {'velocity.R_v': {'min': 1.0}}},
        {'default': dict(DEFAULT), 'override': {'velocity.R_v': {'min': -5.0}}},
    ])
    assert float(cfgs[0]['velocity.R_v'].min) == pytest.approx(1.0)
    assert float(cfgs[1]['velocity.R_v'].min) == pytest.approx(-5.0)


def test_lr_and_fit_are_NOT_inherited():
    '''Varying these per stage is what staging is for: image.R.dx takes lr 0.5 in
    the search and the stage default in the polish, and theta_v is frozen for the
    polish. Inheriting either would break every shipped config.'''
    stages = [
        {'default': dict(DEFAULT), 'override': {'velocity.theta_v': {'fit': False, 'lr': 0.5}}},
        {'default': dict(DEFAULT, lr=0.003)},
    ]
    cfgs = _build(stages, raw={'V_rot': 300.0, 'theta_v': 0.0})
    assert cfgs[0]['velocity.theta_v'].fit is False
    assert cfgs[1]['velocity.theta_v'].fit is True
    assert cfgs[0]['velocity.theta_v'].lr == pytest.approx(0.5)
    assert cfgs[1]['velocity.theta_v'].lr == pytest.approx(0.003)


def test_inherited_bounds_survive_yaml_string_scientific_notation():
    '''PyYAML hands back '-1e10' as a str, so the inheritance must carry the raw
    value and let ensure_float_like convert it, not float() it on the way past.'''
    cfgs = _build([
        {'default': {'lr': 0.05, 'min': '-1e10', 'max': '1e10', 'fit': True},
         'override': {'velocity.R_v': {'min': '1e-3'}}},
        {'default': {'lr': 0.05, 'min': '-1e10', 'max': '1e10', 'fit': True}},
    ])
    assert float(cfgs[1]['velocity.R_v'].min) == pytest.approx(1e-3)
    assert float(cfgs[1]['velocity.R_v'].max) == pytest.approx(1e10)


@pytest.mark.parametrize('cfg_path', CONFIGS, ids=lambda p: os.path.basename(os.path.dirname(p)))
def test_C_grism_offsets_are_frozen_in_every_stage(cfg_path):
    '''image.C.dx/dy must be fixed in every stage of every config.

    Two reasons, and the second is the serious one. With all four offsets free,
    (dx_R, dx_C, x0_v) and (dy_R, dy_C, y0_v) each contain an exactly flat
    direction. And the R-minus-C objective constrains no flux, so it is globally
    minimised at EXACTLY 0 by a solution that maps both grisms off the cutout --
    measured on ID6170, where a 400 px offset gives loss = 0 against 0.0933 at the
    fit. Freezing C makes that unreachable: moving R alone leaves C's flux in
    place and the loss goes up. It is also what makes velocity.v_sys identifiable.
    '''
    cfg = yaml.safe_load(open(cfg_path))
    for i, stage in enumerate(cfg['fitting']):
        ov = stage.get('override') or {}
        frozen = any('image.C' in key and (val or {}).get('fit') is False
                     for key, val in ov.items())
        assert frozen, f'stage {i+1} leaves image.C.dx/dy free'


@pytest.mark.parametrize('cfg_path', CONFIGS, ids=lambda p: os.path.basename(os.path.dirname(p)))
def test_checkpoint_path_stays_inside_the_repo(cfg_path):
    '''Paths resolve from data/, so ../results is right and ../../results escapes
    the repository entirely. Three configs had the latter.'''
    mc = yaml.safe_load(open(cfg_path)).get('mcmc') or {}
    ckpt = mc.get('checkpoint')
    if not ckpt:
        pytest.skip('no checkpoint configured')
    resolved = os.path.normpath(os.path.join(REPO, 'data', ckpt))
    assert resolved.startswith(REPO + os.sep), f'{ckpt} resolves outside the repo'
