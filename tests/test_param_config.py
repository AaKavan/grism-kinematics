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


@pytest.mark.parametrize('cfg_path', CONFIGS, ids=lambda p: os.path.basename(p))
def test_v_sys_is_frozen_unless_the_config_is_a_vsys_variant(cfg_path):
    '''v_sys must exist in every config, and be free only where that is the point.

    Every shipped config has `fit: true` as its stage default, so if v_sys were
    injected through the default block it would become free in all of them and
    silently change every result. It is built with fit=False explicitly instead.

    A config whose name says `vsys` exists precisely to free it, so this asserts
    the opposite for those -- otherwise the variant could stop freeing v_sys and
    the test that is supposed to cover it would pass anyway.
    '''
    want_free = 'vsys' in os.path.basename(cfg_path)
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
        assert st['velocity.v_sys'].fit is want_free, (
            f'stage {i+1}: v_sys is {"free" if st["velocity.v_sys"].fit else "frozen"}, '
            f'expected {"free" if want_free else "frozen"} for {os.path.basename(cfg_path)}')
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


@pytest.mark.data
def test_an_unimplemented_rotation_curve_raises(tmp_path, monkeypatch):
    '''`velocity.model` is not in ALLOWED_PARAM_KEYS, so it was dropped when the
    cfgs were built and never read. Every config says `model: arctan` and would
    have behaved identically saying `model: whatever` -- the arctangent curve is
    hard-wired. A key that does nothing should not be silently accepted.
    '''
    import shutil
    src = os.path.join(REPO, 'data', 'ID21636')
    dst = tmp_path/'ID21636'
    dst.mkdir()
    cfg = yaml.safe_load(open(os.path.join(src, 'config_kinematics.yaml')))
    cfg['velocity']['model'] = 'exponential'
    (dst/'config_kinematics.yaml').write_text(yaml.safe_dump(cfg))
    for f in ('R.fits', 'C.fits'):
        p = os.path.join(src, f)
        if os.path.exists(p):
            os.symlink(p, dst/f)

    cal = os.environ.get('GRISM_CAL_DIR') or os.path.join(os.path.dirname(REPO), 'grism_cal')
    if not os.path.isdir(cal):
        pytest.skip('no grism calibration directory')
    if not os.path.exists(os.path.join(src, 'R.fits')):
        pytest.skip('ID21636 cutouts are not staged (data is not in the repo)')
    monkeypatch.setenv('GRISM_CAL_DIR', cal)
    monkeypatch.chdir(tmp_path)
    from dingo import fitting as _f
    with pytest.raises(NotImplementedError, match='arctan'):
        _f.KinematicsFitter(config_path='ID21636/config_kinematics.yaml')


# --------------------------------------------------------------------------
# V_rot / V_sini: the two names for one quantity.
#
# amplitude_key() covers the chain-name side. These cover the parameter-dict
# side, which is where ID15665 -- the sample's only V_sini config -- died at
# the velocity-field plot after a completed 32-minute chain had been written.
# --------------------------------------------------------------------------
import numpy as np


def test_with_rotation_amplitude_derives_V_rot_from_a_V_sini_config():
    '''The fixed-inclination case: the fit has no inc_v, so the caller supplies it.'''
    out = fitting.with_rotation_amplitude({'V_sini': 272.5, 'R_v': 3.35}, 0.5040)
    assert out['V_rot'] == pytest.approx(272.5/np.sin(0.5040))
    assert out['V_sini'] == 272.5          # the sampled value is never overwritten
    assert out['inc_v'] == pytest.approx(0.5040)
    assert out['R_v'] == 3.35              # everything else passes through


def test_with_rotation_amplitude_derives_V_sini_from_a_V_rot_config():
    out = fitting.with_rotation_amplitude({'V_rot': 506.8591, 'inc_v': 0.5040})
    assert out['V_sini'] == pytest.approx(506.8591*np.sin(0.5040))


def test_with_rotation_amplitude_round_trips():
    a = fitting.with_rotation_amplitude({'V_rot': 506.8591, 'inc_v': 0.5040})
    b = fitting.with_rotation_amplitude({'V_sini': a['V_sini'], 'inc_v': 0.5040})
    assert b['V_rot'] == pytest.approx(506.8591)


def test_with_rotation_amplitude_handles_the_dotted_chain_names():
    out = fitting.with_rotation_amplitude(
        {'velocity.V_sini': 272.5}, 0.5040, prefix='velocity.')
    assert out['velocity.V_rot'] == pytest.approx(272.5/np.sin(0.5040))


def test_with_rotation_amplitude_replaces_a_stored_None_inclination():
    '''get_params() returns None for a parameter the fit held fixed, and
    dict.get(k, default) hands that None back instead of the fallback. The
    plotting code calls float() on this key.'''
    out = fitting.with_rotation_amplitude({'V_sini': 272.5, 'inc_v': None}, 0.5040)
    assert out['inc_v'] == pytest.approx(0.5040)
    assert out['V_rot'] == pytest.approx(272.5/np.sin(0.5040))


def test_with_rotation_amplitude_leaves_unconvertible_dicts_alone():
    '''No inclination from either source, or a non-numeric amplitude: return the
    dict unchanged rather than invent a conversion.'''
    assert fitting.with_rotation_amplitude({'V_sini': 272.5}) == {'V_sini': 272.5}
    assert fitting.with_rotation_amplitude({'V_sini': None, 'inc_v': 0.5}) == \
        {'V_sini': None, 'inc_v': 0.5}
    both = {'V_rot': 1.0, 'V_sini': 0.5, 'inc_v': 0.5}
    assert fitting.with_rotation_amplitude(both) == both


def test_with_rotation_amplitude_does_not_mutate_its_argument():
    src = {'V_sini': 272.5}
    fitting.with_rotation_amplitude(src, 0.5040)
    assert src == {'V_sini': 272.5}


def test_every_sample_config_uses_exactly_one_amplitude_name():
    '''A config carrying both, or neither, would make amplitude_key's answer
    arbitrary. This is the invariant the helpers rely on.'''
    for path in sorted(glob.glob(os.path.join(REPO, 'data', 'ID*', 'config_kinematics.yaml'))):
        raw = yaml.safe_load(open(path))
        vel = raw.get('velocity', {}) or {}
        have = [k for k in ('V_rot', 'V_sini') if k in vel]
        assert len(have) == 1, f'{os.path.basename(os.path.dirname(path))}: {have}'


# --------------------------------------------------------------------------
# Running a source the sample has never seen.
#
# Each of these guards a failure that does not raise: a wrong module fits the
# wrong pixels, and an Adam bound below the MCMC prior floor seeds every walker
# at -inf. Both used to be reachable by generating a config on a fresh machine.
# --------------------------------------------------------------------------
def test_adam_R_v_floor_is_never_below_the_mcmc_prior_floor():
    '''`init: adam` seeds the walkers at the optimizer's answer. An Adam bound
    that admits R_v below the prior's own floor starts every walker outside the
    prior, which is -inf for all of them and no chain at all.'''
    for path in sorted(glob.glob(os.path.join(REPO, 'data', 'ID*', 'config_kinematics.yaml'))):
        cfg = yaml.safe_load(open(path))
        tag = os.path.basename(os.path.dirname(path))
        prior_min = cfg['mcmc']['priors']['velocity.R_v']['min']
        floors = [(st.get('override') or {})['velocity.R_v']['min']
                  for st in cfg['fitting']
                  if 'velocity.R_v' in (st.get('override') or {})
                  and 'min' in (st.get('override') or {})['velocity.R_v']]
        assert floors, f'{tag}: no Adam floor on velocity.R_v at all'
        for f in floors:
            assert f >= prior_min, f'{tag}: Adam floor {f} < MCMC prior floor {prior_min}'


def test_R_v_is_never_floored_at_exactly_zero():
    '''At R_v = 0 the model is finite but its gradient is 0*(-inf) = NaN, and
    nothing downstream reports it: clamp_(min=0) leaves NaN alone and the
    railing test `v < lo or v > hi` is False for NaN.'''
    for path in sorted(glob.glob(os.path.join(REPO, 'data', 'ID*', 'config_kinematics.yaml'))):
        cfg = yaml.safe_load(open(path))
        for st in cfg['fitting']:
            ov = (st.get('override') or {}).get('velocity.R_v') or {}
            if 'min' in ov:
                assert ov['min'] > 0, f'{os.path.basename(os.path.dirname(path))}: R_v floored at 0'


def test_make_config_refuses_to_guess_the_module():
    '''`module` selects the trace and dispersion calibration. It used to fall
    back to 'A' when the 1D spectrum could not be found, and four of the eight
    staged sources are on B.'''
    from dingo import catalogs
    with pytest.raises(ValueError, match='never.*guessed|cannot determine'):
        catalogs._require_module(999999999, 'R', 'F444W', None, 'module_R')


def test_find_spec1d_dir_accepts_the_sibling_layout(tmp_path):
    '''sapphires_edr_catalogs/ and sapphires_edr_1d_spec/ are siblings, so
    joining the spec dirname onto the catalogue dir finds nothing.'''
    from dingo import catalogs
    root = tmp_path / 'EDR'
    (root / 'sapphires_edr_catalogs').mkdir(parents=True)
    (root / catalogs.SPEC1D_DIRNAME).mkdir()
    found = catalogs.find_spec1d_dir(str(root / 'sapphires_edr_catalogs'))
    assert found == str(root / catalogs.SPEC1D_DIRNAME)


def test_find_spec1d_dir_returns_none_when_there_is_nothing(tmp_path):
    from dingo import catalogs
    assert catalogs.find_spec1d_dir(str(tmp_path)) is None


@pytest.mark.parametrize('cfg_path', CONFIGS, ids=lambda p: os.path.basename(os.path.dirname(p)))
def test_the_rotation_amplitude_gets_a_learning_rate_for_its_scale(cfg_path):
    '''The amplitude must override the default lr, or the fit cannot move it.

    Adam steps a parameter by ~lr regardless of the gradient. The default 0.05
    suits theta_v (range 2pi) and R_v (~1); for an amplitude of 0-2500 km/s it
    allows ~98 km/s of travel over the StepLR schedule, which is about one comb
    tooth (spacing 75-95). The optimiser then stays in whichever tooth it was
    seeded in and returns the starting guess as the answer -- measured on
    ID15665, where five starts spanning 50-1100 km/s gave five answers spanning
    933 km/s at lr 0.05, and agreed to 64 km/s on the published value at lr 5.

    Asserts a floor rather than an exact value so a config may tune it, and
    names whichever amplitude that config defines: an override naming the other
    one matches no parameter, which check_unused_overrides raises on.
    '''
    cfg = yaml.safe_load(open(cfg_path))
    amps = [k for k in ('V_rot', 'V_sini') if k in (cfg.get('velocity') or {})]
    assert len(amps) == 1, f'expected exactly one amplitude, found {amps}'
    key = f'velocity.{amps[0]}'

    stage = cfg['fitting'][0]
    lr = ((stage.get('override') or {}).get(key) or {}).get('lr')
    assert lr is not None, (
        f'stage 1 does not override {key}, so it inherits the default '
        f'lr {stage["default"]["lr"]} and can barely move')
    assert lr >= 1.0, f'{key} lr {lr} is too small to cross a comb tooth'
