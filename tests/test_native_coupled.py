"""Probability, calibration-unit, active-set and derivative checks."""
from types import SimpleNamespace

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import nnls

from dingo.native_coupled import PositiveStripContinuum, CoupledNativeModel, integrated_line_response


def test_line_response_counts_frequency_interval_once():
    # A pixel of width 2e10 Hz containing a 1e-17 integrated line has
    # Fnu=5e-5 Jy; sensitivity 20000 DN/s/Jy therefore gives 1 DN/s.
    wavelength = 4.
    derivative = 2.99792458e14 / wavelength**2 / 2e10
    assert integrated_line_response(20000., wavelength, derivative) == pytest.approx(1.)
    assert integrated_line_response(20000., wavelength, -derivative) == pytest.approx(1.)
    with pytest.raises(ValueError):
        integrated_line_response(20000., wavelength, 0.)


def continuum_example():
    rng = np.random.default_rng(22)
    data = rng.normal(.3, .7, (2, 9, 9))
    error = rng.uniform(.5, 1., data.shape)
    valid = rng.uniform(size=data.shape) > .2
    valid[0, 0] = False  # Entirely masked strip: proper prior, zero MAP.
    data[~valid] = np.nan; error[~valid] = np.nan
    return PositiveStripContinuum(data, error, valid, ['R', 'C'], coefficient_scale=.8)


def test_positive_continuum_matches_independent_scipy_nnls_with_masks_and_background():
    m = continuum_example()
    line = np.full(m.data.shape, .4); background = np.array([-.2, .1])
    chi2, penalty, coeff, continuum = m.profile(line, background)
    inv = np.asarray(m.strips(m.inv)); residual = np.asarray(m.strips(m.data-line-background[:,None,None]))
    expected = np.zeros_like(coeff)
    for e in range(2):
        for s in range(9):
            design = np.vstack([np.asarray(m.basis)*inv[e,s,:,None], np.eye(3)/m.coefficient_scale])
            y = np.r_[residual[e,s]*inv[e,s], np.zeros(3)]
            expected[e,s] = nnls(design, y)[0]
    np.testing.assert_allclose(coeff, expected, atol=2e-13)
    assert np.min(continuum) >= 0
    assert np.min(np.asarray(continuum)[0]+background[0]) < 0  # Signed background remains allowed.
    assert np.isfinite([chi2, penalty]).all()


def test_profile_gradient_includes_line_background_and_optimized_continuum():
    m = continuum_example(); pattern = jnp.arange(m.data.size).reshape(m.data.shape)/m.data.size
    def objective(p):
        chi, penalty, _, _ = m.profile(p[0]*pattern, p[1:])
        return .5*chi+penalty
    p = np.array([.3, -.1, .05]); grad = np.asarray(jax.grad(objective)(p))
    for i in range(len(p)):
        step = np.eye(len(p))[i]*1e-5
        numeric = (float(objective(p+step))-float(objective(p-step)))/2e-5
        assert grad[i] == pytest.approx(numeric, rel=1e-7, abs=1e-7)


def small_model():
    from test_native_real_posterior import example
    engine = example()
    cm = engine.likelihood
    inv = np.asarray(cm.inverse_error)
    # A simple target for testing the joint target, with the same native geometry.
    data = np.ones(inv.shape)*.1
    from dingo.native_fitting import StripContinuumLikelihood
    from dingo.native_jax import JaxStripLikelihood
    like = StripContinuumLikelihood(data, 1/inv, np.array(cm.valid), ['R', 'C'])
    engine.likelihood = JaxStripLikelihood(like)
    target = SimpleNamespace(exposures=[SimpleNamespace(science=data[i], error=1/inv[i],
        valid=np.asarray(cm.valid[i]), pupil=p) for i,p in enumerate(('R', 'C'))])
    return CoupledNativeModel(engine, target, [1., 1.2])


def test_calibration_contrasts_remove_flux_gauge_and_forward_amplitudes_agree():
    m = small_model()
    np.testing.assert_allclose(np.asarray(m.gain_basis).sum(0), 0., atol=1e-14)
    np.testing.assert_allclose(np.diag(np.asarray(m.gain_basis@m.gain_basis.T)), 1.)
    p = m.initial(m.engine.initial(.7, 1.11), 200., flux=2.)
    _, parts = m.components(p)
    np.testing.assert_allclose(parts['amplitudes'], [2., 2.4])
    assert len(p) == len(m.bounds)


def test_explicit_numpyro_continuum_target_matches_profile_map_density():
    # Compare changes in probability, with constants cancelling. This catches
    # silently reusing the old amplitude-integrated or flat-continuum target.
    from numpyro.infer.util import log_density
    m = small_model(); engine = m.engine
    offsets = []
    for delta in (0., .1):
        x = engine.initial(.7, 1.11)
        x[3] = delta; x[4] = .3+delta; x[5] = delta; x[10] = delta
        p = m.initial(x, 200+100*delta, flux=2.+delta)
        p[-3:] = [delta, delta, -delta]
        value, parts = m.components(p)
        # Approach boundary coefficients from the interior; NumPyro's positive
        # support excludes exactly zero even though the density has a finite limit.
        params = m.renderer.initial_values(x, 200+100*delta)
        params |= dict(line_flux_1e17=2.+delta, gain_z=np.array([delta]),
                       background_z=np.array([delta, -delta]),
                       continuum_coefficients=np.maximum(parts['continuum_coefficients'], 1e-12))
        logp, _ = log_density(m.model, (), {}, params)
        offsets.append(float(logp+value))
    assert offsets[0] == pytest.approx(offsets[1], abs=1e-8)


def test_legacy_ablation_preserves_original_target_up_to_data_constant():
    m=small_model()
    # Rebuild with fixed auxiliary coordinates, as the worker does.
    m.mode='legacy';offsets=[]
    for v in (100.,400.):
        x=m.engine.initial(.7,1.11)
        p=m.initial(x,v)
        old=m.engine.objective(jnp.asarray(x),v)
        new=m.objective(jnp.asarray(p))
        offsets.append(float(new-old))
    assert offsets[0] == pytest.approx(offsets[1],abs=1e-9)


def test_explicit_detector_mask_is_scoped_and_preserves_all_other_pixels(tmp_path):
    from pathlib import Path
    from dingo.real_workflow import write_json
    from dingo.native_quality import apply_detector_mask
    exposures=[]
    for module,file,origin in [('B','b.fits',(10,20)),('A','b.fits',(10,20)),
                              ('B','other.fits',(10,20)),('B','b.fits',(100,200))]:
        exposures.append(SimpleNamespace(module=module,path=Path(file),origin_xy=origin,
            valid=np.ones((5,5),bool),metadata={},pupil='C'))
    path=tmp_path/'mask.json'
    write_json(path,dict(format='native_detector_mask_v1',coordinate_origin=0,pixels=[dict(
        module='B',x=12,y=23,exposure_basenames=['b.fits'],reason='Independent repeated detector defect')]))
    report=apply_detector_mask(SimpleNamespace(exposures=exposures),path)
    assert report['newly_excluded']==1
    assert not exposures[0].valid[3,2]
    assert sum((~e.valid).sum() for e in exposures)==1
    assert apply_detector_mask(SimpleNamespace(exposures=exposures),path)['newly_excluded']==0
