"""Calibrated shared line flux and positive-continuum real-source model.

This is a new probability model. Existing grids and posterior checkpoints are
not migrated. Optimization profiles positive continuum coefficients; the
NumPyro model samples them explicitly and never treats a profile as a marginal.
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import helmert

from .native_real_posterior import RealNativePosterior
from .native_real import profile_positive_amplitude
from .real_workflow import sha256


FLUX_UNIT = 1.e-17  # erg s^-1 cm^-2, integrated over the emission line
C_UM_S = 2.99792458e14


def integrated_line_response(sensitivity, wavelength_um, pixels_per_um):
    """DN/s per FLUX_UNIT from a per-spectral-pixel DN/s/Jy calibration.

    F_nu[Jy] dnu[Hz] * 1e-23 = F_line[erg/s/cm²]. The pixel's frequency
    interval is c/lambda² / |dp/dlambda|. Do not apply an exposure-time factor
    to SCI rate images, or confuse point spectral sensitivity with line flux.
    """
    sensitivity, wavelength_um, pixels_per_um = np.broadcast_arrays(
        sensitivity, wavelength_um, pixels_per_um)
    if (not all(np.isfinite(a).all() for a in (sensitivity, wavelength_um, pixels_per_um))
            or np.any(sensitivity <= 0) or np.any(wavelength_um <= 0)
            or np.any(pixels_per_um == 0)):
        raise ValueError('Positive sensitivity/wavelength and finite nonzero dispersion required.')
    return FLUX_UNIT / 1.e-23 * sensitivity * wavelength_um**2 * np.abs(pixels_per_um) / C_UM_S


def calibration_response(target, projector, bundle):
    """Evaluate response at the nominal line wavelength and catalog position.

    Within-line response variation is not yet included. Its range is reported
    separately; relative calibration scatter is a sensitivity assumption.
    """
    from astropy.table import Table
    root = Path(bundle) / 'calibration/grism_cal/sensitivity_model'
    coeff = np.asarray(projector.coefficients)
    center = coeff.shape[1] // 2
    wave = target.wavelength_um
    records, response = [], []
    for i, e in enumerate(target.exposures):
        path = root / f'F444W_mod{e.module}_grism{e.pupil}_sensitivity.dat'
        t = Table.read(path, format='ascii')
        wavelengths, values = np.asarray(t['wavelength']), np.asarray(t['DN/s/Jy'])
        if np.any(np.diff(wavelengths) <= 0) or not wavelengths[0] <= wave <= wavelengths[-1]:
            raise ValueError('Invalid sensitivity wavelength grid or extrapolated source wavelength.')
        sens = float(np.interp(wave, wavelengths, values))
        axis = 0 if e.pupil == 'R' else 1
        derivative = float(coeff[i, center, axis, 1])
        h = 1.e-5
        plus = e.calibration.project(*e.direct_xy, wave+h)[axis]
        minus = e.calibration.project(*e.direct_xy, wave-h)[axis]
        np.testing.assert_allclose(derivative, (plus-minus)/(2*h), rtol=1.e-7, atol=1.e-5)
        rate = float(integrated_line_response(sens, wave, derivative))
        response.append(rate)
        records.append(dict(exposure=i+1, pupil=e.pupil, module=e.module,
            sensitivity_path=str(path.resolve()), sensitivity_sha256=sha256(path),
            sensitivity_DN_s_per_Jy=sens, pixels_per_um=derivative,
            DN_s_per_1e17_line_flux=rate,
            sensitivity_ratio_at_v_pm1000=[float(np.interp(wave*(1+v/299792.458), wavelengths, values)/sens)
                                           for v in (-1000., 1000.)]))
    return np.asarray(response), dict(flux_unit_erg_s_cm2=FLUX_UNIT, records=records,
        approximation='Point response at systemic wavelength and catalog position; fixed within each source.',
        absolute_calibration_uncertainty_included=False)


class PositiveStripContinuum:
    """Exact three-coefficient NNLS in every strip, vectorized over exposures.

    A quadratic Bernstein basis on the full cutout has nonnegative weights.
    Nonnegative coefficients are sufficient, but not necessary, for positivity:
    this is a restricted smooth continuum family. A proper HalfNormal prior on
    each coefficient makes even rank-deficient strips well defined. Each pixel
    enters the likelihood once. No fitted sideband is recycled as a prior.
    """
    def __init__(self, science, error, valid, pupils, coefficient_scale=1.):
        science, error, valid = np.asarray(science), np.asarray(error), np.asarray(valid, bool)
        if science.shape != error.shape or science.shape != valid.shape or science.ndim != 3:
            raise ValueError('Matching [exposure,y,x] arrays required.')
        if science.shape[1] != science.shape[2] or len(pupils) != len(science):
            raise ValueError('Square native cutouts and one pupil per exposure required.')
        if set(pupils)-{'R', 'C'} or not np.isfinite(coefficient_scale) or coefficient_scale <= 0:
            raise ValueError('R/C pupils and positive finite coefficient scale required.')
        if np.any(valid & (~np.isfinite(science) | ~np.isfinite(error) | (error <= 0))):
            raise ValueError('Invalid science/error in selected pixels.')
        self.transpose = jnp.asarray(np.array(pupils) == 'C')[:, None, None]
        self.valid = jnp.asarray(valid)
        self.data = jnp.asarray(np.where(valid, science, 0.))
        self.inv = jnp.asarray(np.divide(1., error, out=np.zeros_like(error, dtype=float), where=valid))
        self.coefficient_scale = float(coefficient_scale)
        self.coefficient_shape = (len(science), science.shape[1], 3)
        t = np.linspace(0, 1, science.shape[-1])
        self.basis = jnp.asarray(np.stack(((1-t)**2, 2*t*(1-t), t*t), axis=-1))
        inv = np.asarray(self.strips(self.inv))
        gram = np.einsum('esd,dk,dl->eskl', inv**2, self.basis, self.basis)
        gram += np.eye(3) / coefficient_scale**2
        inverses = np.zeros((8, *gram.shape))
        for active in range(1, 8):
            keep = np.flatnonzero([(active >> i) & 1 for i in range(3)])
            small = np.linalg.inv(gram[..., keep[:, None], keep])
            for j, k in enumerate(keep):
                for l, m in enumerate(keep):
                    inverses[active, ..., k, m] = small[..., j, l]
        self.gram, self.inverses = jnp.asarray(gram), jnp.asarray(inverses)

    def strips(self, image):
        return jnp.where(self.transpose, image.swapaxes(-1, -2), image)

    def continuum(self, coefficients):
        return self.strips(jnp.einsum('dk,esk->esd', self.basis, coefficients))

    def profile(self, line, background):
        residual = self.strips(self.data-line-background[:, None, None])
        inv = self.strips(self.inv)
        rhs = jnp.einsum('esd,esd,dk->esk', residual, inv**2, self.basis)
        candidates = jnp.einsum('aeskl,esl->aesk', self.inverses, rhs)
        feasible = jnp.all(candidates >= 0., axis=-1)
        scores = (.5*jnp.einsum('aesk,eskl,aesl->aes', candidates, self.gram, candidates)
                  - jnp.einsum('aesk,esk->aes', candidates, rhs))
        best = jnp.argmin(jnp.where(feasible, scores, jnp.inf), axis=0)
        coefficients = jnp.take_along_axis(candidates, best[None, ..., None], axis=0)[0]
        # Envelope theorem: differentiate the optimized value, not the active
        # set solver. At exact active-set transitions the value is C1 here.
        coefficients = jax.lax.stop_gradient(coefficients)
        continuum = self.continuum(coefficients)
        chi2 = jnp.sum(((self.data-line-continuum-background[:, None, None])*self.inv)**2)
        penalty = .5*jnp.sum((coefficients/self.coefficient_scale)**2)
        return chi2, penalty, coefficients, continuum


class CoupledNativeModel:
    """Shared physical line flux; zero-sum exposure log-gains; signed background.

    Parameters: velocity/100, old native nuisance vector, F_line/1e-17,
    N-1 gain contrasts and N standardized background offsets. Existing
    dynamical/map/registration/inclination assumptions are retained exactly.
    """
    def __init__(self, engine, target, response, *, mode='positive', calibration_sigma=.1,
                 background_scale=1., continuum_scale=1., center_kind='catalog', center_mean=(0., 0.)):
        if mode not in ('positive', 'coupled', 'legacy') or calibration_sigma <= 0 or background_scale <= 0:
            raise ValueError('Unknown mode or nonpositive nuisance scales.')
        n = len(target.exposures)
        if n < 2 or np.shape(response) != (n,) or np.any(np.asarray(response) <= 0):
            raise ValueError('Positive response for each of at least two exposures required.')
        self.engine, self.mode, self.n = engine, mode, n
        self.renderer = RealNativePosterior(engine, center_kind=center_kind, center_mean=center_mean)
        self.center_kind, self.center_mean = center_kind, jnp.asarray(center_mean)
        self.base_size = len(engine.bounds)
        self.response = jnp.asarray(response)
        # Orthonormal contrasts remove the flux/gain gauge. Each exposure has
        # marginal log-gain SD calibration_sigma, with sum(log_gain)=0.
        self.gain_basis = jnp.asarray(helmert(n).T*np.sqrt(n/(n-1)))
        self.calibration_sigma = float(calibration_sigma)
        data, error, valid = [np.stack([getattr(e, k) for e in target.exposures])
                              for k in ('science', 'error', 'valid')]
        self.continuum_model = PositiveStripContinuum(data, error, valid,
            [e.pupil for e in target.exposures], coefficient_scale=continuum_scale)
        self.background_sigma = jnp.asarray([background_scale*np.median(e.error[e.valid]) for e in target.exposures])
        self.flux_scale = 100.  # HalfNormal, in FLUX_UNIT; tested as a proper weak prior
        self.bounds = [(0., 10.)] + list(engine.bounds) + [(0., None)] + [(None, None)]*(n-1)
        self.bounds += [(None, None) if mode == 'positive' else (0., 0.)]*n
        if mode == 'legacy':
            self.bounds[1+self.base_size:] = [(0.,0.)]*(2*n)
        if center_kind == 'free':
            self.bounds[6] = self.bounds[7] = (-6., 6.)

    def split(self, p):
        stop = 1+self.base_size
        return p[1:stop], 100*p[0], p[stop], p[stop+1:stop+self.n], p[stop+self.n:]

    def base_penalty(self, x):
        m = self.engine
        ce = .1*x[5:7]
        if self.center_kind == 'free':
            center = 0.
        else:
            sd = .05 if self.center_kind == 'stellar' else .15
            center = .5*jnp.sum(((ce-self.center_mean)/sd)**2)
        penalty = -jnp.interp(x[2], m.inclination_grid, m.inclination_logpdf)
        penalty += .5*jnp.sum(x[10:10+m.nmap]**2)+.5*jnp.sum(x[10+m.nmap:]**2)
        penalty += .5*((x[7]-np.log(.35))/.6)**2 + center + .5*((x[3]/3)**2+x[4]**2)
        return penalty

    def components(self, p):
        x, speed, flux_scale, gains_z, bg_z = self.split(p)
        unit, flux, velocity = self.renderer.render(x, speed)
        if self.mode == 'legacy':
            like = self.engine.likelihood
            template = like.residualize(jnp.where(like.valid,unit,0.)*like.inverse_error)
            amplitudes,_ = profile_positive_amplitude(template,like.residual_data)
            line = unit*amplitudes[:,None,None]
            chi2 = like.chi_square(line);penalty = self.base_penalty(x)
            # Relative fluxes here are a diagnostic, not calibration parameters.
            relative = amplitudes/self.response
            gains = relative/jnp.maximum(jnp.median(relative),1e-30)
            return .5*chi2+penalty,dict(chi2=chi2,penalty=penalty,line=line,
                continuum=jnp.zeros_like(line),continuum_coefficients=jnp.zeros((0,)),
                background=jnp.zeros(self.n),amplitudes=amplitudes,gains=gains,
                source_flux=flux,source_velocity=velocity)
        gains = jnp.exp(self.calibration_sigma*(self.gain_basis@gains_z))
        amplitudes = flux_scale*self.response*gains
        line = unit*amplitudes[:, None, None]
        background = self.background_sigma*bg_z
        penalty = self.base_penalty(x)+.5*(flux_scale/self.flux_scale)**2+.5*jnp.sum(gains_z**2)
        penalty += .5*jnp.sum(bg_z**2)
        if self.mode == 'positive':
            chi2, cp, coefficients, continuum = self.continuum_model.profile(line, background)
            penalty += cp
        else:
            chi2 = self.engine.likelihood.chi_square(line)
            coefficients, continuum = jnp.zeros((0,)), jnp.zeros_like(line)
        return .5*chi2+penalty, dict(chi2=chi2, penalty=penalty, line=line, continuum=continuum,
            continuum_coefficients=coefficients, background=background, amplitudes=amplitudes,
            gains=gains, source_flux=flux, source_velocity=velocity)

    def objective(self, p):
        return self.components(p)[0]

    def initial(self, vector, speed, flux=3.):
        return np.r_[speed/100., vector, 0. if self.mode=='legacy' else flux, np.zeros(2*self.n-1)]

    def posterior_initial_values(self, parameters):
        """Interior coordinates for the new target, never old-chain resume."""
        if self.mode == 'legacy':
            raise ValueError('Legacy profiling is an optimization ablation only.')
        x, speed, flux, gain_z, bg_z = self.split(np.asarray(parameters))
        values = self.renderer.initial_values(x, speed)
        values.update(line_flux_1e17=max(float(flux), 1e-5), gain_z=np.asarray(gain_z))
        if self.mode == 'positive':
            _, parts = self.components(jnp.asarray(parameters))
            floor = min(float(np.min(self.background_sigma))*.01,
                        self.continuum_model.coefficient_scale*.001)
            values.update(background_z=np.asarray(bg_z),
                continuum_coefficients=np.maximum(np.asarray(parts['continuum_coefficients']), floor))
        return values

    def model(self):
        """Proper NumPyro target, with explicit positive continuum nuisance draws.

        Optimization's profiled coefficients are NOT marginalized samples.
        The large continuum block needs sampler tuning before long runs.
        """
        import numpyro
        import numpyro.distributions as dist
        if self.mode == 'legacy':
            raise ValueError('Legacy profiling is an optimization ablation only.')
        m = self.engine
        v = numpyro.sample('vrot', dist.Uniform(0., 1000.))
        rt = numpyro.sample('log_rturn', dist.Uniform(np.log(.02), np.log(2.)))
        angle = numpyro.sample('theta', dist.Uniform(-np.pi, np.pi))
        inc = numpyro.sample('inclination', dist.Uniform(m.inclination_grid[0], m.inclination_grid[-1]))
        numpyro.factor('imaging_prior', jnp.interp(inc, m.inclination_grid, m.inclination_logpdf))
        systemic = numpyro.sample('systemic', dist.TruncatedNormal(0., 300., low=-600., high=600.))
        sigma = numpyro.sample('sigma_v', dist.TruncatedNormal(0., 100., low=0., high=200.))
        center_dist = (dist.Uniform(-.6, .6).expand((2,)) if self.center_kind == 'free' else
            dist.TruncatedNormal(self.center_mean, .05 if self.center_kind == 'stellar' else .15, low=-.3, high=.3))
        center = numpyro.sample('center', center_dist.to_event(1))
        width = numpyro.sample('log_width', dist.TruncatedNormal(np.log(.35), .6, low=np.log(.12), high=np.log(.8)))
        q = numpyro.sample('axis_ratio', dist.Uniform(.12, 1.))
        phi = numpyro.sample('emission_angle', dist.Uniform(-np.pi/2, np.pi/2))
        z = numpyro.sample('map_z', dist.Normal(0., 1.).expand((m.nmap,)).to_event(1))
        shifts = numpyro.sample('registration_z', dist.Normal(0., 1.).expand((self.n, 2)).to_event(2))
        total_flux = numpyro.sample('line_flux_1e17', dist.HalfNormal(self.flux_scale))
        gain_z = numpyro.sample('gain_z', dist.Normal(0., 1.).expand((self.n-1,)).to_event(1))
        x = jnp.concatenate((jnp.array([rt, angle, inc, systemic/100, sigma/100,
            center[0]/.1, center[1]/.1, width, q, phi]), z, shifts.ravel()))
        unit, _, _ = self.renderer.render(x, v)
        gains = jnp.exp(self.calibration_sigma*(self.gain_basis@gain_z))
        line = unit*(total_flux*self.response*gains)[:, None, None]
        if self.mode == 'positive':
            bg_z = numpyro.sample('background_z', dist.Normal(0., 1.).expand((self.n,)).to_event(1))
            cm = self.continuum_model
            coeff = numpyro.sample('continuum_coefficients',
                dist.HalfNormal(cm.coefficient_scale).expand(cm.coefficient_shape).to_event(3))
            mean = line+cm.continuum(coeff)+(self.background_sigma*bg_z)[:, None, None]
            chi2 = jnp.sum(((cm.data-mean)*cm.inv)**2)
        else:
            chi2 = self.engine.likelihood.chi_square(line)
        numpyro.factor('native_pixels', -.5*chi2)
