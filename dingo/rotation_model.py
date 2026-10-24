"""Rotation-only flexible R+C model and positive-continuum Gibbs updates.

The target is the calibrated ``CoupledNativeModel`` with intrinsic dispersion
fixed exactly to zero. Its positive continuum is sampled, never substituted by
its optimum when computing posterior uncertainty. Archived models are untouched.
"""
from copy import copy

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtr, ndtri

from .native_coupled import CoupledNativeModel


def positive_normal(rng_key, location, scale):
    """Draw Normal(location, scale), truncated below zero, including deep tails.

    For a positive standardized cutoff, exponential rejection avoids subtracting
    nearly equal CDFs. Elsewhere reflected inverse-CDF sampling is well scaled.
    This is an exact rejection sampler, not clipping a Normal draw to zero.
    """
    location, scale = jnp.broadcast_arrays(jnp.asarray(location), jnp.asarray(scale))
    dtype = jnp.result_type(location, scale, float)
    location, scale = location.astype(dtype), scale.astype(dtype)
    cutoff = -location/scale
    direct_key, tail_key = jax.random.split(rng_key)
    u = jax.random.uniform(direct_key, location.shape, dtype=dtype,
                           minval=jnp.finfo(dtype).tiny)
    # Evaluate this expression only at nonpositive cutoffs, even in tail slots.
    direct = location-scale*ndtri(u*ndtr(-jnp.minimum(cutoff, 0.)))
    a = jnp.maximum(cutoff, 0.)
    rate = .5*(a+jnp.hypot(a, 2.))

    def unfinished(state):
        return jnp.any(~state[2])

    def propose(state):
        key, excess, accepted = state
        key, exponential_key, accept_key = jax.random.split(key, 3)
        proposal = jax.random.exponential(exponential_key, location.shape, dtype=dtype)/rate
        log_uniform = jnp.log(jax.random.uniform(accept_key, location.shape,
                              dtype=dtype, minval=jnp.finfo(dtype).tiny))
        take = (~accepted) & (log_uniform < -.5*(a+proposal-rate)**2)
        return key, jnp.where(take, proposal, excess), accepted | take

    _, excess, _ = jax.lax.while_loop(unfinished, propose,
        (tail_key, jnp.zeros_like(location), cutoff <= 0.))
    return jnp.where(cutoff > 0., scale*excess, direct)


def continuum_gibbs(rng_key, coefficients, gram, rhs, sweeps=4):
    """Invariant coordinate-Gibbs transition for independent positive strips.

    Each strip has density proportional to exp(-c.T G c/2 + h.T c), c >= 0.
    All strips update in parallel; the three coefficients update sequentially.
    Multiple sweeps reduce correlation; they are not independent block draws.
    """
    def update(index, state):
        key, values = state
        key, draw_key = jax.random.split(key)
        column = index % 3
        precision = gram[..., column, column]
        numerator = rhs[..., column]-jnp.sum(gram[..., column, :]*values, axis=-1)
        numerator += precision*values[..., column]
        draw = positive_normal(draw_key, numerator/precision, 1/jnp.sqrt(precision))
        return key, values.at[..., column].set(draw)
    return jax.lax.fori_loop(0, 3*sweeps, update, (rng_key, coefficients))[1]


class RotationOnlyModel(CoupledNativeModel):
    """The flexible joint R+C target, with sigma fixed to zero in every path.

    Full optimizer vectors retain the old sigma slot (index 5), fixed at zero,
    so existing fit artifacts can initialize a new run without changing their
    coordinates. The posterior has no sampled dispersion parameter. A copied
    one-node spectral quadrature is exactly equivalent at sigma=0 and avoids
    depositing the same monochromatic emission repeatedly.
    """
    sigma_index = 5
    scalar_sites = ('vrot', 'log_rturn', 'theta', 'inclination', 'systemic',
                    'center', 'log_width', 'axis_ratio', 'emission_angle',
                    'line_flux_1e17')

    def __init__(self, engine, target, response, *, continuum_sweeps=4, **kwargs):
        if int(continuum_sweeps) != continuum_sweeps or continuum_sweeps < 1:
            raise ValueError('continuum_sweeps must be a positive integer.')
        if kwargs.get('mode', 'positive') != 'positive':
            raise ValueError('Rotation-only posterior requires the positive continuum model.')
        copied_engine = copy(engine)
        # NumPyro's Gibbs wrapper classifies array-valued interval endpoints as
        # dynamic supports. These endpoints are fixed calibration/prior data:
        # retaining the grid as host float64 makes its scalar endpoints static
        # Python-compatible numbers, without changing interpolation or density.
        copied_engine.inclination_grid = np.asarray(engine.inclination_grid, dtype=np.float64)
        copied_engine.projector = copy(engine.projector)
        copied_engine.projector.quadrature = jnp.zeros(1, dtype=engine.projector.quadrature.dtype)
        copied_engine.projector.quadrature_weights = jnp.ones(1, dtype=engine.projector.quadrature_weights.dtype)
        super().__init__(copied_engine, target, response, **kwargs)
        self.bounds[self.sigma_index] = (0., 0.)
        self.continuum_sweeps = int(continuum_sweeps)

    def split(self, parameters):
        x, speed, flux, gain_z, background_z = super().split(parameters)
        return jnp.asarray(x).at[4].set(0.), speed, flux, gain_z, background_z

    def initial(self, vector, speed, flux=3.):
        parameters = super().initial(vector, speed, flux)
        parameters[self.sigma_index] = 0.
        return parameters

    def posterior_initial_values(self, parameters):
        parameters = np.asarray(parameters)
        if parameters.shape != (len(self.bounds),) or parameters[self.sigma_index] != 0.:
            raise ValueError('A matching rotation-only parameter vector with sigma exactly zero is required.')
        values = super().posterior_initial_values(parameters)
        values.pop('sigma_v')
        return values

    def model(self):
        from numpyro.handlers import condition
        # Conditioning leaves an irrelevant fixed prior-density constant. It
        # retains precisely the shared flux/calibration/continuum probability
        # model; no dispersion coordinate enters HMC or the uncertainty report.
        return condition(super().model, data={'sigma_v': jnp.asarray(0.)})()

    def vector_from_sites(self, sites):
        return self.renderer.vector(dict(sites, sigma_v=jnp.asarray(0.)))

    def line_from_sites(self, sites):
        unit, _, _ = self.renderer.render(self.vector_from_sites(sites), sites['vrot'])
        gains = jnp.exp(self.calibration_sigma*(self.gain_basis@sites['gain_z']))
        return unit*(sites['line_flux_1e17']*self.response*gains)[:, None, None]

    def gibbs_fn(self, rng_key, gibbs_sites, hmc_sites):
        """Update positive continuum conditional on the current physical draw."""
        cm = self.continuum_model
        line = self.line_from_sites(hmc_sites)
        background = self.background_sigma*hmc_sites['background_z']
        residual = cm.strips(cm.data-line-background[:, None, None])
        rhs = jnp.einsum('esd,esd,dk->esk', residual, cm.strips(cm.inv)**2, cm.basis)
        coefficients = continuum_gibbs(rng_key, gibbs_sites['continuum_coefficients'],
                                      cm.gram, rhs, self.continuum_sweeps)
        return {'continuum_coefficients': coefficients}

    def prior_description(self):
        prior = self.renderer.prior_description()
        # Remove the old renderer's independently integrated amplitudes and
        # flat continuum, neither of which describes this probability model.
        prior.pop('exposure_amplitudes', None)
        prior['intrinsic_width'] = 'Fixed exactly to zero; not sampled'
        prior['line_flux_1e17'] = f'HalfNormal({self.flux_scale:g}), shared by all exposures'
        prior['relative_calibration'] = (f'Zero-sum log-gains with marginal SD {self.calibration_sigma:g}')
        prior['continuum'] = (f'Positive quadratic strip coefficients with HalfNormal({self.continuum_model.coefficient_scale:g}) '
                              'prior; sampled conditionally, never profiled for posterior inference')
        prior['background'] = 'Independent standard Normal coordinates times saved per-exposure background scales'
        return prior
