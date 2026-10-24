"""A shared exponential emission disk on the existing native R+C operator.

This uses the n=1 member of old DINGO's analytic Sersic family. It is not the
inverse-image objective used by the old Archive6 notebook. Light geometry is
independent of rotation geometry; calibration and continuum are unchanged.
"""
import copy

import jax.numpy as jnp
import numpy as np
from scipy.special import gammaincinv

from .native_coupled import CoupledNativeModel
from .native_jax import projected_velocity

B1 = float(gammaincinv(2., .5))
RADIUS_EPS_ARCSEC = 1e-6


def exponential_weights(east, north, area, log_re, q, angle, center):
    """Normalized quadrature flux; Re is the untruncated major-axis half-light radius.

    Flux normalization is over the same finite sky support as the flexible
    model. Thus the fitted flux is within that support, not extrapolated total
    flux. The tiny central softening only avoids a derivative singularity.
    """
    dx, dy = east-center[0], north-center[1]
    a = jnp.cos(angle)*dx+jnp.sin(angle)*dy
    b = -jnp.sin(angle)*dx+jnp.cos(angle)*dy
    radius = jnp.sqrt(a*a+(b/q)**2+RADIUS_EPS_ARCSEC**2)
    log_light = -B1*radius/jnp.exp(log_re)
    weights = jnp.exp(log_light-jnp.max(log_light))*area
    return weights/weights.sum()


class ExponentialRenderer:
    def __init__(self, engine):
        self.engine = engine

    def render(self, x, speed):
        m = self.engine
        flux = exponential_weights(m.line_map.east, m.line_map.north, m.line_map.area,
                                   x[7], x[8], x[9], .1*x[10:12])
        velocity = projected_velocity(m.line_map.east-.1*x[5], m.line_map.north-.1*x[6],
            speed*jnp.sin(x[2]), jnp.exp(x[0]), x[1], x[2])+100*x[3]
        shifts = m.registration_sigma*x[12:].reshape(m.projector.exposures, 2)
        return m.projector(flux, velocity, sigma_v=100*x[4], shifts=shifts), flux, velocity


class ExponentialCoupledModel(CoupledNativeModel):
    """Five light-shape coordinates instead of hundreds of map coefficients.

    Base vector: log(rt), theta, inc, vsys/100, sigma/100, kin center/.1 (2),
    log(Re), q_light, phi_light, light center/.1 (2), registration_z (2N).
    Full vector keeps velocity/100 first and the existing flux/gain/bg tail.
    """
    def __init__(self, engine, target, response, **kwargs):
        self.original_base_size = len(engine.bounds)
        self.original_nmap = engine.nmap
        compact = copy.copy(engine)
        compact.nmap = 0
        compact.bounds = list(engine.bounds[:10])+[(-3., 3.)]*2+list(engine.bounds[10+engine.nmap:])
        compact.bounds[7] = (np.log(.05), np.log(.8))
        compact.bounds[8] = (.1, 1.)
        super().__init__(compact, target, response, **kwargs)
        if self.mode == 'legacy':
            raise ValueError('The new light model requires the shared physical-flux target.')
        self.renderer = ExponentialRenderer(compact)

    def base_penalty(self, x):
        penalty = -jnp.interp(x[2], self.engine.inclination_grid, self.engine.inclination_logpdf)
        if self.center_kind != 'free':
            sd = .05 if self.center_kind == 'stellar' else .15
            penalty += .5*jnp.sum(((.1*x[5:7]-self.center_mean)/sd)**2)
        penalty += .5*jnp.sum((.1*x[10:12]/.15)**2)
        penalty += .5*jnp.sum(x[12:]**2)+.5*((x[3]/3)**2+x[4]**2)
        # Uniform in log(Re), q, and angle: no inherited Gaussian-envelope prior.
        return penalty

    def from_flexible(self, parameters, source_flux=None):
        """Map an existing solution into an initial exponential, never a fitted result."""
        p = np.asarray(parameters)
        expected = 1+self.original_base_size+2*self.n
        if p.shape != (expected,):
            raise ValueError('Flexible parameter vector has the wrong geometry.')
        old = p[1:1+self.original_base_size]
        center = np.zeros(2)
        re, q, phi = B1*np.exp(old[7])/np.sqrt(3), old[8], old[9]
        if source_flux is not None:
            weights = np.asarray(source_flux, float)
            xy = np.stack((self.engine.line_map.east, self.engine.line_map.north), axis=1)
            if weights.shape != (len(xy),) or np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
                raise ValueError('Expected nonnegative finite source quadrature flux.')
            weights = weights/weights.sum()
            center = weights@xy
            centered = xy-center
            cov = np.einsum('n,ni,nj->ij', weights, centered, centered)
            eigen, vectors = np.linalg.eigh(cov)
            re = B1*np.sqrt(max(eigen[1], 1e-8)/3)
            q = np.sqrt(max(eigen[0], 1e-8)/max(eigen[1], 1e-8))
            phi = np.arctan2(vectors[1, 1], vectors[0, 1])
        x = np.r_[old[:10], np.clip(center/.1, -2.99, 2.99), old[10+self.original_nmap:]]
        x[7:10] = [np.log(np.clip(re, .051, .79)), np.clip(q, .101, .99), phi]
        return np.r_[p[0], x, p[1+self.original_base_size:]]

    def describe(self, parameters):
        x, speed, _, _, _ = self.split(np.asarray(parameters))
        result = self.engine.describe(x, speed)
        result.pop('envelope_width_arcsec'); result.pop('envelope_axis_ratio')
        result.update(emission_model='exponential', sersic_n=1., emission_re_arcsec=float(np.exp(x[7])),
            emission_axis_ratio=float(x[8]), emission_angle_deg=float(np.rad2deg(x[9]) % 180),
            emission_center_east_north_arcsec=(.1*x[10:12]).tolist())
        return result

    def prior_description(self):
        return dict(emission_model='Elliptical Sersic n=1 (exponential); independent light and rotation geometry',
            emission_re='Uniform in log(Re) between 0.05 and 0.8 arcsec; major-axis untruncated half-light radius',
            emission_axis_ratio='Uniform(0.1,1), independent of kinematic inclination',
            emission_angle='Uniform over one pi cycle, independent of kinematic angle',
            emission_center='Independent Normal(0,0.15 arcsec), truncated to +/-0.3 in each axis',
            intrinsic_width='HalfNormal(100 km/s), truncated at 200; scan fixes width outside optimizer',
            source_flux='Flux within the finite quadrature support; no extrapolated total flux',
            central_radius_softening_arcsec=RADIUS_EPS_ARCSEC,
            kinematic_center_kind=self.center_kind, n_emission_shape_parameters=5)

    def posterior_initial_values(self, parameters):
        x, speed, flux, gains, bg = self.split(np.asarray(parameters))
        bound = .6 if self.center_kind == 'free' else .3
        values = dict(vrot=np.clip(speed, 1., 999.), log_rturn=np.clip(x[0], np.log(.0201), np.log(1.999)),
            theta=(x[1]+np.pi)%(2*np.pi)-np.pi,
            inclination=np.clip(x[2], float(self.engine.inclination_grid[0])+.001,
                                float(self.engine.inclination_grid[-1])-.001),
            systemic=np.clip(100*x[3], -599., 599.), sigma_v=np.clip(100*x[4], 5., 195.),
            center=np.clip(.1*x[5:7], -bound+.001, bound-.001),
            log_re=np.clip(x[7], np.log(.0501), np.log(.7999)), axis_ratio=np.clip(x[8], .101, .999),
            emission_angle=(x[9]+np.pi/2)%np.pi-np.pi/2, emission_center=np.clip(.1*x[10:12], -.299, .299),
            registration_z=x[12:].reshape(self.n, 2), line_flux_1e17=max(float(flux), 1e-5), gain_z=gains)
        if self.mode == 'positive':
            _, parts = self.components(jnp.asarray(parameters))
            values.update(background_z=bg, continuum_coefficients=np.maximum(parts['continuum_coefficients'], 1e-8))
        return values

    def model(self):
        """Matching NumPyro density for future sampling; this module launches no MCMC."""
        import numpyro
        import numpyro.distributions as dist
        m = self.engine
        speed = numpyro.sample('vrot', dist.Uniform(0., 1000.))
        rt = numpyro.sample('log_rturn', dist.Uniform(np.log(.02), np.log(2.)))
        angle = numpyro.sample('theta', dist.Uniform(-np.pi, np.pi))
        inc = numpyro.sample('inclination', dist.Uniform(m.inclination_grid[0], m.inclination_grid[-1]))
        numpyro.factor('inclination_prior', jnp.interp(inc, m.inclination_grid, m.inclination_logpdf))
        systemic = numpyro.sample('systemic', dist.TruncatedNormal(0., 300., low=-600., high=600.))
        sigma = numpyro.sample('sigma_v', dist.TruncatedNormal(0., 100., low=0., high=200.))
        center_dist = (dist.Uniform(-.6, .6).expand((2,)) if self.center_kind == 'free' else
            dist.TruncatedNormal(self.center_mean, .05 if self.center_kind == 'stellar' else .15, low=-.3, high=.3))
        center = numpyro.sample('center', center_dist.to_event(1))
        re = numpyro.sample('log_re', dist.Uniform(np.log(.05), np.log(.8)))
        q = numpyro.sample('axis_ratio', dist.Uniform(.1, 1.))
        phi = numpyro.sample('emission_angle', dist.Uniform(-np.pi/2, np.pi/2))
        light_center = numpyro.sample('emission_center', dist.TruncatedNormal(0., .15, low=-.3, high=.3).expand((2,)).to_event(1))
        shifts = numpyro.sample('registration_z', dist.Normal(0., 1.).expand((self.n, 2)).to_event(2))
        flux = numpyro.sample('line_flux_1e17', dist.HalfNormal(self.flux_scale))
        gains_z = numpyro.sample('gain_z', dist.Normal(0., 1.).expand((self.n-1,)).to_event(1))
        x = jnp.concatenate((jnp.array([rt, angle, inc, systemic/100, sigma/100, center[0]/.1,
            center[1]/.1, re, q, phi]), light_center/.1, shifts.ravel()))
        unit, _, _ = self.renderer.render(x, speed)
        line = unit*(flux*self.response*jnp.exp(self.calibration_sigma*(self.gain_basis@gains_z)))[:, None, None]
        if self.mode == 'positive':
            bg = numpyro.sample('background_z', dist.Normal(0., 1.).expand((self.n,)).to_event(1))
            cm = self.continuum_model
            coeff = numpyro.sample('continuum_coefficients', dist.HalfNormal(cm.coefficient_scale).expand(cm.coefficient_shape).to_event(3))
            mean = line+cm.continuum(coeff)+(self.background_sigma*bg)[:, None, None]
            chi2 = jnp.sum(((cm.data-mean)*cm.inv)**2)
        else:
            chi2 = self.engine.likelihood.chi_square(line)
        numpyro.factor('native_pixels', -.5*chi2)
