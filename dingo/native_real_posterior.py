"""Posterior for the real-source native R+C model, independent of mock runs.

The forward operator and finite map match RealVelocityProfile. Exposure line
amplitudes are integrated under independent HalfNormal priors, not profiled.
The fixed linear continuum is integrated with the same flat measure as before;
its determinant is constant in these parameters. This is not model evidence.
"""
import numpy as np
import jax.numpy as jnp
from jax.scipy.special import log_ndtr
import numpyro
import numpyro.distributions as dist

from .native_jax import projected_velocity


def integrated_positive_amplitudes(template, residual_data, scale=100.):
    """Log likelihood relative to zero line, integrating A >= 0 exactly.

    template and residual_data are continuum-projected, ERR-whitened arrays.
    A has HalfNormal(scale), in integrated detector count-rate units (DN/s).
    Returns the log integral and conditional Normal location/scale; the actual
    conditional amplitude distribution is that Normal truncated below zero.
    Unlike profiling, this includes the shape-dependent precision determinant.
    """
    precision = jnp.sum(template**2, axis=(-2, -1)) + 1/scale**2
    numerator = jnp.sum(template*residual_data, axis=(-2, -1))
    sd = 1/jnp.sqrt(precision)
    location = numerator/precision
    standardized = numerator*sd
    log_integral = (jnp.log(2./scale) - .5*jnp.log(precision)
                    + .5*standardized**2 + log_ndtr(standardized))
    return jnp.sum(log_integral), location, sd


class RealNativePosterior:
    """Sample physical parameters, the entire map and exposure registration.

    Map smoothness/variance, PSF/LSF, masks and errors retain the saved grid's
    settings. A conditional posterior under these assumptions is not a total
    systematic error budget. The image-derived inclination density is reused
    exactly, without introducing a second independent orientation constraint.
    """
    scalar_sites = ('vrot', 'log_rturn', 'theta', 'inclination', 'systemic',
                    'sigma_v', 'center', 'log_width', 'axis_ratio', 'emission_angle')

    def __init__(self, engine, *, vmax=1000., amplitude_scale=100.,
                 center_kind='catalog', center_mean=(0., 0.), angle_anchor=0.):
        if not np.isfinite([vmax, amplitude_scale, angle_anchor]).all() or min(vmax, amplitude_scale) <= 0:
            raise ValueError('Positive finite velocity and amplitude prior scales required.')
        if center_kind not in ('catalog', 'stellar', 'free'):
            raise ValueError('Unknown center prior.')
        self.engine = engine
        self.vmax, self.amplitude_scale = float(vmax), float(amplitude_scale)
        self.center_kind, self.center_mean = center_kind, jnp.asarray(center_mean)
        self.angle_anchor = float(angle_anchor)

    def render(self, x, speed):
        """Unit-amplitude line prediction, integrated source flux and LOS speed."""
        m = self.engine
        rt, theta, inc = jnp.exp(x[0]), x[1], x[2]
        ce, cn = .1*x[5], .1*x[6]
        width, q, phi = jnp.exp(x[7]), x[8], x[9]
        a = jnp.cos(phi)*m.east_controls+jnp.sin(phi)*m.north_controls
        b = -jnp.sin(phi)*m.east_controls+jnp.cos(phi)*m.north_controls
        log_controls = -.5*(a*a+(b/q)**2)/width**2 + m.map_sigma*(m.basis@x[10:10+m.nmap])
        log_quad = sum(log_controls[idx]*weight for idx, weight in zip(m.line_map.indices, m.line_map.weights))
        flux = jnp.exp(log_quad-jnp.max(log_quad))*m.line_map.area
        flux /= flux.sum()
        velocity = projected_velocity(m.line_map.east-ce, m.line_map.north-cn,
                                      speed*jnp.sin(inc), rt, theta, inc) + 100*x[3]
        shifts = m.registration_sigma*x[10+m.nmap:].reshape(m.projector.exposures, 2)
        return m.projector(flux, velocity, sigma_v=100*x[4], shifts=shifts), flux, velocity

    def log_likelihood(self, x, speed):
        unit_line, _, _ = self.render(x, speed)
        like = self.engine.likelihood
        template = like.residualize(jnp.where(like.valid, unit_line, 0.)*like.inverse_error)
        return integrated_positive_amplitudes(template, like.residual_data, self.amplitude_scale)[0]

    def vector(self, p):
        return jnp.concatenate((jnp.array([p['log_rturn'], p['theta'], p['inclination'],
            p['systemic']/100, p['sigma_v']/100, p['center'][0]/.1, p['center'][1]/.1,
            p['log_width'], p['axis_ratio'], p['emission_angle']]),
            p['map_z'], p['registration_z'].ravel()))

    def model(self):
        m = self.engine
        p = {}
        p['vrot'] = numpyro.sample('vrot', dist.Uniform(0., self.vmax))
        p['log_rturn'] = numpyro.sample('log_rturn', dist.Uniform(np.log(.02), np.log(2.)))
        p['theta'] = numpyro.sample('theta', dist.Uniform(self.angle_anchor-np.pi, self.angle_anchor+np.pi))
        p['inclination'] = numpyro.sample('inclination', dist.Uniform(m.inclination_grid[0], m.inclination_grid[-1]))
        numpyro.factor('external_inclination', jnp.interp(p['inclination'], m.inclination_grid, m.inclination_logpdf))
        p['systemic'] = numpyro.sample('systemic', dist.TruncatedNormal(0., 300., low=-600., high=600.))
        p['sigma_v'] = numpyro.sample('sigma_v', dist.TruncatedNormal(0., 100., low=0., high=200.))
        if self.center_kind == 'free':
            center_prior = dist.Uniform(-.6, .6).expand((2,)).to_event(1)
        else:
            sd = .05 if self.center_kind == 'stellar' else .15
            center_prior = dist.TruncatedNormal(self.center_mean, sd, low=-.3, high=.3).to_event(1)
        p['center'] = numpyro.sample('center', center_prior)
        p['log_width'] = numpyro.sample('log_width', dist.TruncatedNormal(np.log(.35), .6,
                                        low=np.log(.12), high=np.log(.8)))
        p['axis_ratio'] = numpyro.sample('axis_ratio', dist.Uniform(.12, 1.))
        p['emission_angle'] = numpyro.sample('emission_angle', dist.Uniform(-np.pi/2, np.pi/2))
        p['map_z'] = numpyro.sample('map_z', dist.Normal(0., 1.).expand((m.nmap,)).to_event(1))
        p['registration_z'] = numpyro.sample('registration_z', dist.Normal(0., 1.).expand((m.projector.exposures, 2)).to_event(2))
        numpyro.factor('native_pixels', self.log_likelihood(self.vector(p), p['vrot']))

    def initial_values(self, vector, speed):
        """Map an existing grid branch into finite, interior sampler coordinates."""
        x = np.asarray(vector)
        if x.shape != (10+self.engine.nmap+2*self.engine.projector.exposures,):
            raise ValueError('Grid vector does not match the posterior geometry.')
        theta = self.angle_anchor+(x[1]-self.angle_anchor+np.pi)%(2*np.pi)-np.pi
        bound = .6 if self.center_kind == 'free' else .3
        return dict(vrot=np.clip(speed, .001*self.vmax, .999*self.vmax),
            log_rturn=np.clip(x[0], np.log(.0201), np.log(1.999)), theta=theta,
            inclination=np.clip(x[2], float(self.engine.inclination_grid[0])+.001,
                                float(self.engine.inclination_grid[-1])-.001),
            systemic=np.clip(100*x[3], -599., 599.), sigma_v=np.clip(100*x[4], 5., 195.),
            center=np.clip(.1*x[5:7], -bound+.001, bound-.001),
            log_width=np.clip(x[7], np.log(.1201), np.log(.7999)),
            axis_ratio=np.clip(x[8], .121, .99), emission_angle=(x[9]+np.pi/2)%np.pi-np.pi/2,
            map_z=x[10:10+self.engine.nmap].copy(),
            registration_z=x[10+self.engine.nmap:].reshape(self.engine.projector.exposures, 2).copy())

    def prior_description(self):
        return dict(velocity=f'Uniform(0, {self.vmax:g}) km/s, deprojected V(0.5 arcsec)',
            turnover='LogUniform(0.02, 2) arcsec', kinematic_angle='Uniform over a full 2*pi cycle',
            inclination='Exactly the saved grid inclination density, bounded at 5–85 degrees',
            systemic='Normal(0, 300) km/s, truncated to +/-600',
            intrinsic_width='HalfNormal(100) km/s, truncated at 200',
            center_kind=self.center_kind, center_mean_arcsec=np.asarray(self.center_mean).tolist(),
            center='Saved grid bounds and Gaussian scale; free uses Uniform(-0.6, 0.6) in each axis',
            log_envelope_width='Normal(log(0.35), 0.6), bounded at log(0.12), log(0.8)',
            emission_axis_ratio='Uniform(0.12, 1)', emission_angle='Uniform over one pi cycle',
            map_coefficients='Independent standard Normals in the saved smooth-field basis',
            registration=f'Independent Normal(0, {self.engine.registration_sigma}) detector pixels',
            exposure_amplitudes=f'Independent HalfNormal({self.amplitude_scale:g}) DN/s; analytically integrated',
            continuum='Fixed-design flat-measure integration; constant determinant omitted; not model evidence',
            fixed_map_sigma=self.engine.map_sigma,
            note='Uniform angular intervals cover complete physical cycles; their seams are coordinate choices.')
