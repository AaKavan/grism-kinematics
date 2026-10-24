"""A linked exponential disk used only as a restrictive morphology control.

The light and circular velocity field share center, position angle and
inclination. This is a different probability model from flexible emission;
its agreement or disagreement cannot establish the flexible model's accuracy.
"""
import copy

import jax.numpy as jnp
import numpy as np

from .native_coupled import CoupledNativeModel
from .native_jax import projected_velocity
from .native_simple import B1, RADIUS_EPS_ARCSEC, exponential_weights


def projected_axis_ratio(inclination, thickness=0.):
    """Projected oblate light axis ratio; velocity remains a thin-plane field."""
    return jnp.sqrt(jnp.cos(inclination)**2+thickness**2*jnp.sin(inclination)**2)


class LinkedExponentialRenderer:
    def __init__(self, engine, thickness):
        self.engine, self.thickness = engine, thickness

    def render(self, x, speed):
        m = self.engine
        q = projected_axis_ratio(x[2], self.thickness)
        flux = exponential_weights(m.line_map.east, m.line_map.north, m.line_map.area,
                                   x[7], q, x[1], .1*x[5:7])
        velocity = projected_velocity(m.line_map.east-.1*x[5], m.line_map.north-.1*x[6],
            speed*jnp.sin(x[2]), jnp.exp(x[0]), x[1], x[2])+100*x[3]
        shifts = m.registration_sigma*x[8:].reshape(m.projector.exposures, 2)
        return m.projector(flux, velocity, sigma_v=100*x[4], shifts=shifts), flux, velocity


class LinkedExponentialCoupledModel(CoupledNativeModel):
    """One independent light parameter (Re), with shared disk geometry.

    Base x: log(rt), theta, inc, vsys/100, sigma/100, center/.1 (2),
    log(Re), registration_z (2N). Full p: speed/100, x, physical line
    flux, gain contrasts (N-1), signed backgrounds (N). The light is not
    tied to stellar morphology. Nonzero fixed thickness changes projected
    light ellipticity only; it does not perform a 3D line-of-sight integral.
    """
    def __init__(self, engine, target, response, thickness=0., **kwargs):
        if not np.isfinite(thickness) or not 0 <= thickness < 1:
            raise ValueError('Fixed thickness must be finite and in [0,1).')
        self.thickness = float(thickness)
        self.original_base_size = len(engine.bounds)
        self.original_nmap = engine.nmap
        compact = copy.copy(engine)
        compact.nmap = 0
        compact.bounds = list(engine.bounds[:7])+[(np.log(.05), np.log(.8))]
        compact.bounds += list(engine.bounds[10+engine.nmap:])
        super().__init__(compact, target, response, **kwargs)
        if self.mode == 'legacy':
            raise ValueError('The linked disk requires the shared physical-flux target.')
        self.renderer = LinkedExponentialRenderer(compact, self.thickness)

    def base_penalty(self, x):
        penalty = -jnp.interp(x[2], self.engine.inclination_grid, self.engine.inclination_logpdf)
        if self.center_kind != 'free':
            sd = .05 if self.center_kind == 'stellar' else .15
            penalty += .5*jnp.sum(((.1*x[5:7]-self.center_mean)/sd)**2)
        # The shared center has one prior, not one prior for each role.
        penalty += .5*jnp.sum(x[8:]**2)+.5*((x[3]/3)**2+x[4]**2)
        return penalty

    def from_flexible(self, parameters, source_flux=None):
        """Convert a saved flexible solution to an initializer, never a fit.

        A saved map may use coarser sky quadrature than this renderer. Its
        square-grid coordinates are reconstructed over the same sky support.
        Only its size initializes Re: center, angle and inclination retain
        the kinematic geometry rather than the independent light moments.
        """
        p = np.asarray(parameters, dtype=float)
        if p.shape != (1+self.original_base_size+2*self.n,):
            raise ValueError('Flexible parameter vector has the wrong geometry.')
        old = p[1:1+self.original_base_size]
        re = B1*np.exp(old[7])/np.sqrt(3)
        if source_flux is not None:
            w = np.asarray(source_flux, dtype=float)
            if w.ndim != 1 or np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
                raise ValueError('Expected nonnegative finite source quadrature flux.')
            xy = np.stack((self.engine.line_map.east, self.engine.line_map.north), axis=1)
            if w.shape != (len(xy),):
                side = int(np.sqrt(w.size))
                if side*side != w.size or side < 2:
                    raise ValueError('A different saved quadrature must be a square sky grid.')
                axis = np.linspace(-self.engine.half_extent, self.engine.half_extent, side)
                east, north = np.meshgrid(axis, axis, indexing='xy')
                xy = np.stack((east.ravel(), north.ravel()), axis=1)
            w = w/w.sum()
            centered = xy-w@xy
            cov = np.einsum('n,ni,nj->ij', w, centered, centered)
            largest = np.linalg.eigvalsh(cov)[-1]
            re = B1*np.sqrt(max(largest, 1e-8)/3)
        x = np.r_[old[:7], np.log(np.clip(re, .051, .79)),
                  old[10+self.original_nmap:]]
        return np.r_[p[0], x, p[1+self.original_base_size:]]

    def from_exponential(self, parameters):
        """Retain kinematic geometry and Re from an independent-light fit."""
        p = np.asarray(parameters, dtype=float)
        old_size = 12+2*self.n
        if p.shape != (1+old_size+2*self.n,):
            raise ValueError('Exponential parameter vector has the wrong geometry.')
        x = p[1:1+old_size]
        return np.r_[p[0], x[:8], x[12:], p[1+old_size:]]

    def describe(self, parameters):
        x, speed, _, _, _ = self.split(np.asarray(parameters))
        q = float(projected_axis_ratio(x[2], self.thickness))
        center = (.1*x[5:7]).tolist()
        return dict(vrot_0p5arcsec_kms=float(speed), vproj_0p5arcsec_kms=float(speed*np.sin(x[2])),
            turnover_arcsec=float(np.exp(x[0])), theta_east_to_north_deg=float(np.rad2deg(x[1]) % 360),
            inclination_deg=float(np.rad2deg(x[2])), systemic_offset_kms=float(100*x[3]),
            intrinsic_sigma_kms=float(100*x[4]), center_east_north_arcsec=center,
            emission_model='linked_exponential', sersic_n=1., emission_re_arcsec=float(np.exp(x[7])),
            emission_axis_ratio=q, emission_angle_deg=float(np.rad2deg(x[1]) % 180),
            emission_center_east_north_arcsec=center, emission_thickness=self.thickness,
            geometry_linked=True)

    def prior_description(self):
        return dict(emission_model='Sersic n=1 control; gas light and rotation share center, PA and inclination',
            emission_re='Uniform in log(Re) between 0.05 and 0.8 arcsec; untruncated major-axis half-light radius',
            emission_axis_ratio='sqrt(cos(i)^2 + q0^2 sin(i)^2), with fixed q0',
            emission_thickness=self.thickness,
            thickness_interpretation='Projected oblate light approximation only; no 3D velocity integration',
            emission_angle='Identical to kinematic angle modulo pi; no additional angle prior',
            emission_center='Identical to kinematic center; its prior is applied once',
            kinematic_center_kind=self.center_kind,
            intrinsic_width='HalfNormal(100 km/s), truncated at 200; fixed-width scans bypass the zero-gradient boundary',
            source_flux='Flux within finite quadrature support, not extrapolated total flux',
            central_radius_softening_arcsec=RADIUS_EPS_ARCSEC, n_independent_emission_shape_parameters=1,
            geometry_linked=True)

    def posterior_initial_values(self, parameters):
        x, speed, flux, gains, bg = self.split(np.asarray(parameters))
        bound = .6 if self.center_kind == 'free' else .3
        values = dict(vrot=np.clip(speed, 1., 999.), log_rturn=np.clip(x[0], np.log(.0201), np.log(1.999)),
            theta=(x[1]+np.pi) % (2*np.pi)-np.pi,
            inclination=np.clip(x[2], float(self.engine.inclination_grid[0])+.001,
                                float(self.engine.inclination_grid[-1])-.001),
            systemic=np.clip(100*x[3], -599., 599.), sigma_v=np.clip(100*x[4], 5., 195.),
            center=np.clip(.1*x[5:7], -bound+.001, bound-.001),
            log_re=np.clip(x[7], np.log(.0501), np.log(.7999)),
            registration_z=x[8:].reshape(self.n, 2), line_flux_1e17=max(float(flux), 1e-5), gain_z=gains)
        if self.mode == 'positive':
            _, parts = self.components(jnp.asarray(parameters))
            floor = min(float(np.min(self.background_sigma))*.01,
                        self.continuum_model.coefficient_scale*.001)
            values.update(background_z=bg,
                continuum_coefficients=np.maximum(np.asarray(parts['continuum_coefficients']), floor))
        return values

    def model(self):
        """Matching explicit-continuum NumPyro target; launches no sampler."""
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
        shifts = numpyro.sample('registration_z', dist.Normal(0., 1.).expand((self.n, 2)).to_event(2))
        flux = numpyro.sample('line_flux_1e17', dist.HalfNormal(self.flux_scale))
        gains_z = numpyro.sample('gain_z', dist.Normal(0., 1.).expand((self.n-1,)).to_event(1))
        x = jnp.concatenate((jnp.array([rt, angle, inc, systemic/100, sigma/100,
            center[0]/.1, center[1]/.1, re]), shifts.ravel()))
        unit, _, _ = self.renderer.render(x, speed)
        line = unit*(flux*self.response*jnp.exp(self.calibration_sigma*(self.gain_basis@gains_z)))[:, None, None]
        if self.mode == 'positive':
            bg = numpyro.sample('background_z', dist.Normal(0., 1.).expand((self.n,)).to_event(1))
            cm = self.continuum_model
            coeff = numpyro.sample('continuum_coefficients',
                dist.HalfNormal(cm.coefficient_scale).expand(cm.coefficient_shape).to_event(3))
            mean = line+cm.continuum(coeff)+(self.background_sigma*bg)[:, None, None]
            chi2 = jnp.sum(((cm.data-mean)*cm.inv)**2)
        else:
            chi2 = self.engine.likelihood.chi_square(line)
        numpyro.factor('native_pixels', -.5*chi2)
