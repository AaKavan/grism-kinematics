"""Conditional real-data velocity profiling, separate from the mock posterior.

The instrument kernel is an explicitly approximate separable PSF/LSF model.
Profile scores include nuisance penalties and are not marginalized probabilities.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
import jax
import jax.numpy as jnp

from .native_posterior import log_field_basis
from .native_jax import projected_velocity


def oblate_inclination(q, q0):
    """Axisymmetric oblate disk geometry; reject physically inconsistent q<q0."""
    q, q0 = np.broadcast_arrays(q, q0)
    if np.any(~np.isfinite(q+q0)) or np.any((q0 < 0) | (q0 >= 1) | (q < q0) | (q > 1)):
        raise ValueError("Require 0 <= q0 <= q <= 1, with q0 < 1")
    return np.arccos(np.sqrt((q*q-q0*q0)/(1-q0*q0)))


def broadened_inclination_density(stellar_angles, scatter_deg=10., low_deg=5., high_deg=85.):
    """Phenomenological stellar/gas inclination scatter, explicitly specified.

    Equal mixture of bounded Gaussian conditionals. This is a provisional
    orientation distribution, not a validated physical misalignment population.
    """
    from scipy.special import ndtr
    angles = np.asarray(stellar_angles).ravel()
    if not len(angles) or not np.isfinite(angles).all() or scatter_deg <= 0:
        raise ValueError("Need finite stellar angles and positive scatter")
    grid = np.deg2rad(np.linspace(low_deg, high_deg, 1601))
    sigma = np.deg2rad(scatter_deg)
    norm = ndtr((grid[-1]-angles)/sigma)-ndtr((grid[0]-angles)/sigma)
    logp = logsumexp(-.5*((grid[:,None]-angles)/sigma)**2
                    - np.log(norm)[None,:], axis=1)-np.log(len(angles)*sigma*np.sqrt(2*np.pi))
    logp -= np.log(np.trapezoid(np.exp(logp), grid))
    return grid, logp


def discrete_gaussian_with_variance(variance, half=10):
    """A sampled Gaussian whose discrete second moment equals the request."""
    if not 0 < variance < half*(half+1)/3:
        raise ValueError("Requested variance is outside the finite kernel support")
    x = np.arange(-half, half+1, dtype=float)
    def weights(s):
        p = np.exp(-.5*(x/s)**2)
        return p/p.sum()
    sigma = brentq(lambda s: np.sum(weights(s)*x*x)-variance, .02, 100.)
    return weights(sigma)


def separable_response_kernel(pupil, fraction, along_sigma_pixels, cross_sigma_pixels, half=10):
    """Empirical spectral LSF plus Gaussian cross-dispersion core.

    Remove the cubic source-deposition variance (1/3 pixel^2) from each
    component. This matches second moments, not the complete measured PSF/LSF.
    No extra optical PSF is convolved along the empirical spectral LSF.
    """
    along = np.asarray(along_sigma_pixels, dtype=float)
    if pupil not in ("R", "C") or not 0 < fraction < 1 or along.shape != (2,):
        raise ValueError("Invalid pupil/LSF mixture")
    if min(*along, cross_sigma_pixels) <= np.sqrt(1/3):
        raise ValueError("Response is narrower than the cubic deposition basis")
    spectral = sum(w*discrete_gaussian_with_variance(s*s-1/3, half)
                   for w,s in zip((fraction,1-fraction),along))
    cross = discrete_gaussian_with_variance(cross_sigma_pixels**2-1/3, half)
    return np.outer(cross,spectral) if pupil == "R" else np.outer(spectral,cross)


def profile_positive_amplitude(template, residual_data):
    """Exact nonnegative per-exposure scale after linear continuum projection."""
    numerator = jnp.sum(template*residual_data, axis=(-2,-1))
    denominator = jnp.sum(template*template, axis=(-2,-1))
    amplitude = jnp.maximum(numerator,0.)/jnp.maximum(denominator,1e-300)
    relative = -.5*jnp.sum(amplitude*amplitude*denominator-2*amplitude*numerator)
    return amplitude, relative


class RealVelocityProfile:
    """Shared flexible log-brightness map, all R+C exposures, fixed Vrot(.5").

    Free: turnover, kinematic angle, inclination, systemic velocity, line width,
    kinematic center, independent emission-envelope shape and map coefficients,
    per-exposure registration and line normalization. Continuum is profiled.
    """

    def __init__(self, projector, line_map, likelihood, inclination_grid, inclination_logpdf,
                 half_extent=.9, correlation=.10, map_sigma=2., registration_sigma=.25):
        self.projector, self.line_map, self.likelihood = projector, line_map, likelihood
        self.half_extent, self.map_sigma = half_extent, map_sigma
        self.registration_sigma = registration_sigma
        self.inclination_grid = jnp.asarray(inclination_grid)
        self.inclination_logpdf = jnp.asarray(inclination_logpdf)
        self.basis = jnp.asarray(log_field_basis(line_map.size, half_extent, correlation)[0])
        axis = np.linspace(-half_extent, half_extent, line_map.size)
        y,x = np.meshgrid(axis,axis,indexing="ij")
        self.east_controls, self.north_controls = jnp.asarray(x.ravel()),jnp.asarray(y.ravel())
        self.nmap = self.basis.shape[1]
        # Coordinates are scaled to keep numerical optimization well conditioned.
        self.bounds = [(np.log(.02),np.log(2.)), (None,None), (np.deg2rad(5),np.deg2rad(85)),
            (-6.,6.), (0.,2.), (-3.,3.),(-3.,3.), (np.log(.12),np.log(.8)), (.12,1.),(None,None)]
        self.bounds += [(None,None)]*(self.nmap+2*projector.exposures)

    def initial(self, theta, inclination, rturn=.25):
        x = np.zeros(len(self.bounds))
        x[:10] = [np.log(rturn),theta,inclination,0.,.45,0.,0.,np.log(.35),.65,theta]
        return x

    def components(self, x, speed):
        rt,theta,inc = jnp.exp(x[0]),x[1],x[2]
        v_sys,sigma_v = 100*x[3],100*x[4]
        ce,cn = .1*x[5],.1*x[6]
        width,q,phi = jnp.exp(x[7]),x[8],x[9]
        a = jnp.cos(phi)*self.east_controls+jnp.sin(phi)*self.north_controls
        b = -jnp.sin(phi)*self.east_controls+jnp.cos(phi)*self.north_controls
        z = x[10:10+self.nmap]
        log_controls = -.5*(a*a+(b/q)**2)/width**2+self.map_sigma*(self.basis@z)
        # Interpolate log brightness, avoiding the documented coarse positive-map bias.
        log_quad = sum(log_controls[idx]*weight for idx,weight in
                       zip(self.line_map.indices,self.line_map.weights))
        flux = jnp.exp(log_quad-jnp.max(log_quad))*self.line_map.area
        flux /= jnp.sum(flux)
        velocity = projected_velocity(self.line_map.east-ce,self.line_map.north-cn,
                                      speed*jnp.sin(inc),rt,theta,inc)+v_sys
        shifts = self.registration_sigma*x[10+self.nmap:].reshape(self.projector.exposures,2)
        line = self.projector(flux,velocity,sigma_v=sigma_v,shifts=shifts)
        projected = self.likelihood.residualize(jnp.where(self.likelihood.valid,line,0.)*
                                               self.likelihood.inverse_error)
        amplitudes, relative = profile_positive_amplitude(projected,self.likelihood.residual_data)
        # Explicit regularization, fixed across the whole speed scan.
        inc_penalty = -jnp.interp(inc,self.inclination_grid,self.inclination_logpdf)
        other_penalty = .5*jnp.sum(z*z)+.5*jnp.sum(x[10+self.nmap:]**2)
        other_penalty += .5*((x[7]-np.log(.35))/.6)**2
        other_penalty += .5*((ce/.15)**2+(cn/.15)**2+(v_sys/300.)**2+(sigma_v/100.)**2)
        return -relative+inc_penalty+other_penalty, (relative,inc_penalty,other_penalty,
                         amplitudes,line*amplitudes[:,None,None],flux,velocity)

    def objective(self,x,speed):
        return self.components(x,speed)[0]

    def describe(self,x,speed):
        return {"vrot_0p5arcsec_kms":float(speed), "vproj_0p5arcsec_kms":float(speed*np.sin(x[2])),
                "turnover_arcsec":float(np.exp(x[0])),"theta_east_to_north_deg":float(np.rad2deg(x[1])%360),
                "inclination_deg":float(np.rad2deg(x[2])),"systemic_offset_kms":float(x[3]*100),
                "intrinsic_sigma_kms":float(x[4]*100),"center_east_north_arcsec":(x[5:7]*.1).tolist(),
                "envelope_width_arcsec":float(np.exp(x[7])),"envelope_axis_ratio":float(x[8])}
