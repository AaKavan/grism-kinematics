"""Experimental proper-prior posterior for a shared R/C emission map.

Unknown positive brightness is a normalized log-Gaussian field, separate from
total line flux. A proper, noncentered smoothness prior replaces optimization
over unconstrained positive pixels. It is a scientific assumption to validate,
not a way to hide nuisance parameters at their best-fit values.
"""

from dataclasses import dataclass, asdict

import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .native_jax import projected_velocity


@dataclass(frozen=True)
class PosteriorPrior:
    correlation_arcsec: float = .10
    log_map_sigma_mean: float = np.log(2.)
    log_map_sigma_sd: float = .5
    envelope_width_median: float = .25
    envelope_log_width_sd: float = .5
    log_flux_sd: float = 2.
    v_max: float = 1500.
    turnover_min: float = .02
    turnover_max: float = 2.
    turnover_prior: str = "uniform"
    inclination_min_deg: float = 5.
    inclination_max_deg: float = 85.
    inclination_measurement_rad: float = .9
    inclination_error_deg: float | None = 5.

    def __post_init__(self):
        if self.turnover_prior not in ("uniform", "log_uniform"):
            raise ValueError("Unknown turnover prior")
        if not (0 < self.turnover_min < self.turnover_max and self.v_max > 0):
            raise ValueError("Invalid velocity/turnover support")
        if not 0 < self.inclination_min_deg < self.inclination_max_deg < 90:
            raise ValueError("Invalid inclination support")
        if self.inclination_error_deg is not None and self.inclination_error_deg <= 0:
            raise ValueError("Inclination measurement error must be positive or None")
        if min(self.correlation_arcsec, self.log_map_sigma_sd, self.envelope_width_median,
               self.envelope_log_width_sd, self.log_flux_sd) <= 0:
            raise ValueError("Prior scales must be positive")

    def to_dict(self):
        return asdict(self)


def log_field_basis(size, half_extent, correlation_arcsec):
    """Proper discrete smooth field with no redundant constant brightness mode.

    Neumann-grid Laplacian eigenvectors are orthonormal DCT modes. Variances
    are proportional to (1 + ell^2 lambda)^-2, normalized to unit average
    pointwise variance. Removing DC separates morphology from total flux.
    This is a specified finite-grid prior, not an exact continuum GP claim.
    """
    if size < 2 or min(half_extent, correlation_arcsec) <= 0:
        raise ValueError("Invalid field geometry")
    x = np.arange(size)[:, None]+.5
    k = np.arange(size)[None, :]
    basis = np.cos(np.pi*x*k/size)*np.sqrt(2./size)
    basis[:, 0] /= np.sqrt(2.)
    two_d = np.einsum("ym,xn->yxmn", basis, basis).reshape(size*size, size*size)[:, 1:]
    delta = 2*half_extent/(size-1)
    eigen = 4*np.sin(np.pi*np.arange(size)/(2*size))**2/delta**2
    eigen = (eigen[:, None]+eigen[None, :]).ravel()[1:]
    scale = 1/(1+correlation_arcsec**2*eigen)
    scale *= np.sqrt(size*size/np.sum(scale**2))
    return two_d*scale, two_d, scale


class NativePosterior:
    """Shared map, flux, rotation curve and inclination, with fixed response.

    Registration, source center, line width and PSF/LSF remain conditional.
    Continuum uses the parameter-independent integrated linear likelihood.
    """

    def __init__(self, projector, line_map, likelihood, *, prior=None, half_extent=.63, sigma_v=45.):
        self.projector, self.line_map, self.likelihood = projector, line_map, likelihood
        self.prior = PosteriorPrior() if prior is None else prior
        self.sigma_v, self.half_extent = sigma_v, half_extent
        basis, modes, scales = log_field_basis(line_map.size, half_extent, self.prior.correlation_arcsec)
        self.basis = jnp.asarray(basis)
        self.modes, self.scales = modes, scales
        axis = np.linspace(-half_extent, half_extent, line_map.size)
        north, east = np.meshgrid(axis, axis, indexing="ij")
        self.radius_squared = jnp.asarray((east*east+north*north).ravel())

    def controls(self, log_flux, map_z, map_sigma, envelope_width):
        log_brightness = -self.radius_squared/(2*envelope_width**2) + map_sigma*(self.basis @ map_z)
        shape = jnp.exp(log_brightness-jnp.max(log_brightness)).reshape(self.line_map.size, self.line_map.size)
        unit_integral = jnp.sum(self.line_map(shape))/self.line_map.flux_unit
        return shape/unit_integral*jnp.exp(log_flux)

    def predict(self, v_ref, r_turn, theta, inclination, log_flux, map_z, map_sigma, envelope_width):
        controls = self.controls(log_flux, map_z, map_sigma, envelope_width)
        velocity = projected_velocity(self.line_map.east, self.line_map.north, v_ref, r_turn, theta, inclination)
        return self.projector(self.line_map(controls), velocity, sigma_v=self.sigma_v)

    def model(self, likelihood_weight=1.):
        p = self.prior
        v_ref = numpyro.sample("v_ref", dist.Uniform(0., p.v_max))
        if p.turnover_prior == "uniform":
            r_turn = numpyro.sample("r_turn", dist.Uniform(p.turnover_min, p.turnover_max))
        else:
            log_turnover = numpyro.sample("log_r_turn", dist.Uniform(np.log(p.turnover_min), np.log(p.turnover_max)))
            r_turn = numpyro.deterministic("r_turn", jnp.exp(log_turnover))
        theta = numpyro.sample("theta", dist.Uniform(-np.pi, np.pi))
        if p.inclination_error_deg is None:
            cos_i = numpyro.sample("cos_i", dist.Uniform(np.cos(np.deg2rad(p.inclination_max_deg)),
                                                         np.cos(np.deg2rad(p.inclination_min_deg))))
            inclination = numpyro.deterministic("inclination", jnp.arccos(cos_i))
        else:
            # Explicit external-information prior, rather than multiplying a
            # Gaussian by an undocumented orientation prior. Without external
            # information the branch above uses isotropic orientations.
            inclination = numpyro.sample("inclination", dist.TruncatedNormal(p.inclination_measurement_rad,
                np.deg2rad(p.inclination_error_deg), low=np.deg2rad(p.inclination_min_deg),
                high=np.deg2rad(p.inclination_max_deg)))
        log_flux = numpyro.sample("log_flux", dist.Normal(0., p.log_flux_sd))
        map_z = numpyro.sample("map_z", dist.Normal(0., 1.).expand((self.basis.shape[1],)).to_event(1))
        map_sigma = numpyro.sample("map_sigma", dist.LogNormal(p.log_map_sigma_mean, p.log_map_sigma_sd))
        envelope_width = numpyro.sample("envelope_width", dist.LogNormal(np.log(p.envelope_width_median), p.envelope_log_width_sd))
        # Weight zero is useful for prior-predictive draws; avoids rendering.
        if likelihood_weight != 0:
            prediction = self.predict(v_ref, r_turn, theta, inclination, log_flux, map_z, map_sigma, envelope_width)
            numpyro.factor("native_pixels", likelihood_weight*self.likelihood.relative(prediction))

    def initialize_from_fit(self, fit, controls):
        """Warm starts only; never fix a nuisance parameter at its fitted value.

        Zero optimized pixels have no finite log. A floor and clipped whitened
        coefficients give a finite starting map inside this different prior.
        The floor is not imposed on posterior predictions.
        """
        p = self.prior
        v, rt, theta = fit["fitted_vref_rturn_theta"]
        width, sigma = p.envelope_width_median, np.exp(p.log_map_sigma_mean)
        controls = np.asarray(controls)
        if controls.shape != (self.line_map.size, self.line_map.size):
            raise ValueError("Saved and posterior control grids must match")
        residual = np.log(np.maximum(controls.ravel(), .02*controls.max()))+np.asarray(self.radius_squared)/(2*width**2)
        residual -= residual.mean()
        z = np.clip((self.modes.T @ residual)/(self.scales*sigma), -3., 3.)
        total = float(jnp.sum(self.line_map(jnp.asarray(controls))))/self.line_map.flux_unit
        values = {"v_ref": np.clip(v, .01*p.v_max, .99*p.v_max),
                  "r_turn": np.clip(rt, p.turnover_min+.001, p.turnover_max-.001),
                  "theta": (theta+np.pi)%(2*np.pi)-np.pi,
                  "log_flux": np.log(total), "map_z": z, "map_sigma": sigma, "envelope_width": width}
        values["cos_i" if p.inclination_error_deg is None else "inclination"] = (
            np.cos(p.inclination_measurement_rad) if p.inclination_error_deg is None else p.inclination_measurement_rad)
        if p.turnover_prior == "log_uniform":
            values["log_r_turn"] = np.log(values.pop("r_turn"))
        return values


def radial_profiles(v_ref, r_turn, inclination, radii_arcsec, *, kpc_per_arcsec=None):
    """Draw-by-draw projected/deprojected curves and rotation-only mass proxy.

    Arrays broadcast over chains/draws. Absolute M_rot=r V_rot^2/G is a circular
    speed / spherical-equivalent proxy, not a pressure-corrected mass model.
    Supply a distance explicitly rather than silently assuming one.
    """
    v_ref, r_turn, inclination = map(np.asarray, (v_ref, r_turn, inclination))
    radii = np.asarray(radii_arcsec)
    projected = v_ref[..., None]*np.arctan(radii/r_turn[..., None])/np.arctan(.5/r_turn[..., None])
    deprojected = projected/np.sin(inclination[..., None])
    output = {"projected_velocity_kms": projected, "deprojected_velocity_kms": deprojected,
              "mass_proxy_arcsec_kms2": radii*deprojected**2}
    if kpc_per_arcsec is not None:
        if not np.isfinite(kpc_per_arcsec) or kpc_per_arcsec <= 0:
            raise ValueError("Positive angular-distance scale required")
        output["rotation_only_mass_msun"] = output["mass_proxy_arcsec_kms2"]*kpc_per_arcsec/4.30091727003628e-6
    return output
