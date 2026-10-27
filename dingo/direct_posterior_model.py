"""Joint source-map and kinematic posterior for the direct R+C experiment.

This JAX model implements the same calibrated source-to-grism deposition as
``SourceModel``. The observations stay in their original pixels. Every source
pixel is a latent positive brightness shared by R and C; no Sérsic profile or
spatial smoothing is imposed. The proper half-normal brightness prior is an
explicit modeling assumption, and its scale must be supplied by the caller.

JAX uses 64-bit arithmetic for the small wavelength shifts and their gradients.
Calibration tables and the reference wavelength retain the float32 values used
by the existing PyTorch model, so this is a precision change, not a different
calibration. Exact bit equality with float32 predictions is not expected.
"""
from pathlib import Path
import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from . import grism


jax.config.update("jax_enable_x64", True)


class DirectPosteriorModel:
    """Original-pixel Gaussian likelihood with free positive emission pixels.

    Parameters
    ----------
    source_model : SourceModel
        Prepared direct-source model. Its center, inclination prior, kinematic
        bounds, calibration, fixed finite-pixel masks, and noise are retained.
        ``image.fits`` never enters the prediction or brightness prior.
    emission_prior_scale : positive float
        Standard-deviation parameter of an independent half-normal prior for
        each source pixel, in the input-image intensity units. This keeps the
        posterior proper even for pixels outside both projected cutouts. It
        does not prescribe a disk shape, but can affect the kinematics through
        the allowed brightness maps; sensitivity to this scale needs checking.

    Notes
    -----
    Physical parameters are ``velocity, turnover, pa, rdx, rdy, inclination``
    in that order. When ``source_model.fixed_rdy`` is set, rdy is deterministic
    and only the other five kinematic parameters are sampled. Angles are radians, velocity is km/s, and other coordinates
    are pixels. ``emission`` has the full two-dimensional source-canvas shape.
    ``theta`` always contains all six physical values for saved-array compatibility.

    Create this object inside the desired ``jax.default_device`` context.
    CPU/GPU selection and MCMC/checkpoints belong to the sampling interface.
    There is no intrinsic dispersion, PSF, background, or correlated noise in
    this experimental likelihood. The kinematic center remains fixed.
    """

    parameter_sites = ("velocity", "turnover", "pa", "rdx", "rdy", "inclination")
    site_names = parameter_sites

    def __init__(self, source_model, emission_prior_scale):
        if (isinstance(emission_prior_scale, (bool, np.bool_))
                or np.ndim(emission_prior_scale) != 0
                or not np.isfinite(emission_prior_scale)
                or float(emission_prior_scale) <= 0):
            raise ValueError("emission_prior_scale must be one positive finite intensity scale.")
        if len(source_model.param_names) != 6:
            raise ValueError("DirectPosteriorModel requires the six-parameter SourceModel.")
        source_model.validate_inputs()
        if Path(grism.get_grism_cal_dir()).resolve() != Path(source_model.calibration_directory):
            raise ValueError("The calibration directory changed after preparing the source.")

        self.source_model = source_model
        self.fixed_rdy = source_model.fixed_rdy
        self.site_names = tuple(name for name in self.parameter_sites
                                if name != "rdy" or self.fixed_rdy is None)
        self.shape = tuple(source_model.shape)
        self.emission_prior_scale = float(emission_prior_scale)
        self.theta_bounds = tuple(np.array(v, dtype=float, copy=True)
                                  for v in source_model.theta_bounds)
        self.sigma_by_pupil = dict(source_model.sigma_by_pupil)
        self.data = {p: self._array(source_model.data[p].detach().cpu().numpy())
                     for p in ("R", "C")}
        self.valid_by_pupil = {
            p: jnp.asarray(source_model.valid_by_pupil[p].detach().cpu().numpy(), dtype=bool)
            for p in ("R", "C")
        }
        # Preserve the rounded constants used by the original float32 model.
        self.wavelength = self._array(source_model.wavelength.detach().cpu().numpy().reshape(()))
        self.center = tuple(float(source_model.fixed_velocity[k].detach().cpu())
                            for k in ("x0_v", "y0_v"))
        self.systemic_velocity = float(source_model.fixed_velocity["v_sys"].detach().cpu())
        self.fixed_c_offset = tuple(source_model.fixed_c_offset)
        self.cutouts = dict(source_model.cutouts)
        self.inclination_mean, self.inclination_sigma = np.deg2rad(
            [source_model.inclination_mean, source_model.inclination_sigma])
        self.y, self.x = jnp.meshgrid(jnp.arange(self.shape[0], dtype=jnp.float64),
                                      jnp.arange(self.shape[1], dtype=jnp.float64),
                                      indexing="ij")

        self.calibration = {}
        calibration_digest = hashlib.sha256()
        config = source_model.configuration
        for pupil in ("R", "C"):
            _, spatial, dispersion = grism.load_nircam_wfss_model(
                pupil, config["image"][pupil]["module"], config["summary"]["filter"])
            dx, dy, wavelength = grism.grism_conf_preparation(
                x0=1024, y0=1024, pupil=pupil,
                fit_opt_fit=spatial, w_opt=dispersion)
            dx, dy, wavelength = (np.asarray(a, dtype=np.float32) for a in (dx, dy, wavelength))
            order = np.argsort(wavelength)
            self.calibration[pupil] = tuple(self._array(a[order]) for a in (wavelength, dx, dy))
            for name, array in zip(("wavelength", "x", "y"), (wavelength, dx, dy)):
                calibration_digest.update(f"{pupil}:{name}:{array.shape}".encode())
                calibration_digest.update(np.asarray(array[order], dtype="<f4").tobytes())
        calibration_digest.update(json.dumps(
            {"reference_wavelength": float(self.wavelength),
             "cutouts": {p: list(c) for p, c in self.cutouts.items()}},
            sort_keys=True, separators=(",", ":")).encode())
        self.calibration_hash = calibration_digest.hexdigest()

        self.configuration = {
            "kinematic_sites": list(self.site_names),
            "emission_prior": "independent HalfNormal(scale) per source pixel",
            "emission_prior_scale": self.emission_prior_scale,
            "emission_shape": list(self.shape),
            "spatial_smoothing": "none",
            "brightness_template": "none",
            "arithmetic": "JAX float64 with existing float32 calibration constants",
            "calibration_hash": self.calibration_hash,
            "likelihood": "fixed original-pixel independent Gaussian R+C noise",
        }
        if self.fixed_rdy is not None:
            self.configuration["fixed_rdy"] = self.fixed_rdy
            self.configuration["registration_scope"] = "Conditional on fixed R vertical registration; its uncertainty is excluded."

    @staticmethod
    def _array(value):
        return jnp.asarray(value, dtype=jnp.float64)

    @staticmethod
    def _interpolate(wavelength, grid, values):
        """Match the original piecewise-linear trace interpolation exactly."""
        index = jnp.clip(jnp.searchsorted(grid, wavelength, side="right"), 1, len(grid) - 1)
        fraction = (wavelength - grid[index - 1]) / (grid[index] - grid[index - 1] + 1e-8)
        return values[index - 1] + fraction * (values[index] - values[index - 1])

    def mapping_coordinates(self, theta):
        """Map one six-parameter velocity field into the original R/C cutouts."""
        theta = self._array(theta)
        if theta.shape != (6,):
            raise ValueError("theta must have shape (6,); use jax.vmap for batches.")
        velocity, turnover, angle, rdx, rdy, inclination = theta
        if self.fixed_rdy is not None:
            rdy = self._array(self.fixed_rdy)
        dx, dy = self.x - self.center[0], self.y - self.center[1]
        major = jnp.cos(angle) * dx + jnp.sin(angle) * dy
        minor = -jnp.sin(angle) * dx + jnp.cos(angle) * dy
        radius = jnp.sqrt(major**2 + (minor / jnp.cos(inclination))**2 + 1e-8)
        line_of_sight = (2 / jnp.pi * velocity * jnp.sin(inclination)
                         * jnp.arctan(radius / turnover) * major / (radius + 1e-8)
                         + self.systemic_velocity)
        wavelength = self.wavelength * (1 + line_of_sight / 299792.458)
        coordinates = {}
        for pupil in ("R", "C"):
            grid, trace_x, trace_y = self.calibration[pupil]
            # The torch 'forward' closure returns -trace for a zero input.
            # SourceModel subtracts that closure, so deposition adds the trace.
            dispersion_x = self._interpolate(wavelength, grid, trace_x)
            dispersion_y = self._interpolate(wavelength, grid, trace_y)
            offset_x, offset_y = (rdx, rdy) if pupil == "R" else self.fixed_c_offset
            cutout = self.cutouts[pupil]
            coordinates[pupil] = (self.x + cutout[0] - offset_x + dispersion_x,
                                   self.y + cutout[1] - offset_y + dispersion_y)
        return coordinates

    def _deposit(self, x, y, emission):
        """Bilinearly scatter shared source flux, dropping only off-image flux."""
        height, width = self.shape
        ix, iy = jnp.floor(x).astype(jnp.int32), jnp.floor(y).astype(jnp.int32)
        fx, fy = x - ix, y - iy
        prediction = jnp.zeros(height * width, dtype=jnp.float64)
        for step_x in (0, 1):
            for step_y in (0, 1):
                column, row = ix + step_x, iy + step_y
                valid = ((column >= 0) & (column < width) & (row >= 0) & (row < height))
                weight = (fx if step_x else 1 - fx) * (fy if step_y else 1 - fy)
                index = jnp.clip(row, 0, height - 1) * width + jnp.clip(column, 0, width - 1)
                values = jnp.where(valid, emission * weight, 0.)
                prediction = prediction.at[index.ravel()].add(values.ravel())
        return prediction.reshape(self.shape)

    def predict(self, theta, emission):
        """Predict R and C from the same supplied map; no map is fitted here."""
        emission = self._array(emission)
        if emission.shape != self.shape:
            raise ValueError(f"emission must have shape {self.shape}.")
        coordinates = self.mapping_coordinates(theta)
        return tuple(self._deposit(*coordinates[p], emission) for p in ("R", "C"))

    def chi2(self, theta, emission):
        """Weighted squared residuals on fixed original observed pixels."""
        predictions = self.predict(theta, emission)
        return sum(jnp.sum(jnp.where(self.valid_by_pupil[p],
                                     (prediction - self.data[p]) / self.sigma_by_pupil[p], 0.)**2)
                   for p, prediction in zip(("R", "C"), predictions))

    def kinematic_distributions(self):
        """Normalized priors matching the direct optimizer's bounds/penalty."""
        lower, upper = self.theta_bounds
        return tuple(dist.Uniform(lower[k], upper[k]) for k in range(5)) + (
            dist.TruncatedNormal(self.inclination_mean, self.inclination_sigma,
                                 low=lower[5], high=upper[5]),)

    def log_prior(self, theta, emission):
        """Normalized joint prior in physical coordinates, useful for checks."""
        theta, emission = self._array(theta), self._array(emission)
        kinematic = sum(prior.log_prob(theta[k])
                        for k, prior in enumerate(self.kinematic_distributions())
                        if k != 4 or self.fixed_rdy is None)
        # Keep the density evaluation inside support; the explicit mask below
        # returns -inf for invalid values without emitting NumPyro warnings.
        positive_emission = jnp.where(emission > 0, emission, 1.)
        brightness = jnp.sum(dist.HalfNormal(self.emission_prior_scale).log_prob(positive_emission))
        lower, upper = (self._array(a) for a in self.theta_bounds)
        valid = (jnp.all(jnp.isfinite(theta)) & jnp.all(theta >= lower)
                 & jnp.all(theta <= upper) & jnp.all(jnp.isfinite(emission))
                 & jnp.all(emission > 0))
        if self.fixed_rdy is not None:
            valid &= jnp.isclose(theta[4], self.fixed_rdy, rtol=1e-7, atol=1e-7)
        return jnp.where(valid, kinematic + brightness, -jnp.inf)

    def model(self):
        """NumPyro joint model: sample kinematics AND every emission pixel."""
        values = []
        for name, prior in zip(self.parameter_sites, self.kinematic_distributions()):
            if name == "rdy" and self.fixed_rdy is not None:
                value = numpyro.deterministic(name, self._array(self.fixed_rdy))
            else:
                value = numpyro.sample(name, prior)
            values.append(value)
        theta = jnp.stack(values)
        numpyro.deterministic("theta", theta)
        emission = numpyro.sample(
            "emission", dist.HalfNormal(self.emission_prior_scale).expand(self.shape).to_event(2))
        numpyro.factor("original_R_C_pixels", -.5 * self.chi2(theta, emission))

    def initial_values(self, theta, emission):
        """Build physical starting sites without fixing any source brightness.

        A nonnegative profile fit may have exact zeros. The sampler's standard
        log transform needs strictly positive starts, so zeros are raised to
        0.1 times the smaller input noise. This floor applies only to starting
        coordinates: it is not a lower bound in the sampled posterior.
        Exact kinematic-bound starts are similarly nudged inside their bounds.
        """
        theta = np.array(theta, dtype=float, copy=True)
        emission = np.asarray(emission, dtype=float)
        lower, upper = self.theta_bounds
        if theta.shape != (6,) or not np.isfinite(theta).all():
            raise ValueError("Initial theta must contain six finite values within their bounds.")
        if self.fixed_rdy is not None:
            theta[4] = self.fixed_rdy
        if np.any(theta < lower) or np.any(theta > upper):
            raise ValueError("Initial theta must contain six finite values within their bounds.")
        if (emission.shape != self.shape or not np.isfinite(emission).all()
                or np.any(emission < 0)):
            raise ValueError("Initial emission must be a finite nonnegative full source map.")
        margin = (upper - lower) * 1e-7
        theta = np.clip(theta, lower + margin, upper - margin)
        result = {name: self._array(theta[k]) for k, name in enumerate(self.parameter_sites)
                  if name in self.site_names}
        floor = .1 * min(self.sigma_by_pupil.values())
        result["emission"] = self._array(np.maximum(emission, floor))
        return result
