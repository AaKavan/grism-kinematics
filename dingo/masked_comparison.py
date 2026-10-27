"""Input-region exclusions for the existing three-file R/C comparison.

A DS9 region refers to the original grism stamp. Its pixels move when a trial
velocity rectifies that stamp. We therefore exclude their entire possible
output footprint, over the declared velocity and registration bounds, once.
The resulting common mask is fixed for every candidate and posterior draw.
This sacrifices some otherwise useful pixels, but does not reward a candidate
for moving a difficult residual out of its own comparison region.

No input values are replaced with invented zero emission. The ordinary scalar
and batched likelihoods use the same saved, fixed comparison mask as before.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits

from . import grism
from .fitting import KinematicsFitter, estimate_background_sigma
from .input_masks import read_ds9_exclusion


LOG = logging.getLogger(__name__)


def excluded_scatter_envelope(excluded, x_bounds, y_bounds, output_shape):
    """Cover all bilinear destinations of excluded input pixel centers.

    ``x_bounds`` and ``y_bounds`` bound the *additive* displacement from input
    pixel coordinates to output pixel coordinates, including registration and
    the output cutout origin. Rectangles deliberately overcover a curved trace.
    The extra destination at floor(position)+1 accounts for interpolation.
    Returns a Boolean output mask, with True meaning excluded.
    """
    excluded = np.asarray(excluded, dtype=bool)
    bounds = np.asarray([x_bounds, y_bounds], dtype=float)
    if excluded.ndim != 2 or bounds.shape != (2, 2):
        raise ValueError("Use a 2D exclusion mask and two pairs of displacement bounds.")
    if not np.isfinite(bounds).all() or np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError("Displacement bounds must be finite and ordered.")
    height, width = output_shape
    output = np.zeros((height, width), dtype=bool)
    for y, x in np.argwhere(excluded):
        left = max(0, int(np.floor(x + bounds[0, 0])))
        right = min(width, int(np.floor(x + bounds[0, 1])) + 2)
        bottom = max(0, int(np.floor(y + bounds[1, 0])))
        top = min(height, int(np.floor(y + bounds[1, 1])) + 2)
        if left < right and bottom < top:
            output[bottom:top, left:right] = True
    return output


def _parameter_bounds(config, name, default):
    """Use the posterior support, or the value of an unsampled parameter."""
    prior = config.get("mcmc", {}).get("priors", {}).get(name)
    if prior is None:
        return float(default), float(default)
    bounds = np.asarray([prior.get("min", -np.inf), prior.get("max", np.inf)])
    if not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
        raise ValueError(f"Input masking requires finite ordered support for {name}.")
    return tuple(map(float, bounds))


class InputMaskedFitter(KinematicsFitter):
    """The original fitter with optional, provenance-checked input exclusions.

    ``config['input_masks']`` maps 'R'/'C' to records containing a DS9 region
    ``path`` and its ``sha256``. With no records, numerical behavior is unchanged.
    Region files are read without editing the FITS, and their original-pixel
    masks are also excluded from background-noise estimation.
    """

    def _setup_data(self):
        super()._setup_data()
        self.input_excluded = {}
        for pupil, record in self.config.get("input_masks", {}).items():
            if pupil not in {"R", "C"}:
                raise ValueError(f"Unknown masked grism {pupil!r}; use R or C.")
            path = Path(record["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"The saved input region changed: {path}. Use a new masked run.")
            data = getattr(self, f"true_grism_{pupil}")
            fits_path, extension = self.config["image"][pupil]["path"]
            mask = read_ds9_exclusion(path, tuple(data.shape), fits.getheader(fits_path, extension))
            self.input_excluded[pupil] = torch.as_tensor(mask, device=self.device)
        self.input_comparison_excluded = self._comparison_exclusion()

    def _comparison_exclusion(self):
        """Bound the full remapped region, rather than sampling a few velocities.

        The legacy calibration is a position-independent, piecewise-linear
        displacement as a function of wavelength. Its extrema occur at interval
        endpoints or calibration knots. Evaluating those points bounds every
        allowed rotation field, including different centers and inclinations.
        """
        excluded = np.zeros(tuple(self.true_grism_R.shape), dtype=bool)
        if not self.input_excluded:
            return torch.as_tensor(excluded, device=self.device)

        velocity = self.config["velocity"]
        amplitude = "V_sini" if "V_sini" in velocity else "V_rot"
        lo, hi = _parameter_bounds(self.config, f"velocity.{amplitude}", velocity[amplitude])
        maximum_speed = max(abs(lo), abs(hi))
        sys_lo, sys_hi = _parameter_bounds(self.config, "velocity.v_sys", velocity.get("v_sys", 0.0))
        wavelength = float(self.lambda_rest.item())
        wave_lo = wavelength * (1.0 + (sys_lo - maximum_speed) / 299792.458)
        wave_hi = wavelength * (1.0 + (sys_hi + maximum_speed) / 299792.458)
        if wave_lo <= 0:
            raise ValueError("Input mask wavelength bounds must be positive.")

        for pupil, input_mask in self.input_excluded.items():
            image_config = self.config["image"][pupil]
            _, spatial, dispersion = grism.load_nircam_wfss_model(
                pupil, image_config["module"], self.filter)
            _, _, knots = grism.grism_conf_preparation(
                x0=1024, y0=1024, pupil=pupil,
                fit_opt_fit=spatial, w_opt=dispersion)
            knots = np.asarray(knots, dtype=np.float32)
            inside = knots[(knots >= wave_lo) & (knots <= wave_hi)]
            samples = np.unique(np.concatenate(([wave_lo, wave_hi], inside))).astype(np.float32)
            # Include both sides of knots: the legacy interpolator has a tiny
            # denominator regularizer, so a knot can have a roundoff-sized jump.
            samples = np.unique(np.concatenate((samples,
                np.nextafter(samples, np.float32(-np.inf)),
                np.nextafter(samples, np.float32(np.inf)))))
            waves = torch.as_tensor(samples, device=self.device)
            zeros = torch.zeros_like(waves)
            with torch.no_grad():
                mapped_x, mapped_y = self.fwd_models[pupil](zeros, zeros, waves)
            cutout = getattr(self, f"cutout_{pupil}")
            bounds = []
            for axis, mapped, origin in zip(("x", "y"), (mapped_x, mapped_y), cutout[:2]):
                reg_lo, reg_hi = _parameter_bounds(
                    self.config, f"image.{pupil}.d{axis}", image_config.get(f"d{axis}", 0.0))
                # Margin exceeds fixed-point tolerance and floating-point error.
                margin = max(0.02, 2 * self.config.get("mcmc", {}).get("xy_tol", 0.005))
                bounds.append((float(mapped.min()) + reg_lo - origin - margin,
                               float(mapped.max()) + reg_hi - origin + margin))
            excluded |= excluded_scatter_envelope(
                input_mask.cpu().numpy(), *bounds, excluded.shape)

        LOG.info("Input-region mask: %d/%d common comparison pixels excluded over the full prior support",
                 int(excluded.sum()), excluded.size)
        return torch.as_tensor(excluded, device=self.device)

    def validate_input_mask_support(self):
        """Reject a cached comparison that could still receive masked input."""
        if self.input_excluded and bool((self.mcmc_mask & self.input_comparison_excluded).any()):
            raise ValueError("The saved comparison does not cover the input-region exclusion. Use a new masked run.")

    def setup_noise_model(self, sigma_R=None, sigma_C=None, cov_threshold=0.25, extra_mask=None):
        """Freeze one safe output mask and estimate noise from unmasked inputs."""
        if not self.input_excluded:
            return super().setup_noise_model(sigma_R, sigma_C, cov_threshold, extra_mask)
        noise = self.config.get("mcmc", {}).get("noise", {})
        sigmas = {"R": sigma_R, "C": sigma_C}
        for pupil in ("R", "C"):
            if sigmas[pupil] is None:
                sigmas[pupil] = noise.get(f"sigma_{pupil}")
            if sigmas[pupil] is None:
                sigmas[pupil] = estimate_background_sigma(
                    getattr(self, f"true_grism_{pupil}"), self.input_excluded.get(pupil))
        keep = ~self.input_comparison_excluded
        if extra_mask is not None:
            keep = keep & extra_mask.to(keep.device)
        return super().setup_noise_model(sigmas["R"], sigmas["C"], cov_threshold, keep)

    def setup_adam_compatible_noise_model(self, *args, **kwargs):
        if self.input_excluded:
            raise ValueError("Input-region masks require the Gaussian comparison likelihood.")
        return super().setup_adam_compatible_noise_model(*args, **kwargs)
