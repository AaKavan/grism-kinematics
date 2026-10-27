"""Direct R+C prediction for controlled synthetic recovery experiments.

This module never rectifies observed images. It disperses one source map into
both detector images and compares those predictions with the original pixels.
The default map is the injected truth: that is an explicitly conditional
numerical check, not an assumption available for a real galaxy. Alternative
maps can be supplied to ``predict`` without changing the velocity model.
"""
from pathlib import Path
import tempfile

import numpy as np
import torch
import yaml

from . import kinematics
from .fitting import KinematicsFitter
from .simple_simulation import load_mock
from .three_file_fit import build_config


class ForwardExperiment:
    """One calibrated simulation with a known center and optional inclination prior.

    Parameters are ``[V_inf, R_t, PA, R_dx, R_dy]`` in km/s, pixels, radians,
    pixels, pixels. Supplying ``inclination_mean`` adds a sixth parameter,
    ``inc_v`` in radians, with a Gaussian prior truncated to 5..85 degrees.
    The prior mean and ``inclination_sigma`` arguments are supplied in degrees.
    With ``inclination_mean=None``, inclination remains fixed to injected truth.
    Both grisms use the same velocity field and emission map.
    The C registration stays fixed at its injected value. No PSF or intrinsic
    line width is added, matching the initial ``simple_simulation`` experiment.

    ``log_probability`` is conditional on the known emission map, center,
    calibration and independent input-pixel noise. The brightness map stays
    unchanged when inclination varies: inclination changes the velocity-field
    geometry, not the supplied image. This numerical experiment therefore
    checks implementation before attempting morphology-independent inference.
    It is not the production real-source likelihood.
    """

    param_names = ("velocity.V_rot", "velocity.R_v", "velocity.theta_v",
                   "image.R.dx", "image.R.dy")

    def __init__(self, mock, *, device="cpu", inclination_mean=None,
                 inclination_sigma=5.0):
        from astropy.io import fits

        if (isinstance(inclination_sigma, (bool, np.bool_))
                or not np.isfinite(inclination_sigma) or inclination_sigma <= 0):
            raise ValueError("inclination_sigma must be positive and finite, in degrees.")
        self.sample_inclination = inclination_mean is not None
        if self.sample_inclination and (
                isinstance(inclination_mean, (bool, np.bool_))
                or not np.isfinite(inclination_mean) or not 5. <= inclination_mean <= 85.):
            raise ValueError("inclination_mean must be finite and between 5 and 85 degrees.")
        self.inclination_mean = float(inclination_mean) if self.sample_inclination else None
        self.inclination_sigma = float(inclination_sigma)
        self.param_names = type(self).param_names + (("velocity.inc_v",) if self.sample_inclination else ())
        self.theta_bounds = (np.array([0., .5, 0., -10., -10.]),
                             np.array([1000., 40., 2 * np.pi, 10., 10.]))
        self.parameter_scales = np.array([200., 5., 1., 1., 1.])
        if self.sample_inclination:
            self.theta_bounds = tuple(np.append(bound, np.deg2rad(angle))
                                      for bound, angle in zip(self.theta_bounds, (5., 85.)))
            self.parameter_scales = np.append(self.parameter_scales, np.deg2rad(5.))
        self.mock = load_mock(mock.directory if hasattr(mock, "directory") else mock)
        self.inclination_settings = (
            {"mode": "gaussian", "mean_deg": self.inclination_mean,
             "sigma_deg": self.inclination_sigma, "bounds_deg": [5., 85.]}
            if self.sample_inclination else
            {"mode": "fixed", "value_deg": float(np.rad2deg(self.mock.truth["velocity.inc_v"]))})
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'.")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; choose device='cpu'.")
        self.sigma = float(self.mock.metadata["noise_sigma"])
        source = self.mock.data_dir / f"ID{self.mock.source_id}"
        config_path = source / "config_kinematics.yaml"
        # Reuse the exact generation calibration and cutout conventions. The
        # temporary configuration is removed after retaining its trace closures.
        with tempfile.TemporaryDirectory(prefix="dingo_forward_") as scratch:
            path = Path(scratch)
            config = build_config(self.mock.source_id, source.parent, config_path,
                                  path / "unused", device=self.device.type,
                                  velocity_parameter="V_rot")
            temporary_config = path / "config.yaml"
            temporary_config.write_text(yaml.safe_dump(config, sort_keys=False))
            calibration = KinematicsFitter(str(temporary_config), device=self.device)
        self.forward_models = dict(calibration.fwd_models)
        self.cutouts = {p: getattr(calibration, f"cutout_{p}") for p in ("R", "C")}
        self.wavelength = calibration.lambda_rest  # Already includes (1+z).
        self.truth_theta = np.array([self.mock.truth[k] for k in self.param_names])
        self.data = {p: self._tensor(np.array(fits.getdata(source / f"{p}.fits"), dtype=np.float32, copy=True))
                     for p in ("R", "C")}
        with np.load(self.mock.directory / "truth.npz", allow_pickle=False) as saved:
            self.emission = self._tensor(saved["image"])
            self.clean_data = {p: self._tensor(saved[f"noiseless_{p}"]) for p in ("R", "C")}
        self.shape = tuple(self.emission.shape)
        self.y, self.x = torch.meshgrid(
            torch.arange(self.shape[0], device=self.device, dtype=torch.float32),
            torch.arange(self.shape[1], device=self.device, dtype=torch.float32), indexing="ij")
        truth = self.mock.truth
        self.fixed_velocity = {key: self._tensor(truth[f"velocity.{key}"])
                               for key in ("x0_v", "y0_v", "inc_v", "v_sys")}
        self.fixed_c_offset = (truth["image.C.dx"], truth["image.C.dy"])

    def _tensor(self, value):
        """Keep tensors differentiable while moving them to the selected device."""
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def mapping_coordinates(self, theta):
        """Return ``{'R': (x, y), 'C': (x, y)}`` in detector cutout pixels.

        ``theta`` has shape (n_parameters,) or (batch, n_parameters), with
        five parameters for fixed inclination and six for a Gaussian prior.
        Coordinates refer to the
        centers of source pixels. Bilinear deposition converts these positions
        into flux-conserving pixel weights in ``predict``.
        """
        theta = self._tensor(theta)
        count = len(self.param_names)
        if theta.ndim not in (1, 2) or theta.shape[-1] != count:
            raise ValueError(f"theta must have shape ({count},) or (batch, {count}).")
        parameters = [theta[..., i, None, None] for i in range(5)]
        velocity, turnover, angle, rdx, rdy = parameters
        geometry = dict(self.fixed_velocity)
        if self.sample_inclination:
            geometry["inc_v"] = theta[..., 5, None, None]
        vz = kinematics.arctangent_disk_velocity_model(
            self.x, self.y, V_rot=velocity, R_v=turnover, theta_v=angle,
            **geometry)
        wavelength = self.wavelength * (1.0 + vz / 299792.458)
        zeros = torch.zeros_like(vz)
        result = {}
        for pupil in ("R", "C"):
            mdx, mdy = self.forward_models[pupil](zeros, zeros, wavelength)
            offset_x, offset_y = (rdx, rdy) if pupil == "R" else self.fixed_c_offset
            cutout = self.cutouts[pupil]
            result[pupil] = (self.x + cutout[0] - offset_x - mdx,
                             self.y + cutout[1] - offset_y - mdy)
        return result

    def predict(self, theta, emission=None):
        """Disperse one shared emission map into the R and C images.

        Returns a pair of torch tensors. Passing no map explicitly uses the
        injected map as a conditional numerical check. A supplied map may be
        arbitrary; no exponential or Sérsic constraint is applied here.
        """
        intensity = self.emission if emission is None else self._tensor(emission)
        if tuple(intensity.shape) != self.shape:
            raise ValueError(f"emission must have shape {self.shape}.")
        h, w = self.shape
        coordinates = self.mapping_coordinates(theta)
        return tuple(kinematics.bilinear_interpolte_intensity_torch(
            *coordinates[p], intensity, (0, 0, w, h)) for p in ("R", "C"))

    def residuals(self, theta, *, noisy=True, emission=None):
        """Return original-input residuals, never remapped observed pixels.

        Positive input noise scales noisy residuals by its known standard
        deviation. The noiseless check returns raw residuals with no invented
        uncertainty or posterior interpretation.
        """
        if noisy and self.sigma <= 0:
            raise ValueError("A noisy likelihood requires noise_sigma > 0.")
        data = self.data if noisy else self.clean_data
        scale = self.sigma if noisy else 1.
        return tuple((model - data[p]) / scale
                     for p, model in zip(("R", "C"), self.predict(theta, emission)))

    def chi2(self, theta, *, noisy=True, emission=None):
        """Sum squared residuals; noiseless values are unscaled SSE, not chi2."""
        residuals = self.residuals(theta, noisy=noisy, emission=emission)
        return sum(torch.sum(r.to(torch.float64)**2, dim=(-2, -1)) for r in residuals)

    def log_prior(self, theta):
        """Existing broad kinematic priors, with no morphology prior.

        V is uniform on [0,1000], R_t log-uniform on [.5,40], and PA uniform
        on [0,2pi]. R registration has independent N(0,2 pixel) priors, truncated
        at +/-10 pixels. If sampled, inclination has the configured Gaussian
        prior in degrees, truncated to [5,85]. Normalization constants
        independent of theta are omitted, including the constant conversion
        between degrees and the sampled radians.
        """
        point = np.asarray(theta, dtype=float)
        count = len(self.param_names)
        if point.ndim not in (1, 2) or point.shape[-1] != count:
            raise ValueError(f"theta must have shape ({count},) or (batch, {count}).")
        lower, upper = self.theta_bounds
        valid = np.isfinite(point).all(axis=-1) & (point >= lower).all(axis=-1) & (point <= upper).all(axis=-1)
        # Clip only for safe log evaluation; invalid points remain rejected.
        prior = -np.log(np.clip(point[..., 1], .5, 40.)) - .5 * np.sum((point[..., 3:5] / 2.)**2, axis=-1)
        if self.sample_inclination:
            prior -= .5 * ((np.rad2deg(point[..., 5]) - self.inclination_mean)
                           / self.inclination_sigma)**2
        return np.where(valid, prior, -np.inf)

    def prior_penalty(self, theta):
        """Inclination penalty for the optional free-map profile optimizer.

        Historical free-map mock fits use only hard bounds for their first five
        parameters. Preserve that calculation while applying the new Gaussian
        inclination constraint. The known-map posterior above retains its
        existing turnover-radius and registration priors as well.
        """
        point = self._tensor(theta)
        if not self.sample_inclination:
            return torch.zeros(point.shape[:-1], dtype=point.dtype, device=point.device)
        mean = np.deg2rad(self.inclination_mean)
        sigma = np.deg2rad(self.inclination_sigma)
        return .5 * ((point[..., 5] - mean) / sigma)**2

    def log_probability_batch(self, thetas):
        """Evaluate known-map posterior proposals together on CPU or GPU."""
        thetas = np.asarray(thetas, dtype=float)
        count = len(self.param_names)
        if thetas.ndim != 2 or thetas.shape[1] != count:
            raise ValueError(f"thetas must have shape (batch, {count}).")
        prior = self.log_prior(thetas)
        answer = np.full(len(thetas), -np.inf)
        valid = np.isfinite(prior)
        if valid.any():
            with torch.no_grad():
                score = self.chi2(thetas[valid]).cpu().numpy()
            # Noise and selected input pixels are fixed, so the Gaussian
            # normalization is constant across every candidate.
            answer[valid] = -.5 * score + prior[valid]
        return answer

    def log_probability(self, theta):
        """Scalar conditional posterior; suitable for emcee and diagnostics."""
        return float(self.log_probability_batch(np.asarray(theta)[None, :])[0])

    def fit(self, start_velocity, *, start_turnover=3., start_pa=20.,
            start_offset=(.2, -.2), start_inclination=None,
            noisy=True, max_evaluations=2000):
        """Find one local solution from explicitly supplied starting values.

        Starting PA and inclination are in degrees. A sampled inclination
        starts at its prior mean unless ``start_inclination`` is supplied.
        No injected kinematic values initialize the free fit parameters.
        The known emission map and center remain fixed. Noiseless fits
        minimize raw SSE without priors; noisy fits maximize the conditional
        posterior above. Returned convergence describes this local optimizer.
        """
        from scipy.optimize import minimize

        if (isinstance(max_evaluations, bool) or int(max_evaluations) != max_evaluations
                or max_evaluations < 1):
            raise ValueError("max_evaluations must be a positive integer.")
        start = np.array([start_velocity, start_turnover, np.deg2rad(start_pa), *start_offset], dtype=float)
        if self.sample_inclination:
            angle = self.inclination_mean if start_inclination is None else start_inclination
            start = np.append(start, np.deg2rad(angle))
        elif start_inclination is not None:
            raise ValueError("This experiment fixes inclination; omit start_inclination.")
        if start.shape != (len(self.param_names),) or not np.isfinite(self.log_prior(start)):
            raise ValueError("Starting parameters must lie inside the stated prior bounds.")
        if noisy and self.sigma <= 0:
            raise ValueError("Use noisy=False for a zero-noise simulation.")
        scale = self.parameter_scales
        lower, upper = (bound / scale for bound in self.theta_bounds)
        best = [np.inf, start.copy()]

        def objective(scaled):
            point = np.asarray(scaled) * scale
            with torch.no_grad():
                squared = float(self.chi2(point, noisy=noisy))
            value = .5 * squared - (float(self.log_prior(point)) if noisy else 0.)
            if np.isfinite(value) and value < best[0]:
                best[:] = [value, point.copy()]
            return value

        result = minimize(objective, start / scale, method="Powell",
                          bounds=list(zip(lower, upper)),
                          options={"maxfev": int(max_evaluations), "xtol": 1e-5, "ftol": 1e-8})
        point = best[1]
        with torch.no_grad():
            score = float(self.chi2(point, noisy=noisy))
        return {"theta": point, "start_theta": start, "chi2": score,
                "log_probability": -.5 * score + float(self.log_prior(point)) if noisy else None,
                "success": bool(result.success), "message": str(result.message),
                "nfev": int(result.nfev), "noisy": bool(noisy),
                "emission_mode": "known injected map; conditional recovery"}
