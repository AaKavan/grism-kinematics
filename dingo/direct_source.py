"""Direct R+C prediction for a real three-file source.

The supplied emission map is dispersed into both original grism images. The
observations are never rectified during scoring, and image.fits is retained
only for display. This experimental model has no explicit PSF, intrinsic line
width, background component, or Sérsic constraint on the emitting gas.
"""
from pathlib import Path
import copy
import hashlib

import numpy as np
import torch
import yaml

from . import grism, kinematics, utils
from .fitting import estimate_background_sigma


def _sha256(path):
    """Hash an input without loading the entire file into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SourceModel:
    """A real source with a fixed center and a Gaussian inclination prior.

    Parameters
    ----------
    source_id : int
        Reads ``data_dir/ID<source_id>/{image,R,C}.fits`` and
        ``config_kinematics.yaml``. No previous fit is required or loaded.
    data_dir : path-like
        Directory containing the individual ID folders.
    center : pair of floats
        Fixed kinematic x,y center in zero-based R/C stack pixels. Both grisms
        use this same source-frame center. Its uncertainty is not included.
    inclination_mean, inclination_sigma : float
        Mean and standard deviation of the Gaussian inclination prior, in
        degrees. Inclination is fitted, with hard support from 5 to 85 degrees.
    device : str
        ``cpu`` or an available CUDA device for predictions and derivatives.
        A separate sparse source-map solver can still run on the CPU.
    noise_sigma : positive scalar or (R,C) pair, optional
        Standard deviations in the original input-pixel intensity units.
        By default estimate R and C separately with sigma-clipped statistics.
        Coadd and subtraction noise correlations are not modeled here.
    max_velocity : float
        Upper bound on deprojected asymptotic velocity, at most 1000 km/s.
    fixed_rdy : float or None
        Fix R's vertical registration relative to C to this value in pixels.
        None leaves it fitted. This conditions inference on the chosen
        alignment; it is not a Gaussian registration prior.

    Notes
    -----
    Parameter order is ``[V_inf, R_t, PA, R_dx, R_dy, inclination]``. Internal
    angles are radians; radii and offsets are pixels; velocity is km/s. Only
    inclination has a nonuniform kinematic prior. All other parameters have
    the documented hard bounds. C registration and systemic velocity are zero.

    An emission map must be supplied explicitly to ``predict`` and ``chi2``.
    There is no fallback to the display image or a previously inferred map.
    Finite-pixel masks are fixed once from the input arrays; finite pixels,
    including any externally cleaned artifact regions, are not hidden.
    """

    param_names = ("velocity.V_rot", "velocity.R_v", "velocity.theta_v",
                   "image.R.dx", "image.R.dy", "velocity.inc_v")

    def __init__(self, source_id, *, data_dir, center=(40.25, 41.0),
                 inclination_mean=75., inclination_sigma=5., device="cpu",
                 noise_sigma=None, max_velocity=1000., fixed_rdy=None):
        from astropy.io import fits

        if isinstance(source_id, bool) or int(source_id) != source_id or source_id < 0:
            raise ValueError("source_id must be a nonnegative integer.")
        self.source_id = int(source_id)
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Choose a CPU or CUDA device.")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; choose device='cpu'.")
        if (isinstance(max_velocity, bool) or not np.isfinite(max_velocity)
                or not 0 < max_velocity <= 1000):
            raise ValueError("max_velocity must be positive and no greater than 1000 km/s.")
        if (isinstance(inclination_mean, bool) or not np.isfinite(inclination_mean)
                or not 5 <= inclination_mean <= 85):
            raise ValueError("inclination_mean must be between 5 and 85 degrees.")
        if (isinstance(inclination_sigma, bool) or not np.isfinite(inclination_sigma)
                or inclination_sigma <= 0):
            raise ValueError("inclination_sigma must be positive and finite, in degrees.")
        self.inclination_mean = float(inclination_mean)
        self.inclination_sigma = float(inclination_sigma)
        self.max_velocity = float(max_velocity)
        if fixed_rdy is not None:
            if (isinstance(fixed_rdy, (bool, np.bool_)) or np.ndim(fixed_rdy) != 0
                    or not np.isfinite(fixed_rdy) or not -10 <= fixed_rdy <= 10):
                raise ValueError("fixed_rdy must be None or a finite pixel offset between -10 and 10.")
        self.fixed_rdy = None if fixed_rdy is None else float(fixed_rdy)

        self.source_dir = Path(data_dir).expanduser().resolve() / f"ID{self.source_id}"
        paths = {name: self.source_dir / name for name in
                 ("image.fits", "R.fits", "C.fits", "config_kinematics.yaml")}
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing source inputs:\n" + "\n".join(missing))
        config = yaml.safe_load(paths["config_kinematics.yaml"].read_text())
        summary = config["summary"]
        if summary.get("name") != f"ID{self.source_id}":
            raise ValueError("Source ID disagrees with config_kinematics.yaml.")
        arrays = {p: np.array(fits.getdata(paths[f"{p}.fits"]), dtype=np.float32, copy=True)
                  for p in ("R", "C")}
        if any(a.ndim != 2 for a in arrays.values()) or arrays["R"].shape != arrays["C"].shape:
            raise ValueError("R and C must be two-dimensional arrays with the same shape.")
        self.shape = tuple(arrays["R"].shape)
        height, width = self.shape
        if height != width or height < 3 or height % 2 != 1:
            raise ValueError("This cutout convention requires square, odd-sized R/C images.")
        radius = int(summary.get("r_fit", (height - 1) // 2))
        if 2 * radius + 1 != height:
            raise ValueError("summary.r_fit must match the R/C cutout size.")
        center = np.asarray(center, dtype=float)
        if (center.shape != (2,) or not np.isfinite(center).all()
                or np.any(center < 0) or center[0] > width - 1 or center[1] > height - 1):
            raise ValueError("center must contain two finite x,y coordinates inside the R/C cutout.")
        self.center = tuple(float(x) for x in center)
        self.image = np.array(fits.getdata(paths["image.fits"]), copy=True)
        if self.image.ndim != 2:
            raise ValueError("image.fits must be two-dimensional; it is used for display only.")
        self.valid_by_pupil = {p: torch.as_tensor(np.isfinite(a), device=self.device)
                               for p, a in arrays.items()}
        if any(not bool(mask.any()) for mask in self.valid_by_pupil.values()):
            raise ValueError("Each grism image must contain at least one finite pixel.")
        # Invalid values cannot participate in arithmetic or its derivatives.
        # Their replacement value has no likelihood weight because masks stay fixed.
        self.data = {p: self._tensor(np.where(np.isfinite(a), a, 0.)) for p, a in arrays.items()}
        self.sigma_by_pupil = self._noise_standard_deviations(arrays, noise_sigma)
        self.noise_source = "estimated_from_inputs" if noise_sigma is None else "supplied"
        self.noise_note = (
            "Separate fixed R/C input-pixel standard deviations; nonfinite pixels excluded. "
            "Coadd/subtraction covariance is not modeled. No finite artifact mask is applied."
        )

        redshift = float(summary["z"])
        rest_wavelength = float(summary["lambda_rest"])
        if not np.isfinite([redshift, rest_wavelength]).all() or redshift <= -1 or rest_wavelength <= 0:
            raise ValueError("The source redshift and rest wavelength must give a positive observed wavelength.")
        self.redshift = redshift
        self.wavelength = self._tensor([rest_wavelength]) * (1 + redshift)
        self.calibration_directory = str(Path(grism.get_grism_cal_dir()).resolve())
        self.forward_models, self.cutouts = {}, {}
        for pupil in ("R", "C"):
            image_config = config["image"][pupil]
            _, spatial, dispersion = grism.load_nircam_wfss_model(
                pupil, image_config["module"], summary["filter"])
            self.forward_models[pupil] = utils.get_grism_model_torch(
                spatial, dispersion, pupil, 1024, 1024, direction="forward", device=self.device)
            if "cutout" in image_config:
                cutout = tuple(image_config["cutout"])
                if len(cutout) != 4 or tuple(cutout[2:]) != (width, height):
                    raise ValueError(f"Explicit {pupil} cutout must have origin x,y and matching width,height.")
                if not np.isfinite(cutout).all():
                    raise ValueError(f"Explicit {pupil} cutout contains nonfinite values.")
            else:
                x, y = self.forward_models[pupil](self._tensor(float(radius)),
                                                   self._tensor(float(radius)), self.wavelength)
                cutout = (int(x) - radius, int(y) - radius, width, height)
            self.cutouts[pupil] = cutout
        self.y, self.x = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=self.device),
            torch.arange(width, dtype=torch.float32, device=self.device), indexing="ij")
        self.fixed_velocity = {"x0_v": self._tensor(self.center[0]),
                               "y0_v": self._tensor(self.center[1]), "v_sys": self._tensor(0.)}
        self.fixed_c_offset = (0., 0.)
        self.theta_bounds = (np.array([0., .5, 0., -10., -10., np.deg2rad(5.)]),
                             np.array([self.max_velocity, 40., 2*np.pi, 10., 10., np.deg2rad(85.)]))
        self.parameter_scales = np.array([200., 5., 1., 1., 1., np.deg2rad(5.)])
        self.fixed_parameters = {"velocity.x0_v": self.center[0], "velocity.y0_v": self.center[1],
                                 "velocity.v_sys": 0., "image.C.dx": 0., "image.C.dy": 0.}
        if self.fixed_rdy is not None:
            self.fixed_parameters["image.R.dy"] = self.fixed_rdy
        self.input_records = {name: {"path": str(path), "sha256": _sha256(path)}
                              for name, path in paths.items()}
        self.configuration = copy.deepcopy(config)
        self.configuration["direct_model"] = dict(
            parameters=list(self.param_names), center_pixels=list(self.center),
            inclination_mean_deg=self.inclination_mean, inclination_sigma_deg=self.inclination_sigma,
            bounds=[[float(lo), float(hi)] for lo, hi in zip(*self.theta_bounds)],
            noise_sigma=dict(self.sigma_by_pupil), noise_source=self.noise_source,
            cutouts={p: list(c) for p, c in self.cutouts.items()},
            psf="none", intrinsic_dispersion_kms=0., systemic_velocity_kms=0.,
            imaging_role="display only", calibration_directory=self.calibration_directory,
            calibration_reference_pixel=[1024, 1024], noise_note=self.noise_note)
        # Preserve the existing free-mode record so old saved runs still reopen.
        if self.fixed_rdy is not None:
            self.configuration["direct_model"]["fixed_rdy"] = self.fixed_rdy

    def _tensor(self, value):
        """Move values to this device while preserving autograd connections."""
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    @staticmethod
    def _noise_standard_deviations(arrays, supplied):
        if supplied is None:
            values = np.array([estimate_background_sigma(arrays[p]) for p in ("R", "C")])
        else:
            try:
                values = np.asarray(supplied, dtype=float)
                if values.ndim == 0:
                    values = np.repeat(values, 2)
            except (TypeError, ValueError) as error:
                raise ValueError("noise_sigma must be a positive scalar or an R,C pair.") from error
            if any(isinstance(x, (bool, np.bool_)) for x in np.asarray(supplied, dtype=object).flat):
                raise ValueError("noise_sigma must contain numerical standard deviations, not booleans.")
        if values.shape != (2,) or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("Both R/C noise standard deviations must be finite and positive.")
        return dict(zip(("R", "C"), (float(x) for x in values)))

    def validate_inputs(self):
        """Reject reuse after any of the four source inputs has changed."""
        for name, record in self.input_records.items():
            path = Path(record["path"])
            if not path.is_file() or _sha256(path) != record["sha256"]:
                raise ValueError(f"{name} changed after preparation. Construct a new SourceModel.")

    def mapping_coordinates(self, theta):
        """Map each source-pixel center to each original grism cutout.

        Theta has shape (6,) or (batch,6); angles are radians. The same source
        positions and arctangent velocity field feed both calibrated mappings.
        """
        theta = self._tensor(theta)
        if theta.ndim not in (1, 2) or theta.shape[-1] != len(self.param_names):
            raise ValueError("theta must have shape (6,) or (batch,6).")
        velocity, turnover, angle, rdx, rdy, inclination = [theta[..., k, None, None] for k in range(6)]
        if self.fixed_rdy is not None:
            # Keep the full physical parameter vector for saved-array
            # compatibility, but this coordinate has no fitted degree of freedom.
            rdy = torch.full_like(rdy, self.fixed_rdy)
        vz = kinematics.arctangent_disk_velocity_model(
            self.x, self.y, V_rot=velocity, R_v=turnover, theta_v=angle,
            inc_v=inclination, **self.fixed_velocity)
        wavelength = self.wavelength * (1 + vz / 299792.458)
        zeros = torch.zeros_like(vz)
        result = {}
        for pupil in ("R", "C"):
            mdx, mdy = self.forward_models[pupil](zeros, zeros, wavelength)
            offset_x, offset_y = (rdx, rdy) if pupil == "R" else self.fixed_c_offset
            cutout = self.cutouts[pupil]
            result[pupil] = (self.x + cutout[0] - offset_x - mdx,
                             self.y + cutout[1] - offset_y - mdy)
        return result

    def predict(self, theta, emission):
        """Disperse the explicitly supplied shared brightness map into R and C."""
        intensity = self._tensor(emission)
        if tuple(intensity.shape) != self.shape:
            raise ValueError(f"emission must have shape {self.shape}.")
        coordinates = self.mapping_coordinates(theta)
        height, width = self.shape
        return tuple(kinematics.bilinear_interpolte_intensity_torch(
            *coordinates[p], intensity, (0, 0, width, height)) for p in ("R", "C"))

    def residuals(self, theta, *, emission, noisy=True):
        """Original-pixel residuals with separate R/C noise and fixed masks."""
        if not noisy:
            raise ValueError("Real observations have no saved noiseless counterpart.")
        return tuple(torch.where(self.valid_by_pupil[p],
                                 (prediction - self.data[p]) / self.sigma_by_pupil[p], 0.)
                     for p, prediction in zip(("R", "C"), self.predict(theta, emission)))

    def chi2(self, theta, *, emission, noisy=True):
        """Sum weighted squared residuals without any prior penalty."""
        return sum(torch.sum(r.to(torch.float64)**2, dim=(-2, -1))
                   for r in self.residuals(theta, emission=emission, noisy=noisy))

    def prior_penalty(self, theta):
        """Differentiable negative log inclination prior, omitting constants."""
        theta = self._tensor(theta)
        if theta.ndim not in (1, 2) or theta.shape[-1] != len(self.param_names):
            raise ValueError("theta must have shape (6,) or (batch,6).")
        mean, sigma = np.deg2rad([self.inclination_mean, self.inclination_sigma])
        return .5 * ((theta[..., 5] - mean) / sigma)**2

    def log_prior(self, theta):
        """Gaussian inclination prior inside the documented hard bounds.

        This is used for bound checks by the free-map optimizer; the actual
        inclination penalty must also be added to its objective and gradient.
        """
        point = np.asarray(theta, dtype=float)
        if point.ndim not in (1, 2) or point.shape[-1] != len(self.param_names):
            raise ValueError("theta must have shape (6,) or (batch,6).")
        lower, upper = self.theta_bounds
        valid = np.isfinite(point).all(axis=-1) & (point >= lower).all(axis=-1) & (point <= upper).all(axis=-1)
        if self.fixed_rdy is not None:
            # Bound checks can receive a float32 copy of the exact setting.
            # This tolerance accepts rounding, not a sampled prior width.
            valid &= np.isclose(point[..., 4], self.fixed_rdy, rtol=1e-7, atol=1e-7)
        mean, sigma = np.deg2rad([self.inclination_mean, self.inclination_sigma])
        penalty = .5 * ((point[..., 5] - mean) / sigma)**2
        return np.where(valid, -penalty, -np.inf)
