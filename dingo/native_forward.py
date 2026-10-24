"""Experimental batched, flux-conserving line projection onto native pixels.

This is a line operator, not yet a production science fitter. A science fit must
also supply continuum, contamination, calibrated throughput/PSF/LSF, registration
priors and a tested morphology prior. No observed image is resampled here.

Flux is integrated flux per source node, not surface brightness. The quadrature
and deposition require spatial/spectral convergence checks. Kernels
must describe the intended blur without double-counting the optical PSF in an
empirical LSF. GPU scatter sums can differ at floating-point roundoff level.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

C_KMS = 299792.458


def projection_coefficients(calibration, detector_xy, wavelength_um, origin_xy):
    """Exact polynomial in delta-wavelength, evaluated at every source node.

    Cache the field-dependent spatial terms in float64, then center the result
    on the cutout. This avoids subtracting full detector coordinates in float32
    at every likelihood evaluation. Trace(wavelength) has degree six.
    """
    xy = np.asarray(detector_xy, dtype=float)
    x, y = xy.T - 1024
    basis = np.array([np.ones_like(x), x, y, x*x, x*y, y*y,
                      x**3, x*x*y, x*y*y, y**3]).T
    wave = calibration.dispersion
    a, b, c, d = basis[:, :6] @ wave[:6], basis[:, :6] @ wave[6:12], basis[:, :3] @ wave[12:15], wave[15]
    z = wavelength_um - 3.95
    disp = np.stack([a + b*z + c*z*z + d*z**3, b + 2*c*z + 3*d*z*z,
                     c + 3*d*z, np.full_like(a, d)], axis=-1)
    trace = basis @ calibration.trace.reshape(3, 10).T
    cross = np.zeros((len(xy), 7))
    cross[:, 0] = trace[:, 0]
    cross[:, :4] += trace[:, 1, None] * disp
    for i in range(4):
        for j in range(4):
            cross[:, i+j] += trace[:, 2] * disp[:, i] * disp[:, j]
    along = np.pad(disp, ((0, 0), (0, 3)))
    coeff = np.stack((along, cross) if calibration.pupil == "R" else (cross, along), axis=1)
    coeff[:, :, 0] += xy - np.asarray(origin_xy)
    return coeff


def projected_arctan_velocity(east, north, v_ref, r_turn, theta, inclination,
                             radius_ref=0.5, center_east=0., center_north=0., v_sys=0.):
    """LOS field parameterized by V(radius_ref)*sin(i), in km/s.

    Positions/radii use arcseconds; theta is counterclockwise from east toward
    north (not astronomical PA). Scalar parameters or 1-D batches return [B,N].
    Inclination remains an explicit parameter; deprojection needs its posterior.
    """
    def p(value):
        return torch.as_tensor(value, device=east.device, dtype=east.dtype).reshape(-1, 1)
    x, y = east[None] - p(center_east), north[None] - p(center_north)
    xp = torch.cos(p(theta)) * x + torch.sin(p(theta)) * y
    yp = -torch.sin(p(theta)) * x + torch.cos(p(theta)) * y
    r = torch.sqrt(xp*xp + (yp / torch.cos(p(inclination)))**2 + 1e-16)
    return (p(v_ref) * torch.atan(r / p(r_turn)) / torch.atan(p(radius_ref) / p(r_turn))
            * xp / r + p(v_sys))


class NativeLineProjector(nn.Module):
    """One shared line map, many exposures, optionally many parameter vectors.

    coefficients: [exposure, node, xy, degree=7]
    kernels: [exposure, odd, odd] or one common [odd, odd], supplied explicitly.
    response: fixed relative conversion from source flux to exposure DN/s;
              unity is suitable only for a controlled simulation.
    deposition: bilinear by default; cubic is a smooth positive B-spline with
                an additional, phase-independent 1/3 pixel^2 variance per axis.
                This basis blur is part of the response to be calibrated.
    """

    def __init__(self, coefficients, wavelength_um, image_shape, kernels, response,
                 spectral_samples=5, *, device="cpu", dtype=torch.float64, deposition="bilinear"):
        super().__init__()
        coeff = torch.as_tensor(coefficients, dtype=dtype, device=device)
        if coeff.ndim != 4 or coeff.shape[2:] != (2, 7) or not torch.isfinite(coeff).all():
            raise ValueError("Expected finite coefficients [exposure,node,2,7]")
        self.exposures, self.nodes = coeff.shape[:2]
        if deposition not in ("bilinear", "cubic"):
            raise ValueError("deposition must be bilinear or cubic")
        self.deposition = deposition
        self.image_shape = tuple(image_shape)
        if len(self.image_shape) != 2 or min(self.image_shape) < 1:
            raise ValueError("image_shape must contain positive height and width")
        if wavelength_um <= 0 or not np.isfinite(wavelength_um):
            raise ValueError("wavelength_um must be positive and finite")
        self.wavelength_um = float(wavelength_um)
        kernel = torch.as_tensor(kernels, dtype=dtype, device=device)
        if kernel.ndim == 2:
            kernel = kernel[None].expand(self.exposures, -1, -1)
        if (kernel.ndim != 3 or kernel.shape[0] != self.exposures or
                kernel.shape[1] != kernel.shape[2] or kernel.shape[1] % 2 != 1):
            raise ValueError("kernels must be odd squares, one per exposure or shared")
        if not torch.isfinite(kernel).all() or (kernel < 0).any() or (kernel.sum((-2, -1)) <= 0).any():
            raise ValueError("kernels must have finite nonnegative mass")
        kernel = kernel / kernel.sum((-2, -1), keepdim=True)
        response = torch.as_tensor(response, dtype=dtype, device=device)
        if response.shape != (self.exposures,) or not torch.isfinite(response).all() or (response <= 0).any():
            raise ValueError("response must be positive, finite, one per exposure")
        if spectral_samples < 1:
            raise ValueError("spectral_samples must be >= 1")
        q, weights = np.polynomial.hermite.hermgauss(spectral_samples)
        self.register_buffer("coefficients", coeff)
        self.register_buffer("kernels", kernel)
        self.register_buffer("response", response)
        self.register_buffer("quadrature", torch.as_tensor(q * np.sqrt(2), dtype=dtype, device=device))
        self.register_buffer("quadrature_weights", torch.as_tensor(weights / np.sqrt(np.pi), dtype=dtype, device=device))

    @classmethod
    def from_target(cls, target, east_arcsec, north_arcsec, kernels, response, **kwargs):
        coeff = [projection_coefficients(e.calibration, e.source_detector_xy(east_arcsec, north_arcsec),
                                         target.wavelength_um, e.origin_xy) for e in target.exposures]
        return cls(np.stack(coeff), target.wavelength_um, target.exposures[0].science.shape,
                   kernels, response, **kwargs)

    def forward(self, flux, velocity, sigma_v=None, shifts=None):
        """Return [batch, exposure, y, x]; shifts are detector-pixel (dx,dy).

        Gaussian intrinsic velocity width sigma_v (scalar or [batch]) is optional.
        No flux renormalization occurs at cutout edges. Flux/velocity must end in
        node and have at most one preceding batch dimension.
        """
        flux = torch.as_tensor(flux, dtype=self.coefficients.dtype, device=self.coefficients.device)
        velocity = torch.as_tensor(velocity, dtype=flux.dtype, device=flux.device)
        if flux.ndim not in (1, 2) or velocity.ndim not in (1, 2) or flux.shape[-1] != self.nodes or velocity.shape[-1] != self.nodes:
            raise ValueError("flux and velocity must have shape [node] or [batch,node]")
        flux, velocity = torch.broadcast_tensors(flux.reshape(-1, self.nodes), velocity.reshape(-1, self.nodes))
        batch = len(velocity)
        if sigma_v is None:
            velocity_samples = velocity[..., None]
            weights = self.quadrature_weights.new_ones(1)
        else:
            sigma = torch.as_tensor(sigma_v, dtype=flux.dtype, device=flux.device).reshape(-1, 1, 1)
            velocity_samples = velocity[..., None] + sigma * self.quadrature
            weights = self.quadrature_weights
        dlambda = (self.wavelength_um / C_KMS) * velocity_samples[:, None, :, :, None]
        coeff = self.coefficients[None, :, :, None]
        xy = coeff[..., 6]
        for degree in range(5, -1, -1):
            xy = xy * dlambda + coeff[..., degree]
        if shifts is not None:
            shifts = torch.as_tensor(shifts, dtype=flux.dtype, device=flux.device)
            if shifts.ndim == 2:
                shifts = shifts[None]
            if shifts.shape[-2:] != (self.exposures, 2) or shifts.shape[0] not in (1, batch):
                raise ValueError("shifts must have shape [exposure,2] or [batch,exposure,2]")
            xy = xy + shifts[:, :, None, None, :]
        pad = self.kernels.shape[-1] // 2 + 1
        height, width = (n + 2*pad for n in self.image_shape)
        xy = xy + pad
        x, y = xy[..., 0].flatten(2), xy[..., 1].flatten(2)
        ix, iy = torch.floor(x).long(), torch.floor(y).long()
        dx, dy = x - ix, y - iy
        mass = (flux[:, None, :, None] * weights[None, None, None, :]
                * self.response[None, :, None, None]).expand(batch, self.exposures, self.nodes, len(weights)).flatten(2)
        image = flux.new_zeros((batch, self.exposures, height*width))
        def weights_1d(t):
            if self.deposition == "bilinear":
                return ((0, 1-t), (1, t))
            # Positive cardinal cubic B-spline. C2 across integer coordinates;
            # conserves flux/centroid and adds 1/3 pixel^2 variance per axis.
            # That blur is part of the response, NOT a drop-in calibrated PSF.
            return ((-1, (1-t)**3/6), (0, (3*t**3-6*t*t+4)/6),
                    (1, (-3*t**3+3*t*t+3*t+1)/6), (2, t**3/6))
        for ox, wx in weights_1d(dx):
            for oy, wy in weights_1d(dy):
                weight = wx*wy
                xx, yy = ix + ox, iy + oy
                inside = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
                index = yy.clamp(0, height-1)*width + xx.clamp(0, width-1)
                image = image.scatter_add(2, index, mass * weight * inside)
        image = image.reshape(batch, self.exposures, height, width)
        # conv2d is cross-correlation: flip to preserve an asymmetric PSF's sense.
        image = F.conv2d(image, self.kernels[:, None].flip((-2, -1)), groups=self.exposures)
        return image[..., 1:-1, 1:-1]


class NativeGaussianLikelihood(nn.Module):
    """Fixed-pixel diagonal Gaussian target shared by optimization/sampling.

    This assumption needs validation for processed SCI. It is inappropriate for
    EMLINE with the original SCI errors. Pass a complete prediction, including
    continuum and contaminants, when using actual observations.
    """

    def __init__(self, science, error, valid, *, device="cpu", dtype=torch.float64):
        super().__init__()
        data = torch.as_tensor(science, device=device, dtype=dtype)
        error = torch.as_tensor(error, device=device, dtype=dtype)
        valid = torch.as_tensor(valid, device=device, dtype=torch.bool)
        if data.ndim != 3 or data.shape != error.shape or data.shape != valid.shape:
            raise ValueError("science, error and valid must have shape [exposure,y,x]")
        if not valid.flatten(1).any(1).all():
            raise ValueError("Every exposure must have valid pixels")
        if not (torch.isfinite(data[valid]).all() and torch.isfinite(error[valid]).all() and (error[valid] > 0).all()):
            raise ValueError("Invalid SCI/ERR was included in the likelihood mask")
        self.register_buffer("valid", valid)
        self.register_buffer("data", torch.where(valid, data, 0.))
        self.register_buffer("variance", torch.where(valid, error.square(), 1.))
        self.register_buffer("normalization", torch.where(valid, torch.log(2*torch.pi*self.variance), 0.).sum())

    def forward(self, prediction):
        if prediction.shape[-3:] != self.data.shape:
            raise ValueError("prediction shape differs from data")
        prediction = prediction.to(dtype=self.data.dtype)
        residual = torch.where(self.valid, prediction, 0.) - self.data
        score = -0.5 * ((residual.square() / self.variance).sum((-3, -2, -1)) + self.normalization)
        return torch.where(torch.isfinite(score), score, -torch.inf)

    def relative(self, prediction):
        """log L(prediction) - log L(0), without large constant cancellation.

        The identical statistical target up to a fixed constant. Useful for
        float32 optimizers whose relative stopping rule would otherwise be
        dominated by ~100,000 pixels' normalization and noise-only residuals.
        """
        if prediction.shape[-3:] != self.data.shape:
            raise ValueError("prediction shape differs from data")
        prediction = prediction.to(dtype=self.data.dtype)
        model = torch.where(self.valid, prediction, 0.)
        score = -.5 * ((model.square() - 2*model*self.data) / self.variance).sum((-3, -2, -1))
        return torch.where(torch.isfinite(score), score, -torch.inf)
