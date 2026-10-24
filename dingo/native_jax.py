"""JAX implementation of the validated native-pixel operator for NumPyro.

Calibration, quadrature, pixel deposition and kernels are copied from an
existing NativeLineProjector. No data image is interpolated. The scientific
response limitations of native_forward.py still apply. JAX is optional.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax


def projected_velocity(east, north, v_ref, r_turn, theta, inclination, radius_ref=.5):
    xp = jnp.cos(theta)*east + jnp.sin(theta)*north
    yp = -jnp.sin(theta)*east + jnp.cos(theta)*north
    radius = jnp.sqrt(xp*xp + (yp/jnp.cos(inclination))**2 + 1e-16)
    return v_ref*jnp.arctan(radius/r_turn)/jnp.arctan(radius_ref/r_turn)*xp/radius


class JaxNativeProjector:
    """Single-source operator; use jax.vmap for particles or chains."""

    def __init__(self, torch_projector):
        if str(torch_projector.coefficients.dtype) == "torch.float64" and not jax.config.x64_enabled:
            raise ValueError("Enable JAX x64 before converting a float64 native projector")
        def array(tensor):
            return jnp.asarray(tensor.detach().cpu().numpy())
        self.coefficients = array(torch_projector.coefficients)
        self.kernels = array(torch_projector.kernels)
        self.response = array(torch_projector.response)
        self.quadrature = array(torch_projector.quadrature)
        self.quadrature_weights = array(torch_projector.quadrature_weights)
        self.wavelength_um = torch_projector.wavelength_um
        self.image_shape = torch_projector.image_shape
        self.exposures = torch_projector.exposures
        self.deposition = torch_projector.deposition

    def __call__(self, flux, velocity, sigma_v=45., shifts=None):
        if sigma_v is None:
            velocity_samples, weights = velocity[:, None], jnp.ones(1)
        else:
            velocity_samples = velocity[:, None] + sigma_v*self.quadrature
            weights = self.quadrature_weights
        dlambda = self.wavelength_um/299792.458*velocity_samples[None, :, :, None]
        coeff = self.coefficients[:, :, None]
        xy = coeff[..., 6]
        for degree in range(5, -1, -1):
            xy = xy*dlambda + coeff[..., degree]
        if shifts is not None:
            xy = xy + shifts[:, None, None, :]
        pad = self.kernels.shape[-1]//2 + 1
        height, width = (n+2*pad for n in self.image_shape)
        xy = xy + pad
        x, y = xy[..., 0].reshape(self.exposures, -1), xy[..., 1].reshape(self.exposures, -1)
        ix, iy = jnp.floor(x).astype(jnp.int32), jnp.floor(y).astype(jnp.int32)
        dx, dy = x-ix, y-iy
        mass = (flux[None, :, None]*weights[None, None, :]*self.response[:, None, None]).reshape(self.exposures, -1)

        def basis(t):
            if self.deposition == "bilinear":
                return ((0, 1-t), (1, t))
            return ((-1, (1-t)**3/6), (0, (3*t**3-6*t*t+4)/6),
                    (1, (-3*t**3+3*t*t+3*t+1)/6), (2, t**3/6))

        indices, values = [], []
        offset = jnp.arange(self.exposures)[:, None]*height*width
        for ox, wx in basis(dx):
            for oy, wy in basis(dy):
                xx, yy = ix+ox, iy+oy
                inside = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
                indices.append(offset + jnp.clip(yy, 0, height-1)*width + jnp.clip(xx, 0, width-1))
                values.append(mass*wx*wy*inside)
        indices = jnp.stack(indices).ravel()
        values = jnp.stack(values).ravel()
        image = jnp.zeros(self.exposures*height*width, dtype=flux.dtype).at[indices].add(values)
        image = image.reshape(1, self.exposures, height, width)
        kernel = self.kernels[:, None, ::-1, ::-1]
        image = lax.conv_general_dilated(image, kernel, (1, 1), "VALID",
                                        dimension_numbers=("NCHW", "OIHW", "NCHW"), feature_group_count=self.exposures)
        return image[0, :, 1:-1, 1:-1]


class JaxLineMap:
    """Exact align_corners=True bilinear quadrature of positive controls."""

    def __init__(self, torch_map):
        self.size = torch_map.control_size
        self.east = jnp.asarray(torch_map.east.cpu().numpy())
        self.north = jnp.asarray(torch_map.north.cpu().numpy())
        self.area = jnp.asarray(torch_map.area.cpu().numpy())
        self.flux_unit = torch_map.flux_unit
        coord = np.linspace(0., self.size-1, torch_map.quadrature_size)
        yy, xx = np.meshgrid(coord, coord, indexing="ij")
        ix, iy = np.floor(xx).astype(int), np.floor(yy).astype(int)
        self.indices, self.weights = [], []
        for ox, wx in ((0, 1-(xx-ix)), (1, xx-ix)):
            for oy, wy in ((0, 1-(yy-iy)), (1, yy-iy)):
                self.indices.append(jnp.asarray((np.minimum(iy+oy, self.size-1)*self.size+np.minimum(ix+ox, self.size-1)).ravel()))
                self.weights.append(jnp.asarray((wx*wy).ravel()))

    def __call__(self, controls):
        flat = controls.ravel()
        image = sum(flat[idx]*weight for idx, weight in zip(self.indices, self.weights))
        return image*self.area*self.flux_unit


class JaxStripLikelihood:
    """Identical fixed-error continuum projection, now compiled by JAX.

    Integrating linear continuum coefficients with a fixed flat measure gives
    the same line-parameter dependence as profiling them: the determinant is
    constant because design, errors and masks are fixed. This is not an
    absolute model evidence, or valid for parameter-dependent errors/masks.
    """

    def __init__(self, torch_likelihood):
        self.inverse_error = jnp.asarray(torch_likelihood.inverse_error.cpu().numpy())
        self.valid = jnp.asarray(torch_likelihood.valid.cpu().numpy())
        self.residual_data = jnp.asarray(torch_likelihood.residual_data.cpu().numpy())
        self.groups = []
        for pupil in torch_likelihood.groups:
            self.groups.append((pupil, np.asarray(getattr(torch_likelihood, f"indices_{pupil}").cpu()),
                                jnp.asarray(getattr(torch_likelihood, f"q_{pupil}").cpu().numpy())))

    def residualize(self, whitened):
        output = jnp.zeros_like(whitened)
        for pupil, idx, q in self.groups:
            block = whitened[idx]
            if pupil == "C":
                block = block.swapaxes(-1, -2)
            coef = jnp.einsum("ecdk,ecd->eck", q, block)
            residual = block-jnp.einsum("ecdk,eck->ecd", q, coef)
            if pupil == "C":
                residual = residual.swapaxes(-1, -2)
            output = output.at[idx].set(residual)
        return output

    def relative(self, line):
        projected = self.residualize(jnp.where(self.valid, line, 0.)*self.inverse_error)
        return -.5*jnp.sum(projected**2-2*projected*self.residual_data)

    def chi_square(self, line):
        projected = self.residualize(jnp.where(self.valid, line, 0.)*self.inverse_error)
        return jnp.sum((projected-self.residual_data)**2)
