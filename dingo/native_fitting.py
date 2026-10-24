"""Experimental shared line-map fitting on native R/C exposures.

The continuum is a low-order polynomial along each detector row (R) or column
(C), with separate coefficients per exposure. It is eliminated by weighted
least squares at every trial. This is a conditional, penalized fit, not a
marginal posterior over the line map and not yet a calibrated science pipeline.
"""

import numpy as np
from scipy.optimize import minimize
import torch
from torch import nn
from torch.nn import functional as F

from .native_forward import NativeGaussianLikelihood, projected_arctan_velocity


class StripContinuumLikelihood(nn.Module):
    """Fixed-pixel Gaussian likelihood profiled over polynomial continuum.

    Weighted SVD handles truncated/masked strips, including empty ones. The
    projection depends only on fixed errors/masks, never the trial velocity.
    `relative` subtracts the zero-line profiled log likelihood; its constant
    differs with masks/continuum degree, so don't compare those scores across
    different data or continuum models. The continuum coefficient uncertainty
    is retained through the projection, unlike fixing a subtracted continuum.
    """

    def __init__(self, science, error, valid, pupils, degree=2, *, device="cpu"):
        super().__init__()
        base = NativeGaussianLikelihood(science, error, valid, device=device, dtype=torch.float64)
        if degree < 0 or degree >= min(base.data.shape[-2:]):
            raise ValueError("continuum degree must be >=0 and smaller than each image axis")
        if len(pupils) != len(base.data) or any(p not in ("R", "C") for p in pupils):
            raise ValueError("Supply one R/C pupil per exposure")
        self.register_buffer("data", base.data)
        self.register_buffer("valid", base.valid)
        self.register_buffer("inverse_error", base.valid / base.variance.sqrt())
        self.groups = []
        self.degree = degree
        ranks = []
        for pupil in ("R", "C"):
            indices = [i for i, p in enumerate(pupils) if p == pupil]
            if not indices:
                continue
            idx = torch.tensor(indices, device=device)
            inv = self.inverse_error[idx]
            if pupil == "C":
                inv = inv.transpose(-1, -2)
            basis = torch.as_tensor(np.polynomial.legendre.legvander(
                np.linspace(-1, 1, inv.shape[-1]), degree), device=device, dtype=torch.float64)
            design = inv[..., None] * basis
            left, singular, right = torch.linalg.svd(design, full_matrices=False)
            nonzero = singular > singular[..., :1] * 1e-12
            q = left * nonzero[..., None, :]
            inverse = torch.where(nonzero, 1 / singular.clamp_min(1e-300), 0.)
            pinv = (right.transpose(-1, -2) * inverse[..., None, :]) @ left.transpose(-1, -2)
            covariance = pinv @ pinv.transpose(-1, -2)
            variance = torch.einsum("dk,eckl,dl->ecd", basis, covariance, basis)
            variance = torch.where(nonzero.all(-1)[..., None], variance, torch.inf)
            for name, value in (("indices", idx), ("q", q), ("pinv", pinv), ("basis", basis),
                                ("prediction_variance", variance)):
                self.register_buffer(f"{name}_{pupil}", value)
            ranks.append(nonzero.sum().item())
            self.groups.append(pupil)
        self.continuum_rank = int(sum(ranks))
        self.register_buffer("residual_data", self.residualize(self.data * self.inverse_error))

    def residualize(self, whitened):
        """Remove the continuum subspace from already whitened pixels."""
        single = whitened.ndim == 3
        z = whitened[None] if single else whitened
        if z.ndim != 4 or z.shape[-3:] != self.data.shape:
            raise ValueError("Expected [exposure,y,x] or [batch,exposure,y,x]")
        output = torch.zeros_like(z)
        for pupil in self.groups:
            idx, q = getattr(self, f"indices_{pupil}"), getattr(self, f"q_{pupil}")
            block = z[:, idx]
            if pupil == "C":
                block = block.transpose(-1, -2)
            coefficients = torch.einsum("ecdk,becd->beck", q, block)
            residual = block - torch.einsum("ecdk,beck->becd", q, coefficients)
            if pupil == "C":
                residual = residual.transpose(-1, -2)
            output = output.index_copy(1, idx, residual)
        return output[0] if single else output

    def relative(self, line):
        if line.shape[-3:] != self.data.shape:
            raise ValueError("Line prediction differs from exposure shape")
        whitened = torch.where(self.valid, line.double(), 0.) * self.inverse_error
        projected = self.residualize(whitened)
        score = -.5 * (projected.square() - 2*projected*self.residual_data).sum((-3, -2, -1))
        return torch.where(torch.isfinite(score), score, -torch.inf)

    def chi_square(self, line):
        whitened = torch.where(self.valid, line.double(), 0.) * self.inverse_error
        return (self.residual_data - self.residualize(whitened)).square().sum((-3, -2, -1))

    def continuum(self, line):
        """Best-fitting continuum in SCI units, also evaluated in masked pixels.

        Empty strips receive zero (no information). Other masked pixels are a
        polynomial extrapolation and must not be treated as observed data.
        """
        single = line.ndim == 3
        line = line[None] if single else line
        z = (self.data - torch.where(self.valid, line.double(), 0.)) * self.inverse_error
        output = torch.zeros_like(z)
        for pupil in self.groups:
            idx = getattr(self, f"indices_{pupil}")
            block = z[:, idx]
            if pupil == "C":
                block = block.transpose(-1, -2)
            coef = torch.einsum("eckd,becd->beck", getattr(self, f"pinv_{pupil}"), block)
            continuum = torch.einsum("dk,beck->becd", getattr(self, f"basis_{pupil}"), coef)
            if pupil == "C":
                continuum = continuum.transpose(-1, -2)
            output = output.index_copy(1, idx, continuum)
        return output[0] if single else output

    def continuum_variance(self):
        """Pointwise continuum-prediction variance under independent input noise.

        Useful for held-out pixels. It must not be added to training-pixel ERR:
        fitted continuum and training noise are correlated there. Rank-deficient
        strips have infinite variance rather than fictitious extrapolation
        precision. Off-diagonal prediction covariance is not returned here.
        """
        output = torch.zeros_like(self.data)
        for pupil in self.groups:
            value = getattr(self, f"prediction_variance_{pupil}")
            if pupil == "C":
                value = value.transpose(-1, -2)
            output = output.index_copy(0, getattr(self, f"indices_{pupil}"), value)
        return output


class BilinearLineMap(nn.Module):
    """Nonnegative control-point surface brightness and finer source quadrature.

    Nonnegativity is imposed by the fitter's bounds, not by a saturating sigmoid.
    A constant control map of one integrates to fixed `flux_unit`. The supplied
    unit is fixed before optimization and also defines the regularization scale.
    Neighbor differences approximate the integral of squared brightness gradient
    in 2-D; no normalization by the fitted flux is hidden in the penalty.
    """

    def __init__(self, control_size=17, quadrature_size=49, half_extent=.63,
                 flux_unit=1., *, device="cpu", dtype=torch.float32):
        super().__init__()
        if control_size < 2 or quadrature_size < control_size or half_extent <= 0 or flux_unit <= 0:
            raise ValueError("Invalid map grid or fixed flux unit")
        self.control_size, self.quadrature_size = control_size, quadrature_size
        self.half_extent, self.flux_unit = float(half_extent), float(flux_unit)
        axis = torch.linspace(-half_extent, half_extent, quadrature_size, dtype=dtype, device=device)
        north, east = torch.meshgrid(axis, axis, indexing="ij")
        weights = torch.ones(quadrature_size, dtype=dtype, device=device)
        weights[0] = weights[-1] = .5
        area = weights[:, None] * weights[None, :]
        area /= area.sum()
        self.register_buffer("east", east.flatten())
        self.register_buffer("north", north.flatten())
        self.register_buffer("area", area.flatten())

    def forward(self, controls):
        if controls.shape[-2:] != (self.control_size, self.control_size):
            raise ValueError("Wrong control-map shape")
        image = F.interpolate(controls.reshape(-1, 1, self.control_size, self.control_size),
                              size=(self.quadrature_size, self.quadrature_size),
                              mode="bilinear", align_corners=True)
        return image.flatten(1) * self.area * self.flux_unit

    @staticmethod
    def roughness(controls):
        # Trapezoidal weights on the transverse axis make a linear brightness
        # field's gradient integral independent of control-grid resolution.
        controls = controls.double()
        transverse = controls.new_ones(controls.shape[-1])
        transverse[0] = transverse[-1] = .5
        dx = (torch.diff(controls, dim=-1).square() * transverse[:, None]).sum((-2, -1))
        dy = (torch.diff(controls, dim=-2).square() * transverse[None, :]).sum((-2, -1))
        return .5 * (dx + dy)


class NativeMapFitter:
    """Joint bounded fit of velocity and line map; analytic continuum profiling.

    Inclination/center/registration/blur remain conditional inputs in this stage.
    Held-out exposures evaluate the fixed line prediction with their own
    continuum re-fitted. No line-map parameters are learned from held-out data.
    """

    def __init__(self, projector, line_map, science, error, valid, pupils,
                 train_indices=None, continuum_degree=2, inclination=.9,
                 radius_ref=.5, sigma_v=45., held_indices=None):
        self.projector, self.line_map = projector, line_map
        if projector.nodes != line_map.east.numel():
            raise ValueError("Projector and line map have different source grids")
        self.device, self.dtype = projector.coefficients.device, projector.coefficients.dtype
        if line_map.east.device != self.device or line_map.east.dtype != self.dtype:
            raise ValueError("Projector and line map must share device/dtype")
        n = projector.exposures
        indices = np.arange(n) if train_indices is None else np.asarray(train_indices, dtype=int)
        if len(np.unique(indices)) != len(indices) or len(indices) == 0 or (indices < 0).any() or (indices >= n).any():
            raise ValueError("Invalid training indices")
        held = (np.setdiff1d(np.arange(n), indices) if held_indices is None
                else np.asarray(held_indices, dtype=int))
        if (held.ndim != 1 or len(np.unique(held)) != len(held) or
                (held < 0).any() or (held >= n).any() or np.intersect1d(indices, held).size):
            raise ValueError("Invalid or overlapping held-out indices")
        self.train_indices = torch.as_tensor(indices, device=self.device)
        self.held_indices = torch.as_tensor(held, device=self.device)
        arrays = [np.asarray(a) for a in (science, error, valid)]
        def make(idx):
            return StripContinuumLikelihood(*(a[idx] for a in arrays), [pupils[i] for i in idx],
                                            degree=continuum_degree, device=self.device)
        self.train = make(indices)
        self.held = make(held) if len(held) else None
        self.inclination, self.radius_ref, self.sigma_v = inclination, radius_ref, sigma_v

    def predict(self, vector):
        q = vector[:3]
        controls = vector[3:].reshape(self.line_map.control_size, self.line_map.control_size)
        flux = self.line_map(controls)
        velocity = projected_arctan_velocity(self.line_map.east, self.line_map.north,
                                             q[0]*200, q[1]*.2, q[2], self.inclination,
                                             radius_ref=self.radius_ref)
        return self.projector(flux, velocity, sigma_v=self.sigma_v)[0], controls

    def fit(self, start, initial_map=None, *, strength, maxiter=800, gtol=1e-3,
            callback=None, fixed=None):
        """Fit with optional fixed physical `v_ref`, `r_turn`, and/or `theta`.

        Fixed variables are removed from optimization and its gradient test.
        A nonrotating null fixes v_ref=0 and also r_turn/theta, which are then
        unidentifiable. A velocity profile fixes only v_ref and re-fits all
        other parameters, including the nonnegative emission map.
        """
        if strength < 0:
            raise ValueError("Regularization strength must be nonnegative")
        size = self.line_map.control_size
        initial_map = np.ones((size, size)) if initial_map is None else np.asarray(initial_map)
        if initial_map.shape != (size, size) or not np.isfinite(initial_map).all() or (initial_map < 0).any():
            raise ValueError("Invalid nonnegative initial map")
        initial = np.r_[np.asarray(start) / [200., .2, 1.], initial_map.ravel()]
        bounds = [(-7.5, 7.5), (.1, 10.), (-4*np.pi, 4*np.pi)] + [(0., None)] * size**2
        names, scales = ["v_ref", "r_turn", "theta"], [200., .2, 1.]
        fixed = {} if fixed is None else dict(fixed)
        if set(fixed) - set(names):
            raise ValueError("Unknown fixed kinematic parameter")
        free = np.ones(len(initial), dtype=bool)
        for name, value in fixed.items():
            i = names.index(name)
            scaled = float(value) / scales[i]
            if not np.isfinite(scaled) or not bounds[i][0] <= scaled <= bounds[i][1]:
                raise ValueError("Fixed parameter outside its physical bounds")
            initial[i], free[i] = scaled, False
        free_indices = torch.as_tensor(np.flatnonzero(free), device=self.device)
        fixed_vector = torch.as_tensor(initial, device=self.device, dtype=self.dtype)
        best = {"objective": np.inf}
        evaluations = 0
        def objective(x):
            nonlocal evaluations
            variables = torch.tensor(x, device=self.device, dtype=self.dtype, requires_grad=True)
            vector = fixed_vector.index_copy(0, free_indices, variables)
            line, controls = self.predict(vector)
            loss = -self.train.relative(line[self.train_indices]) + strength*self.line_map.roughness(controls)
            gradient, = torch.autograd.grad(loss, variables)
            value, grad = float(loss.detach()), gradient.detach().cpu().numpy().astype(float)
            evaluations += 1
            if np.isfinite(value) and np.isfinite(grad).all() and value < best["objective"]:
                full_vector, full_gradient = initial.copy(), np.zeros_like(initial)
                full_vector[free], full_gradient[free] = x, grad
                best.update(objective=value, vector=full_vector, gradient=full_gradient)
            if callback is not None and evaluations % 100 == 0:
                callback(evaluations, value)
            return value, grad
        result = minimize(objective, initial[free], jac=True, method="L-BFGS-B",
                          bounds=[b for b, active in zip(bounds, free) if active],
                          options={"maxiter": maxiter, "maxls": 40, "maxcor": 30,
                                   "gtol": gtol, "ftol": 1e-14})
        if "vector" not in best:
            raise RuntimeError("No finite fit state was found")
        vector = torch.as_tensor(best["vector"], device=self.device, dtype=self.dtype)
        with torch.no_grad():
            line, controls = self.predict(vector)
            train_chi2 = float(self.train.chi_square(line[self.train_indices]))
            held_chi2 = None if self.held is None else float(self.held.chi_square(line[self.held_indices]))
            penalty = float(strength*self.line_map.roughness(controls))
            continuum = torch.zeros_like(line, dtype=torch.float64)
            continuum[self.train_indices] = self.train.continuum(line[self.train_indices])
            if self.held is not None:
                continuum[self.held_indices] = self.held.continuum(line[self.held_indices])
        projected_gradient = best["gradient"].copy()
        for i, (low, high) in enumerate(bounds):
            if best["vector"][i] <= low + 1e-7 and projected_gradient[i] > 0:
                projected_gradient[i] = 0
            if high is not None and best["vector"][i] >= high - 1e-7 and projected_gradient[i] < 0:
                projected_gradient[i] = 0
        fitted = best["vector"][:3] * [200., .2, 1.]
        if fitted[0] < 0:
            fitted[0] *= -1; fitted[2] += np.pi
        fitted[2] %= 2*np.pi
        report = {
            "start": list(start), "fitted_vref_rturn_theta": fitted.tolist(), "strength": strength,
            "objective": best["objective"], "penalty": penalty, "train_chi_square": train_chi2,
            "held_chi_square": held_chi2, "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message), "iterations": int(result.nit),
            "evaluations": evaluations, "projected_gradient_inf": float(abs(projected_gradient).max()),
            "gradient_tolerance": gtol, "stationary": bool(abs(projected_gradient).max() <= gtol),
            "fixed_parameters": {name: float(value) for name, value in fixed.items()},
            "boundary_hits": [name for name, v, (lo, hi), active in zip(
                ["signed_vref", "r_turn", "theta"], best["vector"][:3], bounds[:3], free[:3])
                              if active and min(v-lo, hi-v) < 1e-5*(hi-lo)],
            "train_pixels": int(self.train.valid.sum()), "train_continuum_rank": self.train.continuum_rank,
            "held_pixels": 0 if self.held is None else int(self.held.valid.sum()),
            "held_continuum_rank": 0 if self.held is None else self.held.continuum_rank,
        }
        return {"report": report, "vector": best["vector"],
                "controls": controls.cpu().numpy(), "line": line.cpu().numpy(),
                "continuum": continuum.cpu().numpy()}
