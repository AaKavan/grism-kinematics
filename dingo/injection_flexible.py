"""Free-emission-map profile fits for direct R+C predictions.

Every source pixel has its own nonnegative intensity. A sparse linear solve
fits that same map to both original detector images for each trial velocity
field. There is no Sérsic profile, smoothing penalty, or use of the injected
brightness map in these solves. ForwardExperiment keeps the center fixed and
can fix inclination or give it a Gaussian prior. SourceModel also allows an
inclination prior and separate R/C noise. These are profile fits, not marginalized posteriors or
uncertainty estimates for real galaxies.
"""
import time

import numpy as np
import torch


def _check_experiment_inputs(experiment):
    """Reject a stale in-memory experiment after its saved inputs are replaced."""
    if hasattr(experiment, "validate_inputs"):
        experiment.validate_inputs()
        return
    from .simple_simulation import load_mock

    current = load_mock(experiment.mock.directory)
    for key in ("input_sha256", "truth_npz_sha256", "noise_sigma", "truth"):
        if current.metadata[key] != experiment.mock.metadata[key]:
            raise ValueError("The saved simulation has changed. Construct a new ForwardExperiment before fitting or saving.")


def _design_matrix(experiment, theta):
    """Construct the two sparse bilinear deposition operators stacked by row."""
    from scipy.sparse import coo_matrix, vstack

    point = np.asarray(theta, dtype=float)
    if point.shape != (len(experiment.param_names),) or not np.isfinite(experiment.log_prior(point)):
        raise ValueError("theta must contain the model's parameters within their stated bounds.")
    height, width = experiment.shape
    count = height * width
    source_index = np.arange(count)
    matrices = []
    with torch.no_grad():
        coordinates = experiment.mapping_coordinates(point)
    for pupil in ("R", "C"):
        x, y = (a.detach().cpu().numpy().ravel().astype(float)
                for a in coordinates[pupil])
        ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
        fx, fy = x - ix, y - iy
        rows, columns, weights = [], [], []
        for dx in (0, 1):
            for dy in (0, 1):
                col, row = ix + dx, iy + dy
                valid = (col >= 0) & (col < width) & (row >= 0) & (row < height)
                weight = (fx if dx else 1 - fx) * (fy if dy else 1 - fy)
                rows.append((row[valid] * width + col[valid]))
                columns.append(source_index[valid])
                weights.append(weight[valid])
        matrices.append(coo_matrix((np.concatenate(weights),
                                   (np.concatenate(rows), np.concatenate(columns))),
                                  shape=(count, count)).tocsr())
    return vstack(matrices).tocsr()


def _observations(experiment, noisy):
    """Flatten fixed input pixels, validity masks, and their noise scales.

    The image values are never remapped. Invalid pixels are omitted from the
    weighted solve rather than replaced by invented measurements. Injection
    tests retain their existing equal-noise and noiseless behavior.
    """
    if not noisy and not hasattr(experiment, "clean_data"):
        raise ValueError("Real-source inputs have no noiseless counterpart; use noisy=True.")
    data = experiment.data if noisy else experiment.clean_data
    arrays, masks, scales = [], [], []
    for pupil in ("R", "C"):
        pixels = data[pupil].detach().cpu().numpy().ravel().astype(float)
        valid = np.isfinite(pixels)
        if hasattr(experiment, "valid_by_pupil"):
            valid &= experiment.valid_by_pupil[pupil].detach().cpu().numpy().ravel().astype(bool)
        if not valid.any():
            raise ValueError(f"Grism {pupil} has no valid input pixels.")
        if noisy:
            sigma = (experiment.sigma_by_pupil[pupil] if hasattr(experiment, "sigma_by_pupil")
                     else experiment.sigma)
            if not np.isfinite(sigma) or sigma <= 0:
                raise ValueError("Use noisy=False when noise_sigma is zero; input noise must be positive.")
        else:
            sigma = 1.
        arrays.append(pixels)
        masks.append(valid)
        scales.append(np.full(pixels.shape, float(sigma)))
    return np.concatenate(arrays), np.concatenate(masks), np.concatenate(scales)


def _prior_penalty(experiment, theta):
    """Optional differentiable penalty, including a configured inclination prior."""
    if hasattr(experiment, "prior_penalty"):
        return experiment.prior_penalty(theta)
    return torch.zeros((), dtype=theta.dtype, device=theta.device)


def profile_map(experiment, theta, *, noisy=True, tolerance=1e-8,
                max_iterations=100, max_linear_iterations=1000):
    """Fit independent nonnegative source pixels at a specified velocity field.

    Parameters
    ----------
    experiment : ForwardExperiment or SourceModel
        Supplies calibrated coordinate mappings and original R/C input pixels.
        No injected or imaging-derived emission map is used in this solve.
        Sparse map solves use SciPy on the CPU; coordinate predictions can use
        the experiment's selected CPU or GPU device.
    theta : sequence of floats
        V_inf, turnover radius, PA in radians, R dx, R dy, followed by inclination
        in radians when fitted. C registration stays fixed. Any kinematic prior
        penalty is reported separately from the original-pixel chi squared.
    noisy : bool
        Fit the noisy inputs, or the saved noiseless R/C pair for a numerical
        consistency check. The noiseless score is SSE with no error-bar meaning.
    tolerance : float
        Requested bounded least-squares and inner LSMR convergence tolerance.
    max_iterations, max_linear_iterations : int
        Iteration limits for the bounded solver and its sparse linear solves.

    Returns
    -------
    dict
        Fitted ``emission``, ``predicted_R``, ``predicted_C``, raw ``sse``, noisy
        ``chi2`` if available, and explicit solver success/optimality fields.
        The fitted source map is a nuisance maximum, not a posterior draw.
    """
    _check_experiment_inputs(experiment)
    return _profile_map(experiment, theta, noisy=noisy, tolerance=tolerance,
                        max_iterations=max_iterations,
                        max_linear_iterations=max_linear_iterations)


def _profile_map(experiment, theta, *, noisy=True, tolerance=1e-8,
                 max_iterations=100, max_linear_iterations=1000):
    """Linear solve after public entry-point provenance validation."""
    from scipy.optimize import lsq_linear
    from threadpoolctl import threadpool_limits

    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite.")
    for value in (max_iterations, max_linear_iterations):
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError("Solver iteration limits must be positive integers.")
    matrix = _design_matrix(experiment, theta)
    observed, valid, sigma = _observations(experiment, noisy)
    # A common scale cancels from the nuisance optimum. Dividing by the minimum
    # sigma keeps equal-noise injection solves numerically identical to before,
    # while correctly weighting real R/C images with different noise levels.
    relative_weights = float(np.min(sigma[valid])) / sigma[valid]
    weighted_matrix = matrix[valid].multiply(relative_weights[:, None]).tocsr()
    weighted_observed = observed[valid] * relative_weights
    # These sparse solves repeatedly call small BLAS vector operations. Starting
    # many BLAS threads per operation can make one solve tens of seconds slower.
    # Bound only this context; the user's previous thread settings are restored.
    with threadpool_limits(limits=1, user_api="blas"):
        solution = lsq_linear(weighted_matrix, weighted_observed, bounds=(0., np.inf),
                              tol=float(tolerance), lsmr_tol=float(tolerance),
                              max_iter=int(max_iterations),
                              lsmr_maxiter=int(max_linear_iterations))
    predicted = matrix @ solution.x
    count = int(np.prod(experiment.shape))
    residual = predicted[valid] - observed[valid]
    sse = float(np.sum(residual**2))
    # A noiseless numerical check minimizes raw flux SSE. It has no noise
    # scale with which to combine a dimensionless Gaussian prior penalty.
    prior_penalty = 0.
    if noisy:
        with torch.no_grad():
            prior_penalty = float(_prior_penalty(experiment, torch.as_tensor(
                theta, dtype=torch.float32, device=experiment.device)))
    return {"theta": np.array(theta, dtype=float),
            "emission": solution.x.reshape(experiment.shape),
            "predicted_R": predicted[:count].reshape(experiment.shape),
            "predicted_C": predicted[count:].reshape(experiment.shape),
            "sse": sse, "chi2": float(np.sum((residual / sigma[valid])**2)) if noisy else None,
            "prior_penalty": prior_penalty, "valid_pixels": int(valid.sum()),
            "success": bool(solution.success), "message": str(solution.message),
            "optimality": float(solution.optimality), "iterations": int(solution.nit),
            "tolerance": float(tolerance), "noisy": bool(noisy),
            "emission_mode": "independent nonnegative pixels on the full source canvas"}


def scan_velocity(experiment, velocities, *, progress=True, save=True):
    """Refit a free nonnegative emission map at each conditional trial V_inf.

    All other kinematic parameters equal their injected values in this explicit
    diagnostic. The map is fitted from scratch using R+C alone. The difference
    in profile SSE is not a marginalized posterior or an uncertainty interval.
    Both clean and noisy scores are plotted when positive noise is available.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    from tqdm.auto import tqdm

    _check_experiment_inputs(experiment)
    if not hasattr(experiment, "mock"):
        raise ValueError("scan_velocity is an injection-only truth check; real-source scans need explicit fixed parameters.")
    velocities = np.asarray(velocities, dtype=float)
    if (velocities.ndim != 1 or len(velocities) < 2 or not np.isfinite(velocities).all()
            or np.any(velocities < 0) or np.any(velocities > 1000)
            or np.any(np.diff(velocities) <= 0)):
        raise ValueError("velocities must be an increasing grid between 0 and 1000 km/s.")
    modes = {"Noiseless": False}
    if experiment.sigma > 0:
        modes["With noise"] = True
    table = pd.DataFrame({"V_inf": velocities})
    for label, noisy in modes.items():
        scores = []
        for velocity in tqdm(velocities, desc=f"Free map: {label}", disable=not progress):
            point = experiment.truth_theta.copy()
            point[0] = velocity
            result = _profile_map(experiment, point, noisy=noisy)
            if not result["success"]:
                raise RuntimeError(f"Source-map solver failed at V={velocity:g}: {result['message']}")
            scores.append(result["sse"])
        table[label] = scores
    fig, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 4),
                             squeeze=False, layout="constrained")
    for ax, label in zip(axes.flat, modes):
        scores = table[label].to_numpy()
        best = int(np.argmin(scores))
        ax.plot(velocities, scores - scores[best], color="black")
        ax.axvline(experiment.truth_theta[0], color="tab:orange", ls="--", label="Injected truth")
        ax.plot(velocities[best], 0., "o", label=f"Grid minimum: {velocities[best]:g}")
        ax.set(title=label, xlabel="Trial V_inf [km/s]",
               ylabel="Extra original-image squared residual")
        ax.legend(fontsize=9)
    fig.suptitle("One free nonnegative emission map refitted jointly to R+C at each velocity")
    if save:
        directory = experiment.mock.directory / "direct_free_map"
        directory.mkdir(parents=True, exist_ok=True)
        table.to_csv(directory / "velocity_scan.csv", index=False)
        fig.savefig(directory / "velocity_scan.png", dpi=150, bbox_inches="tight")
        (directory / "scan_info.txt").write_text(
            "Conditional profile-likelihood check, not a posterior or error interval.\n"
            "All kinematic parameters except V_inf fixed to injected truth.\n"
            "Every source pixel fitted independently with intensity >= 0; no smoothing or light profile.\n"
            "Original R and C input pixels are compared to forward predictions.\n"
            "Solver: scipy.optimize.lsq_linear, tolerance=1e-8, maximum 100 bounded iterations,\n"
            "maximum 1000 LSMR iterations. Every reported solve returned success.\n")
    return table, fig


def fit(experiment, start_velocity, *, start_turnover=3., start_pa=20.,
        start_offset=(.2, -.2), start_inclination=None, noisy=True,
        max_evaluations=150, progress=True, callback=None):
    """Locally fit kinematics while profiling one shared nonnegative source map.

    Starts are explicit; PA and optional inclination are supplied in degrees.
    Inclination is fitted with its stated Gaussian prior when configured;
    otherwise injection inclination stays fixed. The center is fixed in either
    case. The source-map fit is repeated from scratch
    at every proposal; no generating morphology or fitted-state warm start is
    passed into it. The envelope gradient differentiates the original-image
    residual at the optimized map, including the configured inclination penalty.
    No posterior samples or confidence interval are produced. This optimizer need not find every
    mode, and its success flag is not a scientific validation.
    Noiseless numerical checks omit prior penalties and minimize raw SSE only.
    ``projected_gradient_max`` and ``gradient_converged`` distinguish a small
    gradient from termination merely because successive scores barely change.

    ``callback``, when supplied, receives a dictionary for each new best evaluated
    point, including maps, parameters, score and evaluation count. It may save a
    checkpoint; the fitter itself has no output-directory side effects.
    """
    from scipy.optimize import minimize
    from tqdm.auto import tqdm

    _check_experiment_inputs(experiment)
    if isinstance(max_evaluations, bool) or int(max_evaluations) != max_evaluations or max_evaluations < 1:
        raise ValueError("max_evaluations must be a positive integer.")
    point = np.array([start_velocity, start_turnover, np.deg2rad(start_pa), *start_offset], dtype=float)
    if len(experiment.param_names) == 6:
        angle = experiment.inclination_mean if start_inclination is None else start_inclination
        point = np.append(point, np.deg2rad(angle))
    elif start_inclination is not None:
        raise ValueError("This injection model fixes inclination; omit start_inclination.")
    fixed_rdy = getattr(experiment, "fixed_rdy", None)
    if fixed_rdy is not None:
        point[4] = fixed_rdy
    if point.shape != (len(experiment.param_names),) or not np.isfinite(experiment.log_prior(point)):
        raise ValueError("Starting parameters must lie inside the experiment bounds.")
    _observations(experiment, noisy)  # Validate noise and available pixels once up front.
    if callback is not None and not callable(callback):
        raise ValueError("callback must be callable or None.")
    scale = np.asarray(getattr(experiment, "parameter_scales", [200., 5., 1., 1., 1.]))
    bounds = getattr(experiment, "theta_bounds", ([0., .5, 0., -10., -10.],
                                                [1000., 40., 2 * np.pi, 10., 10.]))
    lower, upper = (np.asarray(bound, dtype=float) / scale for bound in bounds)
    if fixed_rdy is not None:
        # Keep the saved physical vector in its established six-coordinate
        # format, but remove this freedom from the optimizer with equal bounds.
        lower[4] = upper[4] = fixed_rdy / scale[4]
    best = [np.inf, None]
    calls = 0
    started = time.monotonic()

    class EvaluationLimit(Exception):
        pass

    with tqdm(total=int(max_evaluations), desc="Free-map fit evaluations", disable=not progress) as bar:
        def objective(scaled):
            nonlocal calls
            if calls >= max_evaluations:
                raise EvaluationLimit
            parameters = np.asarray(scaled) * scale
            result = _profile_map(experiment, parameters, noisy=noisy)
            calls += 1
            bar.update()
            if not result["success"]:
                raise RuntimeError("The inner source-map solver failed: " + result["message"])
            score = .5 * (result["chi2"] if noisy else result["sse"]) + result["prior_penalty"]
            # At the nuisance optimum, the envelope theorem removes d(map)/d(theta).
            # Holding this fitted map fixed is for this gradient only; the next
            # objective evaluation always refits all map pixels.
            theta = torch.tensor(parameters, dtype=torch.float32,
                                 device=experiment.device, requires_grad=True)
            value = .5 * experiment.chi2(theta, noisy=noisy, emission=result["emission"])
            if noisy:
                value = value + _prior_penalty(experiment, theta)
            gradient = torch.autograd.grad(value, theta)[0].detach().cpu().numpy() * scale
            if score < best[0]:
                projected = gradient.copy()
                scaled = np.asarray(scaled)
                projected[(scaled <= lower + 1e-10) & (gradient > 0)] = 0.
                projected[(scaled >= upper - 1e-10) & (gradient < 0)] = 0.
                result["projected_gradient_max"] = float(np.max(np.abs(projected)))
                result["gradient_converged"] = bool(result["projected_gradient_max"] <= 1e-5)
                best[:] = [score, result]
                bar.set_postfix(best_score=f"{score:.3f}")
                if callback is not None:
                    callback(dict(result, nfev=calls, start_theta=point.copy(),
                                  elapsed_seconds=time.monotonic() - started,
                                  profile_score=score, inner_success=bool(result["success"])))
            return score, gradient

        try:
            optimized = minimize(objective, point / scale, method="L-BFGS-B", jac=True,
                                 bounds=list(zip(lower, upper)),
                                 options={"maxiter": int(max_evaluations), "maxls": 25,
                                          "ftol": 1e-10, "gtol": 1e-5})
            success, message = bool(optimized.success), str(optimized.message)
        except EvaluationLimit:
            success, message = False, "Maximum profile evaluations reached; returning best evaluated point."
    result = dict(best[1])
    has_prior = (noisy and hasattr(experiment, "prior_penalty")
                 and getattr(experiment, "sample_inclination", True))
    result.update(success=success, inner_success=bool(best[1]["success"]),
                  message=message, nfev=calls, start_theta=point,
                  elapsed_seconds=time.monotonic() - started,
                  objective=("profile likelihood plus stated kinematic prior" if has_prior
                             else "profile likelihood; no kinematic prior"), profile_score=best[0])
    return result
