"""Notebook-sized diagnostics for direct R+C injection recovery.

The forward calculation lives in injection_forward. This module keeps plotting,
checkpointing and known-emission-map MCMC out of the demonstration notebook.
All posterior intervals here condition on the supplied true emission map and
center. Inclination can be fixed or sampled with a Gaussian prior. A free-map
profile fit is a separate diagnostic.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import time

import numpy as np


LABELS = [r"$V_\infty$ [km/s]", r"$R_t$ [pixel]", "PA [deg]",
          r"$dx_R$ [pixel]", r"$dy_R$ [pixel]"]
SCALE = np.array([1., 1., 180 / np.pi, 1., 1.])
SCHEMA = "dingo-direct-injection-known-map-v1"


def _display_parameters(experiment):
    """Use degrees for both angles while keeping saved sampler draws in radians."""
    labels = list(LABELS)
    scale = SCALE.copy()
    names = ["V_inf [km/s]", "R_t [pixel]", "PA [deg]", "R_dx [pixel]", "R_dy [pixel]"]
    if experiment.sample_inclination:
        labels.append("i [deg]")
        scale = np.append(scale, 180 / np.pi)
        names.append("i [deg]")
    return labels, scale, names


def _conditions(experiment):
    """Describe what is supplied exactly rather than inferred from the grisms."""
    fixed = "true emission map, true center"
    return fixed if experiment.sample_inclination else fixed + ", true inclination"


def _arrays(prediction):
    """Move a predicted R/C pair to NumPy for inspection."""
    if isinstance(prediction, dict):
        prediction = [prediction[p] for p in ("R", "C")]
    return [a.detach().cpu().numpy() for a in prediction]


def _folder(experiment):
    from .simple_simulation import load_mock
    current = load_mock(experiment.mock.directory)
    if (current.metadata["input_sha256"] != experiment.mock.metadata["input_sha256"]
            or current.metadata["truth_npz_sha256"] != experiment.mock.metadata["truth_npz_sha256"]
            or current.truth != experiment.mock.truth
            or current.metadata["noise_sigma"] != experiment.mock.metadata["noise_sigma"]):
        raise ValueError("The simulation changed after preparation. Create a fresh ForwardExperiment.")
    directory = experiment.mock.directory / "direct_known_map"
    directory.mkdir(exist_ok=True)
    return directory


def scan_velocity(experiment, velocities):
    """Plot direct-image SSE with all parameters except velocity set to truth.

    The noiseless and noisy curves each subtract their own minimum. These are
    conditional score scans, not uncertainty intervals or automatic starts.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    from tqdm.auto import tqdm

    velocities = np.asarray(velocities, dtype=float)
    if (velocities.ndim != 1 or len(velocities) < 2 or
            not np.isfinite(velocities).all() or np.any(np.diff(velocities) <= 0)
            or np.any(velocities < 0) or np.any(velocities > 1000)):
        raise ValueError("Use an increasing velocity grid between 0 and 1000 km/s.")
    data = {"Noiseless": _arrays(experiment.clean_data)}
    if experiment.mock.metadata["noise_sigma"] > 0:
        data["With noise"] = _arrays(experiment.data)
    rows = []
    for velocity in tqdm(velocities, desc="Direct R+C velocity check"):
        theta = experiment.truth_theta.copy()
        theta[0] = velocity
        predicted = _arrays(experiment.predict(theta))
        rows.append({"V_inf": velocity, **{
            label: sum(float(np.sum((a.astype(float) - b)**2))
                       for a, b in zip(observed, predicted))
            for label, observed in data.items()}})
    table = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, len(data), figsize=(6 * len(data), 4),
                             layout="constrained", squeeze=False)
    for ax, label in zip(axes.flat, data):
        score = table[label].to_numpy()
        best = int(np.argmin(score))
        ax.plot(velocities, score - score[best], color="black")
        ax.axvline(experiment.truth_theta[0], color="tab:orange", ls="--", label="Injected truth")
        ax.plot(velocities[best], 0, "o", label=f"Minimum: {velocities[best]:g} km/s")
        ax.set(title=label, xlabel="Trial V_inf [km/s]", ylabel="Extra squared image residual")
        ax.legend()
    fig.suptitle("Direct R+C predictions; emission map and other parameters known")
    folder = _folder(experiment)
    table.to_csv(folder / "velocity_scan.csv", index=False)
    fig.savefig(folder / "velocity_scan.png", dpi=150)
    return table, fig


def compare_starts(experiment, fits):
    """Save actual fitted vectors from independent, explicit starting points."""
    import pandas as pd

    rows = []
    for fit in fits:
        theta = np.asarray(fit["theta"])
        rows.append(dict(start_V_inf=fit["start_theta"][0], fitted_V_inf=theta[0],
                         true_V_inf=experiment.truth_theta[0], fitted_R_t=theta[1],
                         fitted_PA_deg=np.rad2deg(theta[2]), R_dx=theta[3], R_dy=theta[4],
                         chi2=fit["chi2"], finished=fit["success"]))
        if experiment.sample_inclination:
            rows[-1]["start_i_deg"] = np.rad2deg(fit["start_theta"][5])
            rows[-1]["fitted_i_deg"] = np.rad2deg(theta[5])
            rows[-1]["true_i_deg"] = np.rad2deg(experiment.truth_theta[5])
    table = pd.DataFrame(rows)
    table.to_csv(_folder(experiment) / "starting_values.csv", index=False)
    return table


def _residual_figure(experiment, predicted, title, *, noisy=True):
    """Compare predictions with the unchanged detector inputs on shared scales."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from ._simple_mcmc_plots import _brightness_norm

    observed = _arrays(experiment.data if noisy else experiment.clean_data)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
    norm = _brightness_norm(*observed, *predicted)
    sigma = float(experiment.mock.metadata["noise_sigma"]) if noisy else 0.
    residual_label = "(input − prediction) / noise" if sigma > 0 else "input − prediction"
    differences = [(a-b) / (sigma if sigma > 0 else 1.) for a, b in zip(observed, predicted)]
    limit = 4. if sigma > 0 else max(np.max(np.abs(d)) for d in differences)
    limit = max(limit, 1e-8)
    for row, pupil in enumerate(("R", "C")):
        for col, pixels in enumerate((observed[row], predicted[row], differences[row])):
            im = axes[row, col].imshow(pixels, origin="lower", interpolation="none",
                     cmap="RdBu_r" if col == 2 else "afmhot",
                     norm=Normalize(-limit, limit) if col == 2 else norm)
            axes[row, col].set(title=f"{pupil}: " + ["input", "prediction", residual_label][col],
                               xlabel="x [pixel]", ylabel="y [pixel]")
            if col == 2:
                fig.colorbar(im, ax=axes[row, col], shrink=.8)
    fig.suptitle(title)
    return fig


def show_free_result(experiment, result):
    """Save a free-pixel-map profile fit without inventing posterior errors.

    The truth map is used only in the comparison figure, never to solve for
    emission. This stage fits arbitrary nonnegative source-pixel amplitudes.
    Center remains fixed to truth. If configured, inclination is fitted with
    the same Gaussian prior used in the known-map MCMC.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    from ._simple_mcmc_plots import _brightness_norm

    _folder(experiment)  # Validate simulation identity.
    directory = experiment.mock.directory / "direct_free_map"
    directory.mkdir(exist_ok=True)
    theta = np.asarray(result["theta"])
    _, scale, names = _display_parameters(experiment)
    summary = pd.DataFrame(dict(truth=experiment.truth_theta * scale, fitted=theta*scale),
                           index=names)
    predicted = [result["predicted_R"], result["predicted_C"]]
    residuals = _residual_figure(experiment, predicted,
                                f"Free emission map: V_inf={theta[0]:.1f} km/s; profile fit",
                                noisy=result["noisy"])
    emission, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    true_map = experiment.emission.detach().cpu().numpy()
    recovered = result["emission"]
    norm = _brightness_norm(true_map, recovered)
    for ax, pixels, title in zip(axes, (true_map, recovered),
                                 ("Injected emission (comparison only)", "Recovered free pixel map")):
        ax.imshow(pixels, origin="lower", cmap="afmhot", norm=norm, interpolation="none")
        ax.set(title=title, xlabel="x [pixel]", ylabel="y [pixel]")
    emission.suptitle("No Sérsic shape or smoothing prior in the free-map fit")
    np.savez_compressed(directory / "profile_fit.npz", theta=theta,
                        emission=recovered, predicted_R=predicted[0], predicted_C=predicted[1])
    info = dict(schema="dingo-direct-injection-free-map-profile-v1",
                finished_utc=datetime.now(timezone.utc).isoformat(),
                method="joint R+C nonnegative source pixels, profiled over morphology",
                morphology_prior="nonnegative pixels; no smoothing or Sérsic profile",
                posterior_errors_available=False,
                fixed_parameters={k: experiment.mock.truth[k] for k in
                    ("velocity.x0_v", "velocity.y0_v", "image.C.dx", "image.C.dy")},
                inclination_settings=experiment.inclination_settings,
                truth=experiment.mock.truth,
                input_sha256=experiment.mock.metadata["input_sha256"],
                noise_sigma=experiment.mock.metadata["noise_sigma"])
    if not experiment.sample_inclination:
        info["fixed_parameters"]["velocity.inc_v"] = experiment.mock.truth["velocity.inc_v"]
    for key, value in result.items():
        if key in ("emission", "predicted_R", "predicted_C"):
            continue
        info[key] = value.tolist() if isinstance(value, np.ndarray) else value
    (directory / "run_info.txt").write_text(json.dumps(info, indent=2) + "\n")
    summary.to_csv(directory / "truth_comparison.csv")
    residuals.savefig(directory / "residuals.png", dpi=150, bbox_inches="tight")
    emission.savefig(directory / "emission.png", dpi=150, bbox_inches="tight")
    print("Free-map profile fit; no marginalized MCMC errors are reported.")
    print("Outer optimizer:", result["success"], result["message"])
    print("Inner nonnegative solve:", result.get("inner_success", result["success"]),
          "optimality:", result["optimality"])
    if "gradient_converged" in result:
        print("Outer gradient criterion met:", result["gradient_converged"],
              "(a small change in the score can stop the optimizer before this criterion is met)")
    return dict(summary=summary, residuals=residuals, emission=emission)


def sample(experiment, fit, *, warmup=100, draws=500, walkers=32, seed=42,
           progress="notebook", overwrite=False):
    """Sample kinematics/registration with the true map and center fixed.

    This is a conditional numerical recovery test, not the real-source posterior.
    Inclination is the sixth sampled parameter when the experiment specifies a
    Gaussian prior; otherwise the original five-parameter test is preserved.
    Original R/C pixels remain fixed and have their known Gaussian input noise.
    Emcee evaluates the PyTorch forward model in batches on the chosen device.
    Checkpoints contain all retained draws and are written every 50 steps, on
    interruption, and on completion. No walkers are removed from summaries.
    """
    import emcee

    ndim = len(experiment.param_names)
    for name, value, lower in (("warmup", warmup, 0), ("draws", draws, 1),
                               ("walkers", walkers, 2 * ndim)):
        if isinstance(value, bool) or int(value) != value or value < lower:
            raise ValueError(f"{name} must be an integer >= {lower}.")
    if experiment.mock.metadata["noise_sigma"] <= 0:
        raise ValueError("Noiseless images support a score scan, not MCMC error bars.")
    directory = _folder(experiment)
    info_path = directory / "run_info.txt"
    if (directory / "samples.npz").exists() and not overwrite:
        raise FileExistsError("This experiment has saved samples. Set overwrite=True or use a new experiment.")
    if info_path.exists() and json.loads(info_path.read_text()).get("schema") != SCHEMA:
        raise ValueError("Refusing to overwrite an unrelated run record.")
    start = np.asarray(fit["theta"], dtype=float)
    if not np.isfinite(experiment.log_probability(start)):
        raise ValueError("The chosen starting fit has a nonfinite log posterior.")
    rng = np.random.default_rng(seed)
    scatter = np.array([2., .1, .01, .01, .01])
    if experiment.sample_inclination:
        scatter = np.append(scatter, np.deg2rad(min(.5, experiment.inclination_sigma / 10)))
    positions = []
    for _ in range(10000):
        candidate = start + rng.normal(size=ndim) * scatter
        if np.isfinite(experiment.log_probability(candidate)):
            positions.append(candidate)
        if len(positions) == walkers:
            break
    if len(positions) != walkers:
        raise ValueError("Could not initialize all walkers inside the stated bounds.")
    # Remove only posterior products owned by this stage. A fresh interrupted
    # attempt must not accidentally display samples from an older attempt.
    if overwrite:
        for name in ("samples.npz", "corner.png", "corner.pdf", "traces.png",
                     "residuals.png", "rotation_curve.png", "truth_comparison.csv"):
            (directory / name).unlink(missing_ok=True)
    info = dict(schema=SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
                status="warmup", device=str(experiment.device), warmup=int(warmup),
                draws=int(draws), walkers=int(walkers), seed=int(seed),
                model="direct R+C Gaussian likelihood; no data rectification",
                condition_on=_conditions(experiment),
                inclination_settings=experiment.inclination_settings,
                inclination_uncertainty_included=experiment.sample_inclination,
                morphology_uncertainty_included=False,
                priors={"V_inf": "uniform 0..1000 km/s", "R_t": "log-uniform 0.5..40 pixels",
                        "PA": "uniform 0..2pi radians", "R_offsets": "Gaussian 0 +/- 2 pixels, bounded +/-10"},
                truth=experiment.mock.truth, noise_sigma=experiment.mock.metadata["noise_sigma"],
                input_sha256=experiment.mock.metadata["input_sha256"],
                start_theta=start.tolist(), param_names=list(experiment.param_names))
    info["priors"]["inclination"] = experiment.inclination_settings
    info_path.write_text(json.dumps(info, indent=2) + "\n")
    sampler = emcee.EnsembleSampler(walkers, ndim, experiment.log_probability_batch, vectorize=True)
    sampler.random_state = np.random.RandomState(seed).get_state()
    started = time.monotonic()
    retaining = False

    def checkpoint():
        if sampler.iteration:
            path = directory / "samples.npz"
            temporary = directory / "samples.pending.npz"
            np.savez_compressed(temporary, chain=sampler.get_chain(),
                                log_prob=sampler.get_log_prob(),
                                param_names=np.array(experiment.param_names),
                                acceptance_fraction=sampler.acceptance_fraction)
            temporary.replace(path)
            info["retained_steps"] = sampler.iteration
        info["elapsed_seconds"] = time.monotonic() - started
        info_path.write_text(json.dumps(info, indent=2) + "\n")

    try:
        state = np.asarray(positions)
        if warmup:
            state = sampler.run_mcmc(state, warmup, progress=progress)
        sampler.reset()
        retaining = True
        info["status"] = "sampling"
        checkpoint()
        for _ in sampler.sample(state, iterations=draws, progress=progress):
            if sampler.iteration % 50 == 0:
                checkpoint()
        info["status"] = "finished"
        info["finished_utc"] = datetime.now(timezone.utc).isoformat()
        info["acceptance_mean"] = float(np.mean(sampler.acceptance_fraction))
        if sampler.iteration >= 20:
            tau = sampler.get_autocorr_time(tol=0)
            info["autocorrelation_steps"] = [float(x) if np.isfinite(x) else None for x in tau]
            info["steps_over_tau"] = [float(draws / x) if np.isfinite(x) and x > 0 else None for x in tau]
    except BaseException:
        info["status"] = "interrupted"
        # Warmup samples must never be written as retained posterior draws.
        if not retaining:
            sampler.reset()
        raise
    finally:
        checkpoint()
    print("Known-map conditional posterior saved in:", directory)
    return directory


def show_results(experiment, directory):
    """Show original-data residuals, all walkers, truth-marked corner and curve."""
    import corner
    import matplotlib.pyplot as plt
    import pandas as pd
    from .simulation_review import _include_corner_truths

    directory = Path(directory)
    _folder(experiment)  # Recheck saved simulation identity before making plots.
    info = json.loads((directory / "run_info.txt").read_text())
    if info.get("input_sha256") != experiment.mock.metadata["input_sha256"]:
        raise ValueError("Saved posterior and simulation have different inputs.")
    saved_inclination = info.get("inclination_settings")
    if saved_inclination is None:
        # Earlier five-dimensional files implicitly fixed inclination to truth.
        if experiment.sample_inclination:
            raise ValueError("Saved posterior fixed inclination. Recreate its fixed-inclination experiment or run the new prior setup.")
    elif saved_inclination != experiment.inclination_settings:
        raise ValueError("Saved posterior has different inclination settings. Recreate the experiment with the saved prior.")
    with np.load(directory / "samples.npz", allow_pickle=False) as saved:
        chain = saved["chain"]
        logp = saved["log_prob"]
        if "param_names" in saved and saved["param_names"].tolist() != list(experiment.param_names):
            raise ValueError("Saved posterior parameter names do not match this experiment.")
    ndim = len(experiment.param_names)
    if chain.ndim != 3 or chain.shape[-1] != ndim or logp.shape != chain.shape[:2]:
        raise ValueError("Saved posterior dimensions do not match this experiment.")
    if not len(chain):
        raise ValueError("No retained posterior samples are available yet.")
    labels, scale, names = _display_parameters(experiment)
    flat = chain.reshape(-1, ndim)
    values = flat * scale
    truth = experiment.truth_theta * scale
    # PA is periodic: draws at 359 and 1 degrees belong together, not at 180.
    reference = np.rad2deg(np.angle(np.mean(np.exp(1j * flat[:, 2])))) % 360
    values[:, 2] = reference + (values[:, 2] - reference + 180) % 360 - 180
    truth[2] = reference + (truth[2] - reference + 180) % 360 - 180
    displayed_chain = values.reshape(chain.shape)
    q = np.quantile(values, [.16, .5, .84], axis=0)
    summary = pd.DataFrame(dict(truth=truth, median=q[1], minus_16th=q[1]-q[0],
                                plus_84th=q[2]-q[1]), index=names)
    summary.to_csv(directory / "truth_comparison.csv")
    corner_fig = corner.corner(values, labels=labels, color="black", bins=30,
                               plot_datapoints=False, fill_contours=False,
                               levels=(.68, .95), quantiles=(.16, .5, .84),
                               show_titles=True, title_fmt=".2f")
    _include_corner_truths(corner_fig, truth)
    corner.overplot_lines(corner_fig, truth, color="tab:orange", ls="--")
    corner_fig.suptitle("Known-map conditional posterior; orange = injected truth", y=1.02)
    traces, axes = plt.subplots(3, 2, figsize=(11, 7), layout="constrained")
    for j, ax in enumerate(axes.flat):
        if j >= ndim:
            ax.set_visible(False)
            continue
        ax.plot(displayed_chain[:, :, j], color="black", alpha=.12, linewidth=.5)
        ax.axhline(truth[j], color="tab:orange", ls="--")
        ax.set(xlabel="Retained step", ylabel=labels[j])
    traces.suptitle("Every walker; fixed: " + _conditions(experiment) + "; PA unwrapped")

    best = flat[int(np.argmax(logp))]
    predicted = _arrays(experiment.predict(best))
    residuals = _residual_figure(experiment, predicted,
                                f"Direct predictions at maximum-posterior draw: V_inf={best[0]:.1f} km/s")

    radii = np.linspace(0, 20, 121)
    intervals = np.array([np.quantile(2/np.pi*flat[:, 0]*np.arctan(r/flat[:, 1]), [.16, .5, .84])
                          for r in radii])
    rotation, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    ax.fill_between(radii, intervals[:, 0], intervals[:, 2], color="black", alpha=.15, label="16–84%")
    ax.plot(radii, intervals[:, 1], color="black", label="Recovered median")
    ax.plot(radii, 2/np.pi*truth[0]*np.arctan(radii/truth[1]), color="tab:orange", ls="--", label="Injected")
    ax.set(xlabel="Disk radius [pixel]", ylabel="Rotation [km/s]", title="Known-map conditional recovery")
    ax.legend()
    views = dict(summary=summary, residuals=residuals, traces=traces, corner=corner_fig, rotation_curve=rotation)
    for name, fig in views.items():
        if hasattr(fig, "savefig"):
            fig.savefig(directory / f"{name}.png", dpi=150, bbox_inches="tight")
    corner_fig.savefig(directory / "corner.pdf", bbox_inches="tight")
    print("Intervals condition on the " + _conditions(experiment) + ".")
    if experiment.sample_inclination:
        print(f"Inclination sampled: Gaussian {experiment.inclination_mean:g} +/- "
              f"{experiment.inclination_sigma:g} deg, bounded 5..85 deg.")
    print("Status:", info["status"], "Acceptance:", info.get("acceptance_mean"))
    print("Retained steps / estimated autocorrelation time:", info.get("steps_over_tau"))
    return views
