"""Small, explicit recovery checks for the synthetic three-file notebook.

Generation lives in simple_simulation; fitting still uses simple_mcmc.
These helpers only supply known noise, draw comparisons, and save diagnostics.
"""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import json
import tempfile

import numpy as np

from . import simple_mcmc as sm
from .simple_simulation import load_mock


def _check_run_inputs(mock, info):
    """Reject comparisons between a run and another simulation's truth."""
    inputs = info.get("inputs", {})
    for name, expected in mock.metadata["input_sha256"].items():
        if inputs.get(name, {}).get("sha256") != expected:
            raise ValueError("This run was fitted to a different simulation. Choose its matching truth.")


def _include_corner_truths(fig, truths):
    """Keep injected values visible even when the posterior misses them.

    All panels for a parameter use the same expanded range. A placeholder
    figure from an insufficiently sampled run has no corner axes to adjust.
    """
    count = len(truths)
    if len(fig.axes) != count**2:
        return False
    axes = np.asarray(fig.axes).reshape(count, count)
    ranges = []
    for index, truth in enumerate(truths):
        lower, upper = axes[index, index].get_xlim()
        if truth <= lower or truth >= upper:
            lower, upper = min(lower, truth), max(upper, truth)
            padding = .04 * max(upper - lower, 1.)
            lower, upper = lower - padding, upper + padding
        ranges.append((lower, upper))
    for row in range(count):
        for column in range(row + 1):
            axes[row, column].set_xlim(ranges[column])
            if column < row:
                axes[row, column].set_ylim(ranges[row])
    return True


def show_inputs(mock):
    """Show the clean emission, injected velocity field, and noisy R/C inputs."""
    from astropy.io import fits
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from ._simple_mcmc_plots import _brightness_norm

    mock = load_mock(mock.directory)
    source = mock.data_dir / f"ID{mock.source_id}"
    with np.load(mock.directory / "truth.npz", allow_pickle=False) as saved:
        emission = saved["image"]
        velocity = saved["velocity_los"]
    r, c = (fits.getdata(source / f"{p}.fits") for p in ("R", "C"))
    vmax = max(float(np.max(np.abs(velocity))), 1.)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), layout="constrained")
    for ax, pixels, title, cmap, norm in zip(
            axes, (emission, velocity, r, c),
            ("True line emission", "True line-of-sight velocity", "Simulated R", "Simulated C"),
            ("afmhot", "RdBu_r", "afmhot", "afmhot"),
            (_brightness_norm(emission), Normalize(-vmax, vmax),
             _brightness_norm(r, c), _brightness_norm(r, c))):
        im = ax.imshow(pixels, origin="lower", interpolation="none", cmap=cmap, norm=norm)
        ax.set(title=title, xlabel="x [pixel]", ylabel="y [pixel]")
        if title == "True line-of-sight velocity":
            fig.colorbar(im, ax=ax, shrink=.8, label="km/s")
    fig.suptitle(f"Synthetic disk: V_inf={mock.truth['velocity.V_rot']:g} km/s; "
                 f"noise per input pixel={mock.metadata['noise_sigma']:g}; no PSF or intrinsic dispersion")
    fig.savefig(mock.directory / "simulation.png", dpi=150, bbox_inches="tight")
    return fig


def prepare_recovery(mock, *, start_velocity=350., start_turnover=3., start_pa=20.,
                     run_name=None, device="cpu", overwrite=False):
    """Prepare a fresh fit, fixing only the known center and inclination.

    V_inf, turnover radius, orientation and both R registration offsets remain
    free. Starting values are supplied independently of the injected truth.
    Input noise is known, but resampling covariance is still approximated by
    the production likelihood. A noiseless mock supports the SSE scan only;
    its formal posterior uncertainty would have no measurement-noise meaning.
    """
    mock = load_mock(mock.directory)
    sigma = float(mock.metadata["noise_sigma"])
    if sigma <= 0:
        raise ValueError("Use the velocity scan for noise_sigma=0. Add positive noise before MCMC recovery.")
    truth = mock.truth
    return sm.prepare(
        mock.source_id, data_dir=mock.data_dir, output_root=mock.directory / "fits",
        run_name=run_name or f"start_{start_velocity:g}", overwrite=overwrite, device=device,
        start_velocity=start_velocity, start_turnover=start_turnover, start_pa=start_pa,
        start_offset=(.2, -.2), fixed_x0=truth["velocity.x0_v"],
        fixed_y0=truth["velocity.y0_v"],
        fixed_inclination=np.rad2deg(truth["velocity.inc_v"]), noise_sigma=sigma)


def scan_velocity(mock, velocities, *, device="cpu"):
    """Compare noiseless/noisy SSE while all other parameters equal truth.

    This is a conditional numerical check, not a nuisance-profiled likelihood
    or an uncertainty interval. The temporary fitter uses an arbitrary positive
    variance, but the plotted quantity is the unscaled sum of squared residuals.
    No optimizer or sampler runs. Each curve is shifted by its own grid minimum.
    """
    from astropy.io import fits
    import matplotlib.pyplot as plt
    import pandas as pd
    import torch
    from tqdm.auto import tqdm

    mock = load_mock(mock.directory)
    velocities = np.asarray(velocities, dtype=float)
    if (velocities.ndim != 1 or len(velocities) < 2 or not np.isfinite(velocities).all()
            or np.any(velocities < 0) or np.any(velocities > 1000)
            or np.any(np.diff(velocities) <= 0)):
        raise ValueError("velocities must be an increasing 1D grid between 0 and 1000 km/s.")
    truth = mock.truth
    source = mock.data_dir / f"ID{mock.source_id}"
    with np.load(mock.directory / "truth.npz", allow_pickle=False) as saved:
        inputs = {"Noiseless": (saved["noiseless_R"], saved["noiseless_C"])}
    if mock.metadata["noise_sigma"] > 0:
        inputs["With noise"] = tuple(fits.getdata(source / f"{p}.fits") for p in ("R", "C"))
    table = pd.DataFrame({"V_inf": velocities})
    with tempfile.TemporaryDirectory(prefix="dingo_mock_scan_") as scratch:
        # Internal preparation paths are temporary; the actual diagnostic is
        # saved beside truth.json below. Keep the notebook output unambiguous.
        with redirect_stdout(StringIO()):
            run = sm.prepare(
                mock.source_id, data_dir=mock.data_dir, output_root=scratch,
                run_name="scan", device=device, noise_sigma=1.,
                start_velocity=truth["velocity.V_rot"], start_turnover=truth["velocity.R_v"],
                start_pa=np.rad2deg(truth["velocity.theta_v"]), start_offset=(0., 0.),
                fixed_x0=truth["velocity.x0_v"], fixed_y0=truth["velocity.y0_v"],
                fixed_inclination=np.rad2deg(truth["velocity.inc_v"]))
        fitter = run.fitter
        point = np.array([truth[name] for name in fitter.mcmc_param_names])
        vindex = fitter.mcmc_param_names.index("velocity.V_rot")
        for label, pair in inputs.items():
            fitter.true_grism_R, fitter.true_grism_C = (
                torch.as_tensor(np.array(a, dtype=np.float32), device=fitter.device) for a in pair)
            scores = []
            for velocity in tqdm(velocities, desc=f"{label} velocity check"):
                point[vindex] = velocity
                score = fitter.log_probability(point)
                if not np.isfinite(score):
                    scores.append(np.nan)
                    continue
                r, c, _, _ = fitter.get_fitting_results()
                scores.append(float(np.sum((r.astype(float) - c.astype(float))**2)))
            table[label] = scores
    fig, axes = plt.subplots(1, len(inputs), figsize=(6 * len(inputs), 4),
                             layout="constrained", squeeze=False)
    for ax, label in zip(axes.flat, inputs):
        scores = table[label].to_numpy()
        if not np.isfinite(scores).any():
            raise ValueError(f"No finite trial velocities for the {label} check.")
        best = int(np.nanargmin(scores))
        ax.plot(velocities, scores - scores[best], color="black")
        ax.axvline(truth["velocity.V_rot"], color="tab:orange", ls="--", label="Injected truth")
        ax.plot(velocities[best], 0, "o", color="tab:blue", label=f"Grid minimum: {velocities[best]:g}")
        ax.set(title=label, xlabel="Trial V_inf [km/s]", ylabel="Extra squared R−C residual")
        ax.legend(fontsize=9)
    fig.suptitle("Only velocity varies; all other parameters fixed to their true values")
    table.to_csv(mock.directory / "velocity_scan.csv", index=False)
    fig.savefig(mock.directory / "velocity_scan.png", dpi=150, bbox_inches="tight")
    return table, fig


def compare_starts(mock, runs):
    """Tabulate the actual final vectors from independent local optimizations."""
    import pandas as pd

    mock = load_mock(mock.directory)
    rows = []
    for start, run in runs.items():
        _check_run_inputs(mock, run.info)
        p = dict(zip(run.fitter.mcmc_param_names, run.theta))
        rows.append({"start_V_inf": start, "fitted_V_inf": p["velocity.V_rot"],
                     "true_V_inf": mock.truth["velocity.V_rot"], "fitted_R_t": p["velocity.R_v"],
                     "true_R_t": mock.truth["velocity.R_v"],
                     "fitted_PA_deg": np.rad2deg(p["velocity.theta_v"]),
                     "log_posterior": run.info["local_optimizer"]["log_posterior_after"],
                     "optimizer_finished": run.info["local_optimizer"]["success"]})
    table = pd.DataFrame(rows)
    table.to_csv(mock.directory / "starting_values.csv", index=False)
    return table


def compare_posterior(mock, result_directory):
    """Add injected truths to ordinary saved plots and a rotation-curve panel.

    Quantiles use all retained draws, without deleting walkers or selecting
    the solution closest to truth. Errors remain conditional on fixed geometry
    and the existing approximate noise treatment; this is not a coverage test.
    """
    import corner
    import matplotlib.pyplot as plt
    from ._simple_mcmc_plots import _PARAMETERS

    mock = load_mock(mock.directory)
    directory = Path(result_directory)
    info = json.loads((directory / "run_info.txt").read_text())
    _check_run_inputs(mock, info)
    views = sm.show_results(directory, save=False)
    with np.load(directory / "samples.npz", allow_pickle=False) as saved:
        chain = saved["chain"]
        names = saved["param_names"].astype(str).tolist()
    flat = chain.reshape(-1, len(names))
    truth_values = [mock.truth[name] * _PARAMETERS[name][3] for name in names]
    if "velocity.theta_v" in names:
        index = names.index("velocity.theta_v")
        reference = np.rad2deg(np.angle(np.mean(np.exp(1j * flat[:, index])))) % 360
        truth_values[index] = reference + (truth_values[index] - reference + 180) % 360 - 180
    views["summary"].insert(0, "truth", truth_values)
    # A biased recovery can put truth outside every posterior-only axis range.
    if _include_corner_truths(views["corner"], truth_values):
        corner.overplot_lines(views["corner"], truth_values, color="tab:orange", ls="--")
    for ax, truth in zip(views["traces"].axes, truth_values):
        ax.axhline(truth, color="tab:orange", ls="--", linewidth=1)
    for name in ("corner", "traces"):
        fig = views[name]
        heading = fig._suptitle.get_text() if fig._suptitle else ""
        fig.suptitle(heading + "\nOrange dashed lines: injected truth", fontsize=10, y=1.04)

    radii = np.linspace(0, 20, 121)
    v = flat[:, names.index("velocity.V_rot")]
    rt = flat[:, names.index("velocity.R_v")]
    # Evaluate one radius at a time to keep memory independent of chain length.
    intervals = np.array([np.quantile(2 / np.pi * v * np.arctan(r / rt), [.16, .5, .84])
                          for r in radii])
    true_curve = 2 / np.pi * mock.truth["velocity.V_rot"] * np.arctan(radii / mock.truth["velocity.R_v"])
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    ax.fill_between(radii, intervals[:, 0], intervals[:, 2], color="black", alpha=.15,
                    label="Recovered 16–84%")
    ax.plot(radii, intervals[:, 1], color="black", label="Recovered median")
    ax.plot(radii, true_curve, color="tab:orange", ls="--", label="Injected truth")
    fixed = info.get("fixed_parameters", {})
    geometry_names = ("velocity.x0_v", "velocity.y0_v", "velocity.inc_v")
    fixed_to_truth = all(name in fixed and np.isclose(fixed[name], mock.truth[name],
                                                     rtol=0., atol=1e-6)
                         for name in geometry_names)
    title = "Recovery at common radii"
    if fixed_to_truth:
        title += "; center and inclination fixed to truth"
    ax.set(xlabel="Disk radius [pixel]", ylabel="Deprojected rotation [km/s]", title=title)
    ax.legend()
    views["rotation_curve"] = fig
    for name, value in views.items():
        if hasattr(value, "savefig"):
            value.savefig(directory / f"{name}.png", dpi=150, bbox_inches="tight")
    views["corner"].savefig(directory / "corner.pdf", bbox_inches="tight")
    views["summary"].to_csv(directory / "truth_comparison.csv")
    print("Truth comparisons and plots saved in:", directory)
    return views
