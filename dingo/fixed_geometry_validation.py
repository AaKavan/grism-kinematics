"""Injection checks for one saved fixed-geometry (simple_mcmc) run.

A saved real-source run defines everything: its maximum-posterior draw is the
injected truth, its rectified R/C images supply the emission morphology, and
its calibration, noise, fixed geometry, priors, starts and sampler budget are
reused unchanged for recovery. The question each injection answers is narrow:
does this exact configuration return a disk like the one it reported, when
the truth is known? Mocks use the fitter's own pixel deposition with no PSF
or intrinsic line width, so passing is necessary, not sufficient, for real data.

Real inputs and the reference run are only read. Each injection lives in its
own folder: inputs/ID<source>/ holds the mock FITS files, fits/ the recovery.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil

import numpy as np
import yaml

from . import simple_mcmc as sm

SCHEMA = "dingo-fixed-geometry-injection-v1"
REPO = Path(__file__).resolve().parents[1]
NAMES = ("velocity.V_rot", "velocity.R_v", "velocity.theta_v", "velocity.inc_v",
         "image.R.dx", "image.R.dy")
LABELS = {"velocity.V_rot": ("V_inf", "km/s", 1.), "velocity.R_v": ("R_t", "pixel", 1.),
          "velocity.theta_v": ("PA", "deg", 180/np.pi), "velocity.inc_v": ("i", "deg", 180/np.pi),
          "image.R.dx": ("R_dx", "pixel", 1.), "image.R.dy": ("R_dy", "pixel", 1.)}


@dataclass(frozen=True)
class Injection:
    """One saved mock: pass ``data_dir`` and ``source_id`` to simple_mcmc."""
    directory: Path
    data_dir: Path
    source_id: int
    truth: dict
    metadata: dict


def _info(run_dir):
    info = json.loads((Path(run_dir) / "run_info.txt").read_text())
    if info.get("schema") != sm.SCHEMA or "displayed_draw" not in info:
        raise ValueError("Choose a finished simple_mcmc run with saved samples.")
    return info


def load_injection(directory):
    directory = Path(directory).expanduser().resolve()
    record = json.loads((directory / "truth.json").read_text())
    if record.get("schema") != SCHEMA:
        raise ValueError("This folder is not a fixed-geometry injection.")
    return Injection(directory, directory / "inputs", record["source_id"], record["truth"], record)


def make_injection(run_dir, directory, *, seed=1, truth=None, threshold=2., device="cpu",
                   overwrite=False):
    """Save one noisy mock of the reference run's source, plus its noiseless twin.

    Parameters
    ----------
    run_dir : path
        Finished real-source run. Its maximum-posterior draw (``displayed_draw``)
        is the default truth; fixed center values are part of that draw.
    directory : path
        New experiment folder, kept outside the real data directory.
    seed : int
        Noise realization. Geometry and morphology do not depend on it.
    truth : dict or None
        Dotted-name overrides of the default truth, e.g. {"velocity.V_rot": 300.}
        for a disk the real run did not report. Angles are radians. The center
        cannot be overridden: recovery reuses the run's fixed center.
    device : "cpu" or "cuda"
        Used only to rebuild the reference fitter and draw the mock.

    threshold : float
        Emission pixels at or below this many background standard deviations
        of the rectified map are set to zero.

    The emission map is the mean of the R and C images rectified at the
    reference draw. It carries the real source's shape and flux scale (and
    some of its noise), not an idealized profile. Noise is independent
    Gaussian with the run's own sigma_R and sigma_C.
    """
    info = _info(run_dir)
    reference = dict(info["displayed_draw"])
    injected = dict(reference)
    for name, value in (truth or {}).items():
        if name not in NAMES:
            raise ValueError(f"Only {', '.join(NAMES)} can be overridden.")
        injected[name] = float(value)
    directory = Path(directory).expanduser().resolve()
    real_data = REPO / "data"
    if directory == real_data or real_data in directory.parents:
        raise ValueError("Keep injections outside the real data directory.")
    if directory.exists():
        try:
            marked = json.loads((directory / "truth.json").read_text()).get("schema") == SCHEMA
        except (OSError, ValueError):
            marked = False
        if not (overwrite and marked):
            raise FileExistsError(f"{directory} exists. Use another folder, or overwrite=True for an injection folder.")

    run = sm.open_run(run_dir, device=device)
    fitter = run.fitter
    point = np.array([reference[name] for name in fitter.mcmc_param_names])
    if not np.isfinite(fitter.log_probability(point)):
        raise ValueError("The reference draw is no longer evaluable.")
    from astropy.stats import sigma_clipped_stats
    rect_r, rect_c, _, _ = fitter.get_fitting_results()
    mean = 0.5 * (rect_r + rect_c)
    # Clipping noise at zero alone would leave a positive glow over the whole
    # canvas. Keep only pixels clearly above this map's own background noise.
    _, _, background = sigma_clipped_stats(mean)
    emission = np.where(mean > threshold * background, mean, 0.).astype(np.float32)
    sigma = (float(info["noise"]["sigma_R"]), float(info["noise"]["sigma_C"]))
    full_truth = {**injected, "image.C.dx": 0., "image.C.dy": 0.}
    clean = fitter.make_mock_data(emission, theta=full_truth, sigma_R=0., sigma_C=0.,
                                  seed=seed, inplace=False)
    noisy = fitter.make_mock_data(emission, theta=full_truth, sigma_R=sigma[0], sigma_C=sigma[1],
                                  seed=seed, inplace=False)
    clean, noisy = ([a.detach().cpu().numpy() for a in pair] for pair in (clean, noisy))

    from astropy.io import fits
    if directory.exists():
        shutil.rmtree(directory)
    source_id = int(info["source_id"])
    source = directory / "inputs" / f"ID{source_id}"
    source.mkdir(parents=True)
    shutil.copy2(info["inputs"]["image.fits"]["path"], source / "image.fits")
    shutil.copy2(info["inputs"]["config_kinematics.yaml"]["path"], source / "config_kinematics.yaml")
    for pupil, array in zip(("R", "C"), noisy):
        fits.writeto(source / f"{pupil}.fits", array.astype(np.float32))
    np.savez_compressed(directory / "truth.npz", emission=emission,
                        noiseless_R=clean[0], noiseless_C=clean[1])
    record = dict(schema=SCHEMA, source_id=source_id, reference_run=str(Path(run_dir).resolve()),
                  created_utc=datetime.now(timezone.utc).isoformat(), seed=int(seed),
                  truth=injected, overrides=dict(truth or {}), noise_sigma=dict(R=sigma[0], C=sigma[1]),
                  morphology=f"mean rectified R/C of the real source at the reference draw, "
                             f"zeroed at or below {threshold:g} background sigma",
                  note="Same calibration and pixel deposition as the fitter; no PSF or intrinsic dispersion.")
    (directory / "truth.json").write_text(json.dumps(record, indent=2) + "\n")
    return load_injection(directory)


def show_injection(injection):
    """Real R/C beside mock R/C on one brightness scale, plus the emission map."""
    from astropy.io import fits
    import matplotlib.pyplot as plt
    from ._simple_mcmc_plots import _brightness_norm

    record = injection.metadata
    real_dir = Path(_info(record["reference_run"])["inputs"]["R.fits"]["path"]).parent
    source = injection.data_dir / f"ID{injection.source_id}"
    images = {f"Real {p}": fits.getdata(real_dir / f"{p}.fits") for p in ("R", "C")}
    images.update({f"Mock {p}": fits.getdata(source / f"{p}.fits") for p in ("R", "C")})
    with np.load(injection.directory / "truth.npz") as saved:
        emission = saved["emission"]
    norm = _brightness_norm(*images.values())
    fig, axes = plt.subplots(1, 5, figsize=(19, 4), layout="constrained")
    for ax, (title, pixels) in zip(axes, [*images.items(), ("Injected emission", emission)]):
        ax.imshow(pixels, origin="lower", cmap="afmhot", interpolation="none",
                  norm=norm if title != "Injected emission" else _brightness_norm(emission))
        ax.set(title=title, xlabel="x [pixel]", ylabel="y [pixel]")
    t = injection.truth
    fig.suptitle(f"ID{injection.source_id} injection, seed {record['seed']}: "
                 f"V_inf={t['velocity.V_rot']:.0f} km/s, R_t={t['velocity.R_v']:.2f} px, "
                 f"i={np.rad2deg(t['velocity.inc_v']):.1f} deg, PA={np.rad2deg(t['velocity.theta_v']) % 360:.1f} deg")
    return fig


def _settings(info):
    """prepare() arguments that reproduce a saved run's target and start."""
    start, fixed, priors = info["start_parameters"], info["fixed_parameters"], info["priors"]
    inc = priors.get("velocity.inc_v", {})
    center = {}
    for axis in ("x", "y"):
        prior = priors.get(f"velocity.{axis}0_v")
        center[axis] = prior and (prior["mu"], prior["sigma"])
    return dict(
        start_velocity=start["velocity.V_rot"], start_turnover=start["velocity.R_v"],
        start_pa=np.rad2deg(start["velocity.theta_v"]), start_inclination=np.rad2deg(start["velocity.inc_v"]),
        start_center=(start["velocity.x0_v"], start["velocity.y0_v"]),
        start_offset=(start["image.R.dx"], start["image.R.dy"]),
        inclination_mean=np.rad2deg(inc["mu"]) if "mu" in inc else 60.,
        inclination_sigma=np.rad2deg(inc["sigma"]) if "sigma" in inc else 5.,
        center_mean=(center["x"][0], center["y"][0]) if center["x"] and center["y"] else None,
        center_sigma=(center["x"][1], center["y"][1]) if center["x"] and center["y"] else 5.,
        fixed_x0=fixed.get("velocity.x0_v"), fixed_y0=fixed.get("velocity.y0_v"),
        fixed_inclination=np.rad2deg(fixed["velocity.inc_v"]) if "velocity.inc_v" in fixed else None,
        max_velocity=priors["velocity.V_rot"]["max"],
        noise_correlation_area=info["noise"]["correlation_area"])


def recover(injection, *, run_name="recovery", device="cuda", max_evaluations=None,
            warmup=None, draws=None, walkers=None, progress="notebook", overwrite=False):
    """Refit a mock exactly as the reference run fitted the real source.

    Starts, priors, fixed geometry and noise estimation are copied from the
    reference run; nothing is taken from the injected truth. Budgets default
    to the reference run's own (local evaluations, warmup, draws, walkers).
    Smaller budgets are fine for a smoke test but are not the same check.
    Returns the result folder for compare().
    """
    info = _info(injection.metadata["reference_run"])
    sampling = info["sampling"]
    run = sm.prepare(injection.source_id, data_dir=injection.data_dir,
                     output_root=injection.directory / "fits", run_name=run_name,
                     device=device, overwrite=overwrite, **_settings(info))
    sm.fit_local(run, max_evaluations=max_evaluations or info["local_optimizer"]["evaluations"],
                 progress=bool(progress))
    return sm.sample(run, warmup=warmup or sampling["warmup"], draws=draws or sampling["draws"],
                     walkers=walkers or sampling["walkers"], seed=sampling["seed"], progress=progress)


def _derived(values):
    """Quantities the real run reports, computed draw by draw."""
    v, rt, inc = values["velocity.V_rot"], values["velocity.R_v"], values["velocity.inc_v"]
    curve = lambda r: 2 / np.pi * v * np.arctan(r / rt)
    return {"V_inf sin i [km/s]": v * np.sin(inc), "V(3 px) [km/s]": curve(3.),
            "V(5 px) [km/s]": curve(5.), "V(10 px) [km/s]": curve(10.)}


def compare(injection, result_dir):
    """Truth against pooled posterior quantiles, saved as truth_comparison.csv.

    ``z`` is (median - truth) / half the 16-84% width. Across several seeds a
    calibrated setup keeps |z| < 1 about 68% of the time; one seed cannot show it.
    """
    import pandas as pd

    result_dir = Path(result_dir)
    info = json.loads((result_dir / "run_info.txt").read_text())
    if info["inputs"]["R.fits"]["path"] != str(injection.data_dir / f"ID{injection.source_id}" / "R.fits"):
        raise ValueError("This result was not fitted to this injection.")
    with np.load(result_dir / "samples.npz") as saved:
        chain, names = saved["chain"], saved["param_names"].astype(str).tolist()
    flat = chain.reshape(-1, len(names))
    draws = {name: flat[:, k] for k, name in enumerate(names)}
    truth = {name: np.array([injection.truth[name]]) for name in names}
    if "velocity.theta_v" in draws:  # Compare PA on the circle, around the truth.
        t = injection.truth["velocity.theta_v"]
        draws["velocity.theta_v"] = t + np.angle(np.exp(1j * (draws["velocity.theta_v"] - t)))
    rows = {}
    for name in names:
        label, unit, scale = LABELS[name]
        rows[f"{label} [{unit}]"] = (truth[name][0] * scale, draws[name] * scale)
    fixed = {**info.get("fixed_parameters", {})}
    full_draws = {**{n: np.full(len(flat), v) for n, v in fixed.items()}, **draws}
    full_truth = {**{n: np.array([v]) for n, v in fixed.items()}, **truth}
    if "velocity.inc_v" not in full_truth:
        full_truth["velocity.inc_v"] = np.array([injection.truth["velocity.inc_v"]])
    for (label, d), t in zip(_derived(full_draws).items(), _derived(full_truth).values()):
        rows[label] = (float(t[0]), d)
    table = []
    for label, (t, d) in rows.items():
        p16, p50, p84, p2, p97 = np.quantile(d, [.16, .5, .84, .025, .975])
        half = max((p84 - p16) / 2, 1e-12)
        table.append(dict(quantity=label, truth=t, median=p50, minus_16th=p50 - p16,
                          plus_84th=p84 - p50, z=(p50 - t) / half,
                          truth_in_68=bool(p16 <= t <= p84), truth_in_95=bool(p2 <= t <= p97)))
    table = pd.DataFrame(table).set_index("quantity")
    table.to_csv(result_dir / "truth_comparison.csv")
    return table


def summarize(pairs):
    """Stack compare() tables from several (injection, result_dir) pairs.

    Returns one row per quantity: realizations, mean bias, mean |z|, and the
    fraction of truths inside the 68% and 95% intervals.
    """
    import pandas as pd
    tables = [compare(injection, result) for injection, result in pairs]
    stacked = pd.concat(tables, keys=range(len(tables)), names=["realization"])
    stacked["bias"] = stacked["median"] - stacked["truth"]
    grouped = stacked.groupby(level="quantity", sort=False)
    return pd.DataFrame({"realizations": grouped.size(), "truth": grouped["truth"].mean(),
                         "mean_bias": grouped["bias"].mean(), "mean_abs_z": grouped["z"].apply(lambda z: z.abs().mean()),
                         "in_68": grouped["truth_in_68"].mean(), "in_95": grouped["truth_in_95"].mean()})


def scan_velocity(injection, velocities, *, device="cpu"):
    """R-C squared residual of the NOISELESS mock versus V_inf, all else at truth.

    The cheapest check: a working setup has its minimum at the injected V_inf.
    Uses the reference run's fixed geometry; no optimizer or sampler runs.
    """
    import tempfile
    import torch
    import pandas as pd

    info = _info(injection.metadata["reference_run"])
    settings = {**_settings(info), "start_velocity": injection.truth["velocity.V_rot"],
                "noise_sigma": 1.}
    with np.load(injection.directory / "truth.npz") as saved:
        clean = saved["noiseless_R"], saved["noiseless_C"]
    with tempfile.TemporaryDirectory(prefix="dingo_scan_") as scratch:
        run = sm.prepare(injection.source_id, data_dir=injection.data_dir, output_root=scratch,
                         run_name="scan", device=device, **settings)
        fitter = run.fitter
        fitter.true_grism_R, fitter.true_grism_C = (
            torch.as_tensor(a, dtype=torch.float32, device=fitter.device) for a in clean)
        point = np.array([injection.truth[name] for name in fitter.mcmc_param_names])
        index = fitter.mcmc_param_names.index("velocity.V_rot")
        scores = []
        for velocity in velocities:
            point[index] = velocity
            fitter.log_probability(point)
            r, c, _, _ = fitter.get_fitting_results()
            scores.append(float(np.sum((r.astype(float) - c.astype(float))**2)))
    table = pd.DataFrame({"V_inf": np.asarray(velocities, float), "noiseless_sse": scores})
    table.to_csv(injection.directory / "velocity_scan.csv", index=False)
    return table
