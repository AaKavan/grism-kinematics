"""Native-pixel posterior predictive checks and illustrative map draws.

Called by native_posterior_check --predictive-only on a completed run. This
preserves chain identity if convergence failed; no model is selected using
the held-out exposures here.
"""

import json
import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

from dingo.exposures import load_native_target
from dingo.native_forward import NativeLineProjector
from dingo.native_fitting import BilinearLineMap, StripContinuumLikelihood
from dingo.native_jax import JaxNativeProjector, JaxLineMap, JaxStripLikelihood
from dingo.native_posterior import NativePosterior, PosteriorPrior

REPO = Path(__file__).resolve().parents[1]


def run(output, draws_per_chain=16):
    output = Path(output)
    report = json.loads((output/"posterior.json").read_text())
    if not report.get("completed"):
        raise ValueError("Finish sampling before the predictive audit")
    for name, digest in report["code_sha256"].items():
        if name.startswith("dingo/") and hashlib.sha256((REPO/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Predictive operator changed since sampling: {name}")
    for name, digest in report["input_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Predictive input changed since sampling: {name}")
    source_dir = Path(report["source_run"])
    source = json.loads((source_dir/"recovery.json").read_text())
    settings = source["arguments"]
    torch.set_num_threads(settings["threads"])
    target = load_native_target(settings["bundle"], REPO/f"data/ID{source['source_geometry']}/config_kinematics.yaml", settings["size"])
    paths = [str(e.path) for e in target.exposures]
    train = [paths.index(p) for p in report["train_exposures"]]
    held = [] if report["arguments"]["all_exposures"] else [paths.index(p) for p in source["held_exposures"]]
    pupils = [e.pupil for e in target.exposures]
    with np.load(source_dir/"injection.npz") as injected:
        data, error, valid = [injected[k] for k in ("science", "error", "valid")]
        true_line = injected["true_line"]
    with np.load(output/"samples.npz") as saved:
        samples = dict(saved)
    with np.load(output/"prior_samples.npz") as saved:
        prior_samples = dict(saved)
    grid = BilinearLineMap(source["best_by_setting"][0]["grid"], settings["quadrature"],
                           flux_unit=source["fixed_map_flux_unit"], dtype=torch.float64)
    yy, xx = np.mgrid[-4:5, -4:5]
    kernel = np.exp(-.5*(xx*xx+yy*yy)/settings["fit_blur"]**2)
    projector = NativeLineProjector.from_target(target, grid.east.numpy(), grid.north.numpy(), kernel,
        np.ones(len(paths)), spectral_samples=7, dtype=torch.float64, deposition=settings["deposition"])
    likelihood = StripContinuumLikelihood(data, error, valid, pupils, degree=settings["degree"])
    jlike = JaxStripLikelihood(likelihood)
    posterior = NativePosterior(JaxNativeProjector(projector), JaxLineMap(grid), jlike,
                                prior=PosteriorPrior(**report["prior"]))
    fields = ("v_ref", "r_turn", "theta", "inclination", "log_flux", "map_z", "map_sigma", "envelope_width")
    n_chains, n_draws = samples["v_ref"].shape
    chosen = np.linspace(0, n_draws-1, min(n_draws, draws_per_chain), dtype=int)
    predict = jax.jit(jax.vmap(posterior.predict))
    line_draws, map_draws = [], []
    for c in range(n_chains):
        arguments = [jnp.asarray(samples[k][c, chosen]) for k in fields]
        line_draws.append(np.asarray(predict(*arguments)))
        map_draws.append(np.asarray(posterior.controls(samples["log_flux"][c, chosen[0]], samples["map_z"][c, chosen[0]],
            samples["map_sigma"][c, chosen[0]], samples["envelope_width"][c, chosen[0]])))
    line_draws = np.stack(line_draws)
    residual_chi = jax.jit(jax.vmap(jax.vmap(lambda line: jnp.sum(
        (jlike.residual_data-jlike.residualize(line*jlike.inverse_error))**2, axis=(-2, -1)))))
    chi = np.asarray(residual_chi(jnp.asarray(line_draws)))
    truth_residual = jlike.residual_data-jlike.residualize(jnp.asarray(true_line)*jlike.inverse_error)
    truth_chi = np.asarray(jnp.sum(truth_residual**2, axis=(-2, -1)))
    truth_line_residual = jlike.residualize(jnp.asarray(true_line)*jlike.inverse_error)
    line_error_chi = np.asarray(jax.jit(jax.vmap(jax.vmap(lambda line: jnp.sum(
        (truth_line_residual-jlike.residualize(line*jlike.inverse_error))**2, axis=(-2, -1)))))(jnp.asarray(line_draws)))
    ranks = np.zeros(len(paths))
    for pupil in likelihood.groups:
        idx = getattr(likelihood, f"indices_{pupil}").numpy()
        q = getattr(likelihood, f"q_{pupil}").numpy()
        ranks[idx] = np.sum(q*q, axis=(1, 2, 3))
    remaining_rank = np.round(valid.sum((1, 2))-ranks).astype(int)
    audit = {"posterior_convergence_gate_passed": report["diagnostics"]["convergence_gate_passed"],
        "postprocessing_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "draw_indices": chosen.tolist(), "per_chain": [], "train_exposures": len(train), "held_exposures": len(held),
        "note": "Conditional Gaussian replicated residuals after fixed continuum projection; no detection p-value or coverage claim"}
    rng = np.random.default_rng(report["arguments"]["seed"]+901)
    for c in range(n_chains):
        row = {"chain": c}
        for name, indices in (("train", train), ("held", held)):
            if not indices:
                row[name] = None
                continue
            values = chi[c][:, indices].sum(1)
            rank = int(remaining_rank[indices].sum())
            replicated = rng.chisquare(rank, len(values))
            row[name] = {"chi_square_q05_q50_q95": np.quantile(values, [.05, .5, .95]).tolist(),
                         "injected_line_chi_square": float(truth_chi[indices].sum()),
                         "line_prediction_error_chi_square_q05_q50_q95": np.quantile(
                             line_error_chi[c][:, indices].sum(1), [.05, .5, .95]).tolist(),
                         "projected_noise_rank": rank,
                         "replicated_exceeds_observed_fraction": float(np.mean(replicated >= values))}
        audit["per_chain"].append(row)
    (output/"predictive.json").write_text(json.dumps(audit, indent=2))
    (output/"predictive_code.py").write_bytes(Path(__file__).read_bytes())
    np.savez_compressed(output/"predictive.npz", mean_line_by_chain=line_draws.mean(1),
        std_line_by_chain=line_draws.std(1), chi_square_by_chain_draw_exposure=chi,
        line_error_chi_square_by_chain_draw_exposure=line_error_chi,
        draw_indices=chosen, posterior_map_draws=np.stack(map_draws))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, n_chains, figsize=(3*n_chains, 6), squeeze=False)
    for c in range(n_chains):
        prior_map = np.asarray(posterior.controls(prior_samples["log_flux"][c], prior_samples["map_z"][c],
            prior_samples["map_sigma"][c], prior_samples["envelope_width"][c]))
        for ax, values, title in ((axes[0, c], prior_map, f"Prior draw {c}"),
                                  (axes[1, c], map_draws[c], f"Posterior chain {c}, draw {chosen[0]}")):
            image = ax.imshow(np.log10(np.maximum(values/values.max(), 1e-4)), origin="lower",
                              extent=[-.63, .63, -.63, .63], vmin=-4, vmax=0, cmap="magma")
            ax.set(title=title, xlabel="East [arcsec]", ylabel="North [arcsec]")
    fig.suptitle("Emission shape samples; log10 brightness / each map's peak; not uncertainty bands")
    fig.tight_layout()
    fig.savefig(output/"map_draws.png", dpi=160)
    plt.close(fig)
    print(json.dumps(audit), flush=True)
    return audit
