"""Repeated native-exposure injections, used by native_flexible_check --seeds.

These are conditional stress tests. A profile-likelihood reference threshold
is recorded as a diagnostic, not assumed to provide calibrated coverage.
No chi-square null detection calibration is used: angle/turnover are absent
under zero rotation. An empirical null ensemble is required for that purpose.
"""

import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dingo.exposures import load_native_target


def summarize_run(report, scenario, relative_path):
    """Keep failed/boundary fits visible; never interpret estimates as detections."""
    truth = report["truth_vref_rturn_theta"]
    true_inc = report["truth_inclination_rad"]
    fit_inc = report["fixed_inclination_rad"]
    radii = np.array([.1, .2, .3, .5])
    true_curve = truth[0]*np.arctan(radii/truth[1])/np.arctan(.5/truth[1])
    rows = []
    for best in report["best_by_setting"]:
        v, rt, theta = best["fitted_vref_rturn_theta"]
        fitted_curve = v*np.arctan(radii/rt)/np.arctan(.5/rt)
        comparisons = {c["reference"]: c for c in report["comparisons"]
                       if c["grid"] == best["grid"] and c["strength"] == best["strength"]}
        ref = comparisons.get("truth_speed")
        null = comparisons.get("null")
        rows.append({
            "scenario": scenario, "seed": report["seed"], "path": relative_path,
            "grid": best["grid"], "strength": best["strength"],
            "vref": v, "truth_vref": truth[0], "r_turn": rt, "theta": theta,
            "vref_error": v-truth[0], "stationary": best["stationary"],
            "boundary_hits": best["boundary_hits"],
            "deprojected_vref": v/np.sin(fit_inc),
            "truth_deprojected_vref": truth[0]/np.sin(true_inc),
            "radii_arcsec": radii.tolist(),
            "projected_rotation_profile": fitted_curve.tolist(),
            "truth_projected_rotation_profile": true_curve.tolist(),
            "rotation_only_mass_ratio_profile": None if truth[0] == 0 else
                ((fitted_curve/np.sin(fit_inc))/(true_curve/np.sin(true_inc)))**2,
            "twice_delta_null": None if null is None else null["twice_objective_difference"],
            "null_comparison_valid": False if null is None else null["valid_optimization"],
            "twice_delta_truth_speed": None if ref is None else ref["twice_objective_difference"],
            "truth_comparison_valid": False if ref is None else ref["valid_optimization"],
            "nominal_3p84_contains_truth": None if ref is None or best["strength"] != 0
                or not ref["valid_optimization"] else ref["twice_objective_difference"] <= 3.841458820694124,
        })
        if rows[-1]["rotation_only_mass_ratio_profile"] is not None:
            rows[-1]["rotation_only_mass_ratio_profile"] = rows[-1]["rotation_only_mass_ratio_profile"].tolist()
    return rows


def aggregate(rows):
    output = []
    for key in sorted({(r["scenario"], r["grid"], r["strength"]) for r in rows}):
        group = [r for r in rows if (r["scenario"], r["grid"], r["strength"]) == key]
        v = np.array([r["vref"] for r in group])
        checks = [r["nominal_3p84_contains_truth"] for r in group if r["nominal_3p84_contains_truth"] is not None]
        null = [r["twice_delta_null"] for r in group if r["null_comparison_valid"]]
        mass = [r["rotation_only_mass_ratio_profile"] for r in group
                if r["rotation_only_mass_ratio_profile"] is not None]
        output.append({"scenario": key[0], "grid": key[1], "strength": key[2],
            "n": len(group), "stationary": sum(r["stationary"] for r in group),
            "boundary": sum(bool(r["boundary_hits"]) for r in group),
            "vref_median": float(np.median(v)), "vref_range": [float(v.min()), float(v.max())],
            "vref_mean_error": float(np.mean([r["vref_error"] for r in group])),
            "nominal_truth_checks_valid": len(checks), "nominal_truth_checks_included": sum(checks),
            "valid_null_comparisons": len(null),
            "twice_delta_null_range": None if not null else [min(null), max(null)],
            "median_mass_ratio_profile": None if not mass else np.median(mass, axis=0).tolist()})
    return output


def plot_suite(rows, output):
    cases = list(dict.fromkeys(r["scenario"] for r in rows))
    settings = sorted({(r["grid"], r["strength"]) for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for i, setting in enumerate(settings):
        group = [r for r in rows if (r["grid"], r["strength"]) == setting]
        x = [cases.index(r["scenario"]) + .15*(i-(len(settings)-1)/2) for r in group]
        label = f"{setting[0]}x{setting[0]}, lambda={setting[1]:g}"
        color = f"C{i}"
        axes[0].scatter(x, [r["vref_error"] for r in group], color=color, label=label, alpha=.65)
        for xx, r in zip(x, group):
            if r["rotation_only_mass_ratio_profile"] is not None:
                axes[1].scatter(xx, r["rotation_only_mass_ratio_profile"][-1], color=color, alpha=.65)
            if r["twice_delta_null"] is not None:
                axes[2].scatter(xx, r["twice_delta_null"], color=color, alpha=.65,
                                marker="o" if r["null_comparison_valid"] else "x")
    axes[0].axhline(0, color="k", ls="--"); axes[1].axhline(1, color="k", ls="--")
    axes[0].set_ylabel("Recovered minus injected Vref [km/s]")
    axes[1].set_ylabel("Rotation-only mass ratio at 0.5 arcsec")
    axes[2].set_ylabel("2 (null objective - rotating objective)")
    for ax in axes:
        ax.set_xticks(range(len(cases)), cases, rotation=25, ha="right")
    axes[0].legend(fontsize=8)
    fig.suptitle("Synthetic repeated-noise tests; fixed nuisance inputs, not galaxy mass measurements")
    fig.tight_layout(); fig.savefig(output / "suite.png", dpi=150); plt.close(fig)


def run_suite(args):
    from diagnostics.native_flexible_check import run, REPO
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.scenarios)) != len(args.scenarios):
        raise ValueError("Duplicate seeds/scenarios are not independent trials")
    if (args.output / "suite.json").exists():
        raise FileExistsError("Use a fresh suite output directory to preserve previous results")
    args.output.mkdir(parents=True, exist_ok=True)
    target = load_native_target(args.bundle, REPO / f"data/ID{args.source}/config_kinematics.yaml", args.size)
    summary = {"not_science_measurements": True, "source": args.source,
        "seeds": args.seeds, "scenarios": args.scenarios, "completed_runs": 0,
        "expected_runs": len(args.seeds)*len(args.scenarios),
        "noise_note": "Independent seeds within scenarios; same seed paired across scenarios/settings",
        "profile_note": "3.84 is only a nominal 1-parameter profile reference; empirical coverage unestablished",
        "null_note": "No chi-square detection p-value: angle/turnover unidentifiable under null; small null ensemble",
        "mass_note": "V^2 ratio at fixed radius, assuming rotation traces circular speed; not a physical mass measurement",
        "rows": [], "aggregates": []}
    for scenario in args.scenarios:
        for seed in args.seeds:
            child = copy.copy(args)
            child.seed = seed
            child.output = args.output / scenario / f"seed{seed}"
            if scenario == "null":
                child.truth_vref = 0.
            elif scenario in ("smooth", "two_clumps"):
                child.morphology = scenario
            elif scenario == "inclination":
                child.truth_inclination = .65  # fit keeps the declared assumed inclination
            elif scenario == "blur":
                child.truth_blur = 1.15*args.fit_blur
            print(f"SUITE {scenario} seed={seed}", flush=True)
            report = run(child, target)
            summary["rows"].extend(summarize_run(report, scenario, str(child.output.relative_to(args.output))))
            summary["completed_runs"] += 1
            summary["aggregates"] = aggregate(summary["rows"])
            (args.output / "suite.json").write_text(json.dumps(summary, indent=2))
            plot_suite(summary["rows"], args.output)
    return summary
