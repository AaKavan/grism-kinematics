"""Audit and summarize completed paired R/C ablations without rerunning fits."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def summarize(root):
    reports = {mode: json.loads((root/f"nonlinear_{mode}_seed481/recovery.json").read_text())
               for mode in ("R", "C", "RC")}
    if not all(report.get("completed", False) for report in reports.values()):
        raise ValueError("Wait for all three fits to finish before interpreting the comparison")
    reference = reports["RC"]
    rows = []
    radii = np.array([.1, .2, .3, .5])
    truth = np.array(reference["truth_vref_rturn_theta"])
    true_curve = truth[0]*np.arctan(radii/truth[1])/np.arctan(.5/truth[1])
    payload = {}
    injection_differences = {}
    with np.load(root/"nonlinear_RC_seed481/injection.npz") as source:
        reference_arrays = {k: source[k] for k in ("science", "error", "valid", "true_line", "true_continuum")}
    for mode, report in reports.items():
        assert report["held_exposures"] == reference["held_exposures"], "Different held-out data"
        np.testing.assert_allclose(report["fixed_map_flux_unit"], reference["fixed_map_flux_unit"], rtol=1e-13)
        assert report["code_sha256"] == reference["code_sha256"], "Different implementations"
        with np.load(root/f"nonlinear_{mode}_seed481/injection.npz") as injection:
            payload[mode] = {key: hashlib.sha256(injection[key].tobytes()).hexdigest()
                             for key in ("science", "error", "valid", "true_line", "true_continuum")}
            injection_differences[mode] = {}
            for key, expected in reference_arrays.items():
                if key in ("error", "valid", "true_continuum"):
                    np.testing.assert_array_equal(injection[key], expected)
                else:
                    # GPU scatter sums differ at roundoff level. Test in ERR
                    # units, not relative to pixels whose true line is zero.
                    delta = np.max(np.abs(injection[key]-expected)/reference_arrays["error"])
                    assert delta < 1e-12, "Injection differs beyond GPU roundoff"
                    injection_differences[mode][f"max_{key}_difference_in_error_units"] = float(delta)
        assert len(report["best_by_setting"]) == 1
        original_best = report["best_by_setting"][0]
        extra = []
        modes_path = root/f"{mode}_released_modes/modes.json"
        if modes_path.exists():
            released = json.loads(modes_path.read_text())
            if not released.get("completed"):
                raise ValueError("Wait for the released-constraint mode search to finish")
            assert released["source_code_sha256"] == report["code_sha256"]
            assert released.get("training_selection", "original") == "original", "Cannot mix different training data"
            assert Path(released["source_run"]).resolve() == (root/f"nonlinear_{mode}_seed481").resolve()
            extra = released["fits"]
        free_fits = report["fits"] + extra
        best = min(free_fits, key=lambda f: f["objective"])
        speed, turnover, _ = best["fitted_vref_rturn_theta"]
        fitted_curve = speed*np.arctan(radii/turnover)/np.arctan(.5/turnover)
        profiles = sorted([{
            "speed": c["reference_best"]["fixed_parameters"]["v_ref"],
            "twice_delta_nll": 2*(c["reference_best"]["objective"]-best["objective"]),
            "stationary": c["reference_best"]["stationary"],
            "valid_optimization": bool(c["reference_best"]["stationary"] and best["stationary"]
                                       and c["reference_best"]["objective"] >= best["objective"]-5e-5),
            "boundary_hits": c["reference_best"]["boundary_hits"],
            "artifact": c["reference_best"]["artifact"],
        } for c in report["comparisons"]], key=lambda row: row["speed"])
        all_attempts = free_fits+report["reference_fits"]
        alternatives = []
        for f in sorted(free_fits, key=lambda f: f["objective"]):
            if (f["stationary"] and 2*(f["objective"]-best["objective"]) <= 3.84146
                    and all(abs(f["fitted_vref_rturn_theta"][0]-g["fitted_vref_rturn_theta"][0]) > 10
                            for g in [best]+alternatives)):
                alternatives.append(f)
        rows.append({"mode": mode, "n_training_exposures": len(report["train_exposures"]),
                     "best": best, "original_best": original_best, "profiles": profiles,
                     "competitive_alternative_modes": alternatives, "additional_free_fits": extra,
                     "projected_rotation_profile": fitted_curve.tolist(),
                     "rotation_only_mass_ratio": ((fitted_curve/true_curve)**2).tolist(),
                     "fit_attempts": len(all_attempts),
                     "nonstationary_attempts": sum(not f["stationary"] for f in all_attempts),
                     "boundary_attempts": sum(bool(f["boundary_hits"]) for f in all_attempts),
                     "fit_seconds": sum(f["seconds"] for f in all_attempts)})
    summary = {"synthetic_not_science_measurements": True, "completed": True,
               "source_geometry": reference["source_geometry"], "seed": reference["seed"],
               "radii_arcsec": radii.tolist(), "truth_projected_rotation_profile": true_curve.tolist(),
               "injection_array_sha256": payload,
               "injection_roundoff_audit": injection_differences,
               "code_sha256": reference["code_sha256"], "rows": rows,
               "notes": ["One paired noise draw; not interval coverage or a GEKO benchmark",
                         "Profile values re-fit map, turnover and angle; inclination fixed at truth",
                         "Zero-speed statistic has no assumed chi-square null distribution",
                         "Only likelihood differences within one data selection are compared",
                         "All modes predict the same six held-out exposures with continuum refitted"]}
    (root/"comparison.json").write_text(json.dumps(summary, indent=2))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    curve_radii = np.linspace(.03, .5, 100)
    for row in rows:
        best = row["best"]
        color = {"R": "C0", "C": "C1", "RC": "C2"}[row["mode"]]
        x = [p["speed"] for p in row["profiles"]]
        y = [p["twice_delta_nll"] for p in row["profiles"]]
        axes[0].plot(x, y, "o-", color=color, label=row["mode"])
        axes[0].scatter([best["fitted_vref_rturn_theta"][0]], [0], color=color, marker="*")
        for point in row["profiles"]:
            if not point["valid_optimization"]:
                axes[0].scatter(point["speed"], point["twice_delta_nll"], color="red", marker="x", s=90)
        v, rt, _ = best["fitted_vref_rturn_theta"]
        axes[1].plot(curve_radii, v*np.arctan(curve_radii/rt)/np.arctan(.5/rt), color=color, label=row["mode"])
        axes[2].plot(radii, row["rotation_only_mass_ratio"], "o-", color=color, label=row["mode"])
        for alternative in row["competitive_alternative_modes"]:
            v, rt, _ = alternative["fitted_vref_rturn_theta"]
            axes[0].scatter(v, 2*(alternative["objective"]-best["objective"]), marker="*", color=color)
            axes[1].plot(curve_radii, v*np.arctan(curve_radii/rt)/np.arctan(.5/rt), "--", color=color,
                         alpha=.7, label=f"{row['mode']} alternative")
            ratio = (v*np.arctan(radii/rt)/np.arctan(.5/rt)/true_curve)**2
            axes[2].plot(radii, ratio, "--", color=color, alpha=.7, label=f"{row['mode']} alternative")
    axes[0].axvline(truth[0], ls="--", color="k", alpha=.5)
    axes[0].axhline(3.84146, ls=":", color="gray", label="3.84 nominal reference")
    axes[0].set(xlabel="Fixed projected V(0.5 arcsec) [km/s]", ylabel="2 delta negative log likelihood", yscale="symlog", ylim=(-.1, None))
    axes[1].plot(curve_radii, truth[0]*np.arctan(curve_radii/truth[1])/np.arctan(.5/truth[1]), "k--", label="Injected")
    axes[1].set(xlabel="Radius [arcsec] (test coordinates)", ylabel="Best-fit projected rotation [km/s]")
    axes[2].axhline(1., ls="--", color="k")
    axes[2].set(xlabel="Radius [arcsec] (test coordinates)", ylabel="Best-fit / injected rotation-only mass")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle("Same synthetic source and noise: R vs C vs joint; fixed inclination; no calibrated intervals")
    fig.tight_layout()
    fig.savefig(root/"nonlinear_comparison.png", dpi=180)
    plt.close(fig)
    full_path = root/"RC_all_exposures_modes/modes.json"
    if full_path.exists():
        full = json.loads(full_path.read_text())
        if not full.get("completed"):
            raise ValueError("Full-exposure exploration is still running")
        assert full["training_selection"] == "all_exposures"
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        true_curve_dense = truth[0]*np.arctan(curve_radii/truth[1])/np.arctan(.5/truth[1])
        inclination = reference["fixed_inclination_rad"]
        best = min(full["fits"], key=lambda f: f["objective"])
        for f in sorted(full["fits"], key=lambda f: f["objective"]):
            v, rt, _ = f["fitted_vref_rturn_theta"]
            curve = v*np.arctan(curve_radii/rt)/np.arctan(.5/rt)
            delta = 2*(f["objective"]-best["objective"])
            label = f"Vref={v:.1f}; 2 delta NLL={delta:.2f}"
            axes[0].plot(curve_radii, curve/np.sin(inclination), label=label)
            axes[1].plot(curve_radii, (curve/true_curve_dense)**2, label=label)
        axes[0].plot(curve_radii, true_curve_dense/np.sin(inclination), "k--", label="Injected")
        axes[1].axhline(1., ls="--", color="k")
        axes[0].set_ylabel("Deprojected rotation [km/s]; known inclination")
        axes[1].set_ylabel("Fitted / injected rotation-only mass")
        for ax in axes:
            ax.set_xlabel("Radius [arcsec] (test coordinates)")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.suptitle("Exploratory fit to all 18 exposures; two modes; no independent holdout remains")
        fig.tight_layout()
        fig.savefig(root/"all_exposure_profiles.png", dpi=180)
        plt.close(fig)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root)
