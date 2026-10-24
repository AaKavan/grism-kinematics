"""Plot saved conditional injection results; spread is not a posterior interval."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(directory):
    summary = json.loads((directory / "suite.json").read_text())
    scenarios = list(dict.fromkeys(row["scenario"] for row in summary["rows"]))
    settings = sorted({(row["grid"], row["strength"]) for row in summary["rows"]})
    radii = np.linspace(.05, .5, 150)
    fig, axes = plt.subplots(len(scenarios), 2, figsize=(10, 3.3*len(scenarios)), squeeze=False)
    for row_number, scenario in enumerate(scenarios):
        axes[row_number, 0].set_ylabel(f"{scenario}\nProjected rotation [km/s]")
        axes[row_number, 1].set_ylabel("Rotation-only mass ratio")
        for i, (grid, strength) in enumerate(settings):
            rows = [r for r in summary["rows"] if r["scenario"] == scenario
                    and r["grid"] == grid and r["strength"] == strength]
            if not rows:
                continue
            curves, mass_curves = [], []
            for r in rows:
                report = json.loads((directory / r["path"] / "recovery.json").read_text())
                vt, rt, _ = report["truth_vref_rturn_theta"]
                true = vt*np.arctan(radii/rt)/np.arctan(.5/rt)
                fitted = r["vref"]*np.arctan(radii/r["r_turn"])/np.arctan(.5/r["r_turn"])
                curves.append(fitted)
                axes[row_number, 0].plot(radii, fitted, color=f"C{i}", alpha=.3, lw=1)
                if vt > 0:
                    mass = ((fitted/np.sin(report["fixed_inclination_rad"])) /
                            (true/np.sin(report["truth_inclination_rad"])))**2
                    mass_curves.append(mass)
                    axes[row_number, 1].plot(radii, mass, color=f"C{i}", alpha=.3, lw=1)
            label = f"{grid}x{grid}, lambda={strength:g}: median of {len(rows)} seeds"
            axes[row_number, 0].plot(radii, np.median(curves, axis=0), color=f"C{i}", lw=2, label=label)
            if mass_curves:
                axes[row_number, 1].plot(radii, np.median(mass_curves, axis=0), color=f"C{i}", lw=2)
        axes[row_number, 0].plot(radii, true, "k--", label="Injected")
        axes[row_number, 0].legend(fontsize=7)
        if scenario == "null":
            axes[row_number, 1].text(.5, .5, "No mass ratio: injected rotation is zero.\n"
                "Positive fitted speed alone is not a detection.", ha="center", va="center",
                transform=axes[row_number, 1].transAxes)
            axes[row_number, 1].set_axis_off()
        else:
            axes[row_number, 1].axhline(1, color="k", ls="--")
        for ax in axes[row_number]:
            ax.set_xlabel("Test radius [arcsec]")
    fig.suptitle("Synthetic profiles: thin lines are independent-noise fits, not uncertainty bounds", fontsize=11)
    fig.tight_layout(); fig.savefig(directory / "profiles.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    for directory in parser.parse_args().directories:
        plot(directory)
