"""Audit SCI continuum predictability on held-out native sideband pixels.

This does not fit real galaxy rotation. The central +/-20 dispersion pixels
are excluded. Every fourth sideband pixel is held out; its continuum prediction
variance is added to ERR^2 for pointwise residual checks. Neighboring held-out
pixels remain correlated through the shared continuum fit; these summaries
are not a joint likelihood or a calibrated detection statistic.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dingo.exposures import load_native_target
from dingo.native_fitting import StripContinuumLikelihood

REPO = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ids", type=int, nargs="+", default=[3700, 15665])
    p.add_argument("--degrees", type=int, nargs="+", default=[1, 2, 3])
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    records = []
    for sid in args.ids:
        target = load_native_target(args.bundle, REPO / f"data/ID{sid}/config_kinematics.yaml")
        data, error, valid = [np.stack([getattr(e, name) for e in target.exposures]) for name in ("science", "error", "valid")]
        pupils = [e.pupil for e in target.exposures]
        yy, xx = np.indices(data.shape[-2:])
        side, hold = [], []
        for e in target.exposures:
            along = xx if e.pupil == "R" else yy
            axis = 0 if e.pupil == "R" else 1
            distance = along - (e.line_xy[axis]-e.origin_xy[axis])
            side.append(e.valid & (abs(distance) > 20))
            hold.append(along % 4 == 0)
        side, hold = np.array(side), np.array(hold)
        for degree in args.degrees:
            likelihood = StripContinuumLikelihood(data, error, side & ~hold, pupils, degree=degree)
            with torch.no_grad():
                continuum = likelihood.continuum(torch.zeros_like(likelihood.data)).numpy()
                variance = likelihood.continuum_variance().numpy()
            denominator = np.sqrt(error**2 + variance)
            residual = np.where(valid, (data-continuum)/denominator, np.nan)
            keep = side & hold & np.isfinite(residual)
            for pupil in ("R", "C"):
                selection = np.array(pupils) == pupil
                values = residual[selection][keep[selection]]
                median = np.median(values)
                record = {"source_id": sid, "pupil": pupil, "degree": degree, "held_pixels": int(len(values)),
                          "median": float(median), "robust_sigma": float(1.4826*np.median(abs(values-median))),
                          "rms": float(np.sqrt(np.mean(values**2))), "fraction_abs_gt_5": float(np.mean(abs(values)>5))}
                records.append(record); print(json.dumps(record), flush=True)
            np.savez_compressed(args.output / f"ID{sid}_degree{degree}.npz", continuum=continuum,
                                continuum_variance=variance, residual=residual, train=side & ~hold, held=keep)
            fig, axes = plt.subplots(2, 3, figsize=(11, 7))
            for j, pupil in enumerate(("R", "C")):
                e = pupils.index(pupil)
                for ax, values, label in zip(axes[j], [data[e]/error[e], continuum[e]/error[e], residual[e]],
                                             ["SCI / ERR", "Sideband continuum / ERR", "Predictive residual"]):
                    ax.imshow(np.where(valid[e], values, np.nan), origin="lower", vmin=-5, vmax=5, cmap="RdBu_r")
                    ax.set_title(f"{pupil}: {label}"); ax.set_axis_off()
            fig.suptitle(f"ID{sid}: degree {degree}; central dispersion region excluded from continuum fit")
            fig.tight_layout(); fig.savefig(args.output / f"ID{sid}_degree{degree}.png", dpi=140); plt.close(fig)
    (args.output / "sideband_audit.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
