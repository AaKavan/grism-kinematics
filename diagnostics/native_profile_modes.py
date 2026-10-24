"""Release saved velocity constraints to search for alternative free modes.

A competitive constrained solution can lead to a different free optimum even
when the constrained objective does not beat the current best. This complements
the automatic nested-fit consistency guard in native_flexible_check.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from dingo.exposures import load_native_target
from dingo.native_fitting import BilinearLineMap, NativeMapFitter
from dingo.native_forward import NativeLineProjector

REPO = Path(__file__).resolve().parents[1]


def run(args):
    source = json.loads((args.run_dir/"recovery.json").read_text())
    settings = source["arguments"]
    torch.set_num_threads(settings["threads"])
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA access is blocked or unavailable")
    dtype = torch.float64 if settings["precision"] == "float64" else torch.float32
    target = load_native_target(settings["bundle"], REPO/f"data/ID{source['source_geometry']}/config_kinematics.yaml", settings["size"])
    paths = [str(e.path) for e in target.exposures]
    train = [paths.index(p) for p in source["train_exposures"]]
    held = [paths.index(p) for p in source["held_exposures"]]
    if args.all_exposures:
        train, held = list(range(len(paths))), []
    pupils = [e.pupil for e in target.exposures]
    with np.load(args.run_dir/"injection.npz") as data:
        science, error, valid = [data[k] for k in ("science", "error", "valid")]
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/"modes.json").exists():
        raise FileExistsError("Choose a fresh output to preserve the previous mode search")
    report = {"completed": False, "synthetic_not_science_measurements": True,
              "source_run": str(args.run_dir), "source_code_sha256": source["code_sha256"],
              "training_selection": "all_exposures" if args.all_exposures else "original",
              "train_exposures": [paths[i] for i in train], "held_exposures": [paths[i] for i in held],
              "code_sha256": {p: hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in
                  ("dingo/native_fitting.py", "dingo/native_forward.py", "dingo/exposures.py", "diagnostics/native_profile_modes.py")},
              "fits": []}
    for name in ("dingo/native_fitting.py", "dingo/native_forward.py", "dingo/exposures.py"):
        if report["code_sha256"][name] != source["code_sha256"][name]:
            raise ValueError(f"Source operator changed: {name}")
    references = [c for c in source["comparisons"] if c["reference"] in args.references]
    if "free_best" in args.references:
        references.extend({"reference": "free_best", "reference_best": f} for f in source["best_by_setting"])
    if set(c["reference"] for c in references) != set(args.references):
        raise ValueError("Requested reference fits have not been saved")
    y, x = np.mgrid[-4:5, -4:5]
    kernel = np.exp(-.5*(x*x+y*y)/settings["fit_blur"]**2)
    for comparison in references:
        previous = comparison["reference_best"]
        grid_size = previous["grid"]
        grid = BilinearLineMap(grid_size, settings["quadrature"], flux_unit=source["fixed_map_flux_unit"], device=device, dtype=dtype)
        projector = NativeLineProjector.from_target(target, grid.east.cpu().numpy(), grid.north.cpu().numpy(), kernel,
            np.ones(len(paths)), spectral_samples=7, device=device, dtype=dtype, deposition=settings["deposition"])
        fitter = NativeMapFitter(projector, grid, science, error, valid, pupils, train_indices=train,
            held_indices=held, continuum_degree=settings["degree"], inclination=settings["fit_inclination"])
        with np.load(args.run_dir/previous["artifact"]) as artifact:
            controls = artifact["controls"]
        t0 = time.perf_counter()
        fitted = fitter.fit(previous["fitted_vref_rturn_theta"], controls, strength=previous["strength"],
                            maxiter=args.maxiter, gtol=settings["gtol"])
        tag = f"grid{grid_size}_lambda{previous['strength']:g}_release_{comparison['reference']}.npz"
        entry = fitted["report"] | {"grid": grid_size, "released_reference": comparison["reference"],
                                    "seconds": time.perf_counter()-t0, "artifact": tag}
        np.savez_compressed(args.output/tag, vector=fitted["vector"], controls=fitted["controls"],
            line=fitted["line"], continuum=fitted["continuum"], report_json=json.dumps(entry))
        report["fits"].append(entry)
        (args.output/"modes.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(entry), flush=True)
    report["completed"] = True
    (args.output/"modes.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--references", nargs="+", default=["speed_450"])
    parser.add_argument("--maxiter", type=int, default=1600)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--all-exposures", action="store_true",
                        help="Exploratory fit to all exposures; no independent holdout remains")
    run(parser.parse_args())
