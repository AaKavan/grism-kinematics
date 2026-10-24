"""Locate the center of the parameter vector used for saved DINGO images.

This module evaluates only the existing trace calibration at two wavelengths.
It does not fit, rectify, or sample anything. In particular, the marker is not
made by combining posterior medians with images from a different sample.
"""
from pathlib import Path
import json
import os

import numpy as np


SCHEMA = "dingo-three-file-centers-v1"


def saved_parameters(result, pixels=None):
    """Return fixed parameters plus the actual vector that made the images."""
    pixels = result._prediction_pixels() if pixels is None else pixels
    names = np.asarray(pixels["param_names"]).astype(str)
    theta = np.asarray(pixels["theta"], dtype=float)
    if (names.ndim != 1 or theta.shape != names.shape or len(set(names)) != len(names)
            or not np.isfinite(theta).all()):
        raise ValueError("Saved prediction parameters are invalid.")
    params = {key: float(value) for key, value in result.report.get("fixed_params", {}).items()}
    params.update(zip(names.tolist(), theta.tolist()))
    for name in ("velocity.x0_v", "velocity.y0_v"):
        if name not in params or not np.isfinite(params[name]):
            raise ValueError(f"Saved prediction lacks a finite {name}.")
    return params


def _metadata(result):
    bundle = getattr(result, "_final_bundle", None)
    if bundle is None:
        return {}
    from .three_file_export import read_metadata
    return read_metadata(bundle)


def _calibration_records(identity):
    """Find the original recorded calibration in a nested scan-run identity."""
    if not isinstance(identity, dict):
        return None
    if isinstance(identity.get("calibration"), dict):
        return identity["calibration"]
    for key in ("parent_identity", "search_identity"):
        found = _calibration_records(identity.get(key))
        if found is not None:
            return found
    return None


def _project(config, params, identity=None):
    """Apply the inverse of the fitter's source-frame coordinate relation.

At the kinematic center, the rotation contribution is zero, so its wavelength
is lambda_rest*(1+z)*(1+v_sys/c). The fitter maps detector g to source s via
s = forward_model(g + registration, lambda) - cutout_origin.
The trace is position independent within each stamp, hence the exact inverse
is g = s + cutout_origin - registration - forward_model(0,0,lambda).
"""
    import torch
    from . import grism, utils
    from .real_workflow import REPO

    if "GRISM_CAL_DIR" not in os.environ and (REPO.parent / "grism_cal").is_dir():
        os.environ["GRISM_CAL_DIR"] = str(REPO.parent / "grism_cal")

    records = _calibration_records(identity)
    if records:
        from .real_workflow import sha256
        root = Path(grism.get_grism_cal_dir())
        for name, expected in records.items():
            if sha256(root / name) != expected:
                raise ValueError("Grism calibration changed since this run; cannot mark its observed center reliably.")
    summary = config["summary"]
    rfit = summary.get("r_fit")
    # Keep the same float32 operations and integer truncation as KinematicsFitter.
    lambda0 = torch.tensor([float(summary["lambda_rest"])], dtype=torch.float32) * (1 + float(summary["z"]))
    lambda_center = lambda0 * (1 + float(params.get("velocity.v_sys", 0.0)) / 299792.458)
    points, cutouts = {}, {}
    for pupil in ("R", "C"):
        setup = config["image"][pupil]
        _, spatial, dispersion = grism.load_nircam_wfss_model(pupil, setup["module"], summary["filter"])
        forward = utils.get_grism_model_torch(spatial, dispersion, pupil, 1024, 1024,
                                             direction="forward", device="cpu")
        if "cutout" in setup:
            cutout = tuple(setup["cutout"])
        elif rfit is not None:
            center = torch.tensor(float(rfit), dtype=torch.float32)
            cx, cy = forward(center, center, lambda0)
            cutout = (int(cx)-rfit, int(cy)-rfit, 2*rfit+1, 2*rfit+1)
        else:
            raise ValueError("Saved run needs explicit grism cutouts or summary.r_fit.")
        zero = torch.zeros_like(lambda_center)
        mdx, mdy = forward(zero, zero, lambda_center)
        dx = float(params.get(f"image.{pupil}.dx", setup.get("dx", 0.0)))
        dy = float(params.get(f"image.{pupil}.dy", setup.get("dy", 0.0)))
        points[f"observed_{pupil}"] = [params["velocity.x0_v"] + cutout[0] - dx - float(mdx),
                                         params["velocity.y0_v"] + cutout[1] - dy - float(mdy)]
        cutouts[pupil] = list(cutout)
    return points, dict(cutouts=cutouts, wavelength_center_micron=float(lambda_center),
                        calibration="matched saved hashes" if records else "current installed calibration; no saved hashes")


def center_geometry(result, pixels=None):
    """Return center coordinates for raw/rectified images of the saved fit.

Coordinates follow Matplotlib/FITS array display: x=column, y=row, zero based,
origin lower. The direct-image marker is explicitly a *nominal* overlay based
on centered, aligned, equal-scale stamps, not an astrometric gas/star offset.
Raw grism positions are None when an old exported bundle lacks its run config.
New final bundles can embed this complete geometry to remain self-contained.
"""
    pixels = result._prediction_pixels() if pixels is None else pixels
    params = saved_parameters(result, pixels)
    metadata = _metadata(result)
    embedded = metadata.get("kinematic_centers")
    if embedded is not None:
        if (embedded.get("schema") != SCHEMA or embedded.get("source_id") != result.source_id
                or embedded.get("params") != params):
            raise ValueError("Saved center geometry does not match this source's plotted parameter vector.")
        return embedded

    center = [params["velocity.x0_v"], params["velocity.y0_v"]]
    rfit = float(result.report["r_fit_pixels"])
    offset = np.asarray(center) - rfit

    def midpoint(name):
        shape = np.asarray(pixels[name]).shape
        if len(shape) != 2:
            raise ValueError(f"Saved {name} must be a 2D image.")
        return [(shape[1]-1)/2, (shape[0]-1)/2]

    references = {name: midpoint(key) for name, key in
                  (("direct", "image"), ("observed_R", "observed_R"), ("observed_C", "observed_C"))}
    references["rectified"] = [rfit, rfit]
    points = {"rectified": center, "direct": (np.asarray(references["direct"])+offset).tolist(),
              "observed_R": None, "observed_C": None}
    geometry = dict(schema=SCHEMA, source_id=result.source_id, params=params, points=points,
                    reference=references, offset_pixels=offset.tolist(),
                    offset_arcsec=(offset*float(result.report["pixel_scale_arcsec"])).tolist(),
                    offset_pc=(offset*result.physical_scale()["pc_per_pixel"]).tolist(),
                    direct_note="Nominal direct-image overlay assumes centered stamps with matching pixel scale and orientation; no astrometric registration is established.",
                    projection_note="Observed R/C centers use the saved fit's trace calibration, cutouts and registration offsets.",
                    provenance={"parameters": "saved prediction theta, including fixed parameters"})

    original = metadata.get("export", {}).get("original_run_directory")
    run_path = Path(original) if original else Path(result.path)
    config_path = run_path / "run_config.yaml"
    if not config_path.is_file():
        geometry["projection_note"] = "Observed R/C center unavailable: this older final bundle has no saved geometry and its original run_config.yaml is unavailable."
        return geometry
    import yaml
    config = yaml.safe_load(config_path.read_text())
    if (not isinstance(config, dict) or not isinstance(config.get("summary"), dict)
            or not {"name", "z", "lambda_rest", "filter"} <= set(config["summary"])
            or not isinstance(config.get("image"), dict)
            or any(not isinstance(config["image"].get(p), dict)
                   or "module" not in config["image"][p] for p in ("R", "C"))):
        geometry["projection_note"] = "Observed R/C center unavailable: the saved run configuration lacks the wavelength/trace metadata needed for projection."
        return geometry
    summary = config["summary"]
    if (summary.get("name") != f"ID{result.source_id}"
            or not np.isclose(float(summary["z"]), float(result.report["redshift"]), atol=1e-12, rtol=0)
            or not np.isclose(float(summary.get("r_fit", rfit)), rfit, atol=1e-12, rtol=0)):
        raise ValueError("Saved run configuration and plotted source metadata disagree.")
    identity_path = run_path / "identity.json"
    identity = json.loads(identity_path.read_text()) if identity_path.is_file() else None
    projected, provenance = _project(config, params, identity)
    geometry["points"].update(projected)
    geometry["provenance"].update(provenance, config_path=str(config_path))
    return geometry
