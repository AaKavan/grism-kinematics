"""Create small, isolated R/C recovery experiments for the original fitter.

The exponential profile is only a way to invent emission. The recovery fitter
does not receive its morphology parameters or fit a Sérsic model. These mocks
use the existing source-to-grism deposition, without a PSF or intrinsic line
width, so they test numerical consistency before observational realism.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import shutil
import tempfile

import numpy as np
import yaml


SCHEMA = "dingo-simple-simulation-v1"
SOURCE_ID = 9000001
REPO = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class MockData:
    """Locations and known parameters of one saved synthetic experiment.

    ``truth`` uses the fitter's dotted parameter names, radians for angles,
    pixels for radii/positions, and km/s for velocities. ``data_dir`` contains
    the synthetic ID folder and can be passed directly to simple_mcmc.prepare.
    """

    directory: Path
    data_dir: Path
    source_id: int
    truth: dict
    metadata: dict


def load_mock(directory):
    """Open a saved simulation and reject changed inputs or truth arrays."""
    directory = Path(directory).expanduser().resolve()
    record = json.loads((directory / "truth.json").read_text())
    if record.get("schema") != SCHEMA or record.get("source_id") != SOURCE_ID:
        raise ValueError("This folder is not a simple DINGO simulation.")
    source = directory / "inputs" / f"ID{SOURCE_ID}"
    for name, expected in record["input_sha256"].items():
        path = source / name
        if Path(name).name != name or not path.is_file() or _sha(path) != expected:
            raise ValueError(f"Simulation input changed or is missing: {name}.")
    if _sha(directory / "truth.npz") != record["truth_npz_sha256"]:
        raise ValueError("The saved simulation truth arrays have changed.")
    return MockData(directory, directory / "inputs", SOURCE_ID,
                    dict(record["truth"]), record)


def _check_destination(directory, overwrite):
    """Never replace real source data or an unrelated existing folder."""
    original = Path(directory).expanduser().absolute()
    if original.is_symlink():
        raise ValueError("The simulation folder must not be a symlink.")
    directory = original.resolve()
    real_data = REPO / "data"
    cwd = Path.cwd().resolve()
    if (directory == real_data or real_data in directory.parents
            or directory in real_data.parents or directory == cwd
            or directory in cwd.parents):
        raise ValueError("Use a separate simulation experiment folder, outside real source data.")
    if directory.exists():
        if not overwrite:
            raise FileExistsError(f"Simulation already exists: {directory}. Set overwrite=True or change its name.")
        marker = directory / "truth.json"
        try:
            record = json.loads(marker.read_text())
        except (OSError, ValueError) as error:
            raise ValueError("Refusing to replace a folder without a simulation record.") from error
        if record.get("schema") != SCHEMA or record.get("source_id") != SOURCE_ID:
            raise ValueError("Refusing to replace a folder not created by make_mock.")
    return directory


def make_mock(directory, *, velocity=200., turnover=5., inclination=60.,
              pa=35., center=(40.3, 40.7), effective_radius=6., peak=1.,
              noise_sigma=0.02, seed=42, overwrite=False, module="B",
              filter_name="F444W", redshift=1.1231, rest_wavelength=1.875,
              size=81):
    """Save a smooth exponential galaxy dispersed into R and C.

    Parameters
    ----------
    directory : path-like
        One experiment folder. Inputs go into ``inputs/ID9000001``; truth is
        recorded in ``truth.json`` and ``truth.npz``. Real source files are
        never read or edited. Only the installed grism calibration is needed.
    velocity : float
        True deprojected asymptotic speed V_inf, in km/s. Set zero for a null.
    turnover, effective_radius : float
        Arctangent turnover radius and emission half-light radius, in pixels.
    inclination, pa : float
        Inclination and cutout major-axis orientation, in degrees. The smooth
        thin disk has projected axis ratio cos(inclination).
    center : pair of floats
        Common emission and kinematic center in zero-based source pixels.
    peak : float
        Continuous central surface brightness. With a subpixel center, the
        brightest sampled source pixel can be slightly below this value.
    noise_sigma : float
        Known per-pixel standard deviation of independent Gaussian noise in
        each grism image. Zero produces noiseless images for a residual scan;
        it does not define meaningful MCMC uncertainties.
    seed : int
        Seed for independent R and C noise. Geometry is independent of seed.
    overwrite : bool
        Replace only an existing experiment created by this function, including
        its previous recovery runs. Generation is validated before replacement.
    module, filter_name, redshift, rest_wavelength : str or float
        Calibration and line wavelength settings; wavelength is in microns.
        Defaults borrow the observing setup of ID3700, not its measured image.
    size : odd int
        Side length of the source and grism cutouts. The default is 81 pixels.

    Returns
    -------
    MockData
        Saved paths, truth parameters and generation metadata. No optimizer or
        MCMC is run. Recovery must construct a fresh fitter with explicit starts.

    Notes
    -----
    Generation calls KinematicsFitter.make_mock_data with an explicitly supplied
    clean brightness map and inplace=False. That helper uses the fitter's exact
    coordinate convention. Two pixel interpolations are not an exact inverse:
    noiseless truth can still leave a nonzero rectified R-C residual.
    """
    from astropy.io import fits
    import torch
    from .fitting import KinematicsFitter
    from .kinematics import arctangent_disk_velocity_model
    from .three_file_fit import build_config

    values = np.asarray([velocity, turnover, inclination, pa, effective_radius,
                         peak, noise_sigma, redshift, rest_wavelength], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("All numerical simulation settings must be finite.")
    if velocity < 0 or turnover <= 0 or effective_radius <= 0 or peak <= 0 or noise_sigma < 0:
        raise ValueError("Velocity/noise must be nonnegative; radii and peak must be positive.")
    if not 5 <= inclination <= 85:
        raise ValueError("Use an inclination from 5 to 85 degrees, matching the current fitter.")
    if redshift <= -1 or rest_wavelength <= 0:
        raise ValueError("The systemic observed wavelength must be positive.")
    if isinstance(size, bool) or int(size) != size or size < 9 or size % 2 != 1:
        raise ValueError("size must be an odd integer of at least nine pixels.")
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    center = np.asarray(center, dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all() or np.any(center < 0) or np.any(center > size-1):
        raise ValueError("center must contain two finite coordinates inside the source image.")
    if module not in ("A", "B"):
        raise ValueError("module must be 'A' or 'B'.")
    directory = _check_destination(directory, overwrite)
    size, seed = int(size), int(seed)
    angle, inc = np.deg2rad([pa % 360., inclination])
    truth = {"velocity.V_rot": float(velocity), "velocity.R_v": float(turnover),
             "velocity.x0_v": float(center[0]), "velocity.y0_v": float(center[1]),
             "velocity.theta_v": float(angle), "velocity.inc_v": float(inc),
             "velocity.v_sys": 0., "image.R.dx": 0., "image.R.dy": 0.,
             "image.C.dx": 0., "image.C.dy": 0.}

    yy, xx = np.indices((size, size), dtype=float)
    dx, dy = xx-center[0], yy-center[1]
    major = np.cos(angle)*dx + np.sin(angle)*dy
    minor = -np.sin(angle)*dx + np.cos(angle)*dy
    radius = np.hypot(major, minor/np.cos(inc))
    image = (float(peak)*np.exp(-1.67834699001666*radius/effective_radius)).astype(np.float32)
    summary = dict(name=f"ID{SOURCE_ID}", ra=0., dec=0., z=float(redshift),
                   lambda_rest=float(rest_wavelength), filter=filter_name,
                   mode="kinematics", r_fit=(size-1)//2, unit="pix")
    source_final = directory / "inputs" / f"ID{SOURCE_ID}"
    source_config = {"summary": summary, "image": {
        p: dict(module=module, path=[str(source_final/f"{p}.fits"), 0], dx=0., dy=0.)
        for p in ("R", "C")}}
    if "GRISM_CAL_DIR" not in os.environ and (REPO.parent / "grism_cal").is_dir():
        os.environ["GRISM_CAL_DIR"] = str(REPO.parent / "grism_cal")

    # Build in temporary storage: bad calibration or settings cannot destroy a
    # previously completed experiment, even when overwrite=True was requested.
    with tempfile.TemporaryDirectory(prefix="dingo_simulation_") as temporary:
        stage = Path(temporary)
        source = stage / "inputs" / f"ID{SOURCE_ID}"
        source.mkdir(parents=True)
        for name in ("image", "R", "C"):
            fits.writeto(source/f"{name}.fits", image, overwrite=True)
        config_path = source / "config_kinematics.yaml"
        config_path.write_text(yaml.safe_dump(source_config, sort_keys=False))
        config = build_config(SOURCE_ID, source.parent, config_path, stage/"unused",
                              device="cpu", velocity_parameter="V_rot")
        generator_path = stage / "generator.yaml"
        generator_path.write_text(yaml.safe_dump(config, sort_keys=False))
        # Bound CPU threads during generation, then restore the user's setting.
        old_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(min(old_threads, 4))
            generator = KinematicsFitter(str(generator_path), device=torch.device("cpu"))
            clean = generator.make_mock_data(image, theta=truth, sigma_R=0., sigma_C=0.,
                                             sigma_v=0., seed=seed, inplace=False)
            noisy = generator.make_mock_data(image, theta=truth, sigma_R=noise_sigma,
                                             sigma_C=noise_sigma, sigma_v=0., seed=seed,
                                             inplace=False)
            parameters = generator._get_model_params("velocity")
            with torch.no_grad():
                velocity_los = arctangent_disk_velocity_model(
                    torch.as_tensor(xx, dtype=torch.float32),
                    torch.as_tensor(yy, dtype=torch.float32), **parameters).numpy()
            clean = [a.detach().cpu().numpy() for a in clean]
            noisy = [a.detach().cpu().numpy() for a in noisy]
        finally:
            torch.set_num_threads(old_threads)
        if not all(np.isfinite(a).all() for a in [image, velocity_los, *clean, *noisy]):
            raise ValueError("The requested simulation produced nonfinite pixels.")
        for pupil, array in zip(("R", "C"), noisy):
            fits.writeto(source/f"{pupil}.fits", array, overwrite=True)
        np.savez_compressed(stage/"truth.npz", image=image, velocity_los=velocity_los,
                            noiseless_R=clean[0], noiseless_C=clean[1])
        record = dict(schema=SCHEMA, source_id=SOURCE_ID,
                      created_utc=datetime.now(timezone.utc).isoformat(), truth=truth,
                      size=size, effective_radius=float(effective_radius), peak=float(peak),
                      noise_sigma=float(noise_sigma), seed=seed, morphology="exponential thin disk",
                      axis_ratio=float(np.cos(inc)), intrinsic_dispersion_kms=0., psf="none",
                      instrument=summary | {"module": module},
                      source_flux=float(image.sum(dtype=np.float64)),
                      flux_retained={p: float(a.sum(dtype=np.float64)/image.sum(dtype=np.float64))
                                     for p, a in zip(("R", "C"), clean)},
                      input_sha256={p.name: _sha(p) for p in sorted(source.iterdir())},
                      truth_npz_sha256=_sha(stage/"truth.npz"),
                      note="Same-calibration numerical recovery test; no PSF or real source morphology. "
                           "Finite-pixel dispersion and rectification are not exact image inverses.")
        (stage/"truth.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
        generator_path.unlink()
        # Recheck the marker just before replacing an existing experiment.
        _check_destination(directory, overwrite)
        directory.parent.mkdir(parents=True, exist_ok=True)
        if directory.exists():
            shutil.rmtree(directory)
        shutil.copytree(stage, directory)
    return load_mock(directory)
