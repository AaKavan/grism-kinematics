"""Save notebook-facing products without moving or altering fitting checkpoints.

The public workflow owns run creation and locking.  This private module only
reads completed posteriors, renders figures and publishes their final products.
It deliberately reuses the existing, validated final-bundle exporter rather
than introducing another posterior format or another scientific calculation.
"""

from collections.abc import Mapping
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .three_file_export import export_final, read_metadata, _save_plot


PRODUCTS_SCHEMA = "dingo-workflow-products-v1"
POSTERIOR_DETAILS_MARKER = "\n--- POSTERIOR DETAILS ---\n"
_PROTECTED_NAMES = {"run.json", "_checkpoints", "_products.json"}


class _PublicationRollbackError(RuntimeError):
    """Publication failed and its backup must be retained for recovery."""


def all_walker_traces(result, parameters=None):
    """Plot every walker and the ensemble median at each retained step.

    Parameters
    ----------
    result : ThreeFileResults
        A validated saved posterior. No optimization or sampling is performed.
    parameters : sequence of str, optional
        Display names accepted by ``ThreeFileResults._columns``. The default
        uses its normal kinematic parameters and omits fixed inclination.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per parameter. All walkers appear in gray, including any
        walkers in a less populated solution. The black line pools walkers at
        each step; it is not a selected solution or a convergence guarantee.
    """
    import matplotlib.pyplot as plt

    parameters = result._plot_parameters(parameters)
    columns = result._columns(parameters)
    if (not parameters or len(set(parameters)) != len(parameters)
            or any(parameter not in columns for parameter in parameters)):
        raise ValueError(f"Choose distinct parameters from {tuple(columns)}.")

    column_count = min(2, len(parameters))
    row_count = (len(parameters) + column_count - 1) // column_count
    figure, axes = plt.subplots(
        row_count, column_count,
        figsize=(6 * column_count, 2.7 * row_count),
        squeeze=False, layout="constrained",
    )
    for axis, parameter in zip(axes.ravel(), parameters):
        values, label = columns[parameter]
        steps = np.arange(1, values.shape[0] + 1)
        # Plot columns together: matplotlib creates a line for every walker.
        # Rasterizing these thin lines keeps large PDF displays manageable.
        axis.plot(steps, values, color=".55", alpha=.25, linewidth=.4,
                  rasterized=True)
        axis.plot(steps, np.median(values, axis=1), color="black", linewidth=.9)
        axis.set(xlabel="Retained ensemble step", ylabel=label)

    for unused_axis in axes.ravel()[len(parameters):]:
        figure.delaxes(unused_axis)

    title = (f"ID{result.source_id}: all walkers (gray), pooled ensemble median "
             "at each step (black)\n"
             "Walkers interact; they are not independent chains. "
             "Separate solutions are not selected or combined by probability here.")
    if result.fixed_inclination_deg is not None:
        title += "\n" + result._inclination_note()
    figure.suptitle(title)
    return figure


def _read_workflow_identity(directory, source_id):
    """Require an existing workflow-owned folder for this exact source."""
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Choose an existing workflow run directory, not a symlink.")
    settings_path = directory / "run.json"
    if settings_path.is_symlink() or not settings_path.is_file():
        raise ValueError("The run directory must contain its original run.json.")
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    if (settings.get("schema") != "dingo-workflow-v1"
            or settings.get("source_id") != source_id):
        raise ValueError("The run directory belongs to a different source or workflow.")


def _product_names(values):
    """Validate a manifest's flat filenames before touching any paths."""
    if (not isinstance(values, list) or not all(isinstance(value, str) for value in values)
            or len(set(values)) != len(values)):
        raise ValueError("The products manifest must list distinct filenames.")
    for name in values:
        if (not isinstance(name, str) or not name or Path(name).name != name
                or name in {".", ".."} or name in _PROTECTED_NAMES
                or name.startswith((".", "_"))
                or "/" in name or "\\" in name):
            raise ValueError("The products manifest contains a protected or invalid filename.")
    return set(values)


def _existing_products(directory, source_id):
    """Read ownership and refuse to claim arbitrary existing products."""
    manifest_path = directory / "_products.json"
    if manifest_path.is_symlink():
        raise ValueError("The products manifest must not be a symlink.")
    owned = set()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("schema") != PRODUCTS_SCHEMA
                or manifest.get("source_id") != source_id):
            raise ValueError("The products manifest belongs to another source or workflow.")
        owned = _product_names(manifest.get("files"))
        if not {"results.npz", "run_info.txt"}.issubset(owned):
            raise ValueError("The products manifest is missing required final products.")

    # The workflow writes this human-readable summary before fitting begins.
    # It is the only pre-existing product we may replace without a manifest.
    if (directory / "run_info.txt").exists():
        owned.add("run_info.txt")

    for name in owned:
        path = directory / name
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"Refusing to replace a non-file product: {path}")

    results_path = directory / "results.npz"
    if results_path.exists():
        if "results.npz" not in owned or results_path.is_symlink():
            raise ValueError("Existing results.npz is not owned by this workflow export.")
        if read_metadata(results_path)["report"]["source_id"] != source_id:
            raise ValueError("Existing final results belong to another source.")
    return owned


def _workflow_header(directory, metadata):
    """Keep readable settings separate from the backend's scientific notes."""
    return ("DINGO WORKFLOW RUN\n"
            f"Run directory: {directory}\n"
            "Workflow settings and recorded history:\n"
            + json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False)
            + "\n")


def _update_staged_products(result, stage, directory, metadata, radii_pc):
    """Enrich an existing final bundle, retaining all backend caveats."""
    bundle_path = stage / "results.npz"
    with np.load(bundle_path, allow_pickle=False) as bundle:
        arrays = {name: bundle[name].copy() for name in bundle.files}
    bundle_metadata = json.loads(str(arrays["metadata_json"].item()))
    bundle_metadata["workflow"] = metadata

    _save_plot(all_walker_traces(result), stage, "traces")
    notes = [
        "DISPLAY DEFINITIONS",
        "Traces include every walker; the black line is the median over walkers at each step.",
        "Corner plots, summary intervals and profiles pool all retained samples.",
        "Residual images use the highest-posterior retained sample, which need not represent the pooled median.",
        "Distinct solutions are not automatically identified, selected or assigned posterior probabilities.",
    ]
    if radii_pc is not None:
        # This is the same paired-draw calculation used by the legacy exporter.
        # A supplied range is not evidence that the data constrain every radius.
        profiles = result.radial_profiles(radii_pc)
        arrays.update(
            radius_pc=profiles["radius_pc"],
            vsini_kms_p16_p50_p84=profiles["vsini_kms"],
            vrot_kms_p16_p50_p84=profiles["vrot_kms"],
            mass_msun_p16_p50_p84=profiles["mass_msun"],
        )
        rotation_name = ("rotation_curve" if result.velocity_parameter == "V_rot"
                         else "projected_rotation_curve")
        figures = {
            rotation_name: lambda: result.rotation_curve(radii_pc=radii_pc),
            "mass_profile": lambda: result.mass_profile(radii_pc=radii_pc),
        }
        for name, make_figure in figures.items():
            figure = make_figure()
            for annotation in figure.texts:
                annotation.set_text(annotation.get_text().replace(
                    "Illustrative radius range", "User-supplied radius range"))
            _save_plot(figure, stage, name)
        bundle_metadata["export"]["profile_note"] = (
            bundle_metadata["export"]["profile_note"].replace(
                "Illustrative radius range", "User-supplied radius range"))
        bundle_metadata["export"]["profile_radius_source"] = (
            "User-supplied radii in pc; observed radial coverage not established.")
        notes += [
            f"Profile radii: {profiles['radius_pc'][0]:g}–{profiles['radius_pc'][-1]:g} pc "
            f"({len(profiles['radius_pc'])} user-supplied radii).",
            "These radii are not automatically established as observed or resolved.",
            "Mass remains spherical-equivalent r Vrot^2/G without a pressure-support correction.",
        ]
    else:
        notes.append("Profile radii use the legacy illustrative range; observed radial coverage is not established.")

    original_info = (stage / "run_info.txt").read_text(encoding="utf-8")
    (stage / "run_info.txt").write_text(
        _workflow_header(directory, metadata) + POSTERIOR_DETAILS_MARKER
        + original_info + "\n" + "\n".join(notes) + "\n",
        encoding="utf-8",
    )
    bundle_metadata["export"]["files"] = sorted(path.name for path in stage.iterdir())
    arrays["metadata_json"] = np.asarray(json.dumps(bundle_metadata, allow_nan=False))
    np.savez_compressed(bundle_path, **arrays)


def _publish_products(stage, directory, source_id, temporary):
    """Replace owned flat products, rolling back if any publication step fails.

    The caller holds the run lock. This transaction protects against Python
    exceptions during publication; it does not promise a multi-file atomic
    snapshot to readers bypassing that lock or protection from power loss.
    """
    owned = _existing_products(directory, source_id)
    new_names = _product_names([path.name for path in stage.iterdir()])
    for name in new_names:
        destination = directory / name
        if destination.is_symlink() or (destination.exists() and name not in owned):
            raise ValueError(f"An unrelated file would be overwritten: {destination}")

    manifest = {"schema": PRODUCTS_SCHEMA, "source_id": source_id,
                "files": sorted(new_names)}
    (stage / "_products.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    backup = temporary / "previous_products"
    backup.mkdir()
    moved_previous = []
    published = []
    previous_names = owned | {"_products.json"}
    try:
        for name in sorted(previous_names):
            source = directory / name
            if source.exists():
                source.replace(backup / name)
                moved_previous.append(name)
        # Publish ownership last, after every product has reached its destination.
        for name in [*sorted(new_names), "_products.json"]:
            (stage / name).replace(directory / name)
            published.append(name)
    except BaseException as publication_error:
        try:
            for name in reversed(published):
                (directory / name).unlink()
            for name in moved_previous:
                (backup / name).replace(directory / name)
        except BaseException as rollback_error:
            raise _PublicationRollbackError(
                f"Final-product publication and rollback failed. Recoverable files "
                f"are retained in {temporary}. Publication error: {publication_error}; "
                f"rollback error: {rollback_error}"
            ) from publication_error
        raise


def export_run_products(result, run_directory, workflow_metadata, *,
                        comparison_radius_pc=None, radii_pc=None):
    """Publish a completed posterior inside its existing workflow run folder.

    Parameters
    ----------
    result : ThreeFileResults
        Completed posterior loaded from the fitting cache. The underlying
        exporter checks samples and predictions before publishing anything.
    run_directory : path-like
        Existing ``WORKING_DIR/source_id/run_name`` directory with a matching
        ``run.json``. Its checkpoints, settings and unrelated notes are kept.
    workflow_metadata : mapping
        JSON-serializable settings and run history copied into ``results.npz``
        and the human-readable ``run_info.txt``. No implicit global state is
        consulted. A supplied ``source_id`` must match the posterior.
    comparison_radius_pc : float, optional
        Radius used by the velocity summary and residual annotations. This
        does not choose the profile's radial grid.
    radii_pc : sequence of float, optional
        Explicit disk-plane profile radii. They replace the illustrative grid
        in the saved arrays and rotation plot and enable a mass-profile PNG.
        Their inclusion does not establish observed radial coverage.

    Returns
    -------
    pathlib.Path
        The run directory containing final arrays, figures and run information.

    Notes
    -----
    The caller must hold the workflow run lock. No fitting is launched. A
    rendering or publication failure preserves the previous owned products;
    checkpoints are never moved. Final products are tracked in
    ``_products.json`` so unowned files cannot be silently overwritten.
    """
    directory = Path(run_directory).expanduser().absolute()
    _read_workflow_identity(directory, result.source_id)
    _existing_products(directory, result.source_id)
    if not isinstance(workflow_metadata, Mapping):
        raise ValueError("workflow_metadata must be a JSON-serializable mapping.")
    metadata = json.loads(json.dumps(dict(workflow_metadata), allow_nan=False))
    if metadata.get("source_id", result.source_id) != result.source_id:
        raise ValueError("Workflow metadata belongs to a different source.")
    if radii_pc is not None:
        radii_pc = np.asarray(radii_pc, dtype=float)
        if (radii_pc.ndim != 1 or len(radii_pc) < 2
                or not np.isfinite(radii_pc).all() or np.any(radii_pc < 0)
                or np.any(np.diff(radii_pc) <= 0)):
            raise ValueError("Use at least two finite, nonnegative, increasing radii in pc.")

    # Stage beside the run, not inside its checkpoint tree. The legacy exporter
    # deliberately prohibits final folders nested within their source cache.
    temporary = Path(tempfile.mkdtemp(prefix=".dingo_products_", dir=directory.parent))
    keep_backup = False
    try:
        stage = export_final(result, temporary / "bundle",
                             radius_pc=comparison_radius_pc)
        _update_staged_products(result, stage, directory, metadata, radii_pc)
        _read_workflow_identity(directory, result.source_id)
        _publish_products(stage, directory, result.source_id, temporary)
    except _PublicationRollbackError:
        keep_backup = True
        raise
    finally:
        if not keep_backup:
            shutil.rmtree(temporary)
    return directory
