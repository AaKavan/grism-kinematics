"""Workflow exports preserve checkpoints and make their display choices explicit."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from dingo import _workflow_products as products
from dingo import three_file, three_file_export


@pytest.fixture
def workflow_run(tmp_path, monkeypatch):
    """A small completed posterior inside the proposed per-run directory tree."""
    directory = tmp_path / "working" / "15665" / "15665_fixinc_i30_run01"
    cache = directory / "_checkpoints" / "fit"
    cache.mkdir(parents=True)
    (directory / "run.json").write_text(json.dumps({
        "schema": "dingo-workflow-v1", "source_id": 15665, "mode": "fixed",
    }))
    (directory / "run_info.txt").write_text("Run started; keep this if export fails.\n")
    (directory / "inspection_notes.txt").write_text("My scientific inspection notes.\n")
    (cache / "checkpoint.npz").write_bytes(b"Original sampler checkpoint")
    names = ["velocity.V_rot", "velocity.R_v", "velocity.theta_v",
             "velocity.x0_v", "velocity.y0_v", "image.R.dx",
             "image.R.dy", "noise.ln_f"]
    base = np.array([400., 3., .7, 41., 39., .2, -.1, .1])
    chain = np.broadcast_to(base, (20, 12, 8)).copy()
    chain[..., 0] += np.arange(240).reshape(20, 12) / 20
    chain[..., 1] += np.arange(240).reshape(20, 12) / 1000
    np.savez(cache / "samples.npz", chain=chain, param_names=np.asarray(names),
             log_prob=np.ones((20, 12)), acceptance_fraction=np.full(12, .2))
    image = np.arange(25.).reshape(5, 5)
    np.savez(cache / "predictions.npz", image=image, observed_R=image,
             observed_C=image + .1, rectified_R=image, rectified_C=image + .2,
             difference=np.full((5, 5), -.2), velocity_R=image * 0,
             velocity_C=image * 0, mask=np.ones((5, 5), bool),
             variance=np.ones((5, 5)), theta=chain[0, 0], param_names=np.asarray(names))
    report = dict(
        schema="dingo-three-file-v1", source_id=15665, status="complete",
        intrinsic_dispersion_kms=0., redshift=1.1147, pixel_scale_arcsec=.063,
        r_fit_pixels=40, walkers=12, draws=20, warmup=100, param_names=names,
        sampling_pass=False, velocity_parameter="V_rot", inclination_mode="fixed",
        fixed_inclination_deg=30., fixed_params={"velocity.inc_v": np.deg2rad(30.)},
        samples_sha256=three_file_export.sha256(cache / "samples.npz"),
        predictions_sha256=three_file_export.sha256(cache / "predictions.npz"),
    )
    (cache / "report.json").write_text(json.dumps(report))

    # Exercise the real exporter/bundle checks, but avoid repeating its heavy
    # corner rendering in every filesystem-transaction test.
    def render(result, stage, radius_pc, imaging):
        for name in ("fit", "corner", "traces", "rotation_curve", "inputs"):
            (stage / f"{name}.png").write_bytes(b"Fixture image")
    monkeypatch.setattr(three_file_export, "_render_plots", render)
    result = three_file.ThreeFileResults(cache, 15665)
    metadata = {"source_id": 15665, "mode": "fixed", "inclination_deg": 30,
                "draws": 20, "notes": "Conditional inclination experiment"}
    return result, directory, metadata


def file_bytes(directory):
    return {str(path.relative_to(directory)): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


def test_all_walker_traces_include_minority_walkers_and_pooled_median(workflow_run):
    result, _, _ = workflow_run
    values = result._columns(("vrot",))["vrot"][0]
    figure = products.all_walker_traces(result, parameters=("vrot", "inclination"))
    try:
        assert len(figure.axes) == 1  # Fixed inclination is not a sampled dimension.
        lines = figure.axes[0].lines
        assert len(lines) == result.report["walkers"] + 1
        for walker, line in enumerate(lines[:-1]):
            np.testing.assert_array_equal(line.get_ydata(), values[:, walker])
        np.testing.assert_array_equal(lines[-1].get_ydata(), np.median(values, axis=1))
        title = figure._suptitle.get_text()
        assert "all walkers" in title and "pooled ensemble median" in title
        assert "not independent chains" in title
        assert "Fixed i = 30" in title
    finally:
        plt.close(figure)


def test_export_preserves_cache_settings_notes_and_reloads_without_cache(workflow_run):
    result, directory, metadata = workflow_run
    original = file_bytes(directory)
    output = products.export_run_products(result, directory, metadata)
    assert output == directory
    for name, value in original.items():
        if name != "run_info.txt":
            assert (directory / name).read_bytes() == value
    final = three_file.ThreeFileResults(directory, 15665, final_bundle=True)
    np.testing.assert_array_equal(final._samples()[0], result._samples()[0])
    saved = three_file_export.read_metadata(directory / "results.npz")
    assert saved["workflow"] == metadata
    assert "mass_profile.png" not in saved["export"]["files"]
    assert not (directory / "mass_profile.png").exists()
    text = (directory / "run_info.txt").read_text()
    assert products.POSTERIOR_DETAILS_MARKER in text
    assert "Intervals are conditional on this fixed inclination" in text
    assert "highest-posterior retained sample" in text
    assert "Traces include every walker" in text
    manifest = json.loads((directory / "_products.json").read_text())
    assert set(manifest["files"]) == set(saved["export"]["files"])
    assert "run.json" not in manifest["files"]


def test_explicit_radii_update_paired_arrays_and_mass_plot(workflow_run):
    result, directory, metadata = workflow_run
    radii = np.array([0., 500., 1500., 2500.])
    expected = result.radial_profiles(radii)
    products.export_run_products(result, directory, metadata,
                                 comparison_radius_pc=1500., radii_pc=radii)
    with np.load(directory / "results.npz", allow_pickle=False) as bundle:
        np.testing.assert_array_equal(bundle["radius_pc"], radii)
        np.testing.assert_allclose(bundle["mass_msun_p16_p50_p84"], expected["mass_msun"])
        np.testing.assert_allclose(bundle["vrot_kms_p16_p50_p84"], expected["vrot_kms"])
        saved = json.loads(bundle["metadata_json"].item())
    assert saved["export"]["comparison_radius_pc"] == 1500.
    assert "User-supplied" in saved["export"]["profile_radius_source"]
    assert "observed radial coverage not established" in saved["export"]["profile_radius_source"]
    assert (directory / "mass_profile.png").stat().st_size > 100

    # A later default export removes only the now-stale owned mass plot.
    products.export_run_products(result, directory, metadata)
    assert not (directory / "mass_profile.png").exists()
    assert (directory / "inspection_notes.txt").exists()


@pytest.mark.parametrize("filename", ["corner.png", "results.npz"])
def test_unowned_product_collision_never_overwrites_files(workflow_run, filename):
    result, directory, metadata = workflow_run
    (directory / filename).write_bytes(b"An unrelated file")
    original = file_bytes(directory)
    with pytest.raises(ValueError, match="unrelated|not owned"):
        products.export_run_products(result, directory, metadata)
    assert file_bytes(directory) == original


@pytest.mark.parametrize("location", ["run", "metadata", "manifest", "bundle"])
def test_cross_source_export_is_rejected(workflow_run, location):
    result, directory, metadata = workflow_run
    if location in {"manifest", "bundle"}:
        products.export_run_products(result, directory, metadata)
    if location == "run":
        path = directory / "run.json"
        changed = json.loads(path.read_text())
        changed["source_id"] = 3700
        path.write_text(json.dumps(changed))
    elif location == "metadata":
        metadata["source_id"] = 3700
    elif location == "manifest":
        path = directory / "_products.json"
        changed = json.loads(path.read_text())
        changed["source_id"] = 3700
        path.write_text(json.dumps(changed))
    else:
        path = directory / "results.npz"
        with np.load(path, allow_pickle=False) as bundle:
            arrays = {key: bundle[key] for key in bundle.files}
        changed = json.loads(arrays["metadata_json"].item())
        changed["report"]["source_id"] = 3700
        arrays["metadata_json"] = np.asarray(json.dumps(changed))
        np.savez_compressed(path, **arrays)
    original = file_bytes(directory)
    with pytest.raises(ValueError, match="source"):
        products.export_run_products(result, directory, metadata)
    assert file_bytes(directory) == original


def test_failed_render_leaves_previous_final_products_intact(workflow_run, monkeypatch):
    result, directory, metadata = workflow_run
    products.export_run_products(result, directory, metadata)
    original = file_bytes(directory)
    def fail_render(*args, **kwargs):
        raise RuntimeError("A plotting dependency failed")
    monkeypatch.setattr(products, "all_walker_traces", fail_render)
    with pytest.raises(RuntimeError, match="plotting dependency"):
        products.export_run_products(result, directory, metadata)
    assert file_bytes(directory) == original


@pytest.mark.parametrize("first_export", [True, False])
def test_failed_publication_rolls_back_every_owned_file(workflow_run, monkeypatch, first_export):
    result, directory, metadata = workflow_run
    if not first_export:
        products.export_run_products(result, directory, metadata)
    original = file_bytes(directory)
    original_replace = Path.replace
    failed = False

    def fail_one_publish(path, target):
        nonlocal failed
        if Path(target) == directory / "traces.png" and not failed:
            failed = True
            raise OSError("Simulated final-file publication failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_one_publish)
    with pytest.raises(OSError, match="publication failure"):
        products.export_run_products(result, directory, metadata)
    assert failed
    assert file_bytes(directory) == original
    assert not list(directory.parent.glob(".dingo_products_*"))


@pytest.mark.parametrize("radii", [[1.], [0., 2., 1.], [0., np.nan], [-1., 2.]])
def test_invalid_profile_grid_does_not_touch_any_files(workflow_run, radii):
    result, directory, metadata = workflow_run
    original = file_bytes(directory)
    with pytest.raises(ValueError, match="radii"):
        products.export_run_products(result, directory, metadata, radii_pc=radii)
    assert file_bytes(directory) == original


def test_protected_or_traversal_manifest_cannot_claim_settings(workflow_run):
    result, directory, metadata = workflow_run
    (directory / "_products.json").write_text(json.dumps({
        "schema": products.PRODUCTS_SCHEMA, "source_id": 15665,
        "files": ["results.npz", "run_info.txt", "../run.json"],
    }))
    original = file_bytes(directory)
    with pytest.raises(ValueError, match="protected or invalid"):
        products.export_run_products(result, directory, metadata)
    assert file_bytes(directory) == original


def test_failed_rollback_keeps_recoverable_backups(workflow_run, monkeypatch):
    result, directory, metadata = workflow_run
    products.export_run_products(result, directory, metadata)
    original_result = (directory / "results.npz").read_bytes()
    original_replace = Path.replace

    def fail_publish_and_rollback(path, target):
        if Path(target) == directory / "traces.png":
            raise OSError("Simulated publication failure")
        if path.parent.name == "previous_products":
            raise OSError("Simulated rollback failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_publish_and_rollback)
    with pytest.raises(RuntimeError, match="Recoverable files are retained"):
        products.export_run_products(result, directory, metadata)
    backups = list(directory.parent.glob(".dingo_products_*/previous_products"))
    assert len(backups) == 1
    assert (backups[0] / "results.npz").read_bytes() == original_result
    assert (directory / "run.json").exists()
    assert (directory / "_checkpoints" / "fit" / "checkpoint.npz").read_bytes() == b"Original sampler checkpoint"


def test_unowned_symlink_is_never_followed_or_overwritten(workflow_run, tmp_path):
    result, directory, metadata = workflow_run
    outside = tmp_path / "external_notes.txt"
    outside.write_text("Keep these unrelated notes")
    (directory / "corner.png").symlink_to(outside)
    with pytest.raises(ValueError, match="unrelated file"):
        products.export_run_products(result, directory, metadata)
    assert (directory / "corner.png").is_symlink()
    assert outside.read_text() == "Keep these unrelated notes"
