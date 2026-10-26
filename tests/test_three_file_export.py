"""Final source folders retain paired samples and survive failed replacement."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from dingo import three_file, three_file_export as exporting


@pytest.fixture
def completed_run(tmp_path, monkeypatch):
    path = tmp_path/"cache"/"ID6170"/"run"
    path.mkdir(parents=True)
    names = ["velocity.V_sini", "velocity.R_v", "velocity.inc_v", "velocity.theta_v",
             "velocity.x0_v", "velocity.y0_v", "image.R.dx", "image.R.dy", "noise.ln_f"]
    base = np.array([150., 3., np.pi/3, .5, 40., 40., .2, .1, .1])
    chain = np.broadcast_to(base, (20, 4, 9)).copy()
    chain[..., 0] += np.arange(80).reshape(20, 4)
    chain[..., 1] += np.arange(80).reshape(20, 4)/100
    np.savez(path/"samples.npz", chain=chain, param_names=np.asarray(names),
             log_prob=np.ones((20, 4)), warmup=100, acceptance_fraction=np.full(4, .25))
    image = np.arange(25.).reshape(5, 5)
    np.savez(path/"predictions.npz", image=image, observed_R=image, observed_C=image+.1,
        rectified_R=image, rectified_C=image+.2, difference=np.full((5, 5), -.2),
        velocity_R=image*0, velocity_C=image*0, mask=np.ones((5, 5), bool),
        variance=np.ones((5, 5)), theta=chain[0, 0], param_names=np.asarray(names))
    report = dict(schema="dingo-three-file-v1", source_id=6170, status="complete",
        intrinsic_dispersion_kms=0., redshift=1.1147, pixel_scale_arcsec=.063,
        r_fit_pixels=40, walkers=4, draws=20, warmup=100, param_names=names,
        sampling_pass=False, fixed_params={}, chosen_candidate_id=1,
        noise_note="Approximate noise; conditional intervals",
        samples_sha256=exporting.sha256(path/"samples.npz"),
        predictions_sha256=exporting.sha256(path/"predictions.npz"))
    (path/"report.json").write_text(json.dumps(report))
    (path/"checkpoint.npz").write_bytes(b"keep checkpoint")
    (path/"run.log").write_text("keep log")
    def render(result, directory, radius, imaging):
        for name in ("fit", "corner", "traces", "projected_rotation_curve", "inputs"):
            (directory/f"{name}.png").write_bytes(b"plot fixture")
    monkeypatch.setattr(exporting, "_render_plots", render)
    return three_file.ThreeFileResults(path, 6170), tmp_path/"working", chain


def hashes(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob('*') if p.is_file()}


def test_manual_start_export_identifies_start_separately_from_prior(completed_run):
    result, root, _ = completed_run
    result.report.update(seed_kind="manual_start", start_velocity_kms=500.,
        start_inclination_deg=60., base_candidate_id=8, parent_run="/cache/parent",
        inclination_prior=dict(type="gaussian", mu=np.deg2rad(70.), sigma=np.deg2rad(5.),
                               min=np.deg2rad(5.), max=np.deg2rad(85.)))
    output = result.export(root)
    text = (output/"run_info.txt").read_text()
    assert "Manual starting V_inf: 500 km/s; starting inclination: 60 deg; both sampled" in text
    assert "mu=70.000 deg, sigma=5.000 deg" in text
    assert "do not change the prior or fix either parameter" in text
    assert "parent candidate: 8" in text
    assert "Selected candidate:" not in text


def test_fixed_inclination_export_preserves_angle_and_conditional_uncertainties(completed_run):
    result, root, original_chain = completed_run
    names = [name.replace("velocity.V_sini", "velocity.V_rot")
             for name in result.report["param_names"] if name != "velocity.inc_v"]
    chain = np.delete(original_chain, 2, axis=-1)
    for filename in ("samples.npz", "predictions.npz"):
        with np.load(result.path/filename, allow_pickle=False) as saved:
            arrays = {key: saved[key].copy() for key in saved.files}
        arrays["param_names"] = np.asarray(names)
        if filename == "samples.npz":
            arrays["chain"] = chain
        else:
            arrays["theta"] = chain[0, 0]
        np.savez(result.path/filename, **arrays)
    result.report.update(param_names=names, velocity_parameter="V_rot",
        seed_kind="manual_start", manual_start_schema="dingo-three-file-manual-fixed-inclination-v1",
        inclination_mode="fixed", fixed_inclination_deg=30.,
        fixed_params={"velocity.inc_v": np.deg2rad(30.)},
        inclination_prior=dict(type="fixed", value=np.deg2rad(30.)),
        parent_inclination_prior=dict(type="gaussian", mu=np.deg2rad(32.), sigma=np.deg2rad(5.)),
        start_velocity_kms=500., start_inclination_deg=30., base_candidate_id=8,
        parent_run="/cache/parent",
        samples_sha256=exporting.sha256(result.path/"samples.npz"),
        predictions_sha256=exporting.sha256(result.path/"predictions.npz"))
    (result.path/"report.json").write_text(json.dumps(result.report))
    before = hashes(result.path)
    output = result.export(root)
    assert hashes(result.path) == before
    metadata = exporting.read_metadata(output/"results.npz")
    assert metadata["export"]["inclination_mode"] == "fixed"
    assert metadata["export"]["fixed_inclination_deg"] == pytest.approx(30.)
    assert "exclude inclination uncertainty" in metadata["export"]["inclination_uncertainty_note"]
    assert "velocity.inc_v" not in metadata["export"]["chain_parameter_units"]
    assert metadata["report"]["fixed_params"]["velocity.inc_v"] == np.deg2rad(30.)
    text = (output/"run_info.txt").read_text()
    assert "Fixed inclination: 30 deg (not sampled)." in text
    assert "conditional on this fixed inclination" in text
    assert "parent inclination prior is recorded as provenance and is not applied" in text
    assert "both sampled" not in text
    assert "Inclination and the other fitted parameters remain free" not in text
    final = three_file.load_final(6170, working_dir=root)
    assert "inclination" not in final.default_parameters()
    np.testing.assert_allclose(final._columns()["vsini"][0], .5*chain[..., 0])
    assert "Inclination [deg]" not in final.velocity_summary().index
    assert "Fixed i = 30°" in final.status


def test_scan_prior_export_is_separate_and_identifies_sampled_prior(completed_run):
    from dingo.inclination_posterior import final_root
    result, root, _ = completed_run
    original = result.export(root)
    before = hashes(original)
    result.report.update(seed_kind="inclination_scan", seed_inclination_deg=30.,
        parent_run="/cache/parent", scan_dir="/cache/parent_inclination_scan",
        inclination_prior=dict(type="gaussian", mu=np.deg2rad(30.), sigma=np.deg2rad(10.),
                               min=np.deg2rad(5.), max=np.deg2rad(85.)))
    chosen_root = final_root(root, run_name="broad", inclination_deg=30.)
    separate = result.export(chosen_root)
    assert separate == root/"inclination_prior_runs"/"broad_i30_s10"/"ID6170"
    assert hashes(original) == before
    text = (separate/"run_info.txt").read_text()
    assert "inclination scan: 30 deg; inclination sampled" in text
    assert "sigma=10.000 deg" in text
    assert "does not fix inclination" in text
    assert "Selected candidate:" not in text
    assert "Inclination sampled with Gaussian prior 30° ± 10°" in result.status


def test_inherited_imaging_is_recognized_as_context_without_substitution(completed_run, tmp_path):
    result, _, _ = completed_run
    # A missing original file should be reported missing, not silently replaced
    # by another fit or mistaken for absent imaging provenance.
    result.report["inclination_metadata"] = dict(kind="user_gaussian_inclination_prior",
        inherited_imaging_metadata=dict(sources=[
            dict(path=str(tmp_path/"imaging"/variant/"samples.npz"), variant=variant,
                 samples_sha256="0"*64)
            for variant in ("full_pixel", "core_pixel_clean")]))
    imaging, arrays, metadata = exporting._imaging_products(result)
    assert imaging is None and arrays == {}
    assert metadata["status"] == "original imaging files unavailable"
    assert metadata["file_checks"][0]["actual_sha256"] is None
    assert metadata["file_checks"][0]["expected_sha256"] == "0"*64
    assert metadata["file_checks"][0]["status"] == "unavailable"


@pytest.fixture
def recorded_imaging(completed_run, tmp_path):
    result, _, _ = completed_run
    root = tmp_path/"imaging"
    sources = []
    for variant in ("full_pixel", "core_pixel_clean"):
        path = root/variant
        path.mkdir(parents=True)
        samples = {name: np.full((2, 10), value) for name, value in
                   dict(xc=2., yc=2., flux=10., r_eff=2., n=1., ellip=.2,
                        theta=.5, sky_back=0.).items()}
        np.savez(path/"samples.npz", **samples)
        image = np.arange(25.).reshape(5, 5)
        np.savez(path/"fit.npz", image=image, model=image+.2, residual=image*0-.2,
                 error=np.ones_like(image), mask=np.zeros_like(image, dtype=bool))
        report = dict(source_id=result.source_id, completed=True,
                      diagnostics={"convergence_gate_passed": True},
                      orientation_parameterization_ok=True,
                      samples_sha256=exporting.sha256(path/"samples.npz"),
                      fit_sha256=exporting.sha256(path/"fit.npz"))
        (path/"imaging.json").write_text(json.dumps(report))
        sources.append(dict(path=str(path/"samples.npz"), variant=variant,
                            samples_sha256=report["samples_sha256"]))
    result.report["inclination_metadata"] = dict(sources=sources)
    result.report["inclination_prior"] = dict(type="gaussian", mu=np.deg2rad(70.),
        sigma=np.deg2rad(5.), min=np.deg2rad(5.), max=np.deg2rad(85.))
    return root


def test_unchanged_optional_imaging_is_included(completed_run, recorded_imaging):
    result, root, _ = completed_run
    imaging, arrays, metadata = exporting._imaging_products(result)
    assert imaging.path == recorded_imaging
    assert metadata["status"] == "included"
    assert all(check["status"] == "unchanged" for check in metadata["file_checks"])
    for variant in ("full_pixel", "core_pixel_clean"):
        assert f"imaging_{variant}_samples_ellip" in arrays
        assert f"imaging_{variant}_fit_image" in arrays
    output = result.export(root)
    with np.load(output/"results.npz", allow_pickle=False) as bundle:
        assert "imaging_full_pixel_samples_ellip" in bundle


def _declare_single_imaging(result, imaging_root):
    """Select one saved posterior and remove the unselected test fixture."""
    import shutil
    shutil.rmtree(imaging_root/"core_pixel_clean")
    (imaging_root/"imaging_variants.json").write_text(json.dumps(
        dict(source_id=result.source_id, variants=["full_pixel"])))
    result.report["inclination_metadata"]["sources"] = [
        item for item in result.report["inclination_metadata"]["sources"]
        if item["variant"] == "full_pixel"]


def test_declared_single_imaging_is_exported_with_inclination_and_samples(
        completed_run, recorded_imaging):
    result, root, _ = completed_run
    _declare_single_imaging(result, recorded_imaging)
    before = hashes(recorded_imaging)
    imaging, arrays, metadata = exporting._imaging_products(result)
    assert imaging.variants == (("full_pixel", 0.),)
    assert metadata["status"] == "included"
    assert all(check["status"] == "unchanged" for check in metadata["file_checks"])
    assert "imaging_full_pixel_samples_ellip" in arrays
    assert "imaging_full_pixel_fit_image" in arrays
    np.testing.assert_array_equal(arrays["imaging_full_pixel_inclination_q0"], [0., .2])
    summary = arrays["imaging_full_pixel_inclination_summary"]
    assert summary.shape == (2, 4) and np.isfinite(summary).all()
    assert not any("core_pixel_clean" in key for key in arrays)
    output = result.export(root)
    with np.load(output/"results.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["imaging_full_pixel_samples_ellip"], .2*np.ones((2, 10)))
        np.testing.assert_array_equal(saved["imaging_full_pixel_inclination_summary"], summary)
        assert not any("core_pixel_clean" in key for key in saved.files)
    assert hashes(recorded_imaging) == before


def test_legacy_directory_with_only_one_recorded_variant_is_not_reinterpreted(
        completed_run, recorded_imaging):
    result, _, _ = completed_run
    _declare_single_imaging(result, recorded_imaging)
    (recorded_imaging/"imaging_variants.json").unlink()
    imaging, arrays, metadata = exporting._imaging_products(result)
    assert imaging is None and arrays == {}
    assert metadata["status"] == "omitted: original imaging files changed or incomplete"
    assert "declared selection" in metadata["detail"]
    assert not (recorded_imaging/"imaging_variants.json").exists()


def test_declared_selection_must_match_the_recorded_imaging_prior(
        completed_run, recorded_imaging):
    result, _, _ = completed_run
    # Existing files alone do not authorize dropping a variant from the prior
    # recorded when the kinematic fit ran.
    (recorded_imaging/"imaging_variants.json").write_text(json.dumps(
        dict(source_id=result.source_id, variants=["full_pixel"])))
    imaging, arrays, metadata = exporting._imaging_products(result)
    assert imaging is None and arrays == {}
    assert "declared selection" in metadata["detail"]
    assert len(metadata["recorded_sources"]) == 2


@pytest.mark.parametrize("damage", ["missing", "changed", "wrong_source", "wrong_path"])
def test_declared_single_export_keeps_integrity_and_missing_file_status(
        completed_run, recorded_imaging, damage):
    result, _, _ = completed_run
    _declare_single_imaging(result, recorded_imaging)
    sample_path = recorded_imaging/"full_pixel"/"samples.npz"
    if damage == "missing":
        import shutil
        shutil.rmtree(recorded_imaging)
    elif damage == "changed":
        with sample_path.open("ab") as stream:
            stream.write(b"changed posterior")
    elif damage == "wrong_source":
        (recorded_imaging/"imaging_variants.json").write_text(json.dumps(
            dict(source_id=15665, variants=["full_pixel"])))
    else:
        result.report["inclination_metadata"]["sources"][0]["variant"] = "core_pixel_clean"
    imaging, arrays, metadata = exporting._imaging_products(result)
    assert imaging is None and arrays == {}
    if damage == "missing":
        assert metadata["status"] == "original imaging files unavailable"
        assert metadata["file_checks"][0]["status"] == "unavailable"
    else:
        assert metadata["status"] == "omitted: original imaging files changed or incomplete"
    if damage == "changed":
        assert metadata["file_checks"][0]["status"] == "changed"


def test_fixed_inclination_inherited_imaging_is_context_only(completed_run, recorded_imaging):
    result, _, _ = completed_run
    original = result.report["inclination_metadata"]
    result.report["inclination_metadata"] = dict(kind="user_fixed_inclination",
        fixed_inclination_deg=30., inherited_imaging_metadata=dict(
            kind="user_gaussian_inclination_prior", inherited_imaging_metadata=original))
    imaging, arrays, metadata = exporting._imaging_products(result)
    assert imaging.path == recorded_imaging
    assert "imaging_full_pixel_samples_ellip" in arrays
    assert metadata["status"] == "included as context; inclination fixed by user, not an imaging prior"
    (recorded_imaging/"core_pixel_clean"/"samples.npz").unlink()
    imaging, arrays, metadata = exporting._imaging_products(result)
    assert imaging is None and not arrays
    assert metadata["status"] == "original imaging files unavailable"
    assert metadata["recorded_sources"] == original["sources"]


@pytest.mark.parametrize("damage", ["samples_changed", "samples_missing", "fit_changed",
                                     "fit_missing", "fit_incomplete", "not_ready"])
def test_unavailable_optional_imaging_does_not_block_valid_posterior(
        completed_run, recorded_imaging, monkeypatch, damage):
    result, root, chain = completed_run
    # Use the second variant so partial exports would incorrectly include the
    # already-read first variant. No imaging products may survive this case.
    variant = recorded_imaging/"core_pixel_clean"
    filename = "samples.npz" if damage.startswith("samples") else "fit.npz"
    target = variant/filename
    if damage.endswith("changed"):
        with target.open("ab") as stream:
            stream.write(b"changed after the kinematic run")
    elif damage.endswith("missing"):
        target.unlink()
    elif damage == "fit_incomplete":
        # Even if an incomplete file matches its report, it cannot be shown as
        # a complete imaging prediction or copied into the final result.
        np.savez(target, image=np.ones((5, 5)))
        path = variant/"imaging.json"
        report = json.loads(path.read_text())
        report["fit_sha256"] = exporting.sha256(target)
        path.write_text(json.dumps(report))
    else:
        path = variant/"imaging.json"
        report = json.loads(path.read_text())
        report["completed"] = False
        path.write_text(json.dumps(report))
    original_report = json.loads(json.dumps(result.report))
    before = hashes(result.path)
    def render(result, directory, radius, imaging):
        assert imaging is None  # No substituted or partial PySersic figures.
        (directory/"fit.png").write_bytes(b"valid kinematic plot")
    monkeypatch.setattr(exporting, "_render_plots", render)
    output = result.export(root)
    assert hashes(result.path) == before
    assert result.report == original_report
    with np.load(output/"results.npz", allow_pickle=False) as bundle:
        np.testing.assert_array_equal(bundle["chain"], chain)
        assert not any(key.startswith("imaging_") for key in bundle.files)
        metadata = json.loads(bundle["metadata_json"].item())
    assert metadata["report"] == original_report  # Includes actual saved prior.
    imaging = metadata["imaging"]
    assert imaging["recorded_sources"] == result.report["inclination_metadata"]["sources"]
    assert "unchanged" in imaging["note"]
    if damage.endswith("changed"):
        check = next(check for check in imaging["file_checks"] if check["status"] == "changed")
        assert check["path"] == str(target)
        assert check["expected_sha256"] != check["actual_sha256"]
        assert check["actual_sha256"] == exporting.sha256(target)
    text = (output/"run_info.txt").read_text()
    assert "mu=70.000 deg, sigma=5.000 deg" in text
    assert "All imaging arrays and plots omitted" in text
    assert "PySersic export detail:" in text


@pytest.mark.parametrize("filename", ["samples.npz", "predictions.npz"])
def test_optional_imaging_failure_does_not_bypass_core_integrity(
        completed_run, recorded_imaging, filename):
    result, root, _ = completed_run
    output = result.export(root)
    before = hashes(output)
    (recorded_imaging/"full_pixel"/"samples.npz").unlink()
    with (result.path/filename).open("ab") as stream:
        stream.write(b"corrupt core result")
    with pytest.raises(ValueError, match="changed"):
        result.export(root)
    assert hashes(output) == before


def test_export_is_self_contained_preserves_draws_and_does_not_modify_cache(completed_run):
    result, root, chain = completed_run
    before = hashes(result.path)
    output = result.export(root, radius_pc=3000.)
    assert output == root/"ID6170"
    assert all(p.is_file() and (p.suffix in (".npz", ".png") or p.name == "run_info.txt")
               for p in output.iterdir())
    assert hashes(result.path) == before
    assert len(list(root.iterdir())) == 1
    with np.load(output/"results.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["chain"], chain)
        np.testing.assert_array_equal(data["fit_difference"], -.2*np.ones((5, 5)))
        metadata = json.loads(data["metadata_json"].item())
        assert metadata["report"] == result.report
        assert metadata["export"]["comparison_radius_pc"] == 3000.
        assert metadata["export"]["chain_parameter_units"]["velocity.inc_v"] == "radian"
        assert metadata["export"]["primary_velocity"] == "Vrot(r) sin(i), projected rotation at disk-plane radius r"
        assert not metadata["report"]["sampling_pass"]  # Do not promote provisional intervals.
        expected_profiles = result.radial_profiles(data["radius_pc"])
        np.testing.assert_array_equal(data["vsini_kms_p16_p50_p84"], expected_profiles["vsini_kms"])
        np.testing.assert_array_equal(data["vrot_kms_p16_p50_p84"], expected_profiles["vrot_kms"])
        np.testing.assert_array_equal(data["mass_msun_p16_p50_p84"], expected_profiles["mass_msun"])
        for key in data.files:
            assert data[key].dtype.kind != 'O'
        assert "Vrot(3000 pc) sin(i) [km/s]" in data["summary_names"]
        assert "Vrot(3000 pc) [km/s]" not in data["summary_names"]
    assert (output/"projected_rotation_curve.png").exists()
    assert not (output/"mass_profile.png").exists()
    expected_summary = result.velocity_summary(radius_pc=3000.)
    result.path.rename(result.path.with_name("cache_not_accessible"))
    final = three_file.load_final(6170, working_dir=root, candidate_id=1)
    np.testing.assert_array_equal(final._samples()[0], chain)
    np.testing.assert_array_equal(final._prediction_pixels()["difference"], -.2*np.ones((5, 5)))
    np.testing.assert_allclose(final.velocity_summary(radius_pc=3000.), expected_summary)
    # The final bundle is sufficient to recompute all profiles after losing the
    # original cache, including the retained optional mass/deprojection arrays.
    reloaded_profiles = final.radial_profiles(expected_profiles["radius_pc"])
    for key in ("vsini_kms", "vrot_kms", "mass_msun"):
        np.testing.assert_allclose(reloaded_profiles[key], expected_profiles[key])
    with pytest.raises(ValueError, match="different candidate"):
        three_file.load_final(6170, working_dir=root, candidate_id=0)


def test_deprojected_export_records_sampled_velocity_and_recovers_paired_projection(completed_run):
    result, root, original_chain = completed_run
    names = [name.replace("velocity.V_sini", "velocity.V_rot")
             for name in result.report["param_names"]]
    # Represent the same physical draws using genuinely deprojected amplitudes.
    chain = original_chain.copy()
    chain[..., 0] /= np.sin(chain[..., 2])
    for filename in ("samples.npz", "predictions.npz"):
        with np.load(result.path/filename, allow_pickle=False) as saved:
            arrays = {key: saved[key].copy() for key in saved.files}
        arrays["param_names"] = np.asarray(names)
        if filename == "samples.npz":
            arrays["chain"] = chain
        else:
            arrays["theta"] = chain[0, 0]
        np.savez(result.path/filename, **arrays)
    result.report.update(param_names=names,
        samples_sha256=exporting.sha256(result.path/"samples.npz"),
        predictions_sha256=exporting.sha256(result.path/"predictions.npz"))
    (result.path/"report.json").write_text(json.dumps(result.report))
    result = three_file.ThreeFileResults(result.path, 6170)
    before = hashes(result.path)
    output = result.export(root, radius_pc=3000.)
    assert hashes(result.path) == before
    metadata = exporting.read_metadata(output/"results.npz")
    exported = metadata["export"]
    assert exported["sampled_velocity_parameter"] == "velocity.V_rot"
    assert exported["chain_parameter_units"]["velocity.V_rot"] == "km/s"
    assert exported["primary_velocity"] == "Vrot(r), deprojected rotation at disk-plane radius r"
    assert "deprojected asymptotic" in exported["sampled_velocity_definition"]
    text = (output/"run_info.txt").read_text()
    assert "Primary result: deprojected rotation" in text
    assert "Vinf and inclination are separate sampled parameters" in text
    with np.load(output/"results.npz", allow_pickle=False) as bundle:
        np.testing.assert_array_equal(bundle["chain"], chain)
        assert bundle["summary_names"][0] == "Vinf [km/s]"
        assert "Vrot(3000 pc) [km/s]" in bundle["summary_names"]
        assert "Vrot(3000 pc) sin(i) [km/s]" in bundle["summary_names"]
    result.path.rename(result.path.with_name("cache_not_accessible"))
    final = three_file.load_final(6170, working_dir=root)
    np.testing.assert_allclose(final._columns()["vsini"][0], original_chain[..., 0])
    np.testing.assert_array_equal(final._columns()["vrot"][0], chain[..., 0])


@pytest.mark.parametrize("sampled_name, expected_filename", [
    ("velocity.V_sini", "projected_rotation_curve"),
    ("velocity.V_rot", "rotation_curve"),
])
def test_export_plot_filename_tracks_actual_sampled_amplitude(monkeypatch, tmp_path,
                                                             sampled_name, expected_filename):
    from types import SimpleNamespace
    produced = {}
    result = SimpleNamespace(report={"param_names": [sampled_name]},
        input_figure=lambda: "inputs", fit_figure=lambda **kwargs: "fit",
        center_figure=lambda **kwargs: "center", corner=lambda: "corner",
        traces=lambda: "traces", rotation_curve=lambda: "curve")
    monkeypatch.setattr(exporting, "_save_plot", lambda figure, directory, name:
                        produced.update({name: figure}))
    exporting._render_plots(result, tmp_path, None, None)
    assert produced[expected_filename] == "curve"
    assert len([name for name in produced if "rotation_curve" in name]) == 1


def test_failed_export_keeps_previous_final_files(completed_run, monkeypatch):
    result, root, _ = completed_run
    path = result.export(root)
    before = hashes(path)
    def fail(*args):
        raise RuntimeError("Rendering failed")
    monkeypatch.setattr(exporting, "_render_plots", fail)
    with pytest.raises(RuntimeError, match="Rendering failed"):
        result.export(root)
    assert hashes(path) == before
    assert list(root.iterdir()) == [path]
    assert not list(root.parent.glob('.dingo_export_*'))


def test_reexport_replaces_only_this_sources_owned_products(completed_run, monkeypatch):
    result, root, _ = completed_run
    path = result.export(root)
    other = root/"ID3700"
    other.mkdir()
    (other/"results.npz").write_bytes(b"different source must survive")
    monkeypatch.setattr(exporting, "_render_plots",
        lambda result, directory, radius, imaging: (directory/"fit.png").write_bytes(b"new plot"))
    result.export(root)
    assert {p.name for p in path.iterdir()} == {"results.npz", "run_info.txt", "fit.png"}
    assert (path/"fit.png").read_bytes() == b"new plot"
    assert (other/"results.npz").read_bytes() == b"different source must survive"


def test_foreign_files_block_replacement_without_deleting_anything(completed_run):
    result, root, _ = completed_run
    path = result.export(root)
    (path/"my_notes.txt").write_text("keep these")
    before = hashes(path)
    with pytest.raises(ValueError, match="Unrelated files"):
        result.export(root)
    assert hashes(path) == before


def test_incomplete_or_changed_run_cannot_replace_final_results(completed_run):
    result, root, _ = completed_run
    path = result.export(root)
    before = hashes(path)
    result.report["status"] = "interrupted"
    with pytest.raises(ValueError, match="not completed"):
        result.export(root)
    result.report["status"] = "complete"
    with (result.path/"samples.npz").open('ab') as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        result.export(root)
    assert hashes(path) == before


def test_export_cannot_replace_cache_or_an_unrelated_folder(completed_run):
    result, root, _ = completed_run
    with pytest.raises(ValueError, match="separate"):
        result.export(result.path)
    path = root/"ID6170"
    path.mkdir(parents=True)
    (path/"unrelated.png").write_bytes(b"keep")
    with pytest.raises(ValueError, match="unrelated files"):
        result.export(root)
    assert (path/"unrelated.png").read_bytes() == b"keep"


def test_run_info_uses_completion_time_not_export_time_and_actual_counts(completed_run):
    import os
    from datetime import datetime, timezone
    result, root, _ = completed_run
    finished = datetime(2026, 10, 20, 2, 12, 7, tzinfo=timezone.utc).timestamp()
    os.utime(result.path/"report.json", (finished, finished))
    (result.path/"run_config.yaml").write_text('mcmc:\n  seed: 42\n')
    result.report['elapsed_seconds'] = 668.4
    output = result.export(root)
    text = (output/"run_info.txt").read_text()
    assert 'Finished: 2026-10-19 19:12:07 MST (America/Phoenix)' in text
    assert 'estimated from report.json modification time' in text
    assert 'Final files exported:' in text
    assert 'Ensemble walkers: 4 (interacting; not independent chains)' in text
    assert 'Warmup / burn-in steps per walker: 100' in text
    assert 'Retained steps per walker: 20' in text
    assert 'Total steps per walker: 120' in text
    assert 'Total retained draws: 80 (correlated, not independent samples)' in text
    assert 'Random seed: 42' in text
    assert 'NOT PASSED' in text
    metadata = exporting.read_metadata(output/'results.npz')
    assert 'run_info.txt' in metadata['export']['files']
    assert metadata['run_info']['finished_utc'] == '2026-10-20T02:12:07+00:00'
    # Subsequent exports must keep identifying the original completion time.
    result.export(root)
    assert exporting.read_metadata(output/'results.npz')['run_info'] == metadata['run_info']


def test_explicit_completion_timestamp_takes_precedence(completed_run):
    result, root, _ = completed_run
    result.report['completed_utc'] = '2026-10-19T18:00:00Z'
    output = result.export(root)
    text = (output/'run_info.txt').read_text()
    assert 'Finished: 2026-10-19 11:00:00 MST' in text
    assert 'Completion time source: recorded run timestamp' in text
