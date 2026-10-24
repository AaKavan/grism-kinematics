"""Guard against reporting errors from incompatible or unconverged saved runs."""
import hashlib
import json

import numpy as np
import pytest

from dingo.rotation import RotationResults


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def saved_run(tmp_path):
    rng = np.random.default_rng(87)
    shape = (2, 20)
    samples = dict(vrot=rng.normal(200, 10, shape), inclination=rng.normal(1., .02, shape),
        log_rturn=rng.normal(-2., .1, shape), theta=rng.normal(.7, .02, shape),
        systemic=rng.normal(0, 5, shape), center=rng.normal(0, .01, (*shape, 2)),
        line_flux_1e17=rng.normal(5, .2, shape))
    np.savez(tmp_path / "samples.npz", **samples)
    science = np.ones((2, 5, 5))
    metadata = dict(source_id=21636, exposures=[dict(pupil="R"), dict(pupil="C")])
    np.savez(tmp_path / "native_data.npz", science=science, error=science,
        valid=np.ones_like(science, dtype=bool), emline=science, metadata_json=json.dumps(metadata))
    np.savez(tmp_path / "best.npz", source_id=21636, line=science*.6, continuum=science*.3,
        background=np.array([.1, .1]), residual=np.zeros_like(science))
    report = dict(format="dingo_rotation_v1", source_id=21636, completed=True, real_observations=True,
        phase="complete", settings=dict(chains=2), model=dict(intrinsic_sigma_kms=0),
        source_summary=dict(z=1.1906),
        diagnostics=dict(convergence_gate_passed=False), retained_draws_per_chain=20,
        samples_sha256=checksum(tmp_path / "samples.npz"),
        native_data_sha256=checksum(tmp_path / "native_data.npz"),
        best_fit=dict(artifact="best.npz", artifact_sha256=checksum(tmp_path / "best.npz")))
    (tmp_path / "rotation.json").write_text(json.dumps(report))
    return tmp_path, report, samples


def test_pooled_errors_require_completed_accepted_chains(saved_run):
    path, report, _ = saved_run
    result = RotationResults(path, 21636)
    assert "p16" not in result.summary()
    assert set(result.summary().chain) == {1, 2}
    report["diagnostics"]["convergence_gate_passed"] = True
    (path / "rotation.json").write_text(json.dumps(report))
    assert {"p16", "p84", "error_minus", "error_plus"} <= set(result.reload().summary())
    report["completed"] = False
    (path / "rotation.json").write_text(json.dumps(report))
    assert "p16" not in result.reload().summary()


def test_status_surfaces_tree_depth_efficiency_warning(saved_run):
    path, report, _ = saved_run
    report["diagnostics"].update(convergence_gate_passed=True, max_depth_fraction=.25)
    (path / "rotation.json").write_text(json.dumps(report))
    result = RotationResults(path, 21636)
    assert result.converged
    assert "Sampling warning: 25.0%" in result.status
    assert "efficiency" in result.status


def test_wrong_source_and_sampled_dispersion_rejected(saved_run):
    path, report, samples = saved_run
    with pytest.raises(ValueError, match="source's rotation-only"):
        RotationResults(path, 3700)
    np.savez(path / "samples.npz", **samples, sigma_v=np.zeros((2, 20)))
    report["samples_sha256"] = checksum(path / "samples.npz")
    (path / "rotation.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="samples dispersion"):
        RotationResults(path, 21636).summary()


def test_hash_and_retained_shape_are_checked(saved_run):
    path, report, samples = saved_run
    np.savez(path / "samples.npz", **{k: v[:, :10] for k, v in samples.items()})
    with pytest.raises(ValueError, match="changed or checkpoint"):
        RotationResults(path, 21636).summary()
    report["samples_sha256"] = checksum(path / "samples.npz")
    (path / "rotation.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="dimensions disagree"):
        RotationResults(path, 21636).summary()


def test_prediction_checked_against_actual_native_pixels(saved_run):
    path, report, _ = saved_run
    result = RotationResults(path, 21636)
    _, saved, total, _ = result._prediction()
    np.testing.assert_allclose(total, 1.)
    saved["residual"] = np.ones_like(total)
    np.savez(path / "best.npz", **saved)
    report["best_fit"]["artifact_sha256"] = checksum(path / "best.npz")
    (path / "rotation.json").write_text(json.dumps(report))
    with pytest.raises(AssertionError, match="Saved residuals disagree"):
        result.reload()._prediction()


def test_provisional_corner_pools_draws_with_one_monochrome_summary(saved_run):
    pytest.importorskip("corner")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    path, _, samples = saved_run
    figure = RotationResults(path, 21636).corner(parameters=("vrot", "inclination"))
    try:
        colors = [patch.get_edgecolor() for patch in figure.axes[0].patches]
        assert colors == [to_rgba("black")]
        axes = np.asarray(figure.axes).reshape(2, 2)
        for i, values in enumerate((samples["vrot"], np.rad2deg(samples["inclination"]))):
            low, median, high = np.quantile(values, [.16, .5, .84])
            expected = f"${median:.1f}_{{-{median-low:.1f}}}^{{+{high-median:.1f}}}$"
            assert axes[i, i].get_title() == expected
            assert not axes[i, i].texts
        assert not figure.legends
        assert any("provisional intervals" in label.get_text() for label in figure.texts)
        for collection in axes[1, 0].collections:
            for color in [*collection.get_facecolor(), *collection.get_edgecolor()]:
                np.testing.assert_allclose(color[:3], np.full(3, color[0]))
        assert not RotationResults(path, 21636).converged
    finally:
        plt.close(figure)


def test_accepted_corner_titles_match_marginal_errors(saved_run):
    pytest.importorskip("corner")
    import matplotlib.pyplot as plt
    path, report, _ = saved_run
    report["diagnostics"]["convergence_gate_passed"] = True
    (path / "rotation.json").write_text(json.dumps(report))
    result = RotationResults(path, 21636)
    figure = result.corner(parameters=("vrot", "x0"))
    try:
        axes = np.asarray(figure.axes).reshape(2, 2)
        for i, row in enumerate(result.summary(parameters=("vrot", "x0")).itertuples()):
            precision = 3 if row.unit == "arcsec" else 1
            expected = (f"${row.median:.{precision}f}_{{-{row.error_minus:.{precision}f}}}"
                        f"^{{+{row.error_plus:.{precision}f}}}$")
            assert axes[i, i].get_title() == expected
            assert not axes[i, i].texts
    finally:
        plt.close(figure)


def test_four_chains_have_one_pooled_title_per_diagonal_without_header_overlap(saved_run):
    pytest.importorskip("corner")
    import matplotlib.pyplot as plt
    path, report, samples = saved_run
    samples = {key: np.concatenate([values, values], axis=0) for key, values in samples.items()}
    samples["vrot"][2:] += 150  # The requested display must combine even distinct chains.
    np.savez(path / "samples.npz", **samples)
    report["settings"]["chains"] = 4
    report["samples_sha256"] = checksum(path / "samples.npz")
    (path / "rotation.json").write_text(json.dumps(report))
    figure = RotationResults(path, 21636).corner()
    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        header_boxes = [label.get_window_extent(renderer) for label in figure.texts]
        for ax in np.asarray(figure.axes).reshape(6, 6).diagonal():
            assert not ax.texts and ax.get_title()
            box = ax.title.get_window_extent(renderer)
            assert 0 <= box.y0 < box.y1 <= figure.bbox.y1
            assert not any(box.overlaps(header) for header in header_boxes)
        low, median, high = np.quantile(samples['vrot'], [.16, .5, .84])
        assert figure.axes[0].get_title() == f"${median:.1f}_{{-{median-low:.1f}}}^{{+{high-median:.1f}}}$"
    finally:
        plt.close(figure)
