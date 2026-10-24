"""Center overlays retain saved image frames, residual masks and pixel limits."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from types import SimpleNamespace

from dingo.three_file_center_plot import (center_figure, mark_center, mark_corrected_figure,
                                         review_center, review_fit)
from dingo.three_file_plot import velocity_summary


@pytest.fixture
def center_inputs():
    image = np.arange(121., dtype=float).reshape(11, 11)
    r, c = image / 100, image / 110
    mask = np.ones(image.shape, dtype=bool)
    mask[0, 0] = False
    pixels = dict(image=image, observed_R=r + .2, observed_C=c - .1,
                  rectified_R=r, rectified_C=c, difference=r-c, mask=mask)
    params = {"velocity.V_sini": 150., "velocity.inc_v": np.pi/6,
              "velocity.R_v": 2., "velocity.x0_v": 6., "velocity.y0_v": 4.}
    geometry = dict(params=params,
        points={"direct": (6., 4.), "observed_R": (7.25, 4.5),
                "observed_C": (6., 3.75), "rectified": (6., 4.)},
        reference={key: (5., 5.) for key in ("direct", "observed_R", "observed_C", "rectified")},
        offset_pixels=(1., -1.), offset_arcsec=(.063, -.063),
        offset_pc=(500., -500.))
    metadata = dict(source_id=15665, redshift=1.1613, pixel_scale_arcsec=.063)
    return pixels, geometry, metadata


def marker(ax, gid):
    matches = [line for line in ax.lines if line.get_gid() == gid]
    assert len(matches) == 1
    return np.asarray(matches[0].get_xydata()).reshape(-1, 2)[0]


def test_each_panel_uses_its_own_projected_center(center_inputs):
    pixels, geometry, metadata = center_inputs
    fig = center_figure(pixels, geometry, metadata)
    try:
        panels = [ax for ax in fig.axes if ax.images]
        assert len(panels) == 6
        frames = ["direct", "observed_R", "observed_C", "rectified", "rectified", "rectified"]
        for ax, frame in zip(panels, frames):
            np.testing.assert_allclose(marker(ax, "kinematic-center"), geometry["points"][frame])
            np.testing.assert_allclose(marker(ax, "cutout-reference"), geometry["reference"][frame])
        # Labels describe the same parameter vector as the markers, not pooled samples.
        summary = velocity_summary(geometry["params"], metadata)
        title = fig._suptitle.get_text()
        assert f"Vrot({summary['radius_pc']:.0f} pc) sin(i) = {summary['v_sini_at_radius']:.0f} km/s" in title
        assert "V∞ sin(i) = 150 km/s (asymptote)" in title
        assert "i = 30.0°" in title
        assert f"rturn = {summary['rturn_pc']:.0f} pc" in title
        assert f"Vrot({summary['radius_pc']:.0f} pc) =" not in title
        assert "highest-posterior draw" in " ".join(text.get_text() for text in fig.texts)
    finally:
        plt.close(fig)


def test_center_display_preserves_residual_sign_mask_and_input_arrays(center_inputs):
    pixels, geometry, metadata = center_inputs
    original = {key: value.copy() for key, value in pixels.items()}
    fig = center_figure(pixels, geometry, metadata, half_width=None)
    try:
        panels = [ax for ax in fig.axes if ax.images]
        residual = panels[-1].images[0].get_array()
        assert residual.mask[0, 0]
        np.testing.assert_allclose(residual[1:, 1:], pixels["rectified_R"][1:, 1:]-pixels["rectified_C"][1:, 1:])
        assert all(ax.images[0].get_interpolation() == "nearest" for ax in panels)
        assert panels[1].images[0].norm is panels[2].images[0].norm
        assert panels[3].images[0].norm is panels[4].images[0].norm
        for key in original:
            np.testing.assert_array_equal(pixels[key], original[key])
        for ax in panels:
            assert ax.get_xlim() == pytest.approx((-.5, 10.5))
            assert ax.get_ylim() == pytest.approx((-.5, 10.5))
    finally:
        plt.close(fig)


def test_missing_observed_projection_is_labeled_and_never_guessed(center_inputs):
    pixels, geometry, metadata = center_inputs
    geometry["points"]["observed_R"] = None
    geometry["points"]["observed_C"] = None
    fig = center_figure(pixels, geometry, metadata)
    try:
        panels = [ax for ax in fig.axes if ax.images]
        for ax in panels[1:3]:
            assert not any(line.get_gid() == "kinematic-center" for line in ax.lines)
            assert "Projection unavailable" in ax.get_title()
        for ax in [panels[0], *panels[3:]]:
            np.testing.assert_allclose(marker(ax, "kinematic-center"), (6., 4.))
    finally:
        plt.close(fig)


def test_zero_offset_reference_and_center_remain_distinct_markers(center_inputs):
    pixels, geometry, metadata = center_inputs
    for frame in geometry["points"]:
        geometry["points"][frame] = geometry["reference"][frame]
    for key in ("offset_pixels", "offset_arcsec", "offset_pc"):
        geometry[key] = (0., 0.)
    fig = center_figure(pixels, geometry, metadata)
    try:
        assert "distance=0.000″ (0 pc)" in fig._suptitle.get_text()
        for ax in (ax for ax in fig.axes if ax.images):
            np.testing.assert_allclose(marker(ax, "kinematic-center"), marker(ax, "cutout-reference"))
            assert {line.get_marker() for line in ax.lines} == {"x", "o"}
    finally:
        plt.close(fig)


@pytest.mark.parametrize("half_width", [0, -1, np.nan, np.inf])
def test_invalid_zoom_rejected(center_inputs, half_width):
    with pytest.raises(ValueError, match="half_width"):
        center_figure(*center_inputs, half_width=half_width)


def test_center_outside_stamp_never_expands_image_limits():
    fig, ax = plt.subplots()
    try:
        ax.imshow(np.zeros((5, 6)), origin="lower")
        limits = ax.get_xlim(), ax.get_ylim()
        mark_center(ax, (200., -100.), (2., 2.))
        assert (ax.get_xlim(), ax.get_ylim()) == limits
        np.testing.assert_allclose(marker(ax, "kinematic-center"), (200., -100.))
    finally:
        plt.close(fig)


def test_corrected_rows_use_saved_centers_and_skip_legacy_missing_centers():
    fig, axes = plt.subplots(3, 3)
    try:
        for ax in axes.flat:
            ax.imshow(np.zeros((11, 11)), origin="lower")
        records = [dict(params=dict(x0_v=3., y0_v=4.)),
                   dict(params=dict(x0_v=99., y0_v=99.),
                        full_params={"velocity.x0_v": 7., "velocity.y0_v": 8.}),
                   dict(params=dict(V_sini=200.))]
        assert mark_corrected_figure(fig, records, (5., 5.)) is True
        for row, expected in zip(axes[:2], [(3., 4.), (7., 8.)]):
            for ax in row:
                np.testing.assert_allclose(marker(ax, "kinematic-center"), expected)
        assert all(not ax.lines for ax in axes[2])
    finally:
        plt.close(fig)


def test_center_free_legacy_rows_do_not_get_invented_markers():
    fig, axes = plt.subplots(1, 3)
    try:
        for ax in axes:
            ax.imshow(np.zeros((3, 3)))
        assert mark_corrected_figure(fig, [dict(params={})], (1., 1.)) is False
        assert all(not ax.lines for ax in axes)
    finally:
        plt.close(fig)


def test_wrong_residual_sign_rejected(center_inputs):
    pixels, geometry, metadata = center_inputs
    pixels["difference"] *= -1
    with pytest.raises(ValueError, match="corrected R minus corrected C"):
        center_figure(pixels, geometry, metadata)


def test_center_figure_renders_without_fitting(center_inputs, tmp_path):
    fig = center_figure(*center_inputs)
    try:
        path = tmp_path / "center_projection.png"
        fig.savefig(path)
        assert path.stat().st_size > 1000
    finally:
        plt.close(fig)


def test_review_center_supports_result_objects_loaded_before_new_methods(center_inputs, tmp_path):
    pixels, geometry, metadata = center_inputs
    names = list(geometry["params"])
    pixels["param_names"] = np.asarray(names)
    pixels["theta"] = np.asarray([geometry["params"][key] for key in names])
    # No center_figure method: an already running kernel can still display
    # its saved result through the freshly imported module-level helper.
    old_result = SimpleNamespace(_prediction_pixels=lambda: pixels,
        report={**metadata, "r_fit_pixels": 5.}, source_id=15665, path=tmp_path,
        physical_scale=lambda: {"pc_per_pixel": 500.})
    fig = review_center(old_result)
    try:
        panels = [ax for ax in fig.axes if ax.images]
        np.testing.assert_allclose(marker(panels[0], "kinematic-center"), (6., 4.))
        # The absent original config cannot provide observed-image projections.
        assert all("Projection unavailable" in ax.get_title() for ax in panels[1:3])
    finally:
        plt.close(fig)


@pytest.mark.parametrize("already_marked", [False, True])
def test_review_fit_adds_exact_saved_center_once_for_old_or_new_result(center_inputs, already_marked):
    pixels, geometry, metadata = center_inputs
    names = list(geometry["params"])
    pixels["param_names"] = np.asarray(names)
    pixels["theta"] = np.asarray([geometry["params"][key] for key in names])
    fig, axes = plt.subplots(1, 3)
    for ax in axes:
        ax.imshow(np.zeros((11, 11)), origin="lower")
    if already_marked:
        mark_corrected_figure(fig, [{"full_params": geometry["params"]}], (5., 5.))
    called = []

    def old_fit_figure(**kwargs):
        called.append(kwargs)
        return fig

    old_result = SimpleNamespace(_prediction_pixels=lambda: pixels,
        report={**metadata, "r_fit_pixels": 5.}, fit_figure=old_fit_figure)
    try:
        assert review_fit(old_result, radius_pc=2000.) is fig
        assert called == [{"radius_pc": 2000.}]
        for ax in axes:
            np.testing.assert_allclose(marker(ax, "kinematic-center"), (6., 4.))
            assert len(ax.lines) == 2
    finally:
        plt.close(fig)
