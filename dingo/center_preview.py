"""Preview a chosen gas center on the three input images without fitting."""
from pathlib import Path

import numpy as np
import yaml


def _pair(value, name):
    """Validate one explicitly supplied (x, y) coordinate or offset."""
    pair = np.asarray(value, dtype=float)
    if pair.shape != (2,) or not np.isfinite(pair).all():
        raise ValueError(f"{name} must contain two finite numbers: (x, y).")
    return pair


def preview_center(source_id, *, data_dir, center, r_offset=(0.0, 0.0),
                   c_offset=(0.0, 0.0), zoom=12.0):
    """Mark a trial kinematic center on the legacy direct image and raw R/C.

    Parameters
    ----------
    source_id : int
        Source whose ``ID<source_id>`` folder contains the three FITS files and
        ``config_kinematics.yaml``. No saved fit or PySersic result is required.
    data_dir : path-like
        Parent directory of those source folders.
    center : pair of floats
        Trial ``(x0, y0)`` in DINGO's common source-frame pixels, zero based.
        These are the coordinates supplied to ``simple_mcmc.prepare``. They
        are not native F150W pixels or the observed grism-image coordinates.
    r_offset, c_offset : pairs of floats
        Explicit ``(dx, dy)`` registration offsets, in pixels, as used by the
        DINGO model. Use the starting offsets when preparing a new fit, or
        offsets from a particular saved sample to preview that sample. The
        fitting notebook holds C at (0, 0); R offsets are normally sampled.
    zoom : positive float or None
        Display half-width around each cutout's midpoint, in pixels. Use None
        to display the entire cutout. Zoom changes only the displayed limits.

    Returns
    -------
    matplotlib.figure.Figure
        Three heatmaps with cyan center markers. The figure's
        ``dingo_center_preview`` dictionary records the coordinates and trace
        geometry used to draw them. This function writes no files, creates no
        run, and neither fits nor changes any fitting settings.

    Notes
    -----
    R/C markers use the same trace calibration, integer cutout origins and
    registration convention as the simple fitter. At the center the rotation
    contribution is zero, so its projection requires no trial velocity,
    inclination or turnover radius. The systemic velocity is zero, as in the
    current simple fit, and the source's catalog redshift sets the wavelength.

    The direct-image marker is ONLY a nominal reference: it assumes the old
    ``image.fits`` stamp shares the R/C source-frame pixel scale, orientation
    and catalog anchor. It is not a measured stellar/gas astrometric offset.
    This panel does not use the new PySersic/F150W cutout.
    """
    from astropy.io import fits
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as effects

    from ._simple_mcmc_plots import _brightness_norm, _image_axis
    from .three_file_centers import _project

    if isinstance(source_id, (bool, np.bool_)) or int(source_id) != source_id or source_id < 0:
        raise ValueError("source_id must be a nonnegative integer.")
    source_id = int(source_id)
    center = _pair(center, "center")
    r_offset = _pair(r_offset, "r_offset")
    c_offset = _pair(c_offset, "c_offset")
    if zoom is not None and (isinstance(zoom, (bool, np.bool_))
                             or not np.isfinite(zoom) or zoom <= 0):
        raise ValueError("zoom must be positive and finite, or None for the full image.")

    source = Path(data_dir).expanduser().resolve() / f"ID{source_id}"
    original = yaml.safe_load((source / "config_kinematics.yaml").read_text())
    summary = dict(original["summary"])
    if summary.get("name") != f"ID{source_id}":
        raise ValueError("Source ID disagrees with config_kinematics.yaml.")
    arrays = {name: np.asarray(fits.getdata(source / f"{name}.fits"), dtype=float)
              for name in ("image", "R", "C")}
    if any(a.ndim != 2 or not np.isfinite(a).any() for a in arrays.values()):
        raise ValueError("Each FITS file must contain a 2D image with finite pixels.")
    rfit = summary.get("r_fit", (arrays["R"].shape[0] - 1) // 2)
    if isinstance(rfit, bool) or not np.isfinite(rfit) or int(rfit) != rfit or rfit < 1:
        raise ValueError("summary.r_fit must be a positive integer half-width in pixels.")
    rfit = int(rfit)
    expected_shape = (2 * rfit + 1, 2 * rfit + 1)
    if arrays["R"].shape != expected_shape or arrays["C"].shape != expected_shape:
        raise ValueError(f"R/C shapes must match summary.r_fit: expected {expected_shape}.")

    # Mirror build_config: source metadata and modules are retained, while
    # cutout origins are derived from r_fit and redshift rather than copied
    # from possibly stale explicit origins in the legacy source configuration.
    summary.update(unit="pix", r_fit=rfit)
    config = {"summary": summary, "image": {
        p: {"module": original["image"][p]["module"]} for p in ("R", "C")}}
    params = {"velocity.x0_v": float(center[0]), "velocity.y0_v": float(center[1]),
              "velocity.v_sys": 0.0}
    for pupil, offset in (("R", r_offset), ("C", c_offset)):
        params.update({f"image.{pupil}.dx": float(offset[0]),
                       f"image.{pupil}.dy": float(offset[1])})
    projected, provenance = _project(config, params)
    direct_midpoint = (np.asarray(arrays["image"].shape[::-1]) - 1) / 2
    points = {"image": direct_midpoint + center - rfit,
              "R": np.asarray(projected["observed_R"]),
              "C": np.asarray(projected["observed_C"])}

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), layout="constrained")
    heat = plt.get_cmap("afmhot").with_extremes(bad="white")
    grism_norm = _brightness_norm(arrays["R"], arrays["C"])
    titles = ("Legacy direct image\nNominal reference only",
              "Observed R\nProjected trial center", "Observed C\nProjected trial center")
    for ax, name, title in zip(axes, arrays, titles):
        pixels = arrays[name]
        norm = _brightness_norm(pixels) if name == "image" else grism_norm
        _image_axis(ax, pixels, title, norm, heat)
        point = points[name]
        marker, = ax.plot(*point, marker="+", linestyle="none", color="cyan",
                          markersize=16, markeredgewidth=2, zorder=5)
        marker.set_gid("trial-kinematic-center")
        marker.set_path_effects([effects.Stroke(linewidth=4, foreground="black"),
                                effects.Normal()])
        height, width = pixels.shape
        limits = [(-.5, width - .5), (-.5, height - .5)]
        if zoom is not None:
            middle = ((width - 1) / 2, (height - 1) / 2)
            limits = [(max(lo, mid - zoom), min(hi, mid + zoom))
                      for (lo, hi), mid in zip(limits, middle)]
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
        if not all(lo <= coord <= hi for coord, (lo, hi) in zip(point, limits)):
            ax.text(.5, .03, "Trial center outside this view", transform=ax.transAxes,
                    color="cyan", ha="center", bbox={"facecolor": "black", "alpha": .8})
        if name in ("R", "C"):
            ax.set_xlabel("Observed x [pixel, zero-based]")
            ax.set_ylabel("Observed y [pixel, zero-based]")
    fig.suptitle(
        f"ID{source_id}: trial source-frame center x0={center[0]:.2f}, y0={center[1]:.2f} pixels\n"
        f"R registration=({r_offset[0]:g}, {r_offset[1]:g}); "
        f"C=({c_offset[0]:g}, {c_offset[1]:g}) pixels. Preview only; no fit.", fontsize=11)
    fig.dingo_center_preview = {
        "source_id": source_id, "source_center": center.tolist(),
        "points": {name: point.tolist() for name, point in points.items()},
        "r_offset": r_offset.tolist(), "c_offset": c_offset.tolist(),
        "provenance": provenance,
        "direct_note": "Nominal only: equal-scale, aligned, catalog-centered legacy stamps assumed.",
    }
    return fig
