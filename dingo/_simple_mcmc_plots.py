"""Plots for a local three-file fit; no fitting, file access, or automatic saving."""
from __future__ import annotations

import numpy as np


def _image(value, name):
    value = np.asarray(value, dtype=float)
    if value.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional image.")
    return value


def _brightness_norm(*arrays):
    from matplotlib.colors import AsinhNorm

    values = np.concatenate([array[np.isfinite(array)] for array in arrays])
    high = max(float(values.max()) if values.size else 0., np.finfo(float).eps)
    return AsinhNorm(linear_width=high / 15., vmin=0., vmax=high)


def _image_axis(ax, pixels, title, norm, cmap):
    artist = ax.imshow(np.ma.masked_invalid(pixels), origin="lower", interpolation="none",
                       norm=norm, cmap=cmap)
    ax.set(title=title, xlabel="x [pixel, zero-based]", ylabel="y [pixel, zero-based]")
    return artist


def input_figure(image, R, C):
    """Show the direct image and both observed coadds, with a common R/C stretch."""
    import matplotlib.pyplot as plt

    image, R, C = (_image(array, name) for array, name in
                   ((image, "Direct image"), (R, "R"), (C, "C")))
    cmap = plt.get_cmap("afmhot").with_extremes(bad="white")
    norm = _brightness_norm(R, C)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), layout="constrained")
    for ax, pixels, title, stretch in zip(axes, (image, R, C),
            ("Direct image: visual reference", "Observed R coadd", "Observed C coadd"),
            (_brightness_norm(image), norm, norm)):
        _image_axis(ax, pixels, title, stretch, cmap)
    return fig


def residual_figure(R_rect, C_rect, center_xy, title, mask=None):
    """Compare the two rectified images and their signed difference.

    The cyan plus marks the supplied kinematic center in their common source
    coordinates. ``mask=True`` means an included residual pixel. Image values
    are never changed; brightness plots use black at zero and an asinh stretch.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as effects
    from matplotlib.colors import Normalize

    R, C = _image(R_rect, "Rectified R"), _image(C_rect, "Rectified C")
    if R.shape != C.shape:
        raise ValueError("Rectified R and C must have the same shape.")
    center = np.asarray(center_xy, dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("center_xy must contain finite zero-based x and y coordinates.")
    included = np.ones(R.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if included.shape != R.shape:
        raise ValueError("The residual mask must have the same shape as R and C.")
    valid = included & np.isfinite(R) & np.isfinite(C)
    difference = R - C
    limit = max(float(np.max(np.abs(difference[valid]))) if valid.any() else 0.,
                np.finfo(float).eps)
    residual = np.where(valid, difference, np.nan)
    norm = _brightness_norm(R, C)
    heat = plt.get_cmap("afmhot").with_extremes(bad="white")
    diverging = plt.get_cmap("RdBu_r").with_extremes(bad="white")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), layout="constrained")
    artists = []
    for ax, pixels, label, stretch, cmap in zip(axes, (R, C, residual),
            ("Rectified R", "Rectified C", "Rectified R − rectified C"),
            (norm, norm, Normalize(-limit, limit)), (heat, heat, diverging)):
        artists.append(_image_axis(ax, pixels, label, stretch, cmap))
        limits = ax.get_xlim(), ax.get_ylim()
        marker, = ax.plot(*center, marker="+", linestyle="none", color="cyan",
                          markersize=14, markeredgewidth=2, zorder=5)
        marker.set_path_effects([effects.Stroke(linewidth=4, foreground="black"),
                                effects.Normal()])
        marker.set_gid("kinematic-center")
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
    fig.colorbar(artists[1], ax=list(axes[:2]), shrink=.78, label="Image intensity")
    fig.colorbar(artists[2], ax=axes[2], shrink=.78, label="Signed intensity difference")
    excluded = int((~valid).sum())
    note = f"Cyan +: kinematic center (x={center[0]:.2f}, y={center[1]:.2f}) pixels"
    if excluded:
        note += f"; {excluded} excluded/nonfinite residual pixels shown white"
    fig.suptitle(f"{title}\n{note}", fontsize=11)
    return fig


_PARAMETERS = {
    "velocity.V_rot": (r"$V_\infty$ [km/s]", "V_inf", "km/s", 1.),
    "velocity.V_sini": (r"$V_\infty\sin i$ [km/s]", "V_inf_sin_i", "km/s", 1.),
    "velocity.R_v": (r"$R_t$ [pixel]", "R_t", "pixel", 1.),
    "velocity.x0_v": (r"$x_c$ [pixel]", "x_center", "pixel", 1.),
    "velocity.y0_v": (r"$y_c$ [pixel]", "y_center", "pixel", 1.),
    "velocity.theta_v": (r"Cutout PA [deg]", "cutout_PA", "deg", 180. / np.pi),
    "velocity.inc_v": (r"$i$ [deg]", "inclination", "deg", 180. / np.pi),
    "image.R.dx": (r"$dx_R$ [pixel]", "R_x_offset", "pixel", 1.),
    "image.R.dy": (r"$dy_R$ [pixel]", "R_y_offset", "pixel", 1.),
}


def posterior_figures(chain, names, log_prob, source_id):
    """Return pooled quantiles, walker traces, and black 68/95% corner contours.

    ``chain`` contains retained draws only, with shape (draw, walker, parameter).
    Every retained value enters the quantiles and histograms. Traces may thin
    their display to about 2,000 draw positions, while showing every walker.
    The periodic position angle is unwrapped around its circular mean for
    display, so a posterior crossing 0/360 degrees is not split by that seam.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    values = np.array(chain, dtype=float, copy=True)
    names = [str(name) for name in names]
    probability = np.asarray(log_prob, dtype=float)
    if values.ndim != 3 or not all(values.shape) or values.shape[2] != len(names):
        raise ValueError("chain must have nonempty (draw, walker, parameter) axes matching names.")
    if len(set(names)) != len(names):
        raise ValueError("Posterior parameter names must be unique.")
    if probability.shape != values.shape[:2]:
        raise ValueError("log_prob must have shape (draw, walker), matching chain.")
    if not np.isfinite(values).all():
        raise ValueError("Posterior parameters contain nonfinite values; inspect the saved chain.")

    specifications = [_PARAMETERS.get(name, (name, name, "", 1.)) for name in names]
    labels = [row[0] for row in specifications]
    angle_note = ""
    for index, (name, spec) in enumerate(zip(names, specifications)):
        values[:, :, index] *= spec[3]
        if name == "velocity.theta_v":
            angle = values[:, :, index]
            reference = np.rad2deg(np.angle(np.mean(np.exp(1j * np.deg2rad(angle)))))
            reference %= 360.
            values[:, :, index] = reference + (angle - reference + 180.) % 360. - 180.
            angle_note = " PA is unwrapped across the 0/360° seam."
    flat = values.reshape(-1, len(names))
    lower, median, upper = np.quantile(flat, [.16, .5, .84], axis=0)
    summary = pd.DataFrame({
        "parameter": [row[1] for row in specifications],
        "median": median, "minus_16th": median - lower, "plus_84th": upper - median,
        "unit": [row[2] for row in specifications],
    }).set_index("parameter")
    summary.index.name = "parameter (all walkers pooled)"
    summary.attrs.update(n_draws=values.shape[0], n_walkers=values.shape[1],
                         velocity_definition="V_inf is deprojected asymptotic rotation speed",
                         angle_note=angle_note.strip())
    bad_probability = int((~np.isfinite(probability)).sum())
    warning = (f"; {bad_probability} samples have nonfinite log probability" if bad_probability else "")
    unmoved = int(np.all(values == values[:1], axis=(0, 2)).sum()) if len(values) > 1 else 0
    if unmoved:
        warning += f"; {unmoved}/{values.shape[1]} walkers never moved during retained steps"
    heading = f"ID{source_id}: {values.shape[0]} retained draws × {values.shape[1]} walkers{warning}"

    nrows = (len(names) + 1) // 2
    traces, axes = plt.subplots(nrows, 2, figsize=(12., max(3., 1.9 * nrows)),
                                squeeze=False, layout="constrained")
    indices = np.unique(np.r_[np.arange(0, values.shape[0], max(1, int(np.ceil(values.shape[0] / 2000)))),
                              values.shape[0] - 1])
    for index, ax in enumerate(axes.flat):
        if index >= len(names):
            ax.set_visible(False)
            continue
        ax.plot(indices, values[indices, :, index], color="black", alpha=.16, linewidth=.5)
        ax.axhline(median[index], color="black", linewidth=1., linestyle="--")
        ax.set(xlabel="Retained MCMC step", ylabel=labels[index])
        ax.tick_params(labelsize=8)
    traces.suptitle(heading + "\nEvery walker shown; dashed line is the pooled median." + angle_note,
                    fontsize=10)

    constant = [specifications[index][1] for index in range(len(names))
                if np.ptp(flat[:, index]) == 0.]
    no_movement = [specifications[index][1] for index in range(len(names))
                   if np.all(np.ptp(values[:, :, index], axis=0) == 0.)]
    insufficient = values.shape[0] < 20 or flat.shape[0] < max(32, 2 * len(names) + 1)
    if constant or no_movement or insufficient:
        corner_figure, ax = plt.subplots(figsize=(9., 3.5))
        ax.axis("off")
        reason = ("Constant parameters: " + ", ".join(constant) + ".\n" if constant else "")
        if no_movement and not constant:
            reason += "No within-walker movement: " + ", ".join(no_movement) + ".\n"
        if insufficient:
            reason += "Too few retained draws to construct useful contours.\n"
        ax.text(.5, .5, "Corner contours are not available yet.\n\n" + reason
                + "Inspect the traces; these intervals do not establish uncertainty convergence.",
                ha="center", va="center", transform=ax.transAxes, wrap=True)
        corner_figure.suptitle(heading, fontsize=10)
    else:
        import corner

        corner_figure = corner.corner(flat, labels=labels, color="black", bins=30,
            quantiles=[.16, .5, .84], title_quantiles=[.16, .5, .84],
            show_titles=True, title_fmt=".2f", title_kwargs={"fontsize": 9},
            label_kwargs={"fontsize": 9}, smooth=.8, smooth1d=.6,
            plot_datapoints=False, plot_density=False, fill_contours=False,
            levels=(.68, .95), hist_kwargs={"color": "black", "linewidth": 1.1},
            contour_kwargs={"colors": "black", "linewidths": .9})
        corner_figure.suptitle(heading + "\nContours enclose 68% and 95%; titles show pooled median and 16th–84th percentiles."
                              + angle_note, fontsize=10, y=1.01)
    return {"summary": summary, "traces": traces, "corner": corner_figure}
