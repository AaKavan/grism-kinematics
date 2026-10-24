"""Show the center of the exact saved velocity field in each image frame."""
import numpy as np

IMAGE_CMAP = "gist_heat"


def intensity_norm(*arrays):
    """Shared heat-map stretch that preserves the bright core instead of clipping it."""
    from matplotlib.colors import AsinhNorm
    values = np.concatenate([np.asarray(a, dtype=float)[np.isfinite(a)] for a in arrays])
    low = float(np.quantile(values, .01))
    high = max(float(np.max(values)), low+1e-12)
    return AsinhNorm(linear_width=max((high-low)/15, 1e-12), vmin=low, vmax=high)


def color_corrected_figure(fig, pixels):
    """Apply one scale to all R/C candidates; residual signs/scales are retained."""
    panels = [ax for ax in fig.axes if ax.images]
    common = intensity_norm(pixels["rectified_R"], pixels["rectified_C"])
    for index, ax in enumerate(panels):
        if index % 3 != 2:
            ax.images[0].set_cmap(IMAGE_CMAP)
            ax.images[0].set_norm(common)


def review_center(result, *, radius_pc=None, half_width=15.):
    """Also works with a result class already imported in a running notebook."""
    from .three_file_centers import center_geometry
    pixels = result._prediction_pixels()
    return center_figure(pixels, center_geometry(result, pixels), result.report,
                         radius_pc=radius_pc, half_width=half_width)


def review_fit(result, *, radius_pc=None):
    """Add markers even when the kernel still has an older fit_figure method."""
    from .three_file_centers import saved_parameters
    fig = result.fit_figure(radius_pc=radius_pc)
    color_corrected_figure(fig, result._prediction_pixels())
    if not any(line.get_gid() == "kinematic-center" for ax in fig.axes for line in ax.lines):
        reference = float(result.report["r_fit_pixels"])
        mark_corrected_figure(fig, [{"full_params": saved_parameters(result)}], (reference, reference))
    return fig


def mark_center(ax, center, reference=None):
    """High-contrast markers; never change the displayed pixel extent."""
    import matplotlib.patheffects as effects
    limits = ax.get_xlim(), ax.get_ylim()
    if reference is not None:
        line, = ax.plot(*reference, marker="o", markersize=9, markerfacecolor="none",
                        markeredgecolor="white", markeredgewidth=1.4, linestyle="none",
                        label="Cutout reference", zorder=7)
        line.set_path_effects([effects.Stroke(linewidth=3.5, foreground="black"), effects.Normal()])
        line.set_gid("cutout-reference")
    if center is not None:
        line, = ax.plot(*center, marker="x", markersize=12, markeredgewidth=2,
                        color="cyan", linestyle="none", label="Fitted kinematic center", zorder=8)
        line.set_path_effects([effects.Stroke(linewidth=4.5, foreground="black"), effects.Normal()])
        line.set_gid("kinematic-center")
    ax.set_xlim(limits[0]); ax.set_ylim(limits[1])


def mark_corrected_figure(fig, records, reference):
    """Annotate each corrected R/C/residual row without altering saved pixels."""
    panels = [ax for ax in fig.axes if ax.images]
    if len(panels) != 3*len(records):
        raise ValueError("Expected three corrected-image panels per saved fit.")
    marked = False
    for row, record in enumerate(records):
        params = dict(record.get("params", {})); params.update(record.get("full_params", {}))
        x = params.get("velocity.x0_v", params.get("x0_v"))
        y = params.get("velocity.y0_v", params.get("y0_v"))
        if x is None or y is None:
            continue  # Old result tables may not record a center.
        if not np.isfinite([x, y]).all():
            raise ValueError("Saved kinematic center must be finite.")
        for ax in panels[3*row:3*row+3]:
            mark_center(ax, (x, y), reference)
        panels[3*row].text(.03, .97, f"× center ({x:.2f}, {y:.2f}) px\n○ cutout reference",
            transform=panels[3*row].transAxes, va="top", color="white", fontsize=8,
            bbox=dict(facecolor="black", alpha=.65, edgecolor="none"))
        marked = True
    return marked


def center_figure(pixels, geometry, metadata, *, radius_pc=None, half_width=15.):
    """Six panels, with correctly mapped centers and a shared central zoom."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from .three_file_plot import velocity_summary
    summary = velocity_summary(geometry["params"], metadata, radius_pc=radius_pc)
    if half_width is not None and (not np.isfinite(half_width) or half_width <= 0):
        raise ValueError("half_width must be positive, or None for full stamps.")
    direct = np.asarray(pixels["image"], dtype=float)
    low, high = float(np.nanmin(direct)), float(np.nanmax(direct))
    image_norm = Normalize(low, max(high, low+1e-12))
    observed_norm = intensity_norm(pixels["observed_R"], pixels["observed_C"])
    corrected_norm = intensity_norm(pixels["rectified_R"], pixels["rectified_C"])
    mask = np.asarray(pixels["mask"], dtype=bool)
    residual = np.asarray(pixels["difference"], dtype=float)
    if not np.allclose(residual, pixels["rectified_R"]-pixels["rectified_C"], equal_nan=True):
        raise ValueError("Residual must be corrected R minus corrected C.")
    limit = max(float(np.quantile(abs(residual[mask & np.isfinite(residual)]), .99)), 1e-12)
    residual_norm = Normalize(-limit, limit)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.8))
    fig.subplots_adjust(left=.06, right=.94, bottom=.18, top=.86, wspace=.30, hspace=.36)
    titles = ["Direct image · nominal registration", "Observed R", "Observed C",
              "Corrected R", "Corrected C", "Corrected R − corrected C"]
    keys = ["image", "observed_R", "observed_C", "rectified_R", "rectified_C", "difference"]
    frames = ["direct", "observed_R", "observed_C", "rectified", "rectified", "rectified"]
    for ax, title, key, frame in zip(axes.flat, titles, keys, frames):
        data = np.where(mask, residual, np.nan) if key == "difference" else pixels[key]
        cmap = plt.get_cmap("RdBu_r").with_extremes(bad=".65") if key == "difference" else IMAGE_CMAP
        chosen_norm = (residual_norm if key == "difference" else image_norm if key == "image"
                       else observed_norm if key.startswith("observed") else corrected_norm)
        im = ax.imshow(data, origin="lower", interpolation="nearest", cmap=cmap, norm=chosen_norm)
        point, reference = geometry["points"][frame], geometry["reference"][frame]
        if half_width is not None:
            height, width = np.shape(data)
            ax.set_xlim(max(-.5, reference[0]-half_width), min(width-.5, reference[0]+half_width))
            ax.set_ylim(max(-.5, reference[1]-half_width), min(height-.5, reference[1]+half_width))
        mark_center(ax, point, reference)
        coordinate = (f"× x={point[0]:.2f}, y={point[1]:.2f} px" if point is not None
                      else "Projection unavailable: calibration/config missing")
        ax.set(title=f"{title}\n{coordinate}", xlabel="x [pixel]", ylabel="y [pixel]")
        if key == "difference":
            fig.colorbar(im, ax=ax, fraction=.046, pad=.04, extend="both")
    dx, dy = geometry["offset_pixels"]
    arc = float(np.hypot(*geometry["offset_arcsec"]))
    pc = float(np.hypot(*geometry["offset_pc"]))
    if summary.get("sampled_velocity_parameter") == "V_rot":
        velocity_text = (f"V∞ = {summary['v_inf']:.0f} km/s (sampled deprojected asymptote); "
                         f"Vrot({summary['radius_pc']:.0f} pc) = {summary['v_at_radius']:.0f} km/s\n")
    else:
        velocity_text = (f"Vrot({summary['radius_pc']:.0f} pc) sin(i) = {summary['v_sini_at_radius']:.0f} km/s; "
                         f"V∞ sin(i) = {summary['v_sini']:.0f} km/s (asymptote)\n")
    fig.suptitle(f"ID{metadata['source_id']}: center of the highest-posterior retained fit\n"
        f"{velocity_text}"
        f"i = {summary['inclination_deg']:.1f}°; rturn = {summary['rturn_pc']:.0f} pc\n"
        f"Center offset from cutout reference: Δx={dx:+.2f}, Δy={dy:+.2f} px; "
        f"distance={arc:.3f}″ ({pc:.0f} pc)", fontsize=12, y=.98)
    handles = [Line2D([], [], color="darkcyan", marker="x", markersize=10, linestyle="none", label="Fitted kinematic center (cyan ×)"),
               Line2D([], [], color="black", marker="o", markerfacecolor="none", markersize=9, linestyle="none", label="Nominal cutout reference")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .09), ncol=2, frameon=False)
    view = "Full stamps" if half_width is None else f"Central view, ±{half_width:g} pixels; set half_width=None for full stamps"
    fig.text(.5, .025, f"{view}. Zero-based pixel coordinates; gray residual pixels are excluded.\n"
        "Markers and velocities use the same highest-posterior draw as the saved images, not marginal medians.\n"
        "Direct-image placement assumes the nominal stack alignment; no WCS verifies the stellar-center offset.",
        ha="center", va="bottom", fontsize=9)
    return fig
