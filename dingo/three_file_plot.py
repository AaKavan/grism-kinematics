"""Residual-first views of saved, three-file DINGO solutions.

This module neither fits a model nor samples a posterior. Each velocity label
belongs to the parameter vector that produced that row's saved pixels.
"""

import numpy as np


def physical_scale(metadata):
    """Return the proper pc scales used to label a saved source's images."""
    from astropy import units as u
    from astropy.cosmology import Planck18

    redshift = float(metadata["redshift"])
    pixel = float(metadata["pixel_scale_arcsec"])
    if not np.isfinite([redshift, pixel]).all() or redshift <= 0 or pixel <= 0:
        raise ValueError("Positive source redshift and pixel scale are required.")
    scale = float(Planck18.kpc_proper_per_arcmin(redshift).to_value(u.pc/u.arcsec))
    return dict(pc_per_arcsec=scale, pc_per_pixel=scale*pixel,
                reference_radius_pc=.5*scale)


def velocity_summary(params, metadata, *, radius_pc=None):
    """Describe one actual arctan-model parameter vector, not pooled medians.

    The sampled amplitude is either ``V_rot`` (deprojected asymptotic speed)
    or ``V_sini`` (projected asymptotic speed). Names may be plain, dotted
    (``velocity.V_rot``), or in a nested ``velocity`` dict. ``inc_v`` is in
    radians and ``R_v`` is in pixels. Values of both speeds describe this same
    parameter vector; the returned parameter name identifies the sampled one.
    The default finite display radius is 0.5 arcsec converted to proper pc;
    it is not an assertion about where this source has useful emission.
    """
    scale = physical_scale(metadata)

    def parameter(name, *, optional=False):
        if f"velocity.{name}" in params:
            return float(params[f"velocity.{name}"])
        if name in params:
            return float(params[name])
        if isinstance(params.get("velocity"), dict) and name in params["velocity"]:
            return float(params["velocity"][name])
        if optional:
            return None
        raise ValueError(f"Missing saved velocity parameter: {name}")

    vinf = parameter("V_rot", optional=True)
    projected = parameter("V_sini", optional=True)
    if vinf is None and projected is None:
        raise ValueError("Missing saved velocity parameter: V_rot or V_sini")
    sampled_parameter = "V_rot" if vinf is not None else "V_sini"
    inc, rt_pixels = (parameter(key) for key in ("inc_v", "R_v"))
    radius = scale["reference_radius_pc"] if radius_pc is None else float(radius_pc)
    speeds = [value for value in (vinf, projected) if value is not None]
    if (not np.isfinite([*speeds, inc, rt_pixels, radius]).all()
            or min(speeds) < 0 or not 0 < inc < np.pi/2 or rt_pixels <= 0 or radius < 0):
        raise ValueError("Require nonnegative speed/radius, positive turnover and 0 < inclination < 90 degrees.")
    if vinf is not None and projected is not None:
        if not np.isclose(projected, vinf*np.sin(inc), rtol=1e-8, atol=1e-10):
            raise ValueError("Saved V_rot and V_sini disagree: require V_sini = V_rot sin(i).")
        # Old records may retain a derived V_rot in addition to sampled V_sini.
        if ("velocity.V_sini" in metadata.get("param_names", [])
                and "velocity.V_rot" not in metadata.get("param_names", [])):
            sampled_parameter = "V_sini"
    elif vinf is None:
        vinf = projected/np.sin(inc)
    else:
        projected = vinf*np.sin(inc)
    rturn = rt_pixels*scale["pc_per_pixel"]
    return dict(v_inf=vinf, v_sini=projected,
                v_at_radius=(2/np.pi)*vinf*np.arctan(radius/rturn),
                v_sini_at_radius=(2/np.pi)*projected*np.arctan(radius/rturn),
                radius_pc=radius, inclination_deg=np.rad2deg(inc), rturn_pc=rturn,
                sampled_velocity_parameter=sampled_parameter)


def _candidate_records(candidates):
    if isinstance(candidates, dict):
        return {int(key): value for key, value in candidates.items()}
    return {int(value.get("candidate_id", value.get("id", index))): value
            for index, value in enumerate(candidates)}


def candidate_figure(pixels, candidates, metadata, *, ids=None, radius_pc=None,
                     title=None, residual_limit=None):
    """Compare saved corrected R/C images and their unsmoothed differences.

    ``pixels`` contains ``candidate_ids`` (original trial indices),
    ``rectified_R``, ``rectified_C`` and ``difference`` with shape (n, h, w).
    Optional ``mask`` has shape (h, w) or (n, h, w); excluded residual pixels
    are gray. ``candidates`` is the original trials list or an ID-keyed dict;
    each record contains ``full_params`` and/or ``params``.

    Image and residual scales are shared across *all supplied candidates*,
    even when ``ids`` selects a subset. The default symmetric residual limit
    is the pooled 99th absolute percentile. Saturated pixels are marked by
    the colorbar's extensions; ``residual_limit`` sets an explicit limit.
    No image is smoothed or silently cropped, and no velocity is rejected.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import AsinhNorm, Normalize

    candidate_ids = np.asarray(pixels["candidate_ids"])
    if (candidate_ids.ndim != 1 or candidate_ids.size == 0
            or not np.issubdtype(candidate_ids.dtype, np.integer)
            or len(set(candidate_ids.tolist())) != len(candidate_ids)):
        raise ValueError("candidate_ids must be a nonempty vector of distinct integer IDs.")
    arrays = [np.asarray(pixels[key], dtype=float)
              for key in ("rectified_R", "rectified_C", "difference")]
    if (arrays[0].ndim != 3 or arrays[0].shape[0] != len(candidate_ids)
            or min(arrays[0].shape) < 1 or any(a.shape != arrays[0].shape for a in arrays)):
        raise ValueError("Candidate R, C and difference arrays must have matching (n, h, w) shapes.")
    expected = arrays[0]-arrays[1]
    if not np.allclose(arrays[2], expected, rtol=1e-5, atol=1e-8, equal_nan=True):
        raise ValueError("Saved residuals must be corrected R minus corrected C.")
    mask = np.asarray(pixels.get("mask", pixels.get("display_mask",
                      np.ones(arrays[0].shape, dtype=bool))), dtype=bool)
    if mask.shape == arrays[0].shape[1:]:
        mask = np.broadcast_to(mask, arrays[0].shape)
    if mask.shape != arrays[0].shape:
        raise ValueError("Residual mask must match the stamp or the candidate stack.")

    records = _candidate_records(candidates)
    selected = candidate_ids.tolist() if ids is None else list(ids)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Select at least one distinct candidate ID.")
    index = {int(value): i for i, value in enumerate(candidate_ids)}
    if any(value not in index or value not in records for value in selected):
        raise ValueError("A selected candidate ID is absent from pixels or parameter records.")

    image_values = np.concatenate([a[np.isfinite(a)] for a in arrays[:2]])
    residual_values = arrays[2][mask & np.isfinite(arrays[2])]
    if image_values.size == 0 or residual_values.size == 0:
        raise ValueError("Finite images and at least one included residual pixel are required.")
    lo, hi = np.quantile(image_values, [.01, .995])
    hi = max(float(hi), float(lo)+1e-12)
    image_norm = AsinhNorm(linear_width=max((hi-lo)/10, 1e-12), vmin=lo, vmax=hi)
    limit = (max(float(np.quantile(abs(residual_values), .99)), 1e-12)
             if residual_limit is None else float(residual_limit))
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("residual_limit must be finite and positive.")
    residual_norm = Normalize(vmin=-limit, vmax=limit)
    residual_cmap = plt.get_cmap("RdBu_r").with_extremes(bad=".65")

    nrows = len(selected)
    fig, axes = plt.subplots(nrows, 3, squeeze=False,
        figsize=(13, 4.5*nrows+.8), layout="constrained",
        gridspec_kw={"width_ratios": [1, 1, 1.5]})
    for row, candidate_id in enumerate(selected):
        i = index[candidate_id]
        record = records[candidate_id]
        params = dict(record.get("params", {}))
        params.update(record.get("full_params", {}))
        summary = velocity_summary(params, metadata, radius_pc=radius_pc)
        for column, (key, heading) in enumerate(zip(
                ("rectified_R", "rectified_C"), ("Corrected R", "Corrected C"))):
            ax = axes[row, column]
            im = ax.imshow(arrays[column][i], origin="lower", cmap="gray",
                           norm=image_norm, interpolation="nearest")
            ax.set_title(f"Candidate {candidate_id}: {heading}", fontsize=11)
            if column == 0 and "kind" in record:
                kind = record["kind"].replace("; ", ";\n")
                ax.set_title(f"Candidate {candidate_id}: {heading}\n{kind}", fontsize=10)
            ax.set(xlabel="x [pixel]", ylabel="y [pixel]")
        residual = np.where(mask[i], arrays[2][i], np.nan)
        rim = axes[row, 2].imshow(residual, origin="lower", cmap=residual_cmap,
                                norm=residual_norm, interpolation="nearest")
        axes[row, 2].set(xlabel="x [pixel]", ylabel="y [pixel]")
        if summary["sampled_velocity_parameter"] == "V_rot":
            velocity_title = (
                rf"$V_\infty={summary['v_inf']:.0f}$ km/s (sampled asymptote)" "\n"
                rf"$V_{{\rm rot}}({summary['radius_pc']:.0f}\,{{\rm pc}})"
                rf"={summary['v_at_radius']:.0f}$ km/s (deprojected)" "\n")
        else:
            velocity_title = (
                rf"$V_{{\rm rot}}({summary['radius_pc']:.0f}\,{{\rm pc}})\sin i"
                rf"={summary['v_sini_at_radius']:.0f}$ km/s" "\n"
                rf"$V_\infty\sin i={summary['v_sini']:.0f}$ km/s (asymptote)" "\n")
        axes[row, 2].set_title(
            f"Corrected R − corrected C\n" + velocity_title +
            rf"$i={summary['inclination_deg']:.1f}^\circ$; "
            rf"$r_{{\rm turn}}={summary['rturn_pc']:.0f}$ pc", fontsize=12)
    fig.colorbar(im, ax=axes[:, :2].ravel().tolist(), shrink=.85,
                 label="Stack intensity (common image scale)")
    fig.colorbar(rim, ax=axes[:, 2].tolist(), shrink=.85, extend="both",
                 label="R − C (common symmetric scale; ends indicate saturation)")
    radius = velocity_summary(dict(records[selected[0]].get("params", {}),
        **records[selected[0]].get("full_params", {})), metadata, radius_pc=radius_pc)["radius_pc"]
    fig.suptitle((title or "Three-file R+C: compare residual structure and fitted velocity")
        + f"\nFull stamps · unsmoothed residuals · gray residual pixels excluded from comparison"
        + f"\n{radius:.0f} pc is a common display radius, not an emission-coverage measurement",
        fontsize=12)
    return fig
