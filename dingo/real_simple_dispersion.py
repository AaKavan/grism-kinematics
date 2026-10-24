"""CPU-only inspection of saved real-source fixed-dispersion comparisons.

Rendering and optimization stay in the separate worker. This module reads the
actual saved detector predictions, checks their provenance and pixel residuals,
and never imports JAX or interprets profile scores as confidence intervals.
"""
from pathlib import Path
import argparse
import re

import numpy as np

from .real_review import load_pixels
from .real_workflow import read_json, sha256, run_command


MODEL_LABELS = {"flexible": "Flexible emission map", "exponential": "Smooth exponential emission"}
VARIANT_LABELS = {"baseline": "Nominal LSF", "lsf_0p9": "LSF widths × 0.9", "lsf_1p1": "LSF widths × 1.1"}


def fit_simple_dispersion(source, baseline, *, run_name="simple_dispersion_v1",
                          models=("flexible", "exponential"),
                          variants=("baseline", "lsf_0p9", "lsf_1p1"),
                          sigmas=(0, 10, 20, 40, 80, 120, 200),
                          maxiter=1200, refine=4000, device="cuda"):
    """Run or resume a separate worker, leaving the notebook GPU-free."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("Use a simple run name containing only letters, digits, underscores or dashes.")
    baseline = Path(baseline).expanduser().resolve()
    if read_json(baseline / "coupled.json")["source_id"] != source.source_id:
        raise ValueError("Choose this source's own baseline fit.")
    output = source.directory / run_name
    args = ["-m", "diagnostics.native_simple_dispersion", "--source-id", str(source.source_id),
            "--baseline", str(baseline), "--output", str(output),
            "--models", *models, "--variants", *variants, "--sigmas", *map(str, sigmas),
            "--maxiter", str(maxiter), "--refine", str(refine), "--device", device]
    run_command(args, output / "run.log", device=device)
    return output


def _checked(path, expected, label):
    if not Path(path).is_file() or sha256(path) != expected:
        raise ValueError(f"{label} is missing or changed: {path}")


def validation_table(source, run_names=("simple_dispersion_v1", "simple_dispersion_fine_v1",
                                       "simple_dispersion_imaging_v1")):
    """Compare completed nominal-response checks; missing runs are ordinary status."""
    import pandas as pd
    rows, status = [], []
    for name in run_names:
        path = source.directory / name
        checkpoint = path / "dispersion.json"
        if not checkpoint.exists():
            status.append(f"{name}: not available for ID{source.source_id}.")
            continue
        report = read_json(checkpoint)
        if report.get("source_id") != source.source_id:
            raise ValueError("Validation checkpoint belongs to another source.")
        if not report.get("completed"):
            status.append(f"{name}: running or partial; omitted until complete.")
            continue
        result = SimpleDispersionResults(path, source.source_id)
        table = result.table()
        absolute = result.comparison_table()
        grid = report.get("source_grid", {})
        prior = report.get("inclination_prior", {})
        for model in result.models:
            if not any(f["model"] == model and f["variant"] == "baseline" for f in result.fits):
                continue
            fit, _ = result.candidate(model, "baseline")
            point = table.loc[table.artifact == fit["artifact"]].iloc[0]
            comparison = absolute[(absolute.model == model) & (absolute.response == "baseline")].iloc[0]
            rows.append(dict(run=name, model=model,
                prior=prior.get("kind", "imaging" if prior.get("sources") else "unspecified"),
                sky_nodes_per_axis=grid.get("shape", [None])[0],
                spectral_nodes=grid.get("spectral_quadrature_nodes"),
                sigma_kms=point.sigma_kms, V_0p5_kms=point.V_0p5_kms,
                inclination_deg=point.inclination_deg, gas_Re_arcsec=point.gas_Re_arcsec,
                gas_q=point.gas_q, chi2_pixels=point.chi2_pixels,
                chi2_above_flexible_within_run=comparison.chi2_above_flexible_same_response,
                stationary=point.stationary, boundaries=point.boundaries))
        status.append(f"{name}: complete; selected candidates verified against saved native pixels.")
    return pd.DataFrame(rows), status


class SimpleDispersionResults:
    """Inspect an immutable snapshot, including a partially completed scan."""

    def __init__(self, path, source_id):
        self.path = Path(path).expanduser().resolve()
        self.report = read_json(self.path / "dispersion.json")
        self.source_id = int(source_id)
        if (self.report.get("format") != "native_simple_dispersion_v1"
                or self.report.get("real_observations") is not True
                or self.report.get("source_id") != self.source_id):
            raise ValueError("Select this source's own real-data dispersion scan.")
        if not self.report.get("native_data_sha256"):
            raise ValueError("The worker is still preparing native pixels; check run.log and reload later.")
        _checked(self.path / "native_data.npz", self.report["native_data_sha256"], "Native pixels")
        self.pixels, self.metadata = load_pixels(self.path / "native_data.npz", self.source_id)
        for relative, expected in self.report.get("code_sha256", {}).items():
            _checked(self.path / "code" / relative, expected, "Frozen science code")
        snapshots = self.report.get("input_snapshot_paths", {})
        for original, expected in self.report.get("input_sha256", {}).items():
            snapshot = snapshots.get(original)
            actual = self.path / snapshot if snapshot else Path(original)
            _checked(actual, expected, "Saved input")
        self.fits = self.report.get("fits", [])
        if not self.fits:
            raise ValueError("The worker has not saved its first fitted point yet; check run.log.")
        self.models = tuple(dict.fromkeys(f["model"] for f in self.fits))
        self.variants = tuple(dict.fromkeys(f["variant"] for f in self.fits))
        self.pupils = np.array([e["pupil"] for e in self.metadata["exposures"]])
        self._verified = {}
        self._validate_pixel_shapes()

    def _validate_pixel_shapes(self):
        shapes = {a.shape for a in self.pixels.values()}
        if len(shapes) != 1 or self.pixels["science"].ndim != 3:
            raise ValueError("Native image arrays must share exposure, y, x dimensions.")
        valid = self.pixels["valid"]
        if valid.dtype != bool or len(valid) != len(self.pupils):
            raise ValueError("Native valid mask or exposure metadata is inconsistent.")
        if not np.all(np.isfinite(self.pixels["science"][valid])) or not np.all(
                np.isfinite(self.pixels["error"][valid]) & (self.pixels["error"][valid] > 0)):
            raise ValueError("Valid native pixels require finite SCI and positive ERR.")

    def matching(self, model="exponential", variant="baseline", sigma=None):
        fits = [f for f in self.fits if f["model"] == model and f["variant"] == variant
                and (sigma is None or np.isclose(f["fixed_sigma_kms"], sigma, atol=1e-7, rtol=0))]
        if not fits:
            raise ValueError(f"No saved {model}, {variant}, sigma={sigma} point. This scan may still be running.")
        return fits

    def best(self, model="exponential", variant="baseline", sigma=None):
        return min(self.matching(model, variant, sigma), key=lambda f: f["objective"])

    def candidate(self, model="exponential", variant="baseline", sigma=None, start=None):
        fits = self.matching(model, variant, sigma)
        if start is not None:
            fits = [f for f in fits if f["start"] == start or f["artifact"] == start]
            if not fits:
                raise ValueError("That starting branch or artifact is not saved at the selected point.")
        fit = min(fits, key=lambda f: f["objective"])
        artifact = fit["artifact"]
        if artifact not in self._verified:
            _checked(self.path / artifact, fit["artifact_sha256"], "Fit artifact")
            with np.load(self.path / artifact, allow_pickle=False) as a:
                saved = {k: a[k] for k in a.files}
            if int(saved["source_id"]) != self.source_id:
                raise ValueError("Fit artifact belongs to a different source.")
            total = self.total_prediction(saved)
            if total.shape != self.pixels["science"].shape or saved["residual"].shape != total.shape:
                raise ValueError("Fit prediction and native pixels have different dimensions.")
            valid = self.pixels["valid"]
            residual = (self.pixels["science"][valid] - total[valid]) / self.pixels["error"][valid]
            np.testing.assert_allclose(residual, saved["residual"][valid], rtol=1e-9, atol=1e-8,
                                       err_msg="Saved residuals do not match native SCI/ERR.")
            np.testing.assert_allclose(np.sum(residual**2), fit["chi_square"], rtol=1e-9, atol=1e-5,
                                       err_msg="Saved pixel chi-square is inconsistent.")
            for pupil in ("R", "C"):
                mask = valid & (self.pupils == pupil)[:, None, None]
                np.testing.assert_allclose(np.sum(saved["residual"][mask]**2), fit[f"chi_square_{pupil}"],
                                           rtol=1e-9, atol=1e-5)
            self._verified[artifact] = saved
        return fit, self._verified[artifact]

    @staticmethod
    def total_prediction(saved):
        background = saved["background"]
        if background.ndim == 1:
            background = background[:, None, None]
        return saved["line"] + saved["continuum"] + background

    def best_points(self):
        groups = sorted({(f["model"], f["variant"], float(f["fixed_sigma_kms"])) for f in self.fits})
        return [self.best(*group) for group in groups]

    def table(self, all_starts=False):
        import pandas as pd
        rows = []
        for f in self.fits if all_starts else self.best_points():
            center = f["center_east_north_arcsec"]
            gas_center = f.get("emission_center_east_north_arcsec", [np.nan, np.nan])
            boundaries = f.get("boundary_parameters", f.get("boundaries", []))
            if not isinstance(boundaries, str):
                boundaries = ", ".join(boundaries)
            n_r = int(self.pixels["valid"][self.pupils == "R"].sum())
            n_c = int(self.pixels["valid"][self.pupils == "C"].sum())
            rows.append(dict(model=f["model"], response=f["variant"], sigma_kms=f["fixed_sigma_kms"],
                V_0p5_kms=f["vrot_0p5arcsec_kms"], inclination_deg=f["inclination_deg"],
                rturn_arcsec=f["turnover_arcsec"], x_kin_arcsec=center[0], y_kin_arcsec=center[1],
                x_gas_arcsec=gas_center[0], y_gas_arcsec=gas_center[1],
                gas_Re_arcsec=f.get("emission_re_arcsec", np.nan),
                gas_q=f.get("emission_axis_ratio", np.nan),
                gas_angle_deg=f.get("emission_angle_deg", np.nan),
                two_delta_penalized=2 * (f["objective"] - self.best(f["model"], f["variant"])["objective"]),
                chi2_pixels=f["chi_square"], chi2_R=f["chi_square_R"], chi2_C=f["chi_square_C"],
                chi2_per_valid_pixel=f["chi_square"] / (n_r + n_c),
                rms_R_ERR=np.sqrt(f["chi_square_R"] / n_r), rms_C_ERR=np.sqrt(f["chi_square_C"] / n_c),
                penalty=f["penalty"], stationary=f["stationarity_passed"],
                projected_gradient=f["projected_gradient_inf"], boundaries=boundaries,
                **{"start" if all_starts else "winning_start": f["start"]},
                sweep=f.get("sweep", ""), artifact=f["artifact"]))
        return pd.DataFrame(rows)

    def comparison_table(self):
        """Absolute residual fit quality, kept separate from relative profiles."""
        table = self.table()
        rows = [table.loc[g["two_delta_penalized"].idxmin()] for _, g in table.groupby(["model", "response"])]
        import pandas as pd
        summary = pd.DataFrame(rows).reset_index(drop=True)
        flexible = {r["response"]: r["chi2_pixels"] for _, r in summary[summary.model == "flexible"].iterrows()}
        summary["chi2_above_flexible_same_response"] = [
            row["chi2_pixels"] - flexible.get(row["response"], np.nan) for _, row in summary.iterrows()]
        return summary[[
            "model", "response", "sigma_kms", "V_0p5_kms", "chi2_pixels", "chi2_R", "chi2_C",
            "chi2_above_flexible_same_response", "chi2_per_valid_pixel", "rms_R_ERR", "rms_C_ERR",
            "stationary", "boundaries"]]

    def profile_figure(self):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(self.models), squeeze=False, figsize=(7 * len(self.models), 4.7),
                                 layout="constrained")
        table = self.table()
        for ax, model in zip(axes.ravel(), self.models):
            for color, (variant, group) in enumerate(table[table.model == model].groupby("response", sort=False)):
                group = group.sort_values("sigma_kms")
                ax.plot(group.sigma_kms, group.two_delta_penalized, "o-", color=f"C{color}",
                        label=VARIANT_LABELS.get(variant, variant), ms=4)
                failed = group[~group.stationary]
                ax.scatter(failed.sigma_kms, failed.two_delta_penalized, marker="x", s=90, color="red", zorder=3)
            ax.axhline(0, color="0.5", lw=.7)
            ax.set(xlabel="Fixed intrinsic dispersion σ₀ [km/s]", ylabel="2 × change in penalized objective",
                   title=MODEL_LABELS.get(model, model))
            ax.legend(fontsize=9)
        fig.suptitle(f"ID{self.source_id}: each curve has its own zero; other parameters refitted\n"
                     "Lowest saved score = best on this scan; × failed stationarity; no confidence thresholds"
                     + ("\nPARTIAL SCAN — more fitted points are still being added" if not self.report.get("completed") else ""))
        return fig

    def pair(self, r_exposure=None, c_exposure=None):
        pair = []
        for pupil, chosen in (("R", r_exposure), ("C", c_exposure)):
            available = np.flatnonzero(self.pupils == pupil)
            if not len(available):
                raise ValueError(f"No {pupil} exposure in these saved native pixels.")
            chosen = int(available[0]) if chosen is None else int(chosen)
            if chosen not in available:
                raise ValueError(f"Choose an actual zero-based {pupil} exposure index: {available.tolist()}")
            pair.append(chosen)
        return tuple(pair)

    def fit_figure(self, model="exponential", variant="baseline", sigma=None, start=None,
                   *, r_exposure=None, c_exposure=None, full_cutout=False):
        import matplotlib.pyplot as plt
        from matplotlib.colors import AsinhNorm
        fit, saved = self.candidate(model, variant, sigma, start)
        total = self.total_prediction(saved)
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
        for row, e in enumerate(self.pair(r_exposure, c_exposure)):
            valid = self.pixels["valid"][e]
            ny, nx = valid.shape
            ys = slice(0, ny) if full_cutout else slice(max(0, ny//2-18), min(ny, ny//2+19))
            xs = slice(0, nx) if full_cutout else slice(max(0, nx//2-18), min(nx, nx//2+19))
            cut = (ys, xs)
            data = np.where(valid, self.pixels["science"][e], np.nan)
            finite = data[cut][np.isfinite(data[cut])]
            if not len(finite):
                raise ValueError("Selected exposure has no valid pixels in the displayed region.")
            lo, hi = np.quantile(finite, [.02, .995]); hi = max(hi, lo + 1e-12)
            norm = AsinhNorm(linear_width=max((hi-lo)/10, 1e-12), vmin=lo, vmax=hi)
            for col, (values, label) in enumerate(zip((data, total[e], saved["residual"][e]),
                    ("Observed SCI", "Predicted SCI: line + continuum + background", "(Observed − predicted) / ERR"))):
                style = (dict(cmap=plt.get_cmap("RdBu_r").with_extremes(bad="0.65"), vmin=-5, vmax=5)
                         if col == 2 else dict(cmap=plt.get_cmap("gist_heat").with_extremes(bad="0.65"), norm=norm))
                im = axes[row, col].imshow(np.where(valid, values, np.nan)[cut], origin="lower",
                    extent=[xs.start-.5, xs.stop-.5, ys.start-.5, ys.stop-.5], interpolation="none", **style)
                axes[row, col].set_title(f"{self.pupils[e]}, exposure {e+1}:\n{label}", fontsize=10)
                axes[row, col].set(xlabel="Cutout x [pixel]", ylabel="Cutout y [pixel]")
                colorbar = fig.colorbar(im, ax=axes[row, col], shrink=.75, label="ERR" if col == 2 else "DN/s")
                if col != 2:
                    from matplotlib.ticker import FormatStrFormatter
                    colorbar.set_ticks(norm.inverse(np.linspace(0, 1, 5)))
                    colorbar.formatter = FormatStrFormatter("%.2g")
                    colorbar.update_ticks()
        fig.suptitle(f"ID{self.source_id}: {MODEL_LABELS.get(model, model)}, {VARIANT_LABELS.get(variant, variant)}\n"
            f"Fixed σ₀={fit['fixed_sigma_kms']:g}; fitted Vrot(0.5″)={fit['vrot_0p5arcsec_kms']:.1f} km/s; "
            f"{fit['start']}; stationarity {'passed' if fit['stationarity_passed'] else 'FAILED'}\n"
            "Observed/predicted share a stretch in each row; gray pixels are excluded from the fit")
        return fig

    def source_figure(self, model="exponential", variant="baseline", sigma=None, start=None):
        import matplotlib.pyplot as plt
        from matplotlib.colors import AsinhNorm
        fit, saved = self.candidate(model, variant, sigma, start)
        side = int(np.sqrt(saved["source_flux"].size))
        if side * side != saved["source_flux"].size:
            raise ValueError("Expected a square saved source grid.")
        half = self.report.get("source_grid", {}).get("half_extent_arcsec", .9)
        east = saved.get("source_east", np.tile(np.linspace(-half, half, side), side)).reshape(side, side)
        north = saved.get("source_north", np.repeat(np.linspace(-half, half, side), side)).reshape(side, side)
        weights = np.ones(side); weights[[0, -1]] = .5
        brightness = saved["source_flux"].reshape(side, side) / np.outer(weights, weights)
        brightness /= brightness.max()
        velocity = saved["source_velocity"].reshape(side, side) - fit["systemic_offset_kms"]
        velocity = np.where(brightness > .01, velocity, np.nan)
        vmax = max(1., float(np.nanmax(abs(velocity))))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
        for ax, values, style, label, unit in zip(axes, (brightness, velocity),
                (dict(cmap="gist_heat", norm=AsinhNorm(linear_width=.05, vmin=0, vmax=1)),
                 dict(cmap=plt.get_cmap("RdBu_r").with_extremes(bad="0.85"), vmin=-vmax, vmax=vmax)),
                ("Shared intrinsic line-emission brightness", "Shared mean LOS rotation (systemic removed)"),
                ("Relative surface brightness", "km/s")):
            im = ax.pcolormesh(east, north, values, shading="nearest", **style)
            ax.plot(0, 0, "w+", ms=11, mew=2, label="Catalog origin")
            ax.plot(*fit["center_east_north_arcsec"], "cx", ms=9, mew=2, label="Kinematic center")
            if "emission_center_east_north_arcsec" in fit:
                ax.plot(*fit["emission_center_east_north_arcsec"], "+", color="lime", ms=9, mew=2, label="Gas-light center")
            ax.set(xlabel="East from catalog [arcsec]", ylabel="North from catalog [arcsec]", title=label,
                   aspect="equal")
            fig.colorbar(im, ax=ax, label=unit, shrink=.8)
        axes[0].legend(fontsize=8, loc="upper left")
        fig.suptitle(f"ID{self.source_id}: one source model projected into every R and C exposure\n"
                     f"{MODEL_LABELS.get(model, model)}; fixed σ₀={fit['fixed_sigma_kms']:g} km/s")
        return fig

    def save(self, output=None, model="exponential", variant="baseline", sigma=None, start=None):
        import matplotlib.pyplot as plt
        output = Path(output or self.path / "figures"); output.mkdir(parents=True, exist_ok=True)
        fit, _ = self.candidate(model, variant, sigma, start)
        prefix = f"{model}_{variant}_sigma{fit['fixed_sigma_kms']:g}_{Path(fit['artifact']).stem}"
        figures = {"dispersion_profiles": self.profile_figure(),
                   prefix + "_detector": self.fit_figure(model, variant, sigma, start),
                   prefix + "_source": self.source_figure(model, variant, sigma, start)}
        for name, fig in figures.items():
            fig.savefig(output / f"{name}.png", dpi=150)
            fig.savefig(output / f"{name}.pdf")
            plt.close(fig)
        self.table().to_csv(output / "best_by_sigma.csv", index=False)
        self.table(all_starts=True).to_csv(output / "all_starts.csv", index=False)
        self.comparison_table().to_csv(output / "absolute_pixel_fit_comparison.csv", index=False)
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source-id", type=int, required=True)
    parser.add_argument("--model", choices=MODEL_LABELS, default="exponential")
    parser.add_argument("--variant", choices=VARIANT_LABELS, default="baseline")
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--start")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    result = SimpleDispersionResults(args.run, args.source_id)
    print(result.comparison_table().to_string(index=False))
    print("Saved:", result.save(args.output, args.model, args.variant, args.sigma, args.start))


if __name__ == "__main__":
    main()
