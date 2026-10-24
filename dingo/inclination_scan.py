"""Optional inclination diagnostic for the three-file rotation notebook.

Fitting is isolated in a worker; loading and plotting cannot launch MCMC.
The diagnostic has its own cache and export folder, separate from the selected
posterior. Scores profile nuisance parameters; they are not posterior densities.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

import numpy as np

from .real_workflow import read_json, run_command, sha256
from .rotation import _checked
from .three_file import DEFAULT_WORKING, _paths
from .three_file_plot import candidate_figure, velocity_summary


DEFAULT_INCLINATIONS = (5., 7.5, 10., 15., 20., 25., 30., 40., 50., 60., 75., 85.)
SCHEMA = "dingo-inclination-scan-v1"
EXPORT_SCHEMA = "dingo-inclination-scan-export-v1"


def _locations(source_id, run_name, results):
    source, parent = _paths(source_id, run_name, results)
    return source, parent, parent.with_name(parent.name + "_inclination_scan")


def run_scan(source_id, *, run_name, inclinations_deg=DEFAULT_INCLINATIONS,
        iterations=80, override=False, device="cuda", results=None):
    """Run/reuse an inclination scan; never start a search or sample a posterior."""
    source, parent, output = _locations(source_id, run_name, results)
    if device not in ("cpu", "cuda") or not isinstance(override, bool):
        raise ValueError("Use device='cpu' or 'cuda' and a boolean override.")
    if isinstance(iterations, bool) or not isinstance(iterations, (int, np.integer)) or iterations < 1:
        raise ValueError("iterations must be a positive integer.")
    angles = np.asarray(inclinations_deg, dtype=float)
    if (angles.ndim != 1 or not len(angles) or not np.isfinite(angles).all()
            or len(np.unique(angles)) != len(angles)):
        raise ValueError("Supply distinct finite inclinations in degrees.")
    search = read_json(parent / "start_search.json")
    if search.get("source_id") != source.source_id:
        raise ValueError("The saved search belongs to another source.")
    if search.get("search_settings", {}).get("method") != "broad":
        raise ValueError("This diagnostic needs a saved broad search with shared comparison weights.")
    bounds = np.rad2deg([search["inclination_prior"][key] for key in ("min", "max")])
    if np.any(angles < bounds[0] - 1e-10) or np.any(angles > bounds[1] + 1e-10):
        raise ValueError(f"Use inclinations within the saved prior support {bounds[0]:g}–{bounds[1]:g} degrees.")
    if override and output.exists() and any(p.name != "run.log" for p in output.iterdir()):
        identity_path = output / "identity.json"
        if not identity_path.is_file() or read_json(identity_path).get("schema") != SCHEMA:
            raise ValueError("Override can replace only an identified inclination diagnostic, not another run.")
    arguments = ["-m", "dingo.inclination_scan_fit", "--run-dir", str(parent),
                 "--output", str(output), "--device", device, "--iterations", str(iterations),
                 "--inclinations", *[str(float(x)) for x in np.sort(angles)]]
    print(f"ID{source.source_id}: refitting at {len(angles)} fixed inclinations; no MCMC.\n"
          f"Progress and checkpoints: {output}", flush=True)
    from .run_output import output_run
    with output_run(output, source, kind="rotation", override=override):
        run_command(arguments, output / "run.log", device=device)
    return ScanResults(output, source.source_id)


def load(source_id, *, run_name, results=None):
    """Read a completed scan for this source and current saved candidate search."""
    source, parent, output = _locations(source_id, run_name, results)
    result = ScanResults(output, source.source_id)
    # The worker records a snapshot; reject a scan left behind by a fresh search.
    _checked(parent / "start_search.json", result.report["parent_search_sha256"], "Parent candidate search")
    return result


class ScanResults:
    def __init__(self, path, source_id):
        self.path = Path(path).expanduser().resolve()
        self.source_id = int(source_id)
        self.report = read_json(self.path / "report.json")
        if (self.report.get("schema") != SCHEMA or self.report.get("source_id") != self.source_id
                or self.report.get("status") != "complete"):
            raise ValueError("Choose a completed inclination scan for this source.")
        for name in ("scan_sha256", "parent_search_sha256"):
            if not isinstance(self.report.get(name), str) or len(self.report[name]) != 64:
                raise ValueError(f"Saved scan is missing its {name} provenance.")
        self._report_sha = sha256(self.path / "report.json")
        archive = self.path / "inclination_scan.npz"
        _checked(archive, self.report.get("scan_sha256"), "Inclination scan")
        with np.load(archive, allow_pickle=False) as saved:
            self.arrays = {key: saved[key].copy() for key in saved.files}
        self.angles = self.arrays["inclinations_deg"]
        if (self.angles.ndim != 1 or not len(self.angles)
                or not np.isfinite(self.angles).all() or np.any(np.diff(self.angles) <= 0)):
            raise ValueError("Saved inclination grid must be finite and increasing.")
        for key in ("log_likelihood", "log_prior", "log_posterior", "log_inclination_prior"):
            values = self.arrays[key]
            if values.shape != self.angles.shape or not np.isfinite(values).all():
                raise ValueError(f"Invalid saved scan scores: {key}.")
        if not np.allclose(self.arrays["log_posterior"],
                           self.arrays["log_likelihood"] + self.arrays["log_prior"],
                           rtol=0, atol=1e-6):
            raise ValueError("Saved scan likelihood and priors do not sum to the posterior score.")
        trials = self.report["trials"]
        ids = self.arrays["candidate_ids"]
        if (not np.array_equal(ids, np.arange(len(self.angles)))
                or len(trials) != len(self.angles)):
            raise ValueError("Saved scan images and parameter rows disagree.")
        names, points = self.arrays["param_names"].tolist(), self.arrays["points"]
        if (len(set(names)) != len(names) or points.shape != (len(self.angles), len(names))
                or not np.isfinite(points).all()):
            raise ValueError("Saved scan parameter vectors are invalid.")
        for index, (angle, row) in enumerate(zip(self.angles, trials)):
            params = dict(row.get("params", {}), **row.get("full_params", {}))
            if (any(name not in params for name in names)
                    or not np.allclose(points[index], [params[name] for name in names],
                                       rtol=0, atol=1e-10)):
                raise ValueError("Saved parameter vectors and scan records disagree.")
            actual = velocity_summary(params, self.report)["inclination_deg"]
            if not np.isclose(actual, angle, rtol=0, atol=1e-8):
                raise ValueError("A scan row was not fitted at its stated inclination.")
        for key in ("rectified_R", "rectified_C", "difference"):
            if self.arrays[key].ndim != 3 or self.arrays[key].shape[0] != len(self.angles):
                raise ValueError("Invalid saved scan image dimensions.")
        if not np.allclose(self.arrays["difference"],
                           self.arrays["rectified_R"] - self.arrays["rectified_C"],
                           rtol=1e-5, atol=1e-8, equal_nan=True):
            raise ValueError("Saved scan residuals must be corrected R minus corrected C.")

    def table(self, radius_pc=None):
        import pandas as pd
        rows = []
        for index, row in enumerate(self.report["trials"]):
            params = dict(row.get("params", {}), **row.get("full_params", {}))
            v = velocity_summary(params, self.report, radius_pc=radius_pc)
            rows.append(dict(inclination_deg=self.angles[index],
                Vrot_sin_i_kms=v["v_sini_at_radius"], Vrot_kms=v["v_at_radius"],
                radius_pc=v["radius_pc"], rturn_pc=v["rturn_pc"],
                R_dx_pixels=params.get("image.R.dx"), R_dy_pixels=params.get("image.R.dy"),
                **{k: float(self.arrays[k][index]) for k in
                   ("log_likelihood", "log_prior", "log_inclination_prior", "log_posterior")}))
        return pd.DataFrame(rows).set_index("inclination_deg")

    def figure(self, radius_pc=None):
        """One compact overview; every point represents an independently refitted model."""
        import matplotlib.pyplot as plt
        table = self.table(radius_pc)
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
        posterior = self.arrays["log_posterior"]
        likelihood = self.arrays["log_likelihood"]
        axes[0, 0].plot(self.angles, 2 * (posterior.max() - posterior), "ko-",
                        ms=3, label="Total score (includes all priors)")
        axes[0, 0].plot(self.angles, 2 * (likelihood.max() - likelihood), "k.--",
                        label="Data contribution at these fits")
        axes[0, 0].set(title="Fit scores: lower is better", ylabel="Twice the log-score difference")
        axes[0, 0].legend(fontsize=8)
        axes[0, 1].set(title="Inclination prior alone", ylabel="Twice the log-prior difference")
        # Show the continuous prior between points to distinguish a coarse-grid
        # minimum from the actual prior's favored inclination.
        spec = self.report["inclination_prior"]
        x = np.linspace(self.angles[0], self.angles[-1], 300)
        radians = np.deg2rad(x)
        if spec["type"] == "gaussian":
            lp = -.5 * ((radians - spec["mu"]) / spec["sigma"]) ** 2
            grid_lp = -.5 * ((np.deg2rad(self.angles) - spec["mu"]) / spec["sigma"]) ** 2
            peak = -.5 * ((np.clip(spec["mu"], radians[0], radians[-1]) - spec["mu"]) / spec["sigma"]) ** 2
        elif spec["type"] == "sin_i":
            lp, grid_lp = np.log(np.sin(radians)), np.log(np.sin(np.deg2rad(self.angles)))
            peak = lp.max()
        elif spec["type"] == "uniform":
            lp, grid_lp, peak = np.zeros_like(radians), np.zeros_like(self.angles), 0.
        elif spec["type"] == "log_uniform":
            lp, grid_lp = -np.log(radians), -np.log(np.deg2rad(self.angles))
            peak = lp.max()
        else:
            raise ValueError(f"Unsupported inclination prior: {spec['type']}")
        axes[0, 1].plot(self.angles, 2 * (peak - grid_lp), "ko-", ms=3)
        axes[0, 1].plot(x, 2 * (peak - lp), "k-", alpha=.35)
        axes[1, 0].plot(self.angles, table["Vrot_sin_i_kms"], "ko-", ms=3)
        axes[1, 0].set(title="Projected rotation", ylabel=r"$V_{\rm rot}(r)\sin i$ [km/s]")
        axes[1, 1].plot(self.angles, table["Vrot_kms"], "ko-", ms=3)
        axes[1, 1].set(title="Deprojected rotation", ylabel=r"$V_{\rm rot}(r)$ [km/s]")
        for ax in axes.flat:
            ax.set_xlabel("Fixed inclination [degrees]")
            ax.grid(alpha=.15)
        radius = table["radius_pc"].iloc[0]
        fig.suptitle(f"ID{self.source_id}: inclination scan · velocities at {radius:.0f} pc\n"
                     "Projected amplitude, turnover, center, angle, alignment and noise are refitted",
                     fontsize=12)
        fig.supxlabel("Score curves have separate zero points. Nuisance parameters optimized with priors; "
                      "not marginalized probabilities or MCMC intervals.\n"
                      "The display radius does not establish observed emission coverage.", fontsize=9)
        return fig

    def residuals(self, inclination_deg=None, radius_pc=None):
        """Show one saved fit, with color scales shared by the entire scan."""
        if inclination_deg is None:
            index = int(np.argmax(self.arrays["log_posterior"]))
        else:
            match = np.flatnonzero(np.isclose(self.angles, inclination_deg, rtol=0, atol=1e-8))
            if len(match) != 1:
                raise ValueError(f"Choose an inclination in the saved scan: {self.angles.tolist()}")
            index = int(match[0])
        fig = candidate_figure(self.arrays, self.report["trials"], self.report,
            ids=[index], radius_pc=radius_pc,
            title=f"ID{self.source_id}: fixed inclination {self.angles[index]:g}° · diagnostic, not a selected MCMC solution")
        from .three_file_center_plot import mark_corrected_figure, color_corrected_figure
        color_corrected_figure(fig, self.arrays)
        reference = float(self.report.get("r_fit_pixels", (self.arrays["mask"].shape[-1]-1)/2))
        mark_corrected_figure(fig, [self.report["trials"][index]], (reference, reference))
        fig.axes[0].set_title("Corrected R")
        fig.axes[1].set_title("Corrected C")
        return fig

    def export(self, working_dir=DEFAULT_WORKING, *, run_name, radius_pc=None):
        """Publish only scan arrays, plots and run information in a separate folder."""
        import matplotlib.pyplot as plt
        # Validate run_name with the same rules as the main pipeline.
        _paths(self.source_id, run_name)
        _checked(self.path / "inclination_scan.npz", self.report["scan_sha256"], "Inclination scan")
        _checked(self.path / "report.json", self._report_sha, "Inclination scan report")
        destination = Path(working_dir).expanduser().resolve() / "inclination_scans" / f"ID{self.source_id}" / run_name
        if destination == self.path or destination.is_relative_to(self.path) or self.path.is_relative_to(destination):
            raise ValueError("Keep exported scans separate from their checkpoint cache.")
        digest = sha256(self.path / "inclination_scan.npz")
        radius = float(self.table(radius_pc)["radius_pc"].iloc[0])
        expected = ["inclination_scan.npz", "inclination_scan.png", "run_info.txt",
                    *[f"residual_{angle:g}deg.png" for angle in self.angles]]
        identity = dict(scan_sha256=digest, report_sha256=sha256(self.path / "report.json"),
                        radius_pc=radius, run_name=run_name)

        def check_existing():
            if destination.is_symlink():
                raise ValueError("The exported scan directory must not be a symlink.")
            if not destination.exists():
                return None
            if not destination.is_dir() or not (destination / "inclination_scan.npz").is_file():
                raise ValueError(f"Unrelated files exist in {destination}.")
            with np.load(destination / "inclination_scan.npz", allow_pickle=False) as saved:
                metadata = json.loads(saved["metadata_json"].item())
            if (metadata.get("schema") != EXPORT_SCHEMA
                    or metadata.get("report", {}).get("source_id") != self.source_id):
                raise ValueError("Existing diagnostic files belong to another source or product.")
            allowed = metadata["files"]
            if any(p.is_symlink() or not p.is_file() or p.name not in allowed for p in destination.iterdir()):
                raise ValueError(f"Unrelated files exist in {destination}.")
            return metadata

        existing = check_existing()
        if (existing is not None and existing.get("identity") == identity
                and all((destination / name).is_file() for name in expected)):
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".inclination_export_", dir=destination.parent) as tmp:
            stage = Path(tmp) / "new"
            stage.mkdir()
            metadata = dict(schema=EXPORT_SCHEMA, report=self.report, identity=identity,
                            files=expected, exported_utc=datetime.now(timezone.utc).isoformat())
            np.savez_compressed(stage / "inclination_scan.npz", **self.arrays,
                                metadata_json=np.asarray(json.dumps(metadata, allow_nan=False)))
            for name, fig in [("inclination_scan.png", self.figure(radius_pc))]:
                try:
                    fig.savefig(stage / name, dpi=150, bbox_inches="tight")
                finally:
                    plt.close(fig)
            for angle in self.angles:
                fig = self.residuals(angle, radius_pc)
                try:
                    fig.savefig(stage / f"residual_{angle:g}deg.png", dpi=150, bbox_inches="tight")
                finally:
                    plt.close(fig)
            (stage / "run_info.txt").write_text(
                f"DINGO INCLINATION SCAN — ID{self.source_id}\nParent run: {run_name}\n"
                f"Finished (UTC): {self.report.get('finished_utc', 'not recorded')}\n"
                f"Settings: {json.dumps(self.report.get('settings', {}), sort_keys=True)}\n"
                f"Fixed inclinations [degrees]: {self.angles.tolist()}\n"
                f"Comparison radius: {radius:g} pc; emission coverage is not established.\n"
                "No MCMC. Nuisance parameters optimized with their saved priors at each inclination.\n"
                "The likelihood curve is evaluated at those fits; it is not a pure maximum-likelihood profile.\n"
                "No scan point replaces the selected candidate or posterior. No uncertainty intervals inferred.\n"
                f"Original checkpoint directory: {self.path}\n", encoding="utf-8")
            check_existing()
            previous = Path(tmp) / "previous"
            if destination.exists():
                destination.rename(previous)
            try:
                stage.rename(destination)
            except BaseException:
                if previous.exists():
                    previous.rename(destination)
                raise
        return destination

    def display(self, *, radius_pc=None):
        """One overview and one selectable residual view; no fitting or writes."""
        from IPython.display import HTML, display
        import matplotlib.pyplot as plt
        fig = self.figure(radius_pc)
        display(fig)
        plt.close(fig)
        display(HTML("<details><summary>Show all scan values</summary>" +
                     self.table(radius_pc).to_html(float_format=lambda x: f"{x:.3f}") + "</details>"))
        best = float(self.angles[np.argmax(self.arrays["log_posterior"])])
        try:
            import ipywidgets as widgets
        except ImportError:
            fig = self.residuals(best, radius_pc)
            display(fig)
            plt.close(fig)
            print("For another saved angle: display(inclination_review.residuals(15))")
            return
        selector = widgets.Dropdown(options=[(f"{angle:g}°", float(angle)) for angle in self.angles],
                                    value=best, description="Inclination:")
        output = widgets.Output()

        def update(change=None):
            with output:
                output.clear_output(wait=True)
                fig = self.residuals(selector.value, radius_pc)
                display(fig)
                plt.close(fig)

        selector.observe(update, names="value")
        display(selector, output)
        update()
        # Keep widgets and callback alive when callers retain this result.
        self._widgets = (selector, output, update)


def review(source_id, *, run_name, run=False, inclinations_deg=DEFAULT_INCLINATIONS,
           iterations=80, override=False, device="cuda", results=None,
           working_dir=DEFAULT_WORKING, radius_pc=None):
    """Notebook entrypoint: opt in to fitting; otherwise only open a saved scan."""
    if not isinstance(run, bool):
        raise ValueError("run must be True or False.")
    if run:
        result = run_scan(source_id, run_name=run_name,
            inclinations_deg=inclinations_deg, iterations=iterations, override=override,
            device=device, results=results)
    else:
        try:
            result = load(source_id, run_name=run_name, results=results)
        except FileNotFoundError:
            print("Optional inclination scan is off; no saved scan for this source/run.\n"
                  "Set RUN_INCLINATION_SCAN = True and run this cell when needed.")
            return None
    print(f"ID{result.source_id}: {len(result.angles)} saved fixed-inclination fits. No MCMC intervals.\n"
          "Residual selector only displays saved images; it does not change CANDIDATE_ID.", flush=True)
    gains = result.report.get("cross_check_max_logp_gains", [])
    if gains and gains[-1] >= .05:
        print(f"The last optimization check still improved a point by {gains[-1]:.3f} in log score; "
              "the profile may need further refinement.", flush=True)
    if working_dir is not None:
        destination = result.export(working_dir, run_name=run_name, radius_pc=radius_pc)
        print(f"Scan results and plots: {destination}", flush=True)
    result.display(radius_pc=radius_pc)
    return result
