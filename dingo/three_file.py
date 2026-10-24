"""Notebook interface to the original DINGO image/R/C workflow.

Fits run in a worker. Reviewing saved results needs neither Torch nor native
survey exposures. The direct image is a visual reference, not a gas template.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import re

import numpy as np

from .real_workflow import REPO, DEFAULT_RESULTS, source_number, read_json, run_command
from .rotation import _checked


DEFAULT_PARAMETERS = ("vsini", "rturn_pc", "inclination", "theta", "x0", "y0")
DEFAULT_WORKING = REPO.parent/"working"


def _input_figure(pixels):
    import matplotlib.pyplot as plt
    from .three_file_center_plot import IMAGE_CMAP, intensity_norm
    norm = intensity_norm(pixels["observed_R"], pixels["observed_C"])
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.7), layout="constrained")
    for ax, key, title in zip(axes, ("image", "observed_R", "observed_C"),
                              ("Direct image: visual reference", "Input R", "Input C")):
        ax.imshow(pixels[key], origin="lower", cmap=IMAGE_CMAP, interpolation="none",
                  norm=None if key == "image" else norm)
        ax.set(title=title, xlabel="Cutout x [pixel]", ylabel="Cutout y [pixel]")
    return fig


def _paths(source_id, run_name, results=None):
    sid = source_number(source_id)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(run_name)):
        raise ValueError("Use a run name containing letters, digits, underscores or dashes.")
    root = Path(results or os.environ.get("DINGO_RESULTS", DEFAULT_RESULTS)).expanduser().resolve()
    source = SimpleNamespace(source_id=sid, results=root,
                             directory=root/"three_file_sources"/f"ID{sid}")
    return source, source.directory/run_name


def fit(source_id, *, run_name="three_file_v1", warmup=1000, samples=4000,
        walkers=64, override=False, device="cuda", inclination_deg=None,
        inclination_sigma_deg=10., inclination_metadata=None, velocity_parameter=None, data=None, results=None,
        velocity_step=25., seed=42, candidate_id=None, selection_notes="",
        _search_only=False, n_candidates=4, n_angles=16, optimizer_steps=(2500, 1500),
        search_method="legacy", refine_iterations=100, profile_iterations=80,
        profile_step=25., profile_fine_step=1., profile_max=1000., profile_radius_pc=None):
    """Automatically search starts, optimize R/C agreement and run ensemble MCMC.

    Only image.fits, R.fits and C.fits are observation inputs. Source metadata
    and instrument calibration come from the existing local DINGO installation.
    data, if supplied, is the parent directory containing ID<source> folders.
    override=True resets only this named three-file run. Native runs are separate.
    candidate_id chooses a saved candidate for local sampling. Without it, the
    historical automatic selection remains available to programmatic callers.
    """
    source, output = _paths(source_id, run_name, results)
    data_root = Path(data or REPO/"data").expanduser().resolve()
    directory = data_root/f"ID{source.source_id}"
    config = REPO/f"data/ID{source.source_id}/config_kinematics.yaml"
    for path in [config, *(directory/name for name in ("image.fits", "R.fits", "C.fits"))]:
        if not path.is_file():
            raise FileNotFoundError(f"Required three-file input or source metadata missing: {path}")
    if directory.is_relative_to(output.resolve()):
        raise ValueError("Input images cannot be inside the run-output directory.")
    if not isinstance(override, bool) or device not in ("cpu", "cuda"):
        raise ValueError("Use a boolean override and device='cuda' or 'cpu'.")
    for name, value in dict(warmup=warmup, samples=samples, walkers=walkers).items():
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if walkers < 20:
        raise ValueError("Use at least 20 ensemble walkers.")
    if not np.isfinite(velocity_step) or not 0 < velocity_step <= 500:
        raise ValueError("velocity_step must be between 0 and 500 km/s.")
    if (source.results/"three_file_sources").is_symlink():
        raise ValueError("The three-file results directory must not be a symlink.")
    if not isinstance(selection_notes, str):
        raise ValueError("selection_notes must be text.")
    if velocity_parameter not in (None, "V_rot", "V_sini"):
        raise ValueError("velocity_parameter must be 'V_rot' or 'V_sini'.")
    if candidate_id is not None:
        if isinstance(candidate_id, bool) or not isinstance(candidate_id, (int, np.integer)) or candidate_id < 0:
            raise ValueError("candidate_id must be an integer from the candidate table.")
        if override:
            raise ValueError("Run the search with override=True first, inspect its new candidates, then fit with override=False.")
        prepared = CandidateResults(output, source.source_id)
        if velocity_parameter is not None and velocity_parameter != prepared.velocity_parameter:
            raise ValueError("The selected search samples a different velocity parameter; use a new RUN_NAME and search.")
        velocity_parameter = prepared.velocity_parameter
        if candidate_id not in prepared.candidate_ids:
            raise ValueError(f"Candidate {candidate_id} is not in the saved candidate table.")
        settings = prepared.report.get("search_settings", {})
        search_method = settings.get("method", "legacy")
        refine_iterations = settings.get("refine_iterations", refine_iterations)
        profile_iterations = settings.get("profile_iterations", profile_iterations)
        profile_step = settings.get("profile_step", profile_step)
        profile_fine_step = settings.get("profile_fine_step", profile_fine_step)
        profile_max = settings.get("profile_max", profile_max)
        profile_radius_pc = settings.get("profile_radius_pc", profile_radius_pc)
        velocity_step = settings.get("amp_step", velocity_step)
        n_candidates = settings.get("n_keep", n_candidates)
        n_angles = settings.get("n_theta", n_angles)
        optimizer_steps = settings.get("optimizer_steps", optimizer_steps)
        saved = prepared.inclination_settings()
        if inclination_deg is not None and (
                saved["inclination_deg"] is None
                or not np.isclose(inclination_deg, saved["inclination_deg"], rtol=1e-12, atol=1e-12)
                or not np.isclose(inclination_sigma_deg, saved["inclination_sigma_deg"], rtol=1e-12, atol=1e-12)):
            raise ValueError("Use the selected search's saved inclination prior, or run a new search.")
        if inclination_metadata is not None and inclination_metadata != saved["inclination_metadata"]:
            raise ValueError("The supplied imaging provenance differs from the selected search.")
        inclination_deg = saved["inclination_deg"]
        inclination_sigma_deg = saved["inclination_sigma_deg"]
        inclination_metadata = saved["inclination_metadata"]
    velocity_parameter = velocity_parameter or "V_sini"
    if inclination_deg is not None and (not np.isfinite(inclination_deg)
            or not 5 < inclination_deg < 85 or not np.isfinite(inclination_sigma_deg)
            or inclination_sigma_deg <= 0):
        raise ValueError("Supply an inclination between 5 and 85 degrees and a positive uncertainty.")
    from .three_file_fit import validate_inclination_metadata
    inclination_metadata = validate_inclination_metadata(
        inclination_metadata, source.source_id, inclination_deg, inclination_sigma_deg)
    if (not isinstance(n_candidates, (int, np.integer)) or n_candidates < 1
            or not isinstance(n_angles, (int, np.integer)) or n_angles < 2
            or len(optimizer_steps) != 2
            or any(not isinstance(n, (int, np.integer)) or n < 1 for n in optimizer_steps)):
        raise ValueError("Use positive candidate/optimizer counts and at least two search angles.")
    if search_method not in ("legacy", "broad"):
        raise ValueError("search_method must be 'legacy' or 'broad'.")
    module = "dingo.three_file_search" if search_method == "broad" else "dingo.three_file_fit"
    arguments = ["-m", module, "--source", str(source.source_id),
                 "--data-dir", str(data_root), "--config", str(config), "--output", str(output),
                 "--device", device, "--warmup", str(warmup), "--draws", str(samples),
                 "--walkers", str(walkers), "--amp-step", str(velocity_step), "--seed", str(seed),
                 "--velocity-parameter", velocity_parameter,
                 "--n-keep", str(n_candidates), "--n-theta", str(n_angles),
                 "--optimizer-steps", *(str(n) for n in optimizer_steps)]
    if search_method == "broad":
        for name, value in dict(refine_iterations=refine_iterations, profile_iterations=profile_iterations,
                profile_step=profile_step, profile_fine_step=profile_fine_step,
                profile_max=profile_max, profile_radius_pc=profile_radius_pc).items():
            if value is not None:
                if not np.isfinite(value) or value < 0 or (value == 0 and name != "profile_fine_step"):
                    raise ValueError(f"Invalid {name}.")
                arguments += ["--"+name.replace("_", "-"), str(value)]
    if inclination_deg is not None:
        arguments += ["--inclination-deg", str(inclination_deg),
                      "--inclination-sigma-deg", str(inclination_sigma_deg)]
    if inclination_metadata is not None:
        arguments += ["--inclination-metadata-json", json.dumps(inclination_metadata, allow_nan=False)]
    if candidate_id is not None:
        arguments += ["--candidate", str(candidate_id), "--selection-notes", selection_notes]
    if _search_only:
        arguments += ["--search-only"]
    action = ("Search and display candidate residuals; no MCMC." if _search_only else
              f"{'Selected candidate '+str(candidate_id) if candidate_id is not None else 'Automatic candidate'}; "
              f"{walkers} walkers × {warmup:,} burn-in + {samples:,} retained steps.")
    print(f"ID{source.source_id}: image.fits + R.fits + C.fits; intrinsic dispersion = 0.\n"
          f"Sampled velocity: {velocity_parameter}; inclination is sampled separately.\n"
          f"{action}\n"
          f"Progress and checkpoints: {output}", flush=True)
    from .run_output import output_run
    with output_run(output, source, kind="rotation", override=override):
        run_command(arguments, output/"run.log", device=device)
    return (CandidateResults(output, source.source_id) if _search_only else
            ThreeFileResults(output, source.source_id))


def search(source_id, *, run_name="visual_v1", override=False, device="cuda",
           inclination_deg=None, inclination_sigma_deg=10., inclination_metadata=None,
           velocity_parameter=None, data=None, results=None,
           velocity_step=25., seed=42, n_candidates=4, n_angles=16,
           optimizer_steps=(2500, 1500), search_method="legacy", refine_iterations=100,
           profile_iterations=80, profile_step=25., profile_fine_step=1.,
           profile_max=1000., profile_radius_pc=None):
    """Find and render candidate R/C fits without running MCMC."""
    return fit(source_id, run_name=run_name, override=override, device=device,
               inclination_deg=inclination_deg, inclination_sigma_deg=inclination_sigma_deg,
               inclination_metadata=inclination_metadata, velocity_parameter=velocity_parameter,
               data=data, results=results, velocity_step=velocity_step, seed=seed,
               _search_only=True, n_candidates=n_candidates, n_angles=n_angles,
               optimizer_steps=optimizer_steps, search_method=search_method,
               refine_iterations=refine_iterations, profile_iterations=profile_iterations,
               profile_step=profile_step, profile_fine_step=profile_fine_step,
               profile_max=profile_max, profile_radius_pc=profile_radius_pc)


def load_candidates(source_id, *, run_name="visual_v1", results=None):
    """Review saved candidate images without searching, fitting or deleting."""
    source, path = _paths(source_id, run_name, results)
    return CandidateResults(path, source.source_id)


def load(source_id, *, run_name="three_file_v1", results=None, candidate_id=None):
    """Load only saved three-file results, without fitting or deleting anything."""
    source, path = _paths(source_id, run_name, results)
    result = ThreeFileResults(path, source.source_id)
    if candidate_id is not None and result.report.get("chosen_candidate_id") != candidate_id:
        raise ValueError("The saved posterior belongs to a different candidate; choose its candidate ID or a separate run.")
    return result


def load_final(source_id, *, working_dir=DEFAULT_WORKING, candidate_id=None):
    """Read a self-contained final source folder; the run cache is not needed."""
    sid = source_number(source_id)
    path = Path(working_dir).expanduser().resolve()/f"ID{sid}"
    result = ThreeFileResults(path, sid, final_bundle=True)
    if candidate_id is not None and result.report.get("chosen_candidate_id") != candidate_id:
        raise ValueError("The final posterior belongs to a different candidate.")
    return result


class CandidateResults:
    """Saved optimized candidates, with original trial IDs and comparable images."""
    def __init__(self, path, source_id):
        self.path = Path(path).expanduser().resolve()
        self.source_id = source_number(source_id)
        self.report = read_json(self.path/"start_search.json")
        if (self.report.get("schema") != "dingo-three-file-candidates-v1"
                or self.report.get("source_id") != self.source_id):
            raise ValueError("Choose this source's saved visual candidate search.")
        _checked(self.path/"candidates.npz", self.report.get("candidates_sha256"), "Candidate images")
        with np.load(self.path/"candidates.npz", allow_pickle=False) as archive:
            self.pixels = {k: archive[k].copy() for k in archive.files}
        ids = np.asarray(self.pixels["candidate_ids"])
        trials = self.report["trials"]
        if (ids.ndim != 1 or not len(ids) or not np.issubdtype(ids.dtype, np.integer)
                or len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= len(trials))):
            raise ValueError("Invalid candidate IDs in the saved candidate images.")
        self.candidate_ids = ids.astype(int).tolist()
        arrays = [self.pixels[k] for k in ("rectified_R", "rectified_C", "difference")]
        if (any(a.ndim != 3 or a.shape[0] != len(ids) for a in arrays)
                or len({a.shape for a in arrays}) != 1
                or not np.allclose(arrays[2], arrays[0]-arrays[1], equal_nan=True)):
            raise ValueError("Saved candidate image arrays disagree.")

    @property
    def velocity_parameter(self):
        """Amplitude actually optimized in this saved search."""
        amplitudes = set()
        for cid in self.candidate_ids:
            params = self.report["trials"][cid]["params"]
            found = {name for name in ("V_rot", "V_sini") if name in params}
            if len(found) != 1:
                raise ValueError("Saved candidate must contain exactly one velocity amplitude.")
            amplitudes.update(found)
        if len(amplitudes) != 1:
            raise ValueError("Saved candidates use different velocity parameters.")
        parameter = amplitudes.pop()
        if self.report.get("velocity_parameter", parameter) != parameter:
            raise ValueError("Saved search velocity definition disagrees with its parameters.")
        return parameter

    def inclination_settings(self):
        """Reuse the saved search prior exactly, including its imaging provenance."""
        prior = self.report.get("inclination_prior")
        if prior is None:
            # Older visual searches saved the prior in their worker config only.
            import yaml
            path = self.path/"search_config.yaml"
            if not path.exists():
                raise ValueError("The saved search has no recorded inclination prior; run a new search.")
            config = yaml.safe_load(path.read_text())
            prior = config.get("mcmc", {}).get("priors", {}).get("velocity.inc_v")
        if not isinstance(prior, dict) or prior.get("type") not in ("sin_i", "gaussian"):
            raise ValueError("Unrecognized saved inclination prior; inspect this search's configuration.")
        if not np.allclose([prior.get("min", np.nan), prior.get("max", np.nan)],
                           np.deg2rad([5., 85.]), rtol=1e-12, atol=1e-12):
            raise ValueError("The saved inclination support differs from this workflow's 5–85 degrees.")
        metadata = self.report.get("inclination_metadata")
        if prior["type"] == "sin_i":
            mean, sigma = None, 10.
        else:
            exact = self.report.get("inclination_settings") or metadata or {}
            mean = exact.get("inclination_deg", float(np.rad2deg(prior["mu"])))
            sigma = exact.get("inclination_sigma_deg", float(np.rad2deg(prior["sigma"])))
            if (not np.isfinite([mean, sigma]).all() or not 5 < mean < 85 or sigma <= 0
                    or not np.allclose(np.deg2rad([mean, sigma]), [prior["mu"], prior["sigma"]],
                                       rtol=1e-12, atol=1e-12)):
                raise ValueError("The recorded inclination settings disagree with the saved search prior.")
        from .three_file_fit import validate_inclination_metadata
        metadata = validate_inclination_metadata(metadata, self.source_id, mean, sigma)
        return dict(inclination_deg=mean, inclination_sigma_deg=sigma, inclination_metadata=metadata)

    def table(self, radius_pc=None):
        import pandas as pd
        from .three_file_plot import velocity_summary
        rows = []
        for cid in self.candidate_ids:
            trial = self.report["trials"][cid]
            v = velocity_summary(trial["params"], self.report, radius_pc=radius_pc)
            rows.append(dict(candidate=cid, Vrot_at_radius_kms=v["v_at_radius"],
                Vsin_i_at_radius_kms=v["v_sini_at_radius"],
                radius_pc=v["radius_pc"], Vinf_kms=v["v_inf"], Vinf_sin_i_kms=v["v_sini"],
                inclination_deg=v["inclination_deg"], turnover_pc=v["rturn_pc"],
                residual_sum_squares=trial["loss"]))
            if "log_posterior" in trial:
                best = max(t["log_posterior"] for t in self.report["trials"])
                rows[-1].update(kind=trial["kind"],
                    twice_delta_log_posterior=2*(best-trial["log_posterior"]),
                    log_likelihood=trial["log_likelihood"], log_prior=trial["log_prior"],
                    R_dx_pixels=trial["full_params"]["image.R.dx"],
                    R_dy_pixels=trial["full_params"]["image.R.dy"],
                    inclination_near_prior_edge=min(v["inclination_deg"]-5., 85.-v["inclination_deg"]) < 1.)
        return pd.DataFrame(rows).set_index("candidate")

    def search_notes(self):
        if self.report.get("seed_kind") == "manual_start":
            return self.report["selection_note"]
        if self.report.get("search_settings", {}).get("method") != "broad":
            return "Legacy candidate search."
        table = self.table(radius_pc=self.report["search_settings"]["profile_radius_pc"])
        best = table.loc[self.report["selected"]]
        notes = ["Scores use one fixed comparison model. Fixed-velocity rows are constrained comparisons; free MCMC can move."]
        if best["inclination_near_prior_edge"]:
            notes.append("The best-scoring candidate is within 1 degree of an inclination limit: its deprojected velocity is boundary-sensitive.")
        if best["Vrot_at_radius_kms"] > self.report["search_settings"]["profile_max"]:
            notes.append("The best free candidate lies above the plotted velocity range; the profile has not bracketed it.")
        path = self.path/"velocity_profile.npz"
        _checked(path, self.report["velocity_profile_sha256"], "Velocity profile")
        with np.load(path, allow_pickle=False) as saved:
            if "last_pass_max_logp_gain" in saved and float(saved["last_pass_max_logp_gain"]) >= .05:
                notes.append("The last profile cross-check still improved some points; treat the curve as a finite search, not a certified global optimum.")
        return "\n".join(notes)

    def velocity_profile(self, *, save=False):
        """Read the profiled posterior score; this is not a marginalized posterior."""
        import matplotlib.pyplot as plt
        path = self.path/"velocity_profile.npz"
        if "velocity_profile_sha256" not in self.report:
            raise ValueError("This older search has no velocity profile; run a broad search with a new name.")
        _checked(path, self.report["velocity_profile_sha256"], "Velocity profile")
        with np.load(path, allow_pickle=False) as saved:
            velocity, score = saved["velocity_kms"], saved["log_posterior"]
            radius = float(saved["radius_pc"])
        best = max(np.max(score), max(t["log_posterior"] for t in self.report["trials"]))
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6), layout="constrained")
        for ax in axes:
            ax.plot(velocity, 2*(best-score), "k.-", ms=3)
            ax.set_ylabel(r"$2(\ln P_{\rm best}-\ln P_{\rm profile})$")
            ax.set_ylim(bottom=0)
        axes[0].set_title(f"ID{self.source_id}: other parameters refitted at each velocity")
        axes[1].set(title="Same curve, zoomed to score differences below 50", ylim=(0, 50),
                    xlabel=f"Deprojected rotation speed at {radius:.0f} pc [km/s]")
        fig.supxlabel("Lower is better. Profile includes priors; not a probability density or an MCMC error interval.", fontsize=9)
        if save:
            out = self.path/"figures"; out.mkdir(exist_ok=True)
            fig.savefig(out/"velocity_profile.png", dpi=160)
        return fig

    def input_figure(self):
        return _input_figure(self.pixels)

    def figure(self, candidate_ids=None, radius_pc=None, *, save=False):
        from .three_file_plot import candidate_figure
        from .three_file_center_plot import mark_corrected_figure, color_corrected_figure
        fig = candidate_figure(self.pixels, self.report["trials"], self.report,
                              ids=candidate_ids, radius_pc=radius_pc)
        color_corrected_figure(fig, self.pixels)
        selected = self.candidate_ids if candidate_ids is None else list(candidate_ids)
        reference = float(self.report["r_fit_pixels"])
        mark_corrected_figure(fig, [self.report["trials"][i] for i in selected], (reference, reference))
        if save:
            out = self.path/"figures"; out.mkdir(exist_ok=True)
            for ext in ("png", "pdf"):
                fig.savefig(out/f"candidates.{ext}", dpi=160, bbox_inches="tight")
        return fig


class ThreeFileResults:
    def __init__(self, path, source_id, *, final_bundle=False):
        self.path = Path(path).expanduser().resolve()
        self.source_id = source_number(source_id)
        self._final_bundle = self.path/"results.npz" if final_bundle else None
        if final_bundle:
            from .three_file_export import read_metadata
            self.report = read_metadata(self._final_bundle)["report"]
        else:
            self.report = read_json(self.path/"report.json")
        if (self.report.get("schema") != "dingo-three-file-v1"
                or self.report.get("source_id") != self.source_id
                or self.report.get("intrinsic_dispersion_kms") != 0):
            raise ValueError("Choose this source's three-file rotation-only run.")
        self._sample_cache = None

    @property
    def velocity_parameter(self):
        names = self.report.get("param_names", [])
        present = [name for name in ("V_rot", "V_sini") if f"velocity.{name}" in names]
        if len(present) != 1:
            raise ValueError("Saved posterior must sample exactly one velocity amplitude.")
        parameter = present[0]
        if self.report.get("velocity_parameter", parameter) != parameter:
            raise ValueError("Saved posterior velocity definition disagrees with its parameters.")
        return parameter

    def default_parameters(self):
        parameters = (("vrot",)+DEFAULT_PARAMETERS[1:] if self.velocity_parameter == "V_rot"
                      else DEFAULT_PARAMETERS)
        return tuple(p for p in parameters if p != "inclination" or self.fixed_inclination_deg is None)

    @property
    def fixed_inclination_deg(self):
        """Return a genuinely fixed angle; never infer it from a narrow chain."""
        sampled = "velocity.inc_v" in self.report.get("param_names", [])
        fixed = self.report.get("fixed_params", {}).get("velocity.inc_v")
        declared = self.report.get("inclination_mode") == "fixed"
        if sampled:
            if declared or fixed is not None:
                raise ValueError("Saved inclination cannot be both sampled and fixed.")
            return None
        if fixed is None:
            if declared:
                raise ValueError("The fixed inclination is missing from fixed_params.")
            return None
        value = float(np.rad2deg(fixed))
        recorded = self.report.get("fixed_inclination_deg", value)
        if not np.isfinite(value) or not 0 < value < 90 or not np.isclose(value, recorded):
            raise ValueError("The recorded fixed inclination disagrees with fixed_params.")
        return value

    def _inclination_note(self):
        fixed = self.fixed_inclination_deg
        return ("" if fixed is None else
                f"Fixed i = {fixed:g}°; intervals are conditional on this inclination.")

    def _plot_parameters(self, parameters):
        parameters = self.default_parameters() if parameters is None else tuple(parameters)
        # A fixed value is not a posterior dimension, even when an older
        # notebook explicitly requests the usual six-parameter display.
        if self.fixed_inclination_deg is not None:
            parameters = tuple(p for p in parameters if p != "inclination")
        return parameters

    @property
    def status(self):
        r = self.report
        verdict = ("Sampling-length checks passed; errors use the approximate stacked-image noise model."
                   if r.get("sampling_pass") else "Sampling-length checks not passed; intervals are provisional.")
        selected = (f"MCMC initialized at candidate {r['chosen_candidate_id']}; inspect any movement from that solution.\n"
                    if "chosen_candidate_id" in r else "")
        if r.get("seed_kind") == "inclination_scan":
            prior = r["inclination_prior"]
            selected = (f"Initialized from the {r['seed_inclination_deg']:g}° scan fit. "
                        f"Inclination sampled with Gaussian prior {np.rad2deg(prior['mu']):g}° "
                        f"± {np.rad2deg(prior['sigma']):g}° (1σ; 5–85° support).\n")
        if r.get("seed_kind") == "manual_start":
            selected = (f"Manual start: V_inf={r['start_velocity_kms']:g} km/s, "
                        f"i={r['start_inclination_deg']:g}°; "
                        + ("both sampled. " if self.fixed_inclination_deg is None else "V_inf sampled; i fixed. ") +
                        f"Other initial values from parent candidate {r['base_candidate_id']}.\n")
        if self.fixed_inclination_deg is not None:
            selected += self._inclination_note()+"\n"
        return (f"ID{self.source_id} | three-file DINGO | {r.get('status', 'unknown')}\n"
                f"{r.get('walkers', '?')} ensemble walkers × {r.get('draws', '?')} retained steps; σ₀ = 0.\n"
                f"{selected}{verdict}\n{self.path}")

    def __repr__(self):
        return self.status

    def diagnostics(self):
        import pandas as pd
        d = self.report.get("diagnostics", {})
        return pd.DataFrame([dict(parameter=name, autocorrelation_steps=tau,
            retained_steps_per_autocorrelation=d.get("n_steps_over_tau", {}).get(name),
            effective_samples=d.get("n_eff", {}).get(name),
            split_half_shift=d.get("split_half_shift", {}).get(name))
            for name, tau in d.get("tau", {}).items()])

    def _samples(self):
        if self._sample_cache is None:
            path = self._final_bundle or self.path/"samples.npz"
            if self._final_bundle is None:
                _checked(path, self.report.get("samples_sha256"), "Three-file posterior samples")
            with np.load(path, allow_pickle=False) as a:
                chain, names = a["chain"].copy(), a["param_names"].astype(str).tolist()
            if (chain.ndim != 3 or chain.shape[-1] != len(names) or chain.shape[0] < 1
                    or not np.isfinite(chain).all() or len(set(names)) != len(names)
                    or chain.shape[1] != self.report["walkers"]
                    or chain.shape[0] != self.report["draws"]
                    or names != self.report["param_names"]):
                raise ValueError("Invalid retained three-file sample dimensions or values.")
            self._sample_cache = chain, names
        return self._sample_cache

    def physical_scale(self):
        from astropy.cosmology import Planck18
        from astropy import units as u
        z = float(self.report["redshift"])
        pixel = float(self.report["pixel_scale_arcsec"])
        if not np.isfinite([z, pixel]).all() or z <= 0 or pixel <= 0:
            raise ValueError("Positive source redshift and pixel scale required for physical radii.")
        scale = float(Planck18.kpc_proper_per_arcmin(z).to_value(u.pc/u.arcsec))
        return dict(redshift=z, pc_per_arcsec=scale, pc_per_pixel=scale*pixel,
                    reference_radius_pc=.5*scale, cosmology="Planck18")

    def _columns(self, parameters=None):
        """Posterior display columns; pixel-only requests need no distance scale."""
        self.fixed_inclination_deg  # Validate fixed/sample bookkeeping before deriving velocities.
        chain, names = self._samples()
        def column(name):
            if name in names:
                return chain[..., names.index(name)]
            fixed = self.report.get("fixed_params", {})
            if name not in fixed:
                raise ValueError(f"Missing fitted or fixed parameter: {name}")
            return np.full(chain.shape[:2], fixed[name], dtype=float)
        inc = column("velocity.inc_v")
        if self.velocity_parameter == "V_rot":
            vrot = column("velocity.V_rot")
            vsini = vrot*np.sin(inc)
        else:
            vsini = column("velocity.V_sini")
            vrot = vsini/np.sin(inc)
        if np.any((inc <= 0) | (inc >= np.pi/2)) or np.any(vsini < 0) or np.any(column("velocity.R_v") <= 0):
            raise ValueError("Invalid inclination, velocity or turnover radius in posterior.")
        angle = column("velocity.theta_v")
        origin = np.arctan2(np.sin(angle).mean(), np.cos(angle).mean())
        angle = origin+np.angle(np.exp(1j*(angle-origin)))
        center = float(self.report["r_fit_pixels"])
        columns = {
            "vrot": (vrot, r"$V_\infty$ [km/s]"),
            "vsini": (vsini, r"$V_\infty\sin i$ [km/s]"),
            "rturn_pixels": (column("velocity.R_v"), r"$r_{\rm turn}$ [pixel]"),
            "inclination": (np.rad2deg(inc), r"$i$ [deg]"),
            "theta": (np.rad2deg(angle), r"$\theta$ [deg]"),
            "x0": (column("velocity.x0_v")-center, r"$x_0-x_{\rm cut}$ [pixel]"),
            "y0": (column("velocity.y0_v")-center, r"$y_0-y_{\rm cut}$ [pixel]"),
        }
        if parameters is None or "rturn_pc" in parameters:
            columns["rturn_pc"] = (column("velocity.R_v")*self.physical_scale()["pc_per_pixel"],
                                    r"$r_{\rm turn}$ [pc]")
        if not all(np.isfinite(v).all() for v, _ in columns.values()):
            raise ValueError("Nonfinite posterior display parameters.")
        return columns

    def corner(self, parameters=None, *, save=False):
        import corner
        parameters = self._plot_parameters(parameters)
        columns = self._columns(parameters)
        if len(parameters) < 2 or len(set(parameters)) != len(parameters) or any(p not in columns for p in parameters):
            raise ValueError(f"Choose at least two distinct parameters from {tuple(columns)}.")
        draws = np.stack([columns[p][0].ravel() for p in parameters], axis=-1)
        bounds = []
        for low, high in zip(draws.min(axis=0), draws.max(axis=0)):
            pad = max((high-low)*.025, abs(low)*1e-6, 1e-6)
            bounds.append((low-pad, high+pad))
        fig = corner.corner(draws, labels=[columns[p][1] for p in parameters], range=bounds,
            color="black", bins=35, smooth=1., levels=(.68, .95), quantiles=(.16, .5, .84),
            plot_datapoints=False, plot_density=False, fill_contours=True, quiet=True,
            contour_kwargs={"colors": "black"}, contourf_kwargs={"colors": ["white", ".85", ".65"]},
            hist_kwargs={"color": "black", "density": True}, max_n_ticks=3, use_math_text=True)
        for i in range(len(parameters)):
            lo, med, hi = np.quantile(draws[:, i], [.16, .5, .84])
            decimals = 2 if parameters[i] in ("x0", "y0", "rturn_pixels") else 1
            fig.axes[i*len(parameters)+i].set_title(
                f"${med:.{decimals}f}_{{-{med-lo:.{decimals}f}}}^{{+{hi-med:.{decimals}f}}}$", fontsize=11)
        fig.set_size_inches(2.2*len(parameters), 2.2*len(parameters))
        fig.subplots_adjust(top=.87, bottom=.09, left=.10, right=.98, hspace=.08, wspace=.08)
        fig.suptitle(f"ID{self.source_id}: three-file R+C rotation", y=.985, fontsize=15)
        fig.text(.5, .952, "Pooled retained walkers · black 68/95% contours · median and 16–84% intervals", ha="center", fontsize=10)
        note = "Approximate stacked-image noise model" if self.report.get("sampling_pass") else "Provisional: sampling-length checks not passed"
        definitions = []
        if self.fixed_inclination_deg is not None:
            definitions.append(self._inclination_note())
        if "vsini" in parameters:
            kind = "sampled" if self.velocity_parameter == "V_sini" else "derived"
            definitions.append(f"V∞ sin i is the {kind} projected asymptote")
        if "vrot" in parameters:
            kind = "sampled" if self.velocity_parameter == "V_rot" else "derived"
            definitions.append(f"V∞ is the {kind} deprojected asymptote")
        if any(p in parameters for p in ("x0", "y0")):
            center = float(self.report["r_fit_pixels"])
            definitions.append(f"center offsets are in pixels; zero = cutout center ({center:g}, {center:g})")
        fig.text(.5, .925, f"{note}\n" + "; ".join(definitions), ha="center", fontsize=9)
        if save: self._save_figure(fig, "corner")
        return fig

    def traces(self, parameters=None):
        import matplotlib.pyplot as plt
        parameters = self._plot_parameters(parameters)
        columns = self._columns(parameters)
        if not parameters or len(set(parameters)) != len(parameters) or any(p not in columns for p in parameters):
            raise ValueError(f"Choose distinct parameters from {tuple(columns)}.")
        ncols = min(2, len(parameters))
        nrows = (len(parameters)+ncols-1)//ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 2.7*nrows),
                                 squeeze=False, layout="constrained")
        for ax, key in zip(axes.ravel(), parameters):
            values, label = columns[key]
            for j in np.linspace(0, values.shape[1]-1, min(8, values.shape[1]), dtype=int):
                ax.plot(np.arange(1, len(values)+1), values[:, j], color=".6", alpha=.4, lw=.5)
            ax.plot(np.arange(1, len(values)+1), np.median(values, axis=1), color="black", lw=.8)
            ax.set(xlabel="Retained ensemble step", ylabel=label)
        for ax in axes.ravel()[len(parameters):]:
            fig.delaxes(ax)
        fig.suptitle(f"ID{self.source_id}: selected walker traces (gray) and ensemble median (black)\n"
                     "Walkers interact; they are not independent chains"
                     + ("\n"+self._inclination_note() if self.fixed_inclination_deg is not None else ""))
        return fig

    def velocity_summary(self, radius_pc=None, *, radius_pixels=None,
                         include_deprojected=None, include_inclination=True):
        """Lead with the sampled velocity definition; derive others draw by draw."""
        import pandas as pd
        if radius_pc is not None and radius_pixels is not None:
            raise ValueError("Choose a comparison radius in either pixels or pc, not both.")
        if radius_pixels is not None:
            radius, unit, key = float(radius_pixels), "pixel", "rturn_pixels"
        else:
            scale = self.physical_scale()
            radius = scale["reference_radius_pc"] if radius_pc is None else float(radius_pc)
            unit, key = "pc", "rturn_pc"
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError(f"Choose a positive comparison radius in {unit}.")
        columns = self._columns((key,))
        radius_label = f"{radius:g} pixel" if unit == "pixel" else f"{radius:.0f} pc"
        shape = (2/np.pi)*np.arctan(radius/columns[key][0])
        projected_quantities = {
            f"Vrot({radius_label}) sin(i) [km/s]": columns["vsini"][0]*shape,
            "Vinf sin(i) [km/s]": columns["vsini"][0],
        }
        deprojected_quantities = {"Vinf [km/s]": columns["vrot"][0],
                                 f"Vrot({radius_label}) [km/s]": columns["vrot"][0]*shape}
        sampled_v = self.velocity_parameter == "V_rot"
        if include_deprojected is None:
            include_deprojected = sampled_v
        quantities = dict(deprojected_quantities) if sampled_v and include_deprojected else {}
        if not sampled_v:
            quantities.update(projected_quantities)
        if include_inclination and self.fixed_inclination_deg is None:
            quantities["Inclination [deg]"] = columns["inclination"][0]
        if sampled_v:
            quantities.update(projected_quantities)
        elif include_deprojected:
            quantities.update(deprojected_quantities)
        rows = []
        for label, values in quantities.items():
            lo, med, hi = np.quantile(values, [.16, .5, .84])
            rows.append(dict(parameter=label, median=med, minus_1sigma=med-lo, plus_1sigma=hi-med))
        return pd.DataFrame(rows).set_index("parameter")

    def _prediction_pixels(self):
        path = self._final_bundle or self.path/"predictions.npz"
        if self._final_bundle is None:
            _checked(path, self.report.get("predictions_sha256"), "Three-file rectified images")
        with np.load(path, allow_pickle=False) as a:
            return ({k.removeprefix("fit_"): a[k].copy() for k in a.files if k.startswith("fit_")}
                    if self._final_bundle is not None else {k: a[k].copy() for k in a.files})

    def fit_figure(self, radius_pc=None, *, save=False):
        """Large raw R−C residual, with velocities of the actual plotted sample."""
        from .three_file_plot import candidate_figure
        p = self._prediction_pixels()
        names = p["param_names"].astype(str).tolist()
        actual = dict(self.report.get("fixed_params", {}))
        actual.update(zip(names, p["theta"].tolist()))
        params = {key.removeprefix("velocity."): value for key, value in actual.items()
                  if key.startswith("velocity.")}
        cid = self.report.get("chosen_candidate_id", 0)
        pixels = {key: p[key][None] for key in ("rectified_R", "rectified_C", "difference")}
        pixels.update(candidate_ids=np.asarray([cid]), mask=p["mask"])
        fig = candidate_figure(pixels, {cid: {"params": params}}, self.report,
            radius_pc=radius_pc, title=f"ID{self.source_id}: highest-posterior retained fit · σ₀ = 0")
        from .three_file_center_plot import mark_corrected_figure, color_corrected_figure
        color_corrected_figure(fig, pixels)
        reference = float(self.report["r_fit_pixels"])
        mark_corrected_figure(fig, [{"params": params}], (reference, reference))
        if save: self._save_figure(fig, "fit")
        return fig

    def center_geometry(self):
        """Positions of the saved displayed fit's center in each image frame."""
        from .three_file_centers import center_geometry
        return center_geometry(self)

    def center_figure(self, radius_pc=None, *, half_width=15., save=False):
        """Inspect where the fitted center lands; no model fitting or sampling."""
        from .three_file_center_plot import center_figure
        pixels = self._prediction_pixels()
        fig = center_figure(pixels, self.center_geometry(), self.report,
                            radius_pc=radius_pc, half_width=half_width)
        if save: self._save_figure(fig, "kinematic_center")
        return fig

    def input_figure(self):
        """Direct image and input line stacks, separate from residual review."""
        return _input_figure(self._prediction_pixels())

    def radial_profiles(self, radii_pc=None):
        """Paired-draw projected/deprojected curves and derived rotation masses.

        Projected speed uses paired V_rot/inc_v draws or sampled V_sini, never
        products of marginal summaries. Radius remains disk-plane
        radius under the sampled geometry, not an inclination-free sky radius.
        """
        from astropy.constants import G
        from astropy import units as u
        columns = self._columns(); scale = self.physical_scale()
        radii = np.linspace(0., scale["reference_radius_pc"], 200) if radii_pc is None else np.asarray(radii_pc, dtype=float)
        if radii.ndim != 1 or len(radii) < 2 or not np.isfinite(radii).all() or np.any(radii < 0) or np.any(np.diff(radii) <= 0):
            raise ValueError("Use at least two finite, nonnegative, strictly increasing radii in pc.")
        v = columns["vrot"][0].ravel(); rt = columns["rturn_pc"][0].ravel()
        a = columns["vsini"][0].ravel()
        g = G.to_value(u.pc*(u.km/u.s)**2/u.Msun)
        speed, projected, mass = (np.empty((3, len(radii))) for _ in range(3))
        for start in range(0, len(radii), 8):
            block = slice(start, start+8)
            curves = (2/np.pi)*v[:, None]*np.arctan(radii[None, block]/rt[:, None])
            projected_curves = (2/np.pi)*a[:, None]*np.arctan(radii[None, block]/rt[:, None])
            speed[:, block] = np.quantile(curves, [.16, .5, .84], axis=0)
            projected[:, block] = np.quantile(projected_curves, [.16, .5, .84], axis=0)
            mass[:, block] = np.quantile(radii[None, block]*curves**2/g, [.16, .5, .84], axis=0)
        return dict(radius_pc=radii, vsini_kms=projected, vrot_kms=speed, mass_msun=mass, **scale)

    def _profile_figure(self, radii_pc, mass, save, *, projected=False):
        import matplotlib.pyplot as plt
        p = self.radial_profiles(radii_pc)
        y = p["mass_msun" if mass else "vsini_kms" if projected else "vrot_kms"]
        fig, ax = plt.subplots(figsize=(8.5, 5.6))
        fig.subplots_adjust(bottom=.22, top=.84, left=.13, right=.97)
        ax.fill_between(p["radius_pc"], y[0], y[2], color=".8", label="Pointwise 16th–84th percentiles")
        ax.plot(p["radius_pc"], y[1], color="black", label="Pooled median")
        ylabel = (r"$M_{\rm rot}(<r)$ [$M_\odot$]" if mass else
                  r"$V_{\rm rot}(r)\sin i$ [km/s]" if projected else r"$V_{\rm rot}(r)$ [km/s]")
        ax.set(xlabel="Disk-plane radius [pc]", ylabel=ylabel,
               xlim=(p["radius_pc"][0], p["radius_pc"][-1]), ylim=(0, None))
        if mass: ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        ax.legend(fontsize=9); ax.grid(alpha=.18)
        title = ("rotation-derived mass" if mass else
                 "projected rotation curve" if projected else "deprojected rotation curve")
        fig.suptitle(f"ID{self.source_id}: {title}", fontsize=15)
        note = "Approximate stacked-image noise model" if self.report.get("sampling_pass") else "Provisional intervals: sampling-length checks not passed"
        footer = f"{note}\nIllustrative radius range; observed radial coverage not established."
        if self.fixed_inclination_deg is not None:
            footer += "\n"+self._inclination_note()
        if mass: footer += "\nSpherical-equivalent r V²/G; no pressure-support or lensing correction."
        elif projected: footer += "\nV sin i from paired draws; inclination still enters the disk geometry."
        else: footer += f"\nPlanck18 at z = {p['redshift']:g}; distance/lensing uncertainty not included."
        fig.text(.5, .025, footer, ha="center", va="bottom", fontsize=9)
        if save: self._save_figure(fig, "mass_profile" if mass else
                                   "projected_rotation_curve" if projected else "rotation_curve")
        return fig

    def projected_profile(self, radii_pixels=None):
        """Projected speed against the model radius in pixels, from paired draws.

        This uses no redshift or distance conversion. The model radius still
        depends on the sampled disk geometry; it is not a circular sky radius.
        """
        radii = (np.linspace(0., float(self.report["r_fit_pixels"]), 200)
                 if radii_pixels is None else np.asarray(radii_pixels, dtype=float))
        if (radii.ndim != 1 or len(radii) < 2 or not np.isfinite(radii).all()
                or np.any(radii < 0) or np.any(np.diff(radii) <= 0)):
            raise ValueError("Use at least two finite, nonnegative, strictly increasing radii in pixels.")
        columns = self._columns(("vsini", "rturn_pixels"))
        amplitude = columns["vsini"][0].ravel()
        turnover = columns["rturn_pixels"][0].ravel()
        speed = np.empty((3, len(radii)))
        for start in range(0, len(radii), 8):
            block = slice(start, start+8)
            curves = (2/np.pi)*amplitude[:, None]*np.arctan(radii[None, block]/turnover[:, None])
            speed[:, block] = np.quantile(curves, [.16, .5, .84], axis=0)
        return dict(radius_pixels=radii, vsini_kms=speed)

    def rotation_curve(self, radii_pc=None, *, radii_pixels=None, projected=None, save=False):
        """Show the sampled velocity definition by default; old runs stay projected."""
        if projected is None:
            projected = self.velocity_parameter == "V_sini"
        if radii_pixels is not None:
            if radii_pc is not None:
                raise ValueError("Choose curve radii in either pixels or pc, not both.")
            if not projected:
                raise ValueError("Pixel-radius plotting is for projected velocity; use the pc API for deprojection.")
            import matplotlib.pyplot as plt
            p = self.projected_profile(radii_pixels)
            radius, speed = p["radius_pixels"], p["vsini_kms"]
            fig, ax = plt.subplots(figsize=(8.5, 5.6))
            fig.subplots_adjust(bottom=.22, top=.84, left=.13, right=.97)
            ax.fill_between(radius, speed[0], speed[2], color=".8", label="Pointwise 16th–84th percentiles")
            ax.plot(radius, speed[1], color="black", label="Pooled median")
            ax.set(xlabel="Model radius [pixel]", ylabel=r"$V_{\rm rot}(r)\sin i$ [km/s]",
                   xlim=(radius[0], radius[-1]), ylim=(0, None))
            ax.legend(fontsize=9); ax.grid(alpha=.18)
            fig.suptitle(f"ID{self.source_id}: projected rotation curve", fontsize=15)
            note = ("Approximate stacked-image noise model" if self.report.get("sampling_pass")
                    else "Provisional intervals: sampling-length checks not passed")
            footer = (f"{note}\nIllustrative radius range; observed radial coverage not established.\n"
                      "V sin i from paired draws; inclination still enters the disk geometry.")
            if self.fixed_inclination_deg is not None:
                footer += "\n"+self._inclination_note()
            fig.text(.5, .025, footer,
                     ha="center", va="bottom", fontsize=9)
            if save:
                self._save_figure(fig, "projected_rotation_curve_pixels")
            return fig
        return self._profile_figure(radii_pc, False, save, projected=projected)

    def mass_profile(self, radii_pc=None, *, save=False):
        return self._profile_figure(radii_pc, True, save)

    def _save_figure(self, fig, name):
        out = self.path if self._final_bundle is not None else self.path/"figures"
        out.mkdir(exist_ok=True)
        for ext in (("png",) if self._final_bundle is not None else ("png", "pdf")):
            fig.savefig(out/f"{name}.{ext}", dpi=160, bbox_inches="tight")
        return out

    def export(self, working_dir=DEFAULT_WORKING, *, radius_pc=None):
        """Save final results.npz and plots in a clean per-source working folder."""
        from .three_file_export import export_final
        return export_final(self, working_dir, radius_pc=radius_pc)
