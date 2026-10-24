"""Run and inspect real R+C rotation fits with intrinsic dispersion fixed to zero.

The notebook process stays CPU-only. Fitting runs in a separate worker, while
this module checks and plots saved pixels and retained posterior draws.
"""
from pathlib import Path
import hashlib
import os
import re

import numpy as np

from .real_workflow import DEFAULT_BUNDLE, DEFAULT_RESULTS, REPO, RealSource
from .real_workflow import read_json, run_command, source_number


DEFAULT_PARAMETERS = ("vrot", "rturn_pc", "inclination", "theta", "x0", "y0")


def _paths(bundle=None, results=None):
    return (Path(bundle or os.environ.get("DINGO_BUNDLE", DEFAULT_BUNDLE)).expanduser().resolve(),
            Path(results or os.environ.get("DINGO_RESULTS", DEFAULT_RESULTS)).expanduser().resolve())


def _run_path(source_id, run_name, results=None):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(run_name)):
        raise ValueError("Use a run name containing only letters, digits, underscores or dashes.")
    _, root = _paths(results=results)
    return root / "real_sources" / f"ID{source_number(source_id)}" / run_name


def _imaging_path(source):
    from .rotation_imaging import imaging_directory
    return imaging_directory(source)


def _imaging_ready(path, source_id):
    from .rotation_imaging import imaging_ready
    return imaging_ready(path, source_id)


def available_sources(*, bundle=None, results=None):
    """List checked configurations, actual R/C file availability and imaging priors."""
    import pandas as pd
    bundle, results = _paths(bundle, results)
    rows = []
    for config in sorted((REPO / "data").glob("ID*/config_kinematics.yaml")):
        sid = source_number(config.parent.name)
        try:
            source = RealSource(sid, bundle, results, config)
            status = source.availability()
            rows.append(dict(source_id=sid, redshift=status["z"],
                R_exposures=status["R_exposures"], C_exposures=status["C_exposures"],
                native_data_ready=status["ready"],
                imaging_prior_ready=_imaging_ready(_imaging_path(source), sid),
                note="" if status["ready"] else f"Missing {len(status['missing'])} files or unsupported line"))
        except (OSError, ValueError, KeyError) as exc:
            rows.append(dict(source_id=sid, native_data_ready=False, imaging_prior_ready=False, note=str(exc)))
    return pd.DataFrame(rows).sort_values("source_id").reset_index(drop=True) if rows else pd.DataFrame()


def fit(source_id, *, run_name="rotation_v1", warmup=1000, samples=2000, chains=4,
        resume=True, device="cuda", bundle=None, results=None, config=None, imaging=None,
        inclination="imaging", max_tree_depth=9, target_accept=.9, chunk=50, seed=1018,
        override=False):
    """Fit one flexible gas map and one velocity field to all R+C exposures.

    Output is checkpointed by the worker and its progress is streamed here.
    Reusing a run name resumes compatible checkpoints; changing the model or
    adaptation settings requires a new name. ``samples`` is retained draws per
    chain, excluding warmup. Interrupting the notebook terminates the worker.
    ``override=True`` deletes this named run's outputs and logs before fitting;
    it takes precedence over resume and restarts optimization and sampling.
    """
    bundle, results = _paths(bundle, results)
    source = RealSource(source_id, bundle, results, config)
    output = _run_path(source.source_id, run_name, results)
    if not isinstance(override, bool):
        raise ValueError('override must be True or False.')
    if device not in ("cuda", "cpu"):
        raise ValueError("device must be 'cuda' or 'cpu'.")
    if inclination not in ("imaging", "isotropic"):
        raise ValueError("inclination must be 'imaging' or 'isotropic'.")
    for name, value in dict(warmup=warmup, samples=samples, chains=chains,
                            chunk=chunk, max_tree_depth=max_tree_depth).items():
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not .5 < target_accept < 1 or max_tree_depth > 12:
        raise ValueError("Use 0.5 < target_accept < 1 and 1 <= max_tree_depth <= 12.")
    source.require_data()
    imaging = Path(imaging).expanduser().resolve() if imaging is not None else _imaging_path(source)
    if inclination == "imaging" and not _imaging_ready(imaging, source.source_id):
        raise ValueError(f"ID{source.source_id} has no accepted F150W inclination prior at {imaging}. "
                         "Run the notebook's PySersic imaging cell first, or call "
                         "dingo.rotation_imaging.prepare_imaging for this source. "
                         "Choose inclination='isotropic' only if you intend a broad orientation prior.")
    args = ["-m", "dingo.rotation_fit", "--source-id", str(source.source_id),
            "--output", str(output), "--bundle", str(bundle), "--config", str(source.config),
            "--inclination", inclination, "--warmup", str(warmup), "--samples", str(samples),
            "--chains", str(chains), "--chunk", str(chunk), "--seed", str(seed),
            "--max-tree-depth", str(max_tree_depth), "--target-accept", str(target_accept),
            "--device", device]
    if inclination == "imaging":
        args += ["--imaging", str(imaging)]
    if resume and not override:
        args += ["--resume"]
    print(f"ID{source.source_id}: joint R+C, flexible gas emission, intrinsic σ₀ = 0 km/s.\n"
          f"{chains} chains; {warmup:,} warmup + {samples:,} retained draws per chain; {device}.\n"
          f"Results: {output}\nProgress log: {output / 'run.log'}", flush=True)
    from .run_output import output_run
    with output_run(output, source, kind='rotation', override=override):
        run_command(args, output / "run.log", device=device)
    return RotationResults(output, source.source_id)


def load(source_id, *, run_name="rotation_v1", results=None):
    """Inspect this source's saved run without starting a fit or using the GPU."""
    return RotationResults(_run_path(source_id, run_name, results), source_id)


def _checked(path, expected, label):
    if not expected:
        raise ValueError(f"No checksum saved for {label}; reload after the current checkpoint finishes.")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"{label} changed or checkpoint is being replaced: {path}. Reload after it finishes.")


class RotationResults:
    """Verified saved data and plots; no JAX, Torch or GPU imports are needed."""

    def __init__(self, path, source_id):
        self.path = Path(path).expanduser().resolve()
        self.source_id = source_number(source_id)
        self.report = read_json(self.path / "rotation.json")
        if (self.report.get("format") != "dingo_rotation_v1"
                or self.report.get("real_observations") is not True
                or self.report.get("source_id") != self.source_id
                or self.report.get("model", {}).get("intrinsic_sigma_kms") != 0):
            raise ValueError("Choose this source's rotation-only run, with intrinsic sigma fixed to zero.")
        self._columns_cache = None
        self._pixels_cache = None
        self._predictions_cache = {}

    @property
    def converged(self):
        return bool(self.report.get("completed") and
                    (self.report.get("diagnostics") or {}).get("convergence_gate_passed", False))

    @property
    def status(self):
        phase = self.report.get("phase", "unknown")
        draws = self.report.get("retained_draws_per_chain", 0)
        chains = self.report.get("settings", {}).get("chains", "?")
        label = ("Sampling checks passed; intervals are conditional on this model." if self.converged else
                 "NOT CONVERGED: plots and pooled intervals are provisional.")
        diagnostics = self.report.get("diagnostics") or {}
        warnings = list(diagnostics.get("sampling_warnings", []))
        depth = diagnostics.get("max_depth_fraction")
        if not warnings and depth is not None and depth >= .01:
            warnings.append(f"{100*depth:.1f}% of retained transitions reached maximum tree depth. "
                            "This is an efficiency warning; inspect Rhat, ESS and divergences.")
        warning_text = "".join(f"\nSampling warning: {message}" for message in warnings)
        return (f"ID{self.source_id} | {phase} | {chains} chains × {draws:,} retained draws\n"
                f"Intrinsic σ₀ = 0 km/s; one flexible gas map shared by R+C.\n{label}{warning_text}\n{self.path}")

    def __repr__(self):
        return self.status

    def reload(self):
        """Read the latest checkpoint after a worker has saved more samples."""
        return type(self)(self.path, self.source_id)

    def diagnostics(self):
        import pandas as pd
        d = self.report.get("diagnostics") or {}
        return pd.DataFrame([dict(completed=bool(self.report.get("completed")),
            sampling_checks_passed=self.converged, max_Rhat=d.get("max_rank_rhat"),
            min_bulk_ESS=d.get("min_bulk_ess"), min_tail_ESS=d.get("min_tail_ess"),
            divergences=d.get("divergences"), fraction_at_tree_limit=d.get("max_depth_fraction"))])

    def physical_scale(self):
        """Proper transverse scale at this run's redshift, without lensing correction.

        The fitted amplitude remains V(0.5 arcsec). Converting its radius for
        display does not change the saved parameterization, prior or samples.
        """
        from astropy import units as u
        from astropy.cosmology import Planck18
        redshift = self.report.get("source_summary", {}).get("z")
        if (isinstance(redshift, bool) or not isinstance(redshift, (int, float))
                or not np.isfinite(redshift) or redshift <= 0):
            raise ValueError("A positive source redshift saved in rotation.json is required for parsec units.")
        scale = float(Planck18.kpc_proper_per_arcmin(redshift).to_value(u.pc/u.arcsec))
        return dict(redshift=float(redshift), pc_per_arcsec=scale,
                    reference_radius_pc=.5*scale, reference_radius_arcsec=.5,
                    cosmology="Planck18", lensing_corrected=False)

    def _columns(self):
        if self._columns_cache is not None:
            return self._columns_cache
        path = self.path / "samples.npz"
        if not path.exists():
            raise ValueError("No retained samples yet. Check run.log; warmup is not plotted as posterior draws.")
        _checked(path, self.report.get("samples_sha256"), "Retained posterior samples")
        with np.load(path, allow_pickle=False) as a:
            if "sigma_v" in a or "sigma" in a:
                raise ValueError("This file samples dispersion; it is not the rotation-only posterior.")
            v, inc, center = a["vrot"], a["inclination"], a["center"]
            shape = (self.report["settings"]["chains"], self.report["retained_draws_per_chain"])
            if v.shape != shape or inc.shape != shape or center.shape != (*shape, 2):
                raise ValueError("Saved chain dimensions disagree with the checkpoint report.")
            if shape[1] < 1:
                raise ValueError("No retained draws yet.")
            angle = a["theta"]
            origin = np.arctan2(np.sin(angle).mean(), np.cos(angle).mean())
            angle = origin + np.angle(np.exp(1j * (angle - origin)))
            scale = self.physical_scale()["pc_per_arcsec"]
            columns = {
                "vrot": (v, r"$V_{\rm rot}(R_{\rm ref})$ [km/s]", "km/s"),
                "vsini": (v * np.sin(inc), r"$V_{\rm rot}(R_{\rm ref})\sin i$ [km/s]", "km/s"),
                "rturn": (np.exp(a["log_rturn"]), r"$r_{\rm turn}$ [arcsec]", "arcsec"),
                "rturn_pc": (np.exp(a["log_rturn"])*scale, r"$r_{\rm turn}$ [pc]", "pc"),
                "inclination": (np.rad2deg(inc), r"$i$ [deg]", "deg"),
                "theta": (np.rad2deg(angle), r"$\theta$ [deg]", "deg"),
                "systemic": (a["systemic"], r"$v_{\rm sys}$ [km/s]", "km/s"),
                "x0": (center[..., 0], r"$x_0$ [arcsec]", "arcsec"),
                "y0": (center[..., 1], r"$y_0$ [arcsec]", "arcsec"),
                "line_flux": (a["line_flux_1e17"], r"Line flux [$10^{-17}$ erg/s/cm$^2$]", "1e-17 erg/s/cm2"),
            }
        for key, (values, _, _) in columns.items():
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(f"Invalid retained {key} samples.")
        self._columns_cache = columns
        return columns

    def _selection(self, parameters):
        columns = self._columns()
        parameters = tuple(parameters)
        if not parameters or len(set(parameters)) != len(parameters) or any(k not in columns for k in parameters):
            raise ValueError(f"Choose distinct parameters from {tuple(columns)}.")
        return parameters, columns

    def summary(self, parameters=DEFAULT_PARAMETERS):
        """Pooled 16/50/84 percentiles only after checks pass; otherwise chain medians."""
        import pandas as pd
        parameters, columns = self._selection(parameters)
        rows = []
        for key in parameters:
            values, _, unit = columns[key]
            if self.converged:
                low, median, high = np.quantile(values, [.16, .5, .84])
                rows.append(dict(parameter=key, unit=unit, p16=low, median=median, p84=high,
                                 error_minus=median-low, error_plus=high-median))
            else:
                for chain, values_chain in enumerate(values):
                    rows.append(dict(parameter=key, unit=unit, chain=chain+1,
                        diagnostic_median=np.median(values_chain), status="NOT CONVERGED"))
        return pd.DataFrame(rows)

    def traces(self, parameters=("vrot", "inclination", "rturn_pc", "theta", "systemic", "x0", "y0", "line_flux")):
        import matplotlib.pyplot as plt
        parameters, columns = self._selection(parameters)
        fig, axes = plt.subplots((len(parameters)+1)//2, 2, squeeze=False,
                                 figsize=(12, 2.35*((len(parameters)+1)//2)), layout="constrained")
        for ax, key in zip(axes.ravel(), parameters):
            values, label, _ = columns[key]
            for chain, sequence in enumerate(values):
                ax.plot(np.arange(1, len(sequence)+1), sequence, lw=.7, alpha=.85,
                        color=f"C{chain % 10}", label=f"Chain {chain+1}")
            ax.set(xlabel="Retained draw (warmup excluded)", ylabel=label)
        for ax in axes.ravel()[len(parameters):]:
            ax.set_visible(False)
        axes.ravel()[0].legend(fontsize=8)
        ref = self.physical_scale()["reference_radius_pc"]
        fig.suptitle(f"ID{self.source_id}: rotation-only R+C chains; R_ref = {ref:,.0f} pc\n" +
                     ("Sampling checks passed" if self.converged else "NOT CONVERGED — diagnostic traces"))
        return fig

    def corner(self, parameters=DEFAULT_PARAMETERS, *, bins=35, smooth=1., save=False):
        """Monochrome pooled contours with one median and 16–84% interval.

        Only retained draws are combined. If sampling checks fail, the pooled
        summaries are explicitly provisional, not validated uncertainties.
        """
        import corner
        import matplotlib.pyplot as plt
        parameters, columns = self._selection(parameters)
        if len(parameters) < 2:
            raise ValueError("A corner plot needs at least two parameters.")
        if not isinstance(bins, int) or bins < 10:
            raise ValueError("bins must be an integer of at least ten.")
        if smooth is not None and (not np.isfinite(smooth) or smooth <= 0):
            raise ValueError("smooth must be positive or None.")
        data = np.stack([columns[k][0] for k in parameters], axis=-1)
        pooled = data.reshape(-1, len(parameters))
        if len(pooled) < len(parameters):
            raise ValueError(f'At least {len(parameters)} retained draws are needed for this corner plot.')
        ranges = []
        for low, high in zip(pooled.min(axis=0), pooled.max(axis=0)):
            pad = max((high-low)*.025, abs(low)*1e-6, 1e-6)
            ranges.append((low-pad, high+pad))
        labels = [columns[k][1] for k in parameters]
        common = dict(labels=labels, range=ranges, bins=bins, smooth=smooth,
            levels=(.68, .95), plot_datapoints=False, plot_density=False, plot_contours=True,
            max_n_ticks=3, use_math_text=True, show_titles=False, quiet=True,
            hist_kwargs={"density": True, "linewidth": 1.3, "color": "black"},
            contour_kwargs={"colors": "black", "linewidths": 1.0},
            contourf_kwargs={"colors": ["white", ".85", ".65"]})

        def interval_label(values, key):
            low, median, high = np.quantile(values, [.16, .5, .84])
            precision = 3 if columns[key][2] == "arcsec" else 1
            return (f"${median:.{precision}f}_{{-{median-low:.{precision}f}}}"
                    f"^{{+{high-median:.{precision}f}}}$")

        fig = corner.corner(pooled, color="black", fill_contours=True,
                            quantiles=(.16, .5, .84), **common)
        axes = np.asarray(fig.axes).reshape(len(parameters), len(parameters))
        for i, key in enumerate(parameters):
            axes[i, i].set_title(interval_label(pooled[:, i], key), fontsize=11, color="black")
        subtitle = ("Sampling checks passed · chains pooled · 68/95% contours · marginal 16/50/84 percentiles"
                    if self.converged else
                    "Chains pooled · 68/95% contours · provisional intervals: full-fit sampling checks not passed")
        fig.set_size_inches(2.2*len(parameters), 2.2*len(parameters))
        fig.set_facecolor("white")
        for ax in fig.axes:
            ax.set_facecolor("white")
        fig.subplots_adjust(top=.87, bottom=.09, left=.10, right=.98, hspace=.08, wspace=.08)
        fig.suptitle(f"ID{self.source_id}: joint R+C rotation posterior, σ₀ fixed to zero", y=.985, fontsize=15)
        fig.text(.5, .957, subtitle, ha="center", fontsize=10)
        ref = self.physical_scale()["reference_radius_pc"]
        fig.text(.5, .934, f"R_ref = {ref:,.0f} pc at this source's redshift (Planck18)\n"
                 "x₀: east; y₀: north [arcsec] from catalog · θ: velocity-axis angle from east, unwrapped",
                 ha="center", va="top", fontsize=9)
        if save:
            self._save_figure(fig, "corner")
        return fig

    def radial_profiles(self, radii_pc=None):
        """Pointwise pooled 16/50/84 percentiles of paired posterior curves.

        Radius is in the disk plane; vrot is already deprojected. Every curve
        uses V_ref and r_turn from the SAME retained draw. The default range
        (zero to R_ref) is illustrative, not a measured radial coverage limit.
        M_rot = r V_rot**2/G is a spherical-equivalent rotation-only estimate.
        Neither redshift/cosmology uncertainty nor lensing is propagated.
        """
        from astropy import units as u
        from astropy.constants import G
        columns = self._columns()  # Checks sample identity, integrity and shapes.
        scale = self.physical_scale()
        radius = (np.linspace(0., scale["reference_radius_pc"], 200)
                  if radii_pc is None else np.asarray(radii_pc, dtype=float))
        if (radius.ndim != 1 or radius.size < 2 or not np.isfinite(radius).all()
                or np.any(radius < 0) or np.any(np.diff(radius) <= 0)):
            raise ValueError("radii_pc must contain at least two finite, nonnegative, strictly increasing radii.")
        velocity = columns["vrot"][0].ravel()
        turnover = columns["rturn_pc"][0].ravel()
        sin_inc = np.sin(np.deg2rad(columns["inclination"][0].ravel()))
        if np.any(velocity < 0) or np.any(turnover <= 0):
            raise ValueError("Rotation curves require nonnegative speeds and positive turnover radii.")
        normalization = np.arctan(scale["reference_radius_pc"]/turnover)
        gravity = float(G.to_value(u.pc*(u.km/u.s)**2/u.Msun))
        quantiles = np.array([.16, .5, .84])
        vrot, vsini, mass = (np.empty((3, len(radius))) for _ in range(3))
        # Chunk radii rather than subsample draws: preserve all chains and their
        # parameter correlations without allocating three huge draw×radius grids.
        for start in range(0, len(radius), 32):
            block = slice(start, start+32)
            curves = velocity[:, None]*np.arctan(radius[None, block]/turnover[:, None])/normalization[:, None]
            vrot[:, block] = np.quantile(curves, quantiles, axis=0)
            vsini[:, block] = np.quantile(curves*sin_inc[:, None], quantiles, axis=0)
            mass[:, block] = np.quantile(radius[None, block]*curves**2/gravity, quantiles, axis=0)
        return dict(radius_pc=radius, quantiles=quantiles, vrot_kms=vrot,
                    vsini_kms=vsini, mass_msun=mass,
                    rturn_pc_quantiles=np.quantile(turnover, quantiles),
                    vinf_kms_quantiles=np.quantile(velocity*(np.pi/2)/normalization, quantiles),
                    retained_draws=int(velocity.size), sampling_checks_passed=self.converged,
                    radial_coverage="Illustrative model evaluation; observed/resolved radial coverage not established",
                    mass_definition="Spherical-equivalent rotation-only r*Vrot(r)^2/G; no pressure-support correction",
                    **scale)

    def _radial_figure(self, radii_pc, *, mass, save):
        import matplotlib.pyplot as plt
        profiles = self.radial_profiles(radii_pc)
        radius = profiles["radius_pc"]
        values = profiles["mass_msun" if mass else "vrot_kms"]
        title = "Rotation-derived dynamical mass" if mass else "Deprojected rotation curve"
        fig, ax = plt.subplots(figsize=(8.5, 5.8))
        fig.subplots_adjust(left=.13, right=.97, top=.77, bottom=.23 if mass else .19)
        ax.fill_between(radius, values[0], values[2], color=".8", linewidth=0,
                        label="Pointwise 16th–84th percentiles")
        ax.plot(radius, values[1], color="black", lw=1.8, label="Pooled median")
        reference = profiles["reference_radius_pc"]
        if radius[0] <= reference <= radius[-1]:
            ax.axvline(reference, color=".35", linestyle=":", lw=1.2,
                       label=f"Reference radius: {reference:,.0f} pc")
        ax.set(xlabel="Disk-plane radius [pc]",
               ylabel=(r"$M_{\rm rot}(<r)$ [$M_\odot$]" if mass else r"$V_{\rm rot}(r)$ [km/s]"),
               xlim=(radius[0], radius[-1]), ylim=(0, None))
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=.18)
        if mass:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        fig.suptitle(f"ID{self.source_id}: {title}", y=.97, fontsize=15)
        state = ("Sampling checks passed; intervals conditional on this model"
                 if self.converged else "Provisional intervals: full-fit sampling checks not passed")
        fig.text(.5, .915, f"{state}\nAll retained chains pooled · σ₀ = 0 km/s", ha="center", va="top", fontsize=10)
        footnote = (f"z = {profiles['redshift']:g} · Planck18 · 1 arcsec = {profiles['pc_per_arcsec']:,.0f} pc\n"
                    "Illustrative radius range; not independently measured radial bins or established radial coverage.")
        if mass:
            footnote += ("\nSpherical-equivalent r Vrot²/G; no pressure-support or lensing correction."
                         "\nBands exclude distance and mass-geometry systematics.")
        else:
            footnote += "\nPhysical radii assume no lensing correction; distance uncertainty is not included."
        fig.text(.5, .025, footnote, ha="center", va="bottom", fontsize=9)
        if save:
            self._save_figure(fig, "mass_profile" if mass else "rotation_curve")
        return fig

    def rotation_curve(self, radii_pc=None, *, save=False):
        """Plot the deprojected arctangent curve and paired-draw uncertainty in pc."""
        return self._radial_figure(radii_pc, mass=False, save=save)

    def mass_profile(self, radii_pc=None, *, save=False):
        """Plot spherical-equivalent rotation-only enclosed mass, r Vrot(r)**2/G."""
        return self._radial_figure(radii_pc, mass=True, save=save)

    def _pixels(self):
        if self._pixels_cache is None:
            from .real_review import load_pixels
            _checked(self.path / "native_data.npz", self.report.get("native_data_sha256"), "Native detector pixels")
            pixels, metadata = load_pixels(self.path / "native_data.npz", self.source_id)
            shape = pixels["science"].shape
            if len(shape) != 3 or any(a.shape != shape for a in pixels.values()):
                raise ValueError("Saved native arrays have inconsistent dimensions.")
            valid = pixels["valid"]
            if valid.dtype != bool or len(metadata["exposures"]) != shape[0]:
                raise ValueError("Invalid native mask or exposure metadata.")
            if not (np.isfinite(pixels["science"][valid]).all() and
                    np.all(np.isfinite(pixels["error"][valid]) & (pixels["error"][valid] > 0))):
                raise ValueError("Valid detector pixels need finite data and positive uncertainty.")
            self._pixels_cache = pixels, metadata
        return self._pixels_cache

    def exposures(self):
        """Zero-based exposure indices usable in fit_figure(r_exposure=..., c_exposure=...)."""
        import pandas as pd
        _, metadata = self._pixels()
        return pd.DataFrame([dict(index=i, pupil=e["pupil"], name=e.get("name", e.get("path", "")))
                             for i, e in enumerate(metadata["exposures"])])

    def _prediction(self, chain=None):
        key = "best" if chain is None else f"chain_{int(chain)}"
        if key in self._predictions_cache:
            return self._predictions_cache[key]
        if chain is None:
            record = self.report.get("best_fit", {})
            label = "Best optimized fit (not a posterior mean)"
        else:
            if not isinstance(chain, int) or not 0 <= chain < self.report["settings"]["chains"]:
                raise ValueError("chain is a zero-based chain index.")
            predictions = self.report.get("chain_predictions", [])
            record = next((r for r in predictions if r.get("chain") == chain), {})
            if not record:
                raise ValueError("No verified prediction is saved for this chain yet.")
            label = f"Chain {chain+1}: saved posterior draw"
        if not record.get("artifact"):
            raise ValueError("The worker has not saved a detector prediction yet. Check run.log.")
        artifact = self.path / record["artifact"]
        if not artifact.resolve().is_relative_to(self.path):
            raise ValueError("Prediction artifact must be inside this run directory.")
        _checked(artifact, record.get("artifact_sha256"), "Saved detector prediction")
        with np.load(artifact, allow_pickle=False) as a:
            saved = {k: a[k] for k in a.files}
        if int(saved.get("source_id", -1)) != self.source_id:
            raise ValueError("Prediction belongs to a different source.")
        pixels, _ = self._pixels()
        background = saved["background"]
        background = background[:, None, None] if background.ndim == 1 else background
        total = saved["line"] + saved["continuum"] + background
        if total.shape != pixels["science"].shape or saved["residual"].shape != total.shape:
            raise ValueError("Prediction and native detector dimensions disagree.")
        valid = pixels["valid"]
        residual = (pixels["science"][valid] - total[valid]) / pixels["error"][valid]
        np.testing.assert_allclose(saved["residual"][valid], residual, atol=1e-8, rtol=1e-9,
                                   err_msg="Saved residuals disagree with the native SCI/ERR arrays.")
        self._predictions_cache[key] = record, saved, total, label
        return self._predictions_cache[key]

    def fit_figure(self, *, r_exposure=None, c_exposure=None, full_cutout=False, chain=None):
        """Actual observed/predicted R and C cutouts, with matched stretches and ERR residuals."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import AsinhNorm
        record, saved, total, label = self._prediction(chain)
        pixels, metadata = self._pixels()
        pupils = np.array([e["pupil"] for e in metadata["exposures"]])
        pair = []
        for pupil, chosen in (("R", r_exposure), ("C", c_exposure)):
            available = np.flatnonzero(pupils == pupil)
            chosen = int(available[0]) if chosen is None and len(available) else chosen
            if chosen not in available:
                raise ValueError(f"Choose a zero-based {pupil} exposure index from {available.tolist()}.")
            pair.append(chosen)
        fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), layout="constrained")
        for row, e in enumerate(pair):
            valid = pixels["valid"][e]
            ny, nx = valid.shape
            ys = slice(0, ny) if full_cutout else slice(max(0, ny//2-18), min(ny, ny//2+19))
            xs = slice(0, nx) if full_cutout else slice(max(0, nx//2-18), min(nx, nx//2+19))
            cut = (ys, xs)
            finite = pixels["science"][e][cut][valid[cut]]
            if not finite.size:
                raise ValueError("No valid pixels in the selected display region.")
            low, high = np.quantile(finite, [.02, .995])
            high = max(high, low+1e-12)
            norm = AsinhNorm(linear_width=max((high-low)/10, 1e-12), vmin=low, vmax=high)
            for col, (values, title) in enumerate(zip((pixels["science"][e], total[e], saved["residual"][e]),
                    ("Observed SCI", "Predicted: line + continuum + background", "(Observed − predicted) / ERR"))):
                style = (dict(cmap=plt.get_cmap("RdBu_r").with_extremes(bad="0.65"), vmin=-5, vmax=5)
                         if col == 2 else dict(cmap=plt.get_cmap("gist_heat").with_extremes(bad="0.65"), norm=norm))
                im = axes[row, col].imshow(np.where(valid, values, np.nan)[cut], origin="lower",
                    extent=[xs.start-.5, xs.stop-.5, ys.start-.5, ys.stop-.5], interpolation="none", **style)
                axes[row, col].set(title=f"{pupils[e]}, exposure index {e}\n{title}", xlabel="Cutout x [pixel]", ylabel="Cutout y [pixel]")
                bar = fig.colorbar(im, ax=axes[row, col], shrink=.75, label="ERR" if col == 2 else "DN/s")
                if col != 2:
                    from matplotlib.ticker import FormatStrFormatter
                    bar.set_ticks(norm.inverse(np.linspace(0, 1, 5)))
                    bar.formatter = FormatStrFormatter("%.2g")
                    bar.update_ticks()
        velocity = record.get("vrot_0p5arcsec_kms")
        ref = self.physical_scale()["reference_radius_pc"]
        detail = f"; Vrot({ref:,.0f} pc) = {velocity:.1f} km/s" if velocity is not None else ""
        fig.suptitle(f"ID{self.source_id}: {label}{detail}\n"
                     "One source projected into R and C; σ₀ = 0. Gray pixels are excluded from fitting.")
        return fig

    def source_figure(self, *, chain=None):
        import matplotlib.pyplot as plt
        from matplotlib.colors import AsinhNorm
        record, saved, _, label = self._prediction(chain)
        side = int(np.sqrt(saved["source_flux"].size))
        if side*side != saved["source_flux"].size:
            raise ValueError("Expected a square saved source grid.")
        east, north = (saved[k].reshape(side, side) for k in ("source_east", "source_north"))
        weights = np.ones(side); weights[[0, -1]] = .5
        brightness = saved["source_flux"].reshape(side, side) / np.outer(weights, weights)
        brightness /= brightness.max()
        velocity = saved["source_velocity"].reshape(side, side) - record.get("systemic_offset_kms", 0.)
        velocity = np.where(brightness > .01, velocity, np.nan)
        vmax = max(1., float(np.nanmax(abs(velocity))))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
        for ax, values, style, title, unit in zip(axes, (brightness, velocity),
                (dict(cmap="gist_heat", norm=AsinhNorm(linear_width=.05, vmin=0, vmax=1)),
                 dict(cmap=plt.get_cmap("RdBu_r").with_extremes(bad="0.85"), vmin=-vmax, vmax=vmax)),
                ("Shared gas-emission brightness", "Shared mean LOS rotation (systemic removed)"),
                ("Relative surface brightness", "km/s")):
            im = ax.pcolormesh(east, north, values, shading="nearest", **style)
            ax.plot(0, 0, "w+", ms=10, mew=2, label="Catalog position")
            if "center_east_north_arcsec" in record:
                ax.plot(*record["center_east_north_arcsec"], "cx", ms=9, mew=2, label="Rotation center")
            ax.set(xlabel="East from catalog [arcsec]", ylabel="North from catalog [arcsec]", title=title, aspect="equal")
            bar = fig.colorbar(im, ax=ax, label=unit, shrink=.8)
            if "norm" in style:
                from matplotlib.ticker import FormatStrFormatter
                bar.set_ticks(style["norm"].inverse(np.linspace(0, 1, 5)))
                bar.formatter = FormatStrFormatter("%.2g")
                bar.update_ticks()
        axes[0].legend(fontsize=8)
        fig.suptitle(f"ID{self.source_id}: {label}\nFlexible gas brightness fitted jointly to R+C; σ₀ fixed to zero")
        return fig

    def _save_figure(self, figure, name):
        out = self.path / "figures"
        out.mkdir(exist_ok=True)
        for extension in ("png", "pdf"):
            figure.savefig(out / f"{name}.{extension}", dpi=160, bbox_inches="tight")
        return out

