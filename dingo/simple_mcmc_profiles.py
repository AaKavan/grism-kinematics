"""Physical summaries from saved simple-MCMC draws, without fitting or GPU use.

Rotation and mass curves are evaluated for every paired amplitude/turnover
draw before their pointwise percentiles are calculated. This preserves the
posterior covariance; a curve made from marginal median parameters does not.
"""
from __future__ import annotations

import json
from numbers import Real
from pathlib import Path

import numpy as np


PERCENTILES = np.array([.16, .5, .84])
MASS_NOTE = (
    "Spherical-equivalent rotation-based enclosed mass r*Vrot(r)^2/G; "
    "no pressure-support, disk-geometry, or lensing correction."
)
COVERAGE_NOTE = "Model evaluation range; observed radial coverage is not established by these plots."


def _positive_number(value, name):
    """Reject missing or ambiguous physical scale inputs rather than guessing."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number.")
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number.")
    return float(value)


class PhysicalProfiles:
    """Read one saved simple-MCMC run and derive physical rotation profiles.

    Parameters
    ----------
    directory : path
        Run folder containing ``samples.npz`` and ``run_info.txt``. These files
        are only read. No source images, calibration files, or fitter are needed.
    pixel_scale_arcsec : float
        Angular size of a pixel in the fitted R/C stacks, supplied explicitly.
        It must not be the short-wavelength imaging pixel scale. Redshift comes
        from the run's source metadata; Planck18 supplies the proper distance.

    The sampled ``velocity.V_rot`` is already deprojected, so no additional
    division by sin(i) is applied. Fixed geometry, redshift/cosmology and pixel
    scale uncertainties, lensing, and noise-model systematics are not propagated.
    """

    def __init__(self, directory, *, pixel_scale_arcsec):
        from astropy import units as u
        from astropy.cosmology import Planck18

        self.directory = Path(directory).expanduser().resolve()
        self.info = json.loads((self.directory / "run_info.txt").read_text())
        if self.info.get("schema") != "dingo-simple-mcmc-v1":
            raise ValueError("This folder does not contain a simple_mcmc run.")
        redshift = self.info.get("configuration", {}).get("summary", {}).get("z")
        redshift = _positive_number(redshift, "Source redshift")
        pixel = _positive_number(pixel_scale_arcsec, "pixel_scale_arcsec")
        pc_per_arcsec = float(Planck18.kpc_proper_per_arcmin(redshift).to_value(u.pc / u.arcsec))
        self.scale = dict(redshift=redshift, cosmology="Planck18", pixel_scale_arcsec=pixel,
                          pc_per_arcsec=pc_per_arcsec, pc_per_pixel=pc_per_arcsec * pixel)
        self.source_id = self.info["source_id"]
        self.fixed_parameters = dict(self.info.get("fixed_parameters", {}))
        self.noise_note = self.info.get("noise_note", "Noise-model validation is not established.")

        with np.load(self.directory / "samples.npz", allow_pickle=False) as saved:
            names = saved["param_names"]
            chain = np.asarray(saved["chain"], dtype=float)
        if names.ndim != 1 or len(set(names.tolist())) != len(names):
            raise ValueError("Saved parameter names must be a unique one-dimensional list.")
        if chain.ndim != 3 or not all(chain.shape) or chain.shape[-1] != len(names):
            raise ValueError("Saved chain must contain nonempty (draw, walker, parameter) axes.")
        if not np.isfinite(chain).all():
            raise ValueError("Saved chain contains nonfinite values; inspect the sampling result.")
        names = names.tolist()
        if "velocity.V_rot" not in names or "velocity.R_v" not in names:
            raise ValueError("Physical profiles require sampled deprojected velocity.V_rot and velocity.R_v.")
        self._velocity = chain[..., names.index("velocity.V_rot")].ravel().copy()
        self._turnover_pixels = chain[..., names.index("velocity.R_v")].ravel().copy()
        if np.any(self._velocity < 0) or np.any(self._turnover_pixels <= 0):
            raise ValueError("Rotation amplitudes must be nonnegative and turnover radii positive.")
        self._turnover_pc = self._turnover_pixels * self.scale["pc_per_pixel"]
        self.retained_draws = self._velocity.size
        self._profile_cache = None

    def turnover_summary(self, save=False):
        """Summarize each draw at its OWN turnover radius, where V(Rt)=Vinf/2.

        These are pooled median and 16th/84th percentile offsets. In particular,
        turnover_velocity is not velocity evaluated at the median turnover
        radius. Saving writes only ``turnover_summary.txt`` in this run folder.
        """
        import pandas as pd

        rows = []
        for name, draws, unit in (
                ("turnover_velocity", self._velocity / 2., "km/s"),
                ("turnover_radius_pixels", self._turnover_pixels, "pixel"),
                ("turnover_radius_pc", self._turnover_pc, "pc")):
            low, median, high = np.quantile(draws, PERCENTILES)
            rows.append(dict(parameter=name, median=median, minus_16th=median-low,
                             plus_84th=high-median, unit=unit))
        table = pd.DataFrame(rows).set_index("parameter")
        table.attrs.update(self.scale, uncertainty="Conditional pooled 16th–84th percentiles")
        if save:
            table.to_csv(self.directory / "turnover_summary.txt", sep="\t")
        return table

    def radial_profiles(self, radii_pc):
        """Return pointwise 16/50/84% velocity and mass curves at common radii.

        ``radii_pc`` are increasing, nonnegative disk-plane radii in proper pc.
        All retained paired draws are used. Only the radius axis is processed
        in blocks, avoiding a large draw-by-radius allocation or subsampling.
        Repeated requests for the same grid reuse the computed percentiles.
        """
        from astropy import units as u
        from astropy.constants import G

        radius = np.asarray(radii_pc, dtype=float)
        if (radius.ndim != 1 or radius.size < 2 or not np.isfinite(radius).all()
                or np.any(radius < 0) or np.any(np.diff(radius) <= 0)):
            raise ValueError("radii_pc needs at least two finite, nonnegative, strictly increasing values.")
        if self._profile_cache is None or not np.array_equal(radius, self._profile_cache["radius_pc"]):
            gravity = float(G.to_value(u.pc * (u.km / u.s)**2 / u.Msun))
            velocity, mass = (np.empty((3, len(radius))) for _ in range(2))
            for start in range(0, len(radius), 8):
                block = slice(start, start + 8)
                curves = (2 / np.pi) * self._velocity[:, None] * np.arctan(
                    radius[None, block] / self._turnover_pc[:, None])
                velocity[:, block] = np.quantile(curves, PERCENTILES, axis=0)
                mass[:, block] = np.quantile(radius[None, block] * curves**2 / gravity,
                                            PERCENTILES, axis=0)
            self._profile_cache = dict(radius_pc=radius.copy(), vrot_kms=velocity, mass_msun=mass)
        # Return small copies, so callers cannot corrupt a cached result.
        return {name: values.copy() for name, values in self._profile_cache.items()}

    def _fixed_note(self):
        labels = {"velocity.x0_v": ("x0", "px", 1.), "velocity.y0_v": ("y0", "px", 1.),
                  "velocity.inc_v": ("i", "deg", 180. / np.pi)}
        parts = []
        for name, value in self.fixed_parameters.items():
            if name in labels:
                label, unit, factor = labels[name]
                parts.append(f"{label}={float(value)*factor:g} {unit}")
        return "Fixed " + ", ".join(parts) + "; their uncertainty is excluded." if parts else ""

    def _save_profiles(self, profiles):
        """Write an object-free derived NPZ; never alter sampled or fitted files."""
        temporary = self.directory / "physical_profiles.tmp.npz"
        np.savez_compressed(temporary, **profiles, **self.scale, percentiles=PERCENTILES,
            retained_draws=np.array(self.retained_draws),
            fixed_geometry_json=np.array(json.dumps(self.fixed_parameters, sort_keys=True, allow_nan=False)),
            noise_note=np.array(self.noise_note), mass_definition=np.array(MASS_NOTE),
            radial_coverage=np.array(COVERAGE_NOTE),
            uncertainty_note=np.array("Conditional pointwise 16th–84th percentiles; no distance, lensing or systematic uncertainty propagated."))
        temporary.replace(self.directory / "physical_profiles.npz")

    def _figure(self, radii_pc, *, mass, save):
        import matplotlib.pyplot as plt

        profiles = self.radial_profiles(radii_pc)
        radius = profiles["radius_pc"]
        values = profiles["mass_msun" if mass else "vrot_kms"]
        figure, axis = plt.subplots(figsize=(9., 5.8))
        figure.subplots_adjust(left=.13, right=.97, bottom=.25, top=.86)
        axis.fill_between(radius, values[0], values[2], color="steelblue", alpha=.25,
                          label="Conditional 16th–84th percentiles")
        axis.plot(radius, values[1], color="navy", linewidth=2., label="Median of paired-draw curves")
        turnover = float(np.median(self._turnover_pc))
        if radius[0] <= turnover <= radius[-1]:
            axis.axvline(turnover, color=".4", linestyle=":", label=f"Median turnover radius: {turnover:,.0f} pc")
        ylabel = (r"Spherical-equivalent $M(<r)$ [$M_\odot$]" if mass else
                  r"Deprojected $V_{\rm rot}(r)$ [km/s]")
        axis.set(xlabel="Disk-plane radius [pc]", ylabel=ylabel,
                 xlim=(radius[0], radius[-1]), ylim=(0., None))
        if mass:
            axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        axis.legend(fontsize=9)
        axis.grid(alpha=.15)
        title = "rotation-based spherical-equivalent mass" if mass else "rotation curve"
        figure.suptitle(f"ID{self.source_id}: {title}", fontsize=14)
        scale = self.scale
        notes = [f"Planck18, z={scale['redshift']:g}; {scale['pixel_scale_arcsec']:g} arcsec/pixel = {scale['pc_per_pixel']:.2f} pc/pixel.",
                 COVERAGE_NOTE, "Intervals retain the run's approximate noise assumptions."]
        if self._fixed_note():
            notes.append(self._fixed_note())
        notes.append("No pressure-support, disk-geometry or lensing correction." if mass else
                     "Distance/lensing uncertainty is not included.")
        figure.text(.5, .025, "\n".join(notes), ha="center", va="bottom", fontsize=8.5)
        if save:
            figure.savefig(self.directory / ("dynamical_mass.png" if mass else "rotation_curve.png"),
                           dpi=160, bbox_inches="tight")
            self._save_profiles(profiles)
        return figure

    def rotation_curve(self, radii_pc, save=False):
        """Plot the deprojected rotation posterior, optionally saving PNG and NPZ."""
        return self._figure(radii_pc, mass=False, save=save)

    def mass_profile(self, radii_pc, save=False):
        """Plot r*Vrot(r)^2/G, a spherical-equivalent rotation-based mass estimate."""
        return self._figure(radii_pc, mass=True, save=save)
