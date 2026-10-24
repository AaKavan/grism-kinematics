# Research and reproduction scripts

These scripts are retained for the experiments documented under
[docs/](../docs/README.md). They are not a second main fitting pipeline.
The current three-file inverse R+C workflow is described in the
[repository README](../README.md); its existing interfaces are preserved under
[notebooks/legacy/](../notebooks/legacy/).

The proposed notebook for multiple sources is not implemented yet. Repository
cleanup does not change these scripts or rerun their experiments.

| Script group | Purpose and context |
| --- | --- |
| `multistart.py`, `run_mcmc.py`, `validate_mcmc.py`, `regression_snapshot.py` | Earlier optimizer, MCMC, recovery and numerical-regression investigations. See the [original audit](../docs/audit_2026-10-13.md) and [historical validation](../docs/validation.md). |
| `measure_inclination.py`, `scan_inclination.py`, `inclination_budget.py` | Earlier imaging/inclination diagnostics. The current notebook's PySersic interface is [dingo/three_file_imaging.py](../dingo/three_file_imaging.py). |
| `grism_information.py`, `line_tilt.py`, `vsys_separability.py` | Information and degeneracy checks. Apparent line tilt alone is not a rotation measurement. |
| `native_exposure_check.py`, `native_flexible_check.py`, `native_recovery_suite.py`, `native_oracle_check.py`, `native_posterior_*.py` | Native-exposure prototype, synthetic recovery and posterior checks. See the [prototype](../docs/native_forward.md), [flexible model](../docs/native_flexible_2026-10-14.md) and [posterior pilot](../docs/native_posterior_2026-10-15.md). |
| `native_real_*.py`, `native_source_*.py`, `run_source_comparison.py` | Real-source forward-model grids, posterior runs, reports and plots. See the [real posterior](../docs/real_posterior_2026-10-16.md) and [source comparison](../docs/source_comparison_2026-10-18.md). |
| `native_detector_audit.py`, `native_continuum_audit.py`, `native_coupled_fit.py` | Detector and continuum investigations behind the [shared-flux repair](../docs/shared_flux_repair_2026-10-18.md). |
| `native_morphology_*.py`, `native_map_*.py`, `native_simple_*.py`, `native_profile_modes.py` | Morphology, numerical and dispersion controls. See the [morphology comparison](../docs/morphology_controls_2026-10-18.md) and [simpler emission experiment](../docs/simple_emission_dispersion_2026-10-18.md). |
| `native_holdout_*.py`, `native_real_holdout*.py` | Withheld-exposure checks; see the [ID21636 report](../docs/real_holdout_ID21636_2026-10-21.md). |
| `plot_grism_comparison.py`, `plot_native_suite_profiles.py` | Figures for the [R/C comparison](../docs/grism_comparison_2026-10-14.md) and recovery experiments. |
| `validate_real_notebook.py` | Executes the archived real-source notebook with repository-root working directory; it does not validate the current three-file notebook. |

Use each experiment's report and script arguments to identify its input files,
model, output directory and computational cost before reproducing it. Some
scripts launch optimizations or MCMC; others only read and plot saved results.
Legacy paths and launchers, including `run_both_ensembles.sh`, retain their
original machine-specific assumptions. They are historical records, not
portable batch commands for the current workflow.
