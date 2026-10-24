# Documentation index

Start with the [repository README](../README.md). The current method uses the
original three-file architecture: one direct image and coadded grism R and C
images, compared after inverse rectification by a shared rotation field.
Intrinsic dispersion is fixed to zero. The new Python interface supports
multiple sources and a source/run output layout. Start with the
[workflow demo notebook](../workflow_demo.ipynb).

## Current method and preserved notebooks

| Document | Purpose |
| --- | --- |
| [Workflow demo](../workflow_demo.ipynb) | Runnable ID3700 pilot with candidate inspection, MCMC and saved products; optional ID15665 manual-start example. |
| [Python workflow](workflow.md) | New per-run interface: imaging, search/manual starts, sampled or fixed inclination, MCMC, review, export and sequential sources. |
| [Single-source instructions](three_file_workflow.md) | Detailed instructions preserved from the earlier repository README, with updated notebook locations. |
| [Three-file workflow](three_file_2026-10-19.md) | Method, candidate searches, sampled velocity conventions, inclination treatment and later notebook changes. Read the dated updates together. |
| [Rotation notebook](../notebooks/legacy/rotation_only.ipynb) | Preserved single-source imaging, search, MCMC and review interface. |
| [Manual-start notebook](../notebooks/legacy/manual_start_kinematics.ipynb) | Preserved experiments with specified starting velocity and sampled or fixed inclination. |
| [Notebook guide](../notebooks/README.md) | Locations and status of current and historical notebooks. |

The saved notebooks include source-specific settings and outputs. A narrow
posterior at a fixed inclination is conditional on that angle; its errors do
not include inclination uncertainty or competing solutions. Historical
validation claims below do not establish the accuracy of the current masses.

## Dated experiments and audit history

These reports describe the model and data used at the time. In particular,
the native-exposure forward models are a separate experimental branch from
the current three-file inverse method.

| Topic | Reports |
| --- | --- |
| Original-code audit and validation | [October 13 audit](audit_2026-10-13.md), [earlier validation suite](validation.md), [historical workflow](legacy_workflow.md) |
| Native-exposure model development | [Initial prototype](native_forward.md), [flexible shared emission](native_flexible_2026-10-14.md), [repeated-noise checks](native_ensemble_2026-10-14.md) |
| Information from both grism directions | [R/C comparison](grism_comparison_2026-10-14.md) |
| Posterior and real-source grid experiments | [Posterior pilot](native_posterior_2026-10-15.md), [ID3700 velocity grid](real_ID3700_grid_2026-10-15.md), [real-data posterior](real_posterior_2026-10-16.md), [historical notebook guide](real_source_notebook.md) |
| Detector, continuum and morphology investigations | [ID6170 failure](real_ID6170_2026-10-18.md), [shared-flux repair](shared_flux_repair_2026-10-18.md), [source comparison](source_comparison_2026-10-18.md), [morphology controls](morphology_controls_2026-10-18.md) |
| Dispersion experiments | [Instrumental-response comparison](real_dispersion_2026-10-16.md), [simpler emission model](simple_emission_dispersion_2026-10-18.md) |
| Earlier native rotation-only workflow | [October 18 report](rotation_only_2026-10-18.md) |
| Withheld real exposures | [ID21636 validation report](real_holdout_ID21636_2026-10-21.md), [validation notebook](../notebooks/validation/real_rc_validation.ipynb) |

## Data collection and reproduction

- [HPC collection instructions](hpc_collection_instructions.txt) and
  [requested grism files](hpc_grism_files_2026-10-13.txt) record the data-transfer work.
- [Diagnostic scripts](../diagnostics/README.md) support the dated experiments.
- [Historical notebook archive](../notebooks/archive/README.md) records earlier
  notebook locations and checksums.

Scientific input files and fitted results remain outside version control;
paths in dated reports may refer to this workstation's existing data directories.
