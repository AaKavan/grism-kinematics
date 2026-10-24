# Notebook index

Start with [workflow_demo.ipynb](../workflow_demo.ipynb) in the repository root.
It calls `dingo.workflow`, keeps separate run handles for multiple sources,
and saves each experiment in its own directory. The existing single-source
notebooks remain available below, with their saved code, settings, outputs and
execution state unchanged by the October 23, 2026 move.

| Location | Contents |
|---|---|
| [../workflow_demo.ipynb](../workflow_demo.ipynb) | Current demo: ID3700 imaging/search, candidate inspection, MCMC, plots and export; optional ID15665 manual-start run. |
| [legacy/rotation_only.ipynb](legacy/rotation_only.ipynb) | PySersic, candidate search, selected-candidate MCMC and saved-result review. |
| [legacy/manual_start_kinematics.ipynb](legacy/manual_start_kinematics.ipynb) | Manual starting velocity and sampled/fixed inclination, using an existing source search. |
| [validation/real_rc_validation.ipynb](validation/real_rc_validation.ipynb) | The separate ID21636 withheld-exposure prediction experiment. |
| [archive/](archive/README.md) | Earlier notebooks and intermediate snapshots; many large historical files are local-only. |

Use the `grism` kernel with DINGO installed in editable mode. The legacy notebooks
have fitting enabled in their preserved settings. Inspect those settings before
running them; use their `RUN_* = False` switches to review saved results only.
Relative text links within a preserved notebook may refer to its original
repository-root context. Existing result locations have not changed.

The demo enables an ID3700 MCMC pilot by default (1,000 warmup steps, 1,000
retained steps per walker, 64 walkers). It reuses the current accepted imaging
under `../results` when `reuse_imaging_3700=True`; set this to `False` to run
PySersic afresh, with a new run number or explicit overwrite. It creates a new
candidate search on the first run, then reuses it for unchanged settings. The
optional ID15665 example is disabled by default. Pilot runs require inspection
and convergence checks before their intervals can be used scientifically.

[manifest.json](manifest.json) records the original and relocated paths, byte
counts and SHA-256 checksums of all three moved notebooks. These notebooks
remain eligible for version control. Older local-only snapshots are catalogued
separately in [archive/manifest.json](archive/manifest.json).

The default saved-settings reader for `rotation_only.ipynb` now uses its
`notebooks/legacy/` location, so its results cells can still recover the selected
source after a kernel restart. Explicit notebook paths remain supported.
The selected candidate must match the saved posterior. Archiving preserves the
notebook's selection even if a later run has replaced the cache's posterior;
the loader continues to report that mismatch rather than switch candidates.

For the demo's interface, see the [Python workflow guide](../docs/workflow.md).
The [three-file workflow guide](../docs/three_file_workflow.md) retains detailed
instructions for the older notebooks. Historical scientific results have not
been migrated.
