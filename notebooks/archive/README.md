# Historical notebooks

See the [notebook index](../README.md) for the existing single-source notebooks
under `notebooks/legacy/` and the separate exposure-validation notebook.
The multi-source workflow is planned; these older files are historical records.
The original collection was moved from the repository root on October 18,
2026, with its bytes, outputs, and execution state preserved. Later backups
also preserve the original notebooks exactly. No fitted values or scientific
results were changed by archiving them.

Before simplifying the main notebook on October 21, its complete current
version was saved as
[rotation_only_before_simplification_20261021_182638.ipynb](rotation_only_before_simplification_20261021_182638.ipynb).
This backup includes the outputs, inclination scans, alternate prior routes,
and ID15665 selections that were present at that time. It is 7,345,299 bytes;
its SHA-256 is `92bf53722ca263cc2d6396802ebb102851559a8d7f14537b4ea75df74241ef03`.
The simplified notebook still defaults to ID15665; that source and every
other source remain part of the science analysis. Existing fit results and
scientific modules were not removed.

Before making projected velocity the primary output later on October 21,
the completed notebook was saved as
[rotation_only_before_projected_velocity_20261021_194511.ipynb](rotation_only_before_projected_velocity_20261021_194511.ipynb).
It retains the user's ID15665 run outputs and the deprojected rotation/mass
display cells. It is 5,814,558 bytes; its SHA-256 is
`ddb881427afb3286e6b6e6a61d6044e242fe2b0553a42128b5f98f8eecd6f74a`.
That snapshot emphasizes projected velocity. The later notebook in
[legacy/rotation_only.ipynb](../legacy/rotation_only.ipynb) samples V_inf and i
separately. Each snapshot retains its own parameter definitions and settings.

[manifest.json](manifest.json) records each original path, current path, byte
count, SHA-256 checksum, and whether the original file was tracked by Git.
The three previously tracked notebooks remain eligible for version control.
Other archived notebooks and their large saved outputs remain available locally
and are ignored by Git; they are not expected to appear in a fresh clone.

These files describe different stages of the method, including inverse
rectification, earlier forward-model likelihoods, and dispersion experiments.
They are historical records, not additional supported entry points. Their
settings and uncertainty claims must be interpreted with their corresponding
dated reports under [docs/](../../docs/).

To reproduce a historical notebook, start its kernel with the working directory
set to the repository root. Some notebooks assume that directory, and others
contain the original machine's absolute paths. The move intentionally does not
rewrite their code or outputs. Notebook-local relative links may also need the
original repository-root context.

The old validator `diagnostics/validate_real_notebook.py` continues to load
`real_source_kinematics.ipynb` from this archive and executes it with the
repository root as its working directory. It validates that historical example,
not the new rotation-only workflow.
