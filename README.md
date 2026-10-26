<p align="center">
  <img src="fig/logo.png" alt="DINGO Logo" width="240"/>
</p>

# DINGO: galaxy kinematics from joint R+C grism observations

DINGO fits one rotation field to coadded grism R and C emission-line images.
The working three-file method uses `image.fits`, `R.fits` and `C.fits`, and
finds the field that makes the velocity-corrected R/C images agree. It imposes
no fitted gas-light profile and fixes intrinsic dispersion to zero. Imaging
can supply an inclination prior; fixed-inclination experiments are also available.

## Python workflow

The new [`dingo.workflow`](dingo/workflow.py) interface keeps a separate run
handle for every source and experiment. It supports candidate searches, manual
starting velocities, sampled or fixed inclination, live progress bars/checkpoints and
source/run output directories. The Gaussian inclination width is an explicit
argument, `inclination_sigma_deg=5.0`, adjustable for each run.

See the [Python workflow guide](docs/workflow.md) for the prepare, inspect, fit
and save calls, including several configurations of the same source. Each stage
can be called separately with a named run handle and explicit arguments:

```python
run_a = wf.prepare_source(
    15665,
    data_dir=DATA_DIR,
    working_dir=WORKING_DIR,
    run_name="15665_imaging_s5_v400",
    inclination_mode="imaging_prior",
    inclination_sigma_deg=5.0,
    start_velocity_kms=400.0,
)
```

Use `wf.show_inputs(run_a)` and `wf.fit_morphology(run_a)` to inspect the inputs
and fit the morphology. Call `wf.find_candidates(run_a)` and
`wf.show_candidates(run_a)` to search and inspect starting solutions.
Choose a candidate with `wf.prepare_start(run_a, candidate_id=...)`, then call
`wf.fit_source(run_a, warmup=1000, draws=1000, walkers=64)` when ready to sample.
Review with `wf.show_results(run_a)` and save with `wf.save_results(run_a)`.
The guide supplies imports and paths for these calls. Start with the runnable
[workflow demo notebook](workflow_demo.ipynb).

## Notebooks

The [workflow demo](workflow_demo.ipynb) shows direct function calls for
`run_a` and `run_b`: two named experiments for ID15665 with starting velocities
of 400 and 450 km/s. Edit arguments where each function is called, and execute
preparation, morphology, candidate search, start selection, sampling and review
one cell at a time. Copy the calls with a new handle and `run_name` to keep
another configuration or source alongside them.

The demo starts from prepared observations without needing earlier fits.
One PSF-convolved Sérsic morphology fit per source is saved under
`<ID>/morph_fits/F150W/`, with posterior samples, plots, readable parameter
summaries and `pysersic_<ID>_info.cat`; matching fits are reused across experiments.
Each kinematic run makes its own candidate search. Set its chosen `candidate_id`
after visual inspection; sampling and saving are separate calls. Different
run names retain independent kinematic outputs even when the source ID matches.
Arguments on `prepare_source` support manual starts and fixed or sampled inclination.

The default MCMC pilot uses 1,000 warmup steps and 1,000 retained steps per
walker across 64 walkers. These settings let you inspect behavior; they do not
establish paper-quality uncertainties. Live progress bars and checkpoints are
automatic. No source-specific historical result directories are required.

Earlier notebooks were moved out of the repository root on October 23, 2026.
Their saved settings, outputs and execution state were preserved exactly.

| Notebook | Purpose |
|---|---|
| [Workflow demo](workflow_demo.ipynb) | Separate function calls for named experiments: shared morphology, candidate inspection, MCMC, plots and saving. |
| [Rotation-only workflow](notebooks/legacy/rotation_only.ipynb) | Existing single-source PySersic, candidate search, MCMC and results workflow. |
| [Manual starting values](notebooks/legacy/manual_start_kinematics.ipynb) | Existing experiments with user-chosen starting velocity and sampled or fixed inclination. |
| [Exposure validation](notebooks/validation/real_rc_validation.ipynb) | Separate ID21636 experiment fitting some exposures and predicting withheld exposures; no MCMC mass errors. |

Use the **grism** kernel. For the preserved working notebooks, inspect
their saved source and run switches before executing them. The fitting notebooks
currently enable MCMC. For saved-result review, set their fitting switches to
`False`. Moving a notebook does not change its saved result directories.

See the [notebook index](notebooks/README.md) and
[detailed single-source instructions](docs/three_file_workflow.md).

## Repository layout

```text
workflow_demo.ipynb  Runnable demonstration of the current Python interface
dingo/          Importable fitting, review and export functions
notebooks/      Existing workflow notebooks, validation and historical snapshots
tests/          Verification of fitting and review behavior
docs/           Method notes, scientific audits and workflow instructions
diagnostics/    Research and validation scripts retained for reproduction
data/           Source configurations; observational FITS files are local
env/            Environment setup
psf/            Versioned calibration PSF
```

The [documentation index](docs/README.md) distinguishes the three-file workflow
from earlier native-exposure, morphology and dispersion experiments. Those
implementations remain available for reproducing their saved results.

## Model and interpretation

The rotation curve is

\[
V_{\rm rot}(r)=\frac{2}{\pi}V_\infty\arctan(r/r_t).
\]

The grism distortion uses its line-of-sight projection; inclination also changes
the disk geometry. The current notebooks sample deprojected `V_inf` and, unless
fixed explicitly, inclination. `V_inf sin(i)` is derived from paired draws.
The PyTorch likelihood runs on the GPU; emcee supplies ensemble MCMC proposals.

Inspect residuals, all walkers and separate initializations. Stable sampling
within one solution does not establish that competing solutions have been
explored. Fixed-inclination errors are conditional on that exact angle. The
stacked-image noise model is approximate, and plotted reference radii are not
automatically the observed radial coverage.

Existing mass profiles use `r V_rot(r)^2 / G`: a spherical-equivalent,
rotation-derived estimate without pressure-support or lensing corrections.
See [method notes](docs/three_file_2026-10-19.md) for assumptions and limitations.

## Saved results

Existing NPZ files, plots, logs and checkpoints stay at their recorded locations;
the notebook cleanup did not migrate or overwrite them. Observations and run
products are excluded from Git. The relocated notebooks retain their review
outputs; [checksums](notebooks/manifest.json) verify the move.

The Python interface writes each experiment into its own directory:

```text
WORKING_DIR/
  15665/
    morph_fits/F150W/
      fit_001/
        ...shared morphology posterior and plots...
    15665_imaging_s5_v400/
      run_info.txt
      run.json
      results.npz
      ...plots...
      _checkpoints/
    15665_imaging_s5_v450/
      ...independent kinematic outputs...
```

Each run retains its settings, provenance, completion status and diagnostics
in `run_info.txt`. The preserved notebooks retain their existing output layout;
they have not been migrated to the new Python interface.

## Environment

The local **grism** environment already contains the required packages. For a
separate Python 3.12+ environment, install from the repository root:

```bash
python -m pip install -e ".[rotation,mcmc,test]"
```

An editable installation lets the notebooks import `dingo` from their new
locations. Observational data are not bundled with a fresh clone. See
[env/make_grism_env.sh](env/make_grism_env.sh) for the original environment setup.
