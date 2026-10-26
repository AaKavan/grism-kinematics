# Python workflow for multiple sources

`dingo.workflow` provides separate functions to prepare, search, fit, inspect
and save an experiment. Each experiment has its own `Run` handle and directory;
there is no global current source. It uses the existing three-file,
rotation-only R+C model. The scientific likelihood has not changed.

Use the **grism** environment and open the
[workflow demo notebook](../workflow_demo.ipynb). Set `DATA_DIR`,
`WORKING_DIR` and `SOURCE_IDS`, then run its seven code cells in order.
The notebook has no dependency on previous fits. A new source gets its own
morphology fit and candidate search. It pauses for explicit candidate choices,
then samples the chosen starts sequentially. You can supply the candidate IDs
in `CANDIDATE_IDS` after inspecting the residuals.

The default pilot runs 1,000 warmup steps and 1,000 retained steps per walker
across 64 walkers. These are inspection settings, not a claim of reportable
uncertainties. `RUN_MCMC=False` disables sampling in the MCMC cell; it does not
disable the preceding morphology/search stages. For read-only review, execute
only Settings and the final saved-results cell.

The [previous demo](notebook_backups/workflow_demo_before_shared_morphology_20261024.ipynb)
is preserved byte for byte. The [older notebooks](../notebooks/README.md) still
use their existing interfaces and save locations.

## Inputs and settings

Prepared inputs use this layout:

```text
DATA_DIR/
  ID3700/
    image.fits
    R.fits
    C.fits
    config_kinematics.yaml
  ID6170/
    ...
```

The configuration supplies source coordinates and grism metadata. Set
`CONFIG_DIR` in the notebook if configurations are in a separate directory with
the same `ID<id>/config_kinematics.yaml` structure. `prepare_source` also accepts
an explicit `config_path` for an individual source.

A new F150W morphology fit needs calibrated science/error imaging and a matching
PSF. The `science`, `error`, `psf` and `bundle` arguments can specify these paths;
`None` uses the local survey defaults. These inputs are separate from the
three kinematic cutouts. Data and calibration files are not bundled in Git.

The notebook's `SOURCE_OPTIONS` dictionary overrides `prepare_source` settings
for individual IDs, for example:

```python
SOURCE_IDS = [3700, 6170, 15665]
SOURCE_OPTIONS = {
    6170: dict(inclination_sigma_deg=10.0),
    15665: dict(inclination_mode="fixed", inclination_deg=30.0,
                start_velocity_kms=400.0, run_number=2),
}
```

These are example experimental choices, not recommended angles or velocities
for those sources. The notebook stores run handles and products in dictionaries
keyed by source ID, so changing the source list needs no source-specific code.
Ordinary errors are displayed and collected in `errors`; later sources continue.
A keyboard interrupt stops execution. No jobs are launched in the background.

## Prepare and inspect morphology

The individual package calls are useful both inside and outside the notebook:

```python
from pathlib import Path
from dingo import workflow as wf

source_id = 3700
working_dir = Path("/home/aakavan/kinematics/field_kinematics/edr_kinematics/PaA/dingo_runs")
run = wf.prepare_source(
    source_id,
    working_dir=working_dir,
    data_dir=Path("/path/to/prepared/data"),
    inclination_sigma_deg=5.0,
)
wf.show_inputs(run)
morphology = wf.fit_morphology(run, warmup=1000, draws=2000, display=True)
print(morphology.path)
```

`prepare_source` creates or reopens the kinematic experiment and checks its
inputs. `show_inputs` displays the direct image and R/C cutouts.
`fit_morphology` fits **one PSF-convolved Sérsic component with a constant sky**,
or reopens a matching saved fit. It does not make a second core-masked fit or
combine alternative morphology estimates. Its display includes the actual
fitting image, model/residuals, parameter summaries, native PySersic corner plot,
and inclinations derived from axis ratio for both `q0=0` and `q0=0.2`.

The shared morphology directory belongs to the source, not a kinematic run:

```text
WORKING_DIR/
  15665/
    morph_fits/
      F150W/
        current.json
        fit_001/
          pysersic_15665_info.cat
          parameters.txt
          ...posterior samples, plots and checkpoints...
        fit_002/                 # An intentional later refit
    15665_fixinc_i30_vstart400_run02/
      run_info.txt
      run.json
      results.npz
      ...plots...
      _checkpoints/
```

`parameters.txt` gives readable medians and 16th/84th percentiles, including the
inclination conversions. `pysersic_<ID>_info.cat` records input paths, settings, sampler
counts, timing, status and provenance. Posterior samples remain available for
reopening the corner plot and deriving inclination without refitting.

Unchanged settings and inputs reuse a matching morphology fit automatically.
`refit=True` requests a fresh revision and preserves earlier revisions.
Each kinematic run records the exact revision it uses; an existing run remains
pinned to that revision even if another experiment later refits the morphology.
To attach a new fit after a run already has an imaging prior, choose a new
kinematic run number or explicitly overwrite that kinematic run first. In the
notebook, set `REFIT_MORPHOLOGY=False` again after the intended refit.

The default `imaging_prior` mode uses the morphology-derived Gaussian mean
and the requested total width. The gas morphology and center are not forced
to follow the stars. Inspect whether a single Sérsic profile represents the
stellar disk before using its flattening as an inclination constraint.

The legacy `fit_imaging`, `use_imaging` and `show_imaging` APIs remain available
for reproducing the earlier full/core morphology workflow. The new notebook
uses `fit_morphology` throughout and requires no historical imaging directories.

## Search, choose and fit

```python
wf.find_candidates(run)
wf.show_candidates(run)

chosen_id = int(input("Candidate ID selected after inspecting residuals: "))
initial = wf.prepare_start(run, candidate_id=chosen_id)
initial.figure()

posterior = wf.fit_source(run, warmup=1000, draws=1000, walkers=64, seed=42)
wf.show_results(run)
wf.save_results(run)
```

`show_candidates` displays all saved candidates and their matching residuals.
Choose an **original candidate ID from its table**, rather than a row's position
in a sorted table. `prepare_start` renders the exact initial model without
optimizing or sampling and records the selection. A search alone does not start
MCMC or choose a scientific result. An exhaustive guarantee of all solutions
is not possible with this finite search.

`draws` is the total retained target **per walker**, after warmup. Live `tqdm`
bars show completed steps, elapsed time, estimated time remaining and speed.
The MCMC total counts warmup plus retained ensemble steps, not multiplied by
the number of walkers. PySersic and candidate searches show their own stage
progress. Bars update when the worker reports completed work, so they can pause
during compilation or expensive blocks. Detailed text logs and checkpoints
remain in the corresponding run/morphology directory.

## Inclination choices and manual starts

Set these arguments on `prepare_source`, independently for each experiment:

| Mode | Required choices | Meaning |
| --- | --- | --- |
| `"imaging_prior"` (default) | Morphology fit; optional `inclination_sigma_deg` | Sample inclination with an imaging-centered Gaussian prior. |
| `"gaussian_prior"` | `inclination_deg`; optional `inclination_sigma_deg` | Sample inclination with your chosen Gaussian mean and width. |
| `"fixed"` | `inclination_deg` | Hold inclination exactly at that angle; the Gaussian width is unused. |

`inclination_sigma_deg=5.0` is the **total** Gaussian 1σ width. It is not extra
scatter added to the imaging uncertainty, and it remains adjustable. The prior
is truncated to the existing 5–85° support. Use a new run name or number when
changing a prior. Imaging in fixed/Gaussian modes provides context; it does not
replace the angle or prior you specified.

Manual starts specify `start_velocity_kms` and, optionally,
`start_inclination_deg`. These are initial values, not constraints around which
the posterior must remain. The remaining coordinates come from a candidate
search for the same source. The notebook makes this search inside the new run
and asks which candidate should supply those coordinates. No external search
is required.

The broad search varies inclination even for a fixed-inclination experiment;
its rows are exploratory starts. `prepare_start` previews the exact fixed-angle
model, and `fit_source` holds that angle fixed throughout sampling.

For advanced reuse, `parent_search=earlier_run.search_directory` attaches an
external search without changing it. A sampled-inclination run requires a
matching prior, including its width and imaging provenance. A manual-velocity
run can omit `candidate_id` through the package API, in which case it records
the highest-posterior finite parent candidate for the other coordinates. The
demo instead asks for an explicit choice in every mode.

## Resume, restart and review

Calling `prepare_source` again with identical settings reopens the same run.
Calling `fit_source` with unchanged settings reuses/resumes its output. To
extend sampling, increase only `draws`, for example from 1000 to 8000. This
sets the total retained target to 8000; it does not add another 8000. Keep the
same warmup, walker count, seed, start and prior.

`overwrite=True` deletes only the identified kinematic run and its checkpoints
and saved products. It preserves sibling runs, observations and the source's
shared `morph_fits` directory. The workflow refuses to replace a run that is
still active or an unidentified directory. To compare experiments while keeping
their results, change `run_number` or `run_name` instead.

Each kinematic run has a readable `run_info.txt` and machine-readable
`run.json`, updated with timestamps, settings, selected start, actual/requested
sampling counts, diagnostics and provenance. `save_results` writes the final
NPZ and plots alongside them; internal sampler products stay in `_checkpoints`.

Reopening results requires no fitting:

```python
saved = wf.load_run(working_dir / "15665/15665_fixinc_i30_vstart400_run02")
posterior = wf.load_results(saved)
wf.show_results(saved)
```

`fit_batch({"source A": run_a, "source B": run_b}, **settings)` also supports
sequential MCMC after starts have been selected. It returns a `BatchResult`
per label and keeps ordinary failures separate. Review/export are explicit.
The demo uses short loops so each source's outputs are visible immediately.

## Reading velocities and errors

The sampled velocity is the deprojected asymptotic amplitude `V_inf` in
`V_rot(r) = (2/pi) V_inf arctan(r/r_t)`. It is not automatically the velocity at
the outermost observed radius. The model retains zero intrinsic dispersion and
fits one rotation field jointly to R+C, without a fitted gas-light profile.
PyTorch evaluates the likelihood on CUDA by default; emcee supplies interacting
ensemble walkers.

`show_results` shows residuals, a pooled monochrome corner plot, diagnostics and
**all** walkers. Residuals use the highest-posterior saved draw; the corner and
table use pooled medians and 16th–84th percentiles. These can describe different
locations when a run contains competing solutions. Completed sampling is not a
convergence verdict. Fixed-inclination intervals exclude inclination uncertainty;
intervals confined to one solution do not describe alternatives not explored.

`wf.show_results(run, display=False)` returns figures and tables for separate
notebook cells. `wf.save_results(run, radii_pc=observed_radii_pc)` additionally
exports rotation and mass profiles on your explicit radius array. Default
exports use illustrative radii, not established observed radial coverage.
Reportable profiles need radii justified from the data; supplying an array
does not establish that coverage. Masses use paired posterior draws in
`M(r) = r V_rot(r)^2/G`, a spherical-equivalent rotation-supported estimate
without pressure-support or lensing corrections.
