# Python workflow for multiple sources

`dingo.workflow` provides separate functions to prepare, search, fit, inspect
and save an experiment. Each experiment has its own `Run` handle and directory;
there is no global current source. It uses the existing three-file,
rotation-only R+C model. The scientific likelihood has not changed.

Use the **grism** environment and open the
[workflow demo notebook](../workflow_demo.ipynb). The preserved
[older notebooks](../notebooks/README.md) still use their existing interfaces
and save locations.

The demo's main ID3700 run reuses the current accepted imaging under `../results`
when `reuse_imaging_3700=True`. Set that switch to `False` to run PySersic afresh;
use a new run number or explicit overwrite when changing between these
configurations. It creates a new candidate search for this imaging prior. This
first search takes time; later calls with the same run and settings reuse it.
MCMC is enabled by default with
1,000 warmup steps, 1,000 retained steps per walker and 64 walkers. This is a
pilot for inspecting behavior, not a claim of reportable uncertainties. A
separate, optional ID15665 example starts at 400 km/s with inclination fixed at
30°; it is disabled by default. The final saved-result example can reopen an
explicit run directory without fitting.

## Search, inspect and fit

```python
from dingo import workflow as wf

working_dir = "/home/aakavan/kinematics/working_results"

source3700 = wf.prepare_source(
    3700,
    working_dir=working_dir,
    inclination_sigma_deg=5.0,
)

wf.fit_imaging(source3700)
wf.show_imaging(source3700)

wf.find_candidates(source3700)
wf.show_candidates(source3700)
```

`prepare_source` only creates or reopens the run. Its default inclination
mode uses an imaging-derived Gaussian mean. `fit_imaging` runs PySersic;
`show_imaging` displays the input, fit, native PySersic corner plot and the
inclinations inferred with `q0=0` and `q0=0.2`. To reuse an accepted fit instead
of refitting imaging, call
`wf.use_imaging(source3700, "/path/to/accepted/imaging_directory")` before the
search.

`show_candidates` displays all saved candidates and their matching residuals.
Choose an **original candidate ID from its table**, rather than a row's position
in a sorted table. The following cell asks for that choice explicitly:

```python
chosen_id = int(input("Candidate ID selected after inspecting residuals: "))

initial = wf.prepare_start(source3700, candidate_id=chosen_id)
initial.figure()
```

`prepare_start` renders the exact initial model without optimizing or sampling.
It records the selected start for this run. After inspecting it:

```python
posterior3700 = wf.fit_source(
    source3700,
    warmup=1000,
    draws=8000,
    walkers=64,
    seed=42,
)
wf.show_results(source3700)
wf.save_results(source3700)
```

`draws` is the total retained target **per walker**, after warmup. Progress is
printed while running; logs and checkpoints stay inside the run directory.
You may skip `prepare_start` and pass `candidate_id` directly to `fit_source`.
A search alone never starts MCMC or selects a scientific result for you.

## Inclination choices and manual starts

Set these arguments on `prepare_source`, separately for each run:

| Mode | Required choices | Meaning |
| --- | --- | --- |
| `"imaging_prior"` (default) | Accepted imaging fit; optional `inclination_sigma_deg` | Sample inclination with an imaging-centered Gaussian prior. |
| `"gaussian_prior"` | `inclination_deg`; optional `inclination_sigma_deg` | Sample inclination with your specified Gaussian mean and width. |
| `"fixed"` | `inclination_deg` | Hold inclination exactly at that angle; ignore the Gaussian width. |

`inclination_sigma_deg=5.0` is adjustable, for example
`inclination_sigma_deg=10.0`. It is the **total** Gaussian 1σ width, rather than
extra scatter added to the imaging uncertainty. The Gaussian prior is truncated
to the existing 5–85° inclination support. Supply a new run name or number when
changing a prior.

Manual starts specify `start_velocity_kms` and, optionally,
`start_inclination_deg`. These are initial values, not prior constraints. The
remaining initial coordinates come from a candidate search for the same source.
For example, a fixed-inclination experiment can reuse an existing search:

```python
source15665 = wf.prepare_source(
    15665,
    working_dir=working_dir,
    inclination_mode="fixed",
    inclination_deg=30.0,
    start_velocity_kms=400.0,
    parent_search="/path/to/15665/search",  # Contains start_search.json
    notes="Fixed 30-degree sensitivity experiment, starting at 400 km/s.",
)

initial15665 = wf.prepare_start(source15665)
initial15665.figure()
```

Use `earlier_run.search_directory` as `parent_search` when the search was made
with this interface. The parent is read without modification. If a manual run
does not specify `candidate_id`, it uses the highest-posterior finite parent
candidate for the other starting coordinates and records that choice. You can
instead pass an explicit candidate ID to `prepare_start` or `fit_source`.

For sampled inclination, use `inclination_mode="imaging_prior"` or
`"gaussian_prior"` with the corresponding settings. A reused parent search must
already have the requested prior, including its width. Changing the initial
inclination does **not** replace that prior. If its prior differs, make a new
search. Without `parent_search`, call `find_candidates` on the new run first.

The existing broad search varies inclination even for a run configured as
fixed. Those candidate rows are exploratory starts. `prepare_start` shows the
exact fixed-angle model, and `fit_source` then holds the angle fixed throughout
sampling.

## Several runs in one notebook

Keep separate handles such as `source3700` and `source15665`. After inspecting
and recording each start, fit them in successive cells or use a sequential batch:

```python
outcomes = wf.fit_batch(
    {"3700 imaging prior": source3700, "15665 fixed 30": source15665},
    warmup=1000,
    draws=8000,
    walkers=64,
    seed=42,
)

for label, outcome in outcomes.items():
    if outcome.succeeded:
        print(label, "finished; inspect before interpreting its intervals")
    else:
        print(label, "failed:", outcome.error)
```

If starts have not yet been recorded, supply
`candidate_ids={"3700 imaging prior": chosen_id}`. A manual-velocity run can
omit its candidate ID. Ordinary failures are recorded and later batch entries
continue. Interrupting the batch stops it; jobs do not continue in the
background. Review and export remain explicit calls for each successful run.

## Files, reopening and resuming

The default names describe the source, inclination choice, start and run number:

```text
WORKING_DIR/
  15665/
    15665_fixinc_i30_vstart400_run01/
      run_info.txt
      run.json
      results.npz
      ...plots...
      _checkpoints/
    15665_priorinc_s5_vstart400_run01/
  3700/
    3700_priorinc_s5_search_run01/
```

`run_info.txt` is created immediately and updated with status, UTC timestamps,
settings, selected start, prior, requested/completed sampling steps and
provenance. `run.json` is its machine-readable companion. `save_results` adds
the final NPZ and plots to the same directory. Checkpoint details remain in
`_checkpoints`; a product manifest tracks exported files.

To review saved output without launching work:

```python
saved = wf.load_run(
    working_dir + "/15665/15665_fixinc_i30_vstart400_run01"
)
posterior = wf.load_results(saved)
wf.show_results(saved)
```

Calling `prepare_source` again with identical settings reopens the same run.
Calling `fit_source` again with unchanged settings resumes/reuses its output.
Increasing only `draws`, for example from 8000 to 12000, extends the retained
target to 12000; it does not add another 12000. Keep the same warmup, walker
count, seed, start and prior. Use a new `run_number` or `run_name` to compare a
different experiment.

For an intentional fresh restart, pass `overwrite=True` to `prepare_source`.
It deletes only that identified run, including its checkpoints and saved plots.
It does not delete sibling runs, observations or an external parent search, and
it refuses to replace a run that is still active or an unidentified directory.

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
intervals confined to one solution do not describe alternatives that were not
explored.

`wf.show_results(run, display=False)` returns the figures and tables in a
dictionary for placing them in separate notebook cells. `wf.save_results(run,
radii_pc=observed_radii_pc)` also exports rotation and mass profiles on your
explicit radius array. Those radii must be justified from the data; supplying
them does not establish radial coverage. Masses use paired posterior draws in
`M(r) = r V_rot(r)^2/G`, a spherical-equivalent rotation-supported estimate
without pressure-support or lensing corrections. No mass plot is exported by
default.
