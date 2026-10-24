> These are the existing single-source workflow instructions, relocated on October 23, 2026.
> The notebooks now live under `notebooks/legacy/`; saved settings and outputs were preserved.
> The planned multi-source workflow and source-first output layout are not implemented here.
> Start with the [repository overview](../README.md) or [notebook index](../notebooks/README.md).

<p align="center">
  <img src="../fig/logo.png" alt="DINGO Logo" width="240"/>
</p>

# DINGO: rotation from image.fits, R.fits and C.fits

Open **[rotation_only.ipynb](../notebooks/legacy/rotation_only.ipynb)** in the **grism** kernel.
The main notebook uses the original DINGO approach: find one rotation field
that makes the velocity-corrected, coadded R and C line images agree. It has
seven code cells: settings, PySersic imaging, candidate search, candidate
inspection, MCMC, posterior inspection, and rotation curve.
The notebook now samples **deprojected asymptotic velocity `V_rot = V_inf`
and inclination `i` as separate parameters**. `V_inf sin(i)` is derived from
each paired draw. Inclination has an imaging-centered Gaussian prior with
a total standard deviation of **5 degrees**; it remains free.

**ID15665 remains the default source and an important science target.** The
notebook was simplified on October 21 without changing the kinematic model
or removing any source's existing results. The complete previous notebook,
including outputs and settings, is preserved in the
[backup](../notebooks/archive/rotation_only_before_simplification_20261021_182638.ipynb).
The completed notebook before switching to projected-velocity displays is
also saved [with its outputs](../notebooks/archive/rotation_only_before_projected_velocity_20261021_194511.ipynb).
Inclination scans and alternate prior-selection routes are retained in the
pre-simplification backup and the scientific modules; they are no longer part
of the main notebook.

## Run or inspect a source

1. Use the **grism** kernel and set `SOURCE_ID`, `RUN_NAME`, and `WORKING_DIR` in the first cell.
2. Run through candidate inspection. PySersic fits or reuses this source's
   calibrated F150W imaging before the R+C search. All candidate residuals
   are displayed with their deprojected velocities, inclinations, and fitted centers.
3. Set `CANDIDATE_ID` in the MCMC cell, then run that cell and the result
   cells below it. No candidate is selected automatically.

The defaults are source **15665**, new run **`rotation_v_i5_v1`** with `VELOCITY_PARAMETER = "V_rot"`,
`RUN_SEARCH = True`, `RUN_MCMC = True`, `CANDIDATE_ID = None`, and
`OVERRIDE = False`. Run through the search and inspect its candidates before
choosing one for MCMC. `INCLINATION_SIGMA_DEG = 5.0` sets the total Gaussian
inclination-prior width; its center still comes from the accepted imaging fit
(32.15 degrees for ID15665). Inclination stays free over 5–85 degrees;
5 degrees is not a hard bound or an extra scatter added in quadrature.
Changing this prior requires a new named search and fresh MCMC, not an
extension of an existing chain. The amplitude prior is uniform from 0 to
1500 km/s in **deprojected V_inf**, not in projected V_inf sin(i). Therefore
this is a fresh posterior fit, not relabeling or rescaling old samples.
MCMC uses 64 emcee walkers, 1,000 warmup steps, and 8,000 retained steps,
with progress and checkpoints.

To reopen saved results, set both run switches to `False` and select the
matching source, run name, and candidate ID. The result-display cells can
also reload that selection after a kernel restart; save edited settings
first. They do not launch fits. `OVERRIDE = True` rebuilds the named search
and refits that source's PySersic imaging when those earlier cells execute.
The MCMC cell preserves the selected search. Other named runs, sources, and
raw input files remain separate.

The broad candidate search tries multiple velocities, orientations,
inclinations, turnover radii, and center/alignment starts. Refinement uses
the same posterior as MCMC. Search and MCMC share a fixed comparison mask
and variance prescription constructed at zero velocity and registration.
All candidates are shown on matching color scales; no candidate is discarded
solely for having a high speed. Finite search cannot guarantee that every
solution has been found. A fine search-grid spacing is not measurement
precision or an MCMC uncertainty.

## Saved results

For experiments with user-chosen starting velocity and inclination, open
**[manual_start_kinematics.ipynb](../notebooks/legacy/manual_start_kinematics.ipynb)**. Enter
`V_START_KMS` (deprojected asymptotic V_inf) and `I_START_DEG`. A saved broad
V_rot search supplies the other initial parameters and unchanged priors;
with `FIX_INCLINATION=False`, both V and i remain free in MCMC.
Set `FIX_INCLINATION=True` to hold i exactly at `I_START_DEG` and sample the
other parameters. Fixed-i runs have a separate `_fixedi` folder, omit i from
corner plots, and label velocity errors as conditional on that angle.
The notebook renders the exact starting
residual before sampling, then shows residuals, traces and black corner plots.
Different starts save separately under `WORKING_DIR/<experiment>/ID<source>/`.
It preserves the main notebook and its runs. `RUN_MCMC=False` only reads
saved results; `OVERRIDE=True` resets this manual experiment in its preparation
cell. To repeat identical starts separately, change `RUN_LABEL`.

`WORKING_DIR` now defaults to `/home/aakavan/kinematics/working/rotation_v_i5`.
The completed projected-amplitude run `rotation_simple_i5_v1` (candidate 6)
remains under `working/simple_rotation_i5/ID15665`. It used the 5-degree
inclination prior but sampled V_inf sin(i); it is not a result of the new setup.
The completed wider-prior run remains under `working/simple_rotation/ID15665`
and the cached run name `rotation_simple_v1` (candidate 9).
Each completed selected run exports to **`WORKING_DIR/ID<source>/`**:
`results.npz`, `run_info.txt`, and PNG plots of the inputs, fit and center,
traces, corner, rotation curve, and matching PySersic results
when available. Internal logs, checkpoints, and candidate trials remain
under `../results/`.

The source's final folder holds its latest completed export. Older named
runs remain in the internal cache; failed or interrupted runs do not replace
a completed export. `run_info.txt` records completion/export times, source,
run and candidate, sampler/device, walkers, warmup and retained steps, seed,
runtime, sampled parameters, checks, and velocity/inclination summaries.
Estimated completion times are labeled. Re-exporting does not reset them.

The NPZ contains samples, log probabilities, parameter names, input/fit/residual
images, profile percentiles, and embedded JSON metadata. New V_rot runs save
`rotation_curve.png` and `vrot_kms_p16_p50_p84`. Legacy V_sini runs retain
`projected_rotation_curve.png` and projected primary labels. Both velocity
definitions remain available in the NPZ; no mass plot is exported by default.
It opens with
`allow_pickle=False` and can be reviewed without the internal run cache:

```python
from dingo.three_file import load_final
result = load_final(15665, working_dir="/home/aakavan/kinematics/working/rotation_v_i5")
result.velocity_summary()
result.corner()
```

## Model and interpretation

Inputs are `data/ID<source>/image.fits`, `R.fits`, and `C.fits`; source
metadata and grism calibrations are read from existing repository files.
The R+C fit has no fitted gas-light profile, intrinsic dispersion, continuum
coefficients, or background offsets. It does not force gas morphology or
center to follow the stellar image. One velocity field rectifies both line
stacks, and the fit compares those corrected images.

The main notebook samples the **deprojected asymptotic velocity**, `V_inf`
(`velocity.V_rot` in saved chains), and inclination `i` separately. Default
summaries, corner plots, and traces show those actual sampled quantities.
The disk-plane rotation curve is

\[
V_{\rm rot}(r)=\frac{2}{\pi}V_\infty\arctan(r/R_v).
\]

The grism prediction projects this field as
`v_los = V_rot(r) sin(i) cos(phi)`; inclination also enters the deprojected
radius. `V_inf sin(i)` is derived draw by draw. Earlier runs that sampled
`V_sini` retain projected primary labels when loaded; they are not relabeled
as new deprojected fits.

`R_v` is where speed reaches half its asymptotic value. This amplitude differs
from the native model's speed at 0.5 arcsec; compare curves at the same radius.
`VELOCITY_RADIUS_PC = None` uses the physical equivalent of 0.5 arcsec as a
labeled comparison radius, not a measured emission extent. Parsecs use the
source redshift, Planck18 angular-diameter distance, and stack pixel scale.

PySersic uses calibrated F150W science, uncertainty, and PSF images. It shows
the fitted image, residuals, its built-in corner, and derived inclination.
The old direct-image stacks were display-stretched before coaddition and are
visual references, not calibrated linear-flux PySersic inputs. PySersic fits
ellipticity and an imaging sky level; inclination requires a disk assumption.
The block prints median and 16th–84th percentile intervals for `q0 = 0` and
`q0 = 0.2`, flagging incompatible axis ratios instead of clipping them.

The soft gas-inclination prior combines full-image and core-masked imaging
fits with equal weight as alternative measurements. It allows intrinsic
thickness `q0 = 0.05–0.30` when estimating the mean. The notebook explicitly
sets the **total** Gaussian sigma to `INCLINATION_SIGMA_DEG = 5.0`; it does
not add the former 10° scatter. The prior is truncated to 5–85° by the grism
fit. Its printed mean and
width are before truncation, not the bounded prior's median and interval.
MCMC retains the search's saved prior; inclination remains free. Failed
imaging checks stop new notebook searches rather than silently changing the
prior. Deprojected speeds and masses inherit these inclination assumptions.

All kinematic parameters remain free during selected-candidate MCMC; no
artificial boundary confines sampling to that solution. Inspect residuals,
traces, and diagnostics for movement to another solution and poor mixing.
The noise prescription is approximate: the coadds lack propagated error maps,
and interpolation correlates pixels. Intervals are conditional on the
selected model and this noise treatment. Visual selection does not validate
those assumptions or establish that a mode has converged.

The GPU batches PyTorch rectification likelihoods; emcee supplies interacting
ensemble walkers, not independent NUTS chains. Black/gray corner contours
show pooled 68% and 95% regions, with one median and asymmetric 16th–84th
percentile interval per diagonal. Image panels use `gist_heat`; residuals
are red–blue. A cyan × marks the fitted kinematic center and a white ○ the
cutout reference. Direct-image overlays assume nominal alignment because
legacy stacks lack WCS; they do not measure a stellar–gas center separation.
Corner and trace center axes show pixel offsets `x0 - x_cut` and `y0 - y_cut`.
Zero is the cutout center: `(40, 40)` for an 81×81 stamp, or `(20, 20)` for
a 41×41 stamp, using zero-based coordinates. Image overlays show absolute pixels.

`result.rotation_curve()` follows the run's sampled velocity parameter:
new V_rot runs show deprojected velocity; older V_sini runs show projected velocity.
Use `projected=True` or `projected=False` to select explicitly.
`result.mass_profile()` remains available outside the minimal notebook.
It transforms each posterior curve using
`Mrot(<r) = r Vrot(r)^2 / G`, a spherical-equivalent rotation estimate. It
assumes circular motion and excludes pressure support and a flattened-disk
potential. The plotted extent is not an established observed radius.

The existing **grism** environment has the required packages. For a separate
Python 3.12+ installation:

```bash
python -m pip install -e ".[rotation,mcmc,test]"
```

## Optional diagnostic and earlier work

[real_rc_validation.ipynb](../notebooks/validation/real_rc_validation.ipynb) is a separate **ID21636
individual-exposure prediction diagnostic**, not the main pipeline. Run All
opens its saved experiment: exposure splits, four starting solutions,
R/C residuals with centers, gas/velocity maps, and covariance-aware prediction
scores. `RUN_NEW_FIT = True` repeats that experiment with progress and fresh
outputs controlled by its own `WORKING_DIR`. It fixes dispersion to zero and
does not run MCMC or provide posterior errors. It has not established that
coaddition causes the extreme velocities or that this alternative is more
accurate than the coadded workflow.

The pre-simplification notebook and older native notebook are preserved in
[notebooks/archive/](../notebooks/archive/README.md), with checksums in the
[manifest](../notebooks/archive/manifest.json). Existing scientific modules,
native posteriors, and previous source results remain intact. Observational
FITS files and posterior results are local, not bundled with Git.

See the [three-file method notes](three_file_2026-10-19.md),
[native workflow and results](rotation_only_2026-10-18.md), and
[legacy command-line workflow](legacy_workflow.md).
