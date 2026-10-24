# Three-file rotation workflow — 19 October 2026

## Current notebook: sample V and i separately — 21 October 2026

The user's intended change is now implemented in the actual search and sampler,
not only in plot labels. `VELOCITY_PARAMETER = "V_rot"` selects the deprojected
arctangent amplitude V_inf. `velocity.inc_v` remains a separate sampled
coordinate. V_inf sin(i) is derived from paired draws. Both describe the same
underlying disk velocity field, and the samples can still be correlated.

The inclination prior uses the imaging-derived mean (32.1458 degrees for
ID15665) and **total Gaussian sigma = 5 degrees**, truncated to 5–85 degrees.
The amplitude prior is uniform in **V_inf from 0 to 1500 km/s**. This differs
from the previous uniform prior in V_inf sin(i); no Jacobian is applied to
preserve that previous prior. A fresh search and posterior are required.

Notebook defaults: `RUN_NAME = "rotation_v_i5_v1"`,
`WORKING_DIR = /home/aakavan/kinematics/working/rotation_v_i5`,
`CANDIDATE_ID = None`, `RUN_SEARCH = True`, `RUN_MCMC = True`,
`OVERRIDE = False`, 64 walkers, 1000 warmup and 8000 retained steps.
The search still explores velocity and geometry automatically. The user
inspects all candidates before choosing an MCMC start. Corner plots and
traces lead with sampled V_inf; the separate i axis remains visible. The
rotation curve and residual labels use deprojected velocity for new runs.
Legacy V_sini results keep their original definitions when loaded.

The completed previous 5-degree-prior run, `rotation_simple_i5_v1`, candidate
6, **still sampled V_inf sin(i)**. Its samples and final exports remain
unchanged. The complete notebook is backed up at
`../../results/notebook_backups/rotation_only_before_sampled_v_i5_20261021_225329.ipynb`.
No new source search or MCMC was run during this correction. Existing saved
runs remain readable, but changed code fingerprints protect them from being
silently resumed under a different configuration.

The sections below are development history and may describe earlier defaults.

## Earlier projected-velocity setup — 21 October 2026

The main [rotation_only.ipynb](../notebooks/legacy/rotation_only.ipynb) now has seven
code cells: settings, PySersic, candidate search, candidate inspection, MCMC,
posterior inspection, and projected rotation curve. The original coadded
R+C inverse-rectification model is unchanged. Intrinsic dispersion remains
zero; there is no fitted gas-light profile. Inclination scans and alternate
prior-selection routes were removed from the main notebook, not from the
scientific modules.

The complete previous notebook, including outputs and settings, is preserved
byte-for-byte in
[rotation_only_before_simplification_20261021_182638.ipynb](../notebooks/archive/rotation_only_before_simplification_20261021_182638.ipynb).
Its byte count and checksum are recorded in the archive manifest. No previous
fit results were removed. ID15665 remains the default and an important science
target; simplifying the workflow does not remove difficult sources from the
analysis.

Current defaults prepare a narrower inclination-prior comparison:
`SOURCE_ID = 15665`, `RUN_NAME = "rotation_simple_i5_v1"`,
`RUN_SEARCH = True`, `RUN_MCMC = True`, `CANDIDATE_ID = None`, and
`OVERRIDE = False`. The user inspects new candidates before selecting one.
`INCLINATION_SIGMA_DEG = 5.0` is the **total** Gaussian standard deviation,
not extra scatter added to the imaging uncertainty. The imaging-derived mean
remains 32.1458 degrees for ID15665. Inclination is sampled, with the same
5–85 degree support. This changes the prior, not the disk velocity law.
The notebook before this change, including its completed 8,000-step outputs,
is backed up at
`../../results/notebook_backups/rotation_only_before_i5_20261021_215251.ipynb`.
The completed wider-prior run `rotation_simple_v1`, candidate 9, and its final
folder `working/simple_rotation/ID15665` are preserved for comparison.
Its actual Gaussian sigma was 10.2077 degrees, including imaging uncertainty
and the former 10-degree additional scatter. It passed the sampling-length
checks after extension to 8,000 retained steps; that does not resolve the
inclination disagreement. No narrower-prior fit has been run as part of setup.
PySersic runs/reuses that source's imaging before the candidate search and
supplies the saved soft inclination prior. All candidates are displayed.
MCMC waits for an explicit candidate ID; then it uses 64 walkers, 1,000 warmup
steps, and 8,000 retained steps, with progress and checkpoints. Restart the
grism kernel once when opening the updated version.

The primary result is now projected velocity. The sampler already fits
`A = V_inf sin(i)`; its parameters and likelihood have not changed. The new
comparison changes only the inclination-prior width. Default corner plots/traces show `A`, and residual plots
distinguish it from `Vrot(r) sin(i)` at a labeled radius. The projected curve
is `(2/pi) A arctan(r/R_v)`. Inclination still affects disk geometry and remains
free alongside the center, orientation, turnover, and registration parameters;
this is not a one-parameter fit with everything else frozen. Deprojection and
mass interpretation are deferred. The minimal notebook no longer has a mass
cell, but `rotation_curve(projected=False)` and `mass_profile()` remain
available through the result API.

The complete notebook immediately before this display change, including the
user's saved outputs, is preserved as
[rotation_only_before_projected_velocity_20261021_194511.ipynb](../notebooks/archive/rotation_only_before_projected_velocity_20261021_194511.ipynb).
Its verified checksum and byte count are recorded in the archive manifest.

Final products go to
`/home/aakavan/kinematics/working/simple_rotation_i5/ID<source>/`, controlled by
`WORKING_DIR`. Exports now include `projected_rotation_curve.png` and the
`vsini_kms_p16_p50_p84` NPZ array. The NPZ retains legacy deprojected and mass
arrays for compatibility; no mass figure is exported automatically.
Set both run switches to `False` with the matching run name and
candidate ID to reopen a saved run. The posterior display cells can reload
saved selections after a kernel restart. This cleanup does not constitute a
new scientific fit or resolve the previously documented inclination/noise
limitations.

The earlier projected-display update passed 138 focused tests. All seven notebook
code cells were executed in saved-results mode for the user's completed
`rotation_simple_v1` candidate 9, with fitting entry points disabled. The
notebook now contains 20 rendered figures. Hashes of all 46 original run and
imaging files remained unchanged. The previous final export is preserved at
`../results/review_backups/ID15665_before_projected_20261021_194841/` relative
to the repository; the refreshed plots and NPZ are in the usual working folder.

At the earlier 4,000-step checkpoint, the pooled projected asymptote was
`Vinf sin(i) = 200.63 (+2.20 / -2.17) km/s`, and
`Vrot(4238 pc) sin(i) = 159.62 (+1.26 / -1.23) km/s`.
These are 16th–84th percentile intervals from the existing draws, not a new
fit. That checkpoint remained provisional: the center coordinates had only about
33–34 estimated autocorrelation times of retained sampling, below the existing
length criterion. The 4238 pc comparison radius is not a measured emission extent.

The earlier `broad_candidates_v3` run gives approximately 159.50 km/s at the
same radius, but the scan-initialized `broad_candidates_v3_i30_s10` solution
gives 183.20 (+1.54 / -1.34) km/s. That latter run changes both initialization
and inclination prior, so this difference cannot be attributed to either alone.
All use the same saved comparison mask/noise prescription. Projected velocity
avoids dividing by a small sine, but is still dependent on geometry and the
selected solution. Agreement between repeated fits of one solution is not
independent validation; uncertainty in stack processing is also not captured
by these narrow conditional intervals.

The earlier eight-cell simplification passed 85 focused tests and executed
all eight code cells in
saved-results mode against ID15665's existing `broad_candidates_v3` candidate
9. The check rendered 21 figures and exported 13 files to a temporary directory,
with fitting entry points disabled. Hash checks confirmed that all 58 existing
search/imaging files were unchanged. Those checks describe the earlier
simplification, not a new science fit or validation of the model's uncertainties.

**The sections below retain the development history. Their source selections,
run names, paths, and scan instructions describe earlier versions, not the
current notebook defaults.** The [README](../README.md) describes the current
entry point.

## Earlier workflow and results

The main [rotation_only.ipynb](../notebooks/legacy/rotation_only.ipynb) now uses the original
DINGO inverse-rectification approach with `image.fits`, `R.fits` and `C.fits`.
The user requested this simpler workflow instead of the native-exposure
continuum/background model. The selected source remains ID6170 and
`OVERRIDE` remains `False`.

## Final working directory

Final notebook products now go to `../working/ID<source>/`, controlled by the
single notebook setting `WORKING_DIR`. That folder contains `results.npz`,
`run_info.txt`, and PNG plots. It is updated automatically after a completed selected MCMC run
or when that completed run is reopened in the selected-fit cell. Internal
logs, checkpoints and candidate trials remain under `../results/`.

The NPZ is self-contained: retained samples and log probabilities, parameter
names/units, diagnostic and prior metadata, final fit images, rotation/mass
profile percentiles, and the matching imaging samples/images when available.
It can be opened with `numpy.load(..., allow_pickle=False)` or
`three_file.load_final(source_id, working_dir=WORKING_DIR)` without the run
cache. A new completed export replaces the previous final copy for that source;
older named runs remain in the cache. An interrupted fit or failed export
leaves the previous final folder intact.

The user's completed ID6170 candidate-1 run (64 walkers, 1,000 warmup and
4,000 retained steps) has been exported to `/home/aakavan/kinematics/working/ID6170/`.
That folder has one NPZ, a readable run-information file, and ten plots. Its reopened samples, residual images
and velocity summary were checked against the original run; the imaging and
kinematic caches were unchanged. No new fit was launched for this export.

`run_info.txt` distinguishes completion time from export time, lists actual
walkers/warmup/retained steps and the seed, and includes model, device, diagnostic
status and velocity summaries. Times are shown in America/Phoenix with UTC also
recorded for completion. ID6170's completion time is estimated from the original
report modification time (2026-10-19 19:12:07 MST); the estimate is labeled.

## What is fitted

One rotating velocity field rectifies both continuum-subtracted line stacks.
The fit compares their rectified R and C images. It does not fit a gas-emission
profile, force gas to follow stellar light, or add intrinsic dispersion,
continuum coefficients or background offsets. Existing source configurations
supply redshift, coordinates, module and filter; the grism calibration remains
part of the model.

The projected asymptotic amplitude `V_inf sin(i)`, arctangent turnover radius,
center, orientation, inclination and relative registration describe the
kinematics. A velocity/angle search and optimization produce candidate fits.
The user inspects their residual maps and velocities, selects a candidate,
and then runs ensemble MCMC within that solution. A small search-grid spacing
describes search resolution, not velocity uncertainty. The search cannot
guarantee that every possible solution has been found.

The curve is

\[
V_{\rm rot}(r)=\frac{2}{\pi}V_\infty\arctan(r/R_v),
\qquad V_\infty=\frac{V_{\rm projected}}{\sin i}.
\]

`R_v` is the half-asymptotic-speed radius; it is not the stellar-light scale
radius. The asymptotic amplitude must not be directly compared with the
earlier native model's `V(0.5 arcsec)`. Compare velocities at a common radius.

## The direct image and inclination

Inspection of the archived `Archive6/1-fit-kinematics.ipynb` shows that each
F444W direct-image exposure was transformed with
`arcsinh(image / sigma_clipped_std)` before the cutouts were coadded and saved.
The present ID15665 `image.fits` is array-identical to that archived product.
The inspected direct-image FITS files contain only an 81×81 primary array,
without WCS, filter, pixel-scale or uncertainty metadata.

These arrays are useful visual references but are not calibrated linear-flux
images for a conventional PSF-convolved Sérsic likelihood. The notebook's
PySersic block instead locates the existing calibrated F150W science,
uncertainty and PSF images. It shows the actual fitting image **before** starting
PySersic, then sampling diagnostics, image/model/residual panels, PySersic's
default built-in corner, and a derived inclination panel. The corner now calls
`PySersicResults.corner()` without contour, label or title overrides; the first
version had customized its appearance. This block runs before the
velocity/center search and reuses an accepted source-specific imaging fit
unless `OVERRIDE = True`. Imaging defaults to 1,000 warmup and 2,000 retained
draws; `RUN_IMAGING = False` requests a saved imaging fit.

PySersic fits ellipticity rather than inclination. The displayed inclination
panel uses the thin-disk conversion. The block additionally prints both fixed
`q0 = 0` and `q0 = 0.2` inclinations, with median and asymmetric 16th–84th
percentile intervals from the same imaging draws. These intervals include no
stellar–gas scatter. If any draws have `b/a < q0` beyond floating-point
roundoff, that assumption is flagged with the incompatible fraction and no
interval, rather than silently clipping those draws to 90°. The prior passed to the gas fit instead
allows oblate-disk thickness `q0` uniformly from 0.05 to 0.30 before rejecting
incompatible axis-ratio/thickness pairs. Full-image and core-masked fits have
equal mixture weight because they are alternatives to the same data, not
independent measurements. The mean and variance of that mixture, plus
`GAS_STELLAR_SCATTER_DEG = 10` degrees of additional scatter in quadrature,
define a Gaussian approximation. The grism fitter truncates it to 5–85°.
The block prints Gaussian `mu` and `sigma` **before truncation**; they are not
the bounded prior's median and uncertainty. This is a soft gas-inclination
prior, not a PySersic measurement of the gas inclination. PySersic samples an
imaging sky level (`sky_back`); the R+C kinematic fit has no background term.

The prior and its provenance are saved with the candidate search. Selected
MCMC uses that saved prior even when the imaging files are unavailable during
later review. Neither the stellar center nor the Sérsic light profile fixes
the gas center or gas morphology. New searches require accepted imaging
results for the selected source; they do not silently switch to an isotropic
prior if imaging fails. Programmatic fitting without an imaging prior remains
available. Deprojected speeds and masses inherit the inclination assumptions.

The stack pixel scale is 0.063 arcsec/pixel, as used for these detector cutouts.
A later archived plotting cell switches to 0.031 arcsec/pixel for a scale bar;
that inconsistent plotting value is not adopted here. Parsecs additionally
require each source's actual redshift and angular-diameter distance, calculated
with Planck18.

## Sampling and review

The main notebook now selects run `visual_imaging_v1`, separate from the
earlier isotropic `visual_v1` search. Its first Run All reviews imaging,
constructs the soft prior, and displays candidates; `CANDIDATE_ID = None`
deliberately prevents R+C MCMC from
starting before visual inspection. Set that ID from the displayed table,
optionally record `SELECTION_NOTES`, and run from the selection cell onward.
`RUN_SEARCH = False` opens saved candidates and makes the imaging block
read-only; missing imaging does not block saved R+C review.
`RUN_MCMC = False` opens the selected posterior. `OVERRIDE` replaces the named
imaging and R+C search runs when their respective fitting cells execute.
Read-only review never applies it. The selected-fit cell preserves the
candidate search and its prior.

The initial search ranks R−C residuals with inclination restricted to Gaussian
`mu ± 2 sigma`, clipped to 5–85°. Its optimizer does not include the Gaussian
prior penalty. MCMC applies that penalty over the full 5–85° inclination
support and may move away from the visually selected candidate. The final
residual maps and velocities therefore need inspection too.

Residual panels share a color scale across candidates. Velocity labels belong
to each displayed candidate, not posterior marginal medians. Compare both
deprojected `V(r)` and projected `V(r) sin(i)` at the same stated radius.
`VELOCITY_RADIUS_PC = None` uses the physical equivalent of 0.5 arcsec,
explicitly labeled as a comparison radius, not measured emission coverage.
A large asymptotic velocity can reflect an unconstrained turnover radius or
inclination; investigate it with the residuals and the speed over the source.
There is no automatic rejection of high-velocity candidates. The selected
posterior view separately shows a pooled velocity median and uncertainty.

MCMC defaults to 1,000 warmup steps and 4,000 retained steps for 64 emcee
walkers, with batched PyTorch likelihood evaluations on CUDA.
The ensemble walkers interact; they are not independent NUTS chains.
The input line stacks have no propagated ERR maps. Their estimated-noise
likelihood approximates the effects of stacking and inverse interpolation;
the comparison mask and noise prescription are fixed for the selected run.
Posterior intervals are conditional on the visually selected solution, noise
approximation and rotation model. They do not summarize the probability of
all alternative solutions. Visual inspection cannot establish that the noise
estimate sizes the intervals correctly, or replace checks of mixing within
the selected solution. This workflow is not a demonstrated correction of
ID6170's previous high-velocity fit.

Sampling starts at the selected candidate but has no artificial boundary
around it. If walkers move between distinct solutions, their pooled intervals
must not be presented as the uncertainty of only the original candidate.

Inspect observed and corrected R/C images, coherent structure in their
difference, source centering and edges, as well as sampling diagnostics and
traces. Corners pool retained walkers into black contours with gray 68%/95%
regions and one median with 16th–84th percentile interval per diagonal.

The separate rotation and mass cells read saved draws only. They evaluate
paired posterior parameters, preserving their correlations. Mass is
`r Vrot(r)^2 / G`, a spherical-equivalent rotation mass assuming circular
motion, without a flattened-disk potential or pressure-support correction.
The default plot extent corresponds to 0.5 arcsec for illustration, not a
measured limit of resolved emission or a claim that every radius is constrained.

## Preservation

The prior main notebook, including ID6170 outputs, is archived exactly as
[rotation_only_native_2026-10-19.ipynb](../notebooks/archive/rotation_only_native_2026-10-19.ipynb).
Its SHA-256 is
`615c3713c6dcb818cb00086725ffac9983bf0bff71380ea229a40eeebfb7f7dc`.
A second timestamped copy is under `../results/notebook_backups/` relative to
the repository. The archive manifest includes the new file.

The new notebook has no carried-over figures. Its run name is `visual_imaging_v1`,
separate from earlier `visual_v1` and `three_file_v1` work. Native scientific modules, reports
and posterior samples remain available through the archived notebook. The
final three plotting cells only load the chosen source, run and candidate;
they never fit or delete data. A copy of the preceding three-file notebook
was saved to `../results/notebook_backups/` before this visual-selection edit.
The notebook's full science run has not been launched.

Before the PySersic-block edit, the preceding visual-selection notebook was
backed up at
`../results/notebook_backups/rotation_only_before_pysersic_20261019_183417.ipynb`.

## Software validation

The preceding automatic-selection version passed 91 focused tests covering worker configuration, posterior quantities,
automatic-start logic, the existing likelihood, output isolation, and notebook
fit/load dispatch. A short real-ID6170 GPU run completed using 20 walkers,
10 warmup steps and 25 retained steps, with a deliberately coarse start search.
All five review figures rendered. Reopening the completed final smoke run
preserved its checkpoint, samples, images and report byte-for-byte. Extending
it by five retained steps preserved all previous draws and appended exactly
five steps. These checks establish software operation, not convergence or a
new scientific velocity measurement.

The final smoke artifacts are under
`../results/three_file_sources/ID6170/three_file_smoke_final_20261019` relative to
the repository, separate from the notebook's `three_file_v1` run.

For the visual-selection notebook, 43 notebook tests passed. They verify that
an unselected candidate cannot launch MCMC, the selected-fit cell never
replaces the search, saved-only plots validate the candidate and never fit,
and a missing posterior cannot display a stale result. The notebook schema
and all code cells also validate. These checks exercise notebook behavior,
not scientific convergence; no new science fit was launched for this edit.

The full visual-selection integration passed 103 focused tests. A separate
real-ID6170 software check at
`../results/three_file_sources/ID6170/visual_smoke_20261019` first searched and
saved candidate images without creating any posterior or checkpoint. Explicit
candidate selection then ran 10 warmup and 30 retained steps with 20 GPU
walkers, reused the saved search, and rendered the new residual/velocity view.
Those short runs and their velocities are not scientific measurements. The
notebook's `visual_v1` science run remains unstarted.

The PySersic notebook integration passes 55 notebook tests. They cover the
imaging-before-search order, reuse-only review, same-source prior requirement,
failure without stale priors, loading saved R+C results without imaging files,
and using the saved search prior for selected MCMC. The notebook schema and
code cells validate, and its old outputs are cleared. These checks do not
launch an imaging or kinematic science fit.

The settings cell selects the CPU JAX backend before importing imaging review
modules so that displaying a PySersic corner does not reserve GPU memory.
Isolated fitting workers select their own backend from `DEVICE`, which remains
`"cuda"` by default.

The complete PySersic/three-file integration passed 165 focused tests. The
actual notebook imaging block also ran against the accepted ID6170 F150W
samples with fitting workers forbidden: all four figures displayed, the
saved imaging files remained byte-for-byte unchanged, and the resulting
Gaussian parameters (before truncation) were `mu = 77.2638°` and
`sigma = 10.6474°`. These values transferred correctly into the MCMC
configuration. No new imaging fit, velocity search or kinematic posterior was
run for this change. Reload the notebook and restart its kernel once after
updating, so the new plotting methods and CPU review backend are loaded.

## Broad candidate search (2026-10-19)

The current notebook uses `SEARCH_METHOD="broad"` in a separate named run.
The original `visual_imaging_v1` searches and posterior files are preserved.
Legacy workers were not edited, so their existing identity checks remain valid.

The new worker (`dingo/three_file_search.py`) uses the original inverse R/C
rectification and the same nine sampled parameters. It introduces no gas
morphology template, dispersion, continuum, or background parameters.

Candidate discovery evaluates projected asymptotic speeds every 25 km/s and
16 position angles at five inclination starts and three turnover radii
(1, 3, 8 stack pixels). It retains seeds in separate amplitude bands, then
performs batched posterior refinement. Additional starts offset R registration
by neighbouring pixels and the velocity center by two pixels. These are starting
points, not new parameter constraints. Refinement retains the full existing
prior support, including 5–85 degrees for inclination.

A common comparison model is built at zero velocity and zero registration,
independently of the candidates, and saved in `comparison.npz`. Its mask,
variance, correlation-area tempering, and input footprints are reused by all
search evaluations and subsequent MCMC, including checkpoint resumes. The
reference is a noise/coverage convention, not a zero-rotation prior. This
changes the previous candidate-specific noise approximation: raw posterior
scores from the old and new runs must not be compared. Neither approximation
fully propagates covariance in the continuum-subtracted stacks.

The GPU search evaluates the same cold-started fixed-iteration likelihood as
MCMC, including the Gaussian inclination prior and the other existing priors.
During search, the uniform-prior log noise scale is maximized analytically,
clipped to its existing bounds. MCMC samples it as before. Candidate output
separates likelihood and prior contributions and retains the unweighted
residual sum of squares for inspection.

The physical-velocity profile fixes Vrot at the displayed radius and refits the
other parameters from multiple starts at every grid point. The default radius
remains the physical equivalent of 0.5 arcsec, not a claimed resolved-emission
boundary. Its default speed range is 0–1000 km/s in 25 km/s steps, with 1 km/s
refinement near the first best grid point. Neighbouring profile solutions and
competitive free fits are subsequently propagated across the grid in two
cross-check passes. The last pass's largest score improvement is saved and
reported when still appreciable. A finite search is not a guarantee of the
global profile envelope.

The plotted quantity is twice the difference in maximized log posterior,
using the original sampler parameterization. This is not a marginalized
posterior or a confidence-interval construction; no Jacobian/prior change is
introduced by merely constraining the plotted derived velocity. Grid spacing
is not measurement precision. The best free candidate can lie beyond the
plotted range, in which case the notebook says that the profile has not
bracketed it.

Free candidates and representative fixed-velocity comparisons are displayed
with identical image/residual scales. The latter are explicitly labeled:
they need not be stationary solutions, and free MCMC may leave them. Fits
near inclination boundaries are flagged, not hidden or automatically rejected.

Progress is logged for geometry grids, refinement iterations, and profile
cross-checks. Completed phases are saved separately and reused on rerun with
matching code/settings. An incomplete phase restarts; `search_progress.npz`
retains its most recent parameters and scores for inspection. Search products
stay in the named run cache; the final working folder is updated only after
an explicitly selected MCMC completes.

### ID15665 validation and interpretation

The final search is saved at
`../results/three_file_sources/ID15665/broad_candidates_v3/`. It completed in
4.72 minutes on the RTX 4090, produced 11 candidate rows, and ran no MCMC.
Re-running the same request successfully reused the completed search without
optimization. The notebook selects this run and leaves `CANDIDATE_ID=None`.
The older `visual_imaging_v1` posterior and predictions still match their
original SHA-256 checksums; the final working folder was not replaced.

At the 4237.6 pc comparison radius:

| Candidate | Type | Vrot(r), km/s | Vinf sin(i), km/s | Inclination | 2 delta log posterior |
|---|---|---:|---:|---:|---:|
| 0 | Free candidate | 1733.2 | 200.6 | 5.27 deg | 0 |
| 8 | Fixed-velocity comparison | 225 | 202.7 | 45.54 deg | 143.50 |
| 9 | Fixed-velocity comparison | 475 | 200.6 | 19.59 deg | 16.26 |
| 10 | Fixed-velocity comparison | 725 | 200.4 | 12.70 deg | 4.39 |

These are optimized fits, not posterior medians or error bars. Moderate
velocities now receive nuisance-parameter optimization and are available for
residual inspection. Their fixed-velocity labels matter: they are not established
free local modes, and selecting one does not constrain MCMC to that speed.

Candidate 0 still approaches the 5-degree inclination floor. Its asymptotic
speed is about 2183 km/s. The profile over 0–1000 km/s at 4238 pc therefore does
not bracket the best free candidate. The last profile cross-check improved at
least one grid point by 2.08 in log posterior; remaining optimization dependence
is explicitly reported rather than treating the curve as an exact envelope.

Central residual structure remains in the high-speed fit as well as in the
moderate-speed comparisons. This search improves exploration, but does not
establish a credible deprojected velocity or dynamical mass for ID15665. More
MCMC alone would not resolve the inclination sensitivity or demonstrate the
adequacy of the residual model.

To isolate search improvement from the changed noise convention, the old
candidate and old MCMC best-fit parameters were rescored with the new shared
comparison model and an optimized noise scale. The old candidate 0 scored
10431.39 and the old MCMC best point scored 10659.82, compared with 10677.41 for
the new free candidate 0 (CPU/GPU rounding is much smaller than these gaps).
Only these common-objective scores are directly comparable.

Saved inspection products include `candidate_summary.csv`,
`figures/candidates.png`, `figures/candidate_comparison.png` (rows 0, 8, 9, 10),
and `figures/velocity_profile.png`. The profile figure includes a full-range
panel and a zoom below a score difference of 50, without interpreting those
values as confidence thresholds.

Validation: 152 focused tests passed. New tests cover profiled-noise bounds,
multistart movement, fixed physical-velocity constraints, neighbouring-solution
cross-checks, exact shared-mask reuse across starts, scalar/batched posterior
agreement on the real ID15665 inputs, selected-candidate handoff to MCMC
preparation without drawing samples, saved-only profile plotting, and checksum
rejection. The notebook validates as nbformat and all Python cells parse.
