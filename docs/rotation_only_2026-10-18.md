# Rotation-only notebook: 18 October 2026

## Completed ID15665 run reviewed on 19 October 2026

The refreshed `real_sources/ID15665/rotation_v1` completed four chains with
1,000 warmup steps and 2,000 retained draws each, in about 142 minutes.
The displayed rotation parameters now agree closely across chains. Independent
diagnostics computed from the retained arrays agree with the saved report.

| Parameter | Median and 16th–84th percentile range | Rank R-hat | Bulk ESS |
|---|---:|---:|---:|
| Vrot at 0.5 arcsec [km/s] | 283.04, 274.24–294.09 | 1.00122 | 1,966 |
| Vrot sin(i) [km/s] | 194.20, 191.92–196.55 | 1.00069 | 2,655 |
| Inclination [deg] | 43.35, 40.99–45.49 | 1.00108 | 1,923 |
| Turnover radius [arcsec] | 0.07858, 0.07125–0.08594 | 1.00197 | 2,819 |
| Velocity-axis angle [deg] | 74.59, 73.87–75.33 | 0.99997 | 2,504 |
| Center x [arcsec] | −0.04728, −0.05309–−0.04157 | 1.00668 | 1,066 |
| Center y [arcsec] | −0.04972, −0.05419–−0.04536 | 1.00775 | 576 |

These are descriptive pooled statistics of the saved draws, **not yet validated
measurement uncertainties**. Velocity medians for the four individual chains
are 282.79, 283.70, 282.98 and 282.52 km/s. The corner plot's agreement is real;
the full model fails checks for identifiable reasons elsewhere:

- Nine of 8,000 retained transitions diverged.
- Worst background R-hat is 1.895 with bulk ESS 5.65; worst continuum R-hat
  is 1.569 with ESS 6.87. Systemic velocity and some registration parameters
  also fall short (R-hat 1.023/1.031; bulk ESS 273/253).
- 99.89% of retained trajectories reach depth 9. This indicates inefficient
  trajectories; it is not itself the acceptance veto.

The continuum/background decomposition has a poorly explored direction: the
quadratic Bernstein basis sums to one, so a common constant added to its three
coefficients can be offset by the opposite background change without changing
the predicted pixels. Priors and coefficient positivity still constrain this
direction, but conditional updates can move along it slowly. At the worst
continuum coordinate, inspecting continuum plus background improves R-hat
from 1.569 to 1.018. The worst background coordinate's pooled correlation with
rotation speed is −0.0066. This points toward improving that parameterization
and sampler while retaining the independent flexible gas morphology; it does
not prove that remaining sampling bias is absent. No sampler was changed or
rerun during this review.

Both new imaging fits pass sampling checks at 2,000 warmup and 4,000 retained
draws per chain: maximum R-hat <1.001, minimum bulk ESS >11,800, and no
divergences or radius/angle boundary pileup. Their residual RMS remains
4.78/4.30 times ERR, and their axis-ratio medians differ (0.86986/0.83477).
These are alternative stellar descriptions in the inclination prior, not two
independent imaging measurements.

The saved grism draws reproduce the detector images broadly well (residual
RMS about 1.11 for R and 1.13 for C). However, 10.1–10.6% of modeled gas flux
lies in the outer 0.1 arcsec of the source square, and bright emission reaches
its northern edge. Source extent and contamination still need checking before
final dynamical masses are interpreted.

The corner plot now annotates every diagonal histogram with the median and
asymmetric 16th–84th percentile interval (central 68%, the usual approximately
1-sigma equivalent). At the user's request, all retained chains are now pooled
into one monochrome plot: black contour edges with light/dark gray fills for
the 95%/68% regions, and one median with its interval above each diagonal.
While full-model checks fail, those pooled intervals are explicitly provisional;
the numerical convergence checks are unchanged. The notebook's corner and
profile cells reload saved results and redraw only; they never start a fit
or apply `OVERRIDE`. Original fit samples,
checkpoints and reports were preserved.

## Physical radii and posterior profiles added on 19 October 2026

The notebook now has separate rotation-curve and rotation-based mass-profile
cells. Both load saved results independently of the fitting cells. The source
redshift and Planck18 angular-diameter distance determine parsecs per arcsec;
the imaging pixel scale supplies only the conversion from pixels to angles.
The internal reference radius remains 0.5 arcsec to preserve the fitted model,
but its source-specific physical value is printed and used in plot labels.
The default turnover coordinate in the corner and trace plots is now in
parsecs. Center offsets and source-image axes remain angular coordinates for
comparison with the observations.

For the saved ID15665 run at z = 1.1613, 1 arcsec is 8,475.24 pc and
the reference radius is 4,237.62 pc. The pooled turnover radius is
665.99 pc (16th–84th percentiles: 603.86–728.35 pc). The spherical-equivalent
rotation mass inside the reference radius is 7.893 × 10¹⁰ solar masses
(7.410–8.521 × 10¹⁰); these remain provisional summaries of the same draws.

The original `Archive6/1-fit-kinematics.ipynb` also converts angle using
angular-diameter distance before evaluating r V²/G, but its mass cell
hardcodes z = 1.123 and FlatLambdaCDM(H0=70, Om0=0.3) for the saved ID15665
example. The archived repository's `utils.kpc_pix_scale` instead uses
Planck18. The new displays consistently use Planck18 and the actual source
redshift recorded with each fit; they do not copy that notebook's fixed distance.

The current rotation family already has a free turnover radius, equivalent
to the old arctangent model's turnover radius: it is the radius where the
curve reaches half its asymptotic velocity. It is distinct from a light-profile
scale radius. For each retained posterior draw, in consistent physical units,

\[
V_{\rm rot}(r)=V_{\rm ref}
\frac{\arctan(r/r_{\rm turn})}{\arctan(r_{\rm ref}/r_{\rm turn})},
\qquad M_{\rm rot}(<r)=\frac{r\,V_{\rm rot}(r)^2}{G}.
\]

The plotted median and pointwise 16th–84th percentile bands are computed
after transforming each paired velocity/turnover draw, preserving their
correlation. They pool all retained chains and are explicitly provisional
when full-model sampling checks fail. These are inferred arctangent profiles,
not independent measurements in radial bins. `RADIUS_MAX_PC` controls only the
display range; by default it ends at the reference radius for illustration.
That endpoint is not a claim that emission constrains the full displayed
range. Source-edge flux and contamination noted above remain unresolved.

The mass is a spherical-equivalent, rotation-only estimate, rather than a
full flattened-disk gravitational model. Interpreting it as enclosed dynamical
mass assumes circular motions and spherical gravity. Pressure support,
lensing and model systematics are not corrected; intrinsic dispersion stays
zero. Converting display units does not refit the data or change saved draws.

Automatic initialization currently optimizes fresh starts at 80, 250, 500 and
850 km/s with different velocity-axis angles. Velocity is free during each
optimization and MCMC; 200 km/s is not a fixed value. The earlier 1 km/s scan
was a separate profile-optimization experiment, not a current initialization
step and not the source of posterior errors. The present model keeps velocity
at a finite reference radius rather than switching to asymptotic velocity;
the latter can be derived from the same draws but depends more on the curve
at radii beyond the data.

The primary entry point is [`rotation_only.ipynb`](../notebooks/legacy/rotation_only.ipynb).
It runs on real native R+C exposures, with intrinsic dispersion fixed exactly
to zero. Earlier notebooks are preserved byte-for-byte in
[`notebooks/archive`](../notebooks/archive/); their manifest records checksums.
Existing scientific modules and old result directories were preserved.

## What is fitted

One positive gas-emission map and one mean velocity field are projected into
every usable R and C exposure. The map is a flexible 21×21 log-brightness field,
integrated on a 121×121 source grid. Its shape is independent of the stellar
image and the velocity field. The default inclination prior transfers the
accepted F150W fits with additional 10-degree scatter; it does not supply a
gas-light template. The finite map, regularization and response remain model
assumptions, rather than being eliminated by using two grisms.

With coordinates rotated into the velocity-axis frame,

\[
r^2=x'^2+(y'/\cos i)^2,\qquad
v_{\rm los}=v_{\rm sys}+V_{0.5}\sin i\,
\frac{\arctan(r/r_t)}{\arctan(0.5''/r_t)}\frac{x'}{r}.
\]

The central limit is evaluated smoothly by the renderer. The reported velocity
is **deprojected rotation at 0.5 arcsec**, not the asymptotic rotation speed.
The detector mean for exposure e is

\[
\mu_e=F_{\rm line}\,R_e\,g_e\,
\mathcal P_e[I,v_{\rm los},\sigma_0=0]+C_e+B_e.
\]

Here R is the calibrated count-rate response, g is a relative exposure gain,
P includes its trace, dispersion, optical response, registration and detector
integration, C is positive smooth strip continuum, and B is a signed background.
SCI, ERR, DQ and the recorded manual exclusions determine the pixel likelihood.
Spatial extent, rotational shear and instrumental broadening remain present
when intrinsic dispersion is zero. The predictions for R and C therefore need
not look identical, although their underlying galaxy is shared.

The model is a forward pixel likelihood. It preserves the shared-source R+C
idea but is not the legacy inverse-rectification objective.

## Inference and display

Several velocity/angle starts are optimized first. Verified old zero-dispersion
fits may initialize an additional start, but are evaluated and refitted under
the current target. A source does not require an old fit directory. Neither
optimizer curvature nor a velocity scan supplies the notebook's error bars.

JAX/NumPyro NUTS samples rotation, inclination, center, gas morphology, line
flux, gains, registration and background. Exact conditional Gibbs transitions
sample the positive continuum coefficients. These transitions preserve the
proper posterior; they are not independent draws. Continuum is not held at its
optimum during MCMC. Sigma is absent from the sampled coordinates and corners.

The one-node spectral renderer is numerically equivalent to the old multi-node
renderer at sigma=0, avoiding repeated identical deposits. Tests compare both
predictions and gradients. Warmup and sampling share one global adaptation
state. Each completed block saves the state, random-number state and draw
chunks before publishing its checkpoint pointer. Restart tests cover both
mid-warmup and post-warmup continuations without resetting adaptation or RNGs.
Changing scientific code, data, priors, library versions or adaptation settings
requires a fresh run; increasing retained draws can resume the same run.
Use `OVERRIDE = True` to replace the selected run, or use a new run name to
preserve the previous experiment. Resume checks apply with `OVERRIDE = False`.

The longer-run settings are four chains, 1,000 warmup steps and 2,000 retained
draws per chain, with progress/checkpoints every 50 steps. The notebook currently
preserves those settings for the user's selected ID15665. Compilation has a separate
elapsed-time message. Step timings advance all four chains together. Warmup
is excluded from the reported convergence diagnostics.

The notebook's images default to an actual saved draw from Chain 1, including
its sampled continuum. Set `IMAGE_CHAIN` to another zero-based chain index to
inspect it, or to `None` to inspect the optimized fit. Gray detector pixels are
excluded from fitting. Gray regions in the intrinsic velocity map instead hide
brightness below 1% of the peak.

The convergence gate checks every sampled coordinate, including morphology
and continuum, plus derived rotation: rank R-hat <1.01, bulk/tail ESS ≥400,
no retained divergences, and at least four chains. Frequent tree-depth
saturation is reported prominently as an efficiency warning, not itself a
veto on intervals when mixing and effective sample sizes are satisfactory;
this follows the distinction in the
[Stan diagnostic guidance](https://mc-stan.org/learn-stan/diagnostics-warnings.html#maximum-treedepth).
Passing is necessary, not proof of complete mode exploration or scientific
accuracy. Ordinary NUTS BFMI is not used as a validated diagnostic for the
changing Gibbs conditional target.

The corner plot pools retained chains into black/gray 68%/95% pairwise contours
and displays one marginal 16th/50th/84th percentile summary per parameter.
When full-model checks fail, its subtitle explicitly marks intervals provisional.
Chain traces remain available for diagnosing disagreements. The summary table
continues to withhold validated pooled errors until the sampling checks pass.
Instrumental and model systematics are not included in those conditional
posterior intervals.

## Real-source checks

On ID21636, all six optimized starts in `rotation_validation_v1` reached
Vrot(0.5 arcsec) = 187.49–187.55 km/s. Fresh starts included 80, 250, 500 and
850 km/s with velocity-axis angles 0, 90, 180 and −90 degrees. The maximum
objective difference was 0.00022; all projected gradient norms were <0.03.
This supports initialization stability of this optimized solution for this
source and target. It does not establish posterior uncertainty or rule out
every alternative mode.

The first GPU smoke run used four chains, 100 warmup steps and 20 retained
draws per chain, then resumed to 40 retained draws without repeating warmup.
It had no retained divergences, but failed convergence checks and saturated
its deliberately short depth-6 trajectories. Its chains moved away from their
initial velocities. Those draws are software validation, not a reportable
velocity interval. Its executed review notebook is outside Git, under
`results/real_sources/ID21636/rotation_smoke_v1/review.ipynb`.

The final GPU check, `rotation_validation_v1`, used four chains, 200 warmup
steps and 40 retained draws per chain at the notebook's depth-9 setting.
It finished in **9.8 minutes**. Retained draws had zero divergences and no
tree-depth saturation, but were still far too short to pass mixing checks:
maximum rank R-hat 3.70, minimum bulk ESS 5.4, minimum tail ESS 13.2.
The four velocity medians were 196, 193, 190 and 192 km/s; these are diagnostic
chain summaries, **not four independent measurements or a pooled uncertainty**.
The last sampling block took 1.22 seconds per iteration for all four chains
together. Earlier warmup blocks took about 2.3 seconds per iteration. Allow
roughly one to two hours for the full default run on this RTX 4090, with the
actual time depending on adaptation and source complexity.

The notebook was executed against real saved results, producing all four
figures without errors. Its Run All fitting path also reopened the completed
validation checkpoint without changing the report, state or draws. Numerical
tests cover the zero-width forward model and gradient, proper density,
conditional continuum draws, warmup/sampling restart equivalence, corruption
rejection, convergence gates, and posterior prediction/display. Tests were
repeated for the final diagnostic-only change that treats tree-depth warnings
as efficiency information. No scientific model or saved pilot draws changed
in that correction. The final short pilot still fails R-hat/ESS independently
of that warning rule.

Source preparation was checked on all seven locally available sources.
ID21636's native pixels, masks, inclination density, calibration and response
match the previously validated flexible model. ID8910 correctly retains 3R+6C
usable exposures and records three excluded off-detector C exposures.

Start with ID21636, then ID3700. Other available IDs are 21619, 16887, 6170,
15665 and 8910. Older imaging fits for the last two failed checks. The new
notebook can run PySersic again, or the user can explicitly select isotropic
inclination. Input availability does not imply an
accepted rotation measurement. In particular, previous ID3700 fits had emission
near the source-grid edge, and some other sources had boundary-sensitive fits.

## F150W fitting in the notebook

The notebook now calls `dingo.rotation_imaging.prepare_imaging` before a new
grism fit when `INCLINATION = "imaging"`. It checks accepted imaging products
first when `OVERRIDE = False`. With `RUN_IMAGING = True`, a missing or rejected
prior triggers the PySersic worker; `OVERRIDE = True` rebuilds the selected
imaging run from scratch. `RUN_IMAGING = False` checks saved imaging only and
does not delete it. `RUN_MCMC = False` skips
this preparation altogether and opens the saved grism result without launching
either sampler or deleting anything. Explicit `IMAGING` paths are supported
when reusing imaging; override ignores them for a fresh fit and always uses
the selected named `imaging_runs/IMAGING_RUN_NAME` output folder.

PySersic fits a single Sérsic model to the F150W stellar image. Its two fits
use the full fitting region and a version excluding the central pixels. Both
must pass the existing source identity, completion, sampling, and orientation
checks before their axis-ratio samples can be used. The two distributions are
treated as alternative stellar models and broadened for gas–star inclination
differences; the R+C gas-emission map remains independent of the Sérsic model.

The imaging controls are separate: `IMAGING_WARMUP = 2000`,
`IMAGING_DRAWS = 4000`, and `IMAGING_RUN_NAME = "pysersic_v1"`. New imaging
uses a 181-pixel cutout with a 2-arcsec fitting radius. The notebook shows a
summary table and observed/model/residual panels before checking readiness.
Failure remains visible and stops the grism fit; rerunning PySersic is not
a guarantee that a source admits a reliable photometric inclination. Imaging
requires four chains, rank R-hat <1.01, bulk/tail ESS ≥400, no divergences,
and ordinary NUTS BFMI ≥0.3. Frequent maximum tree depth is an efficiency
warning, consistent with the rotation sampler's reporting. Angle-seam checks
remain required; the new worker also rejects effective-radius boundary pileup.
ID15665's old run fails R-hat/ESS independently of its tree-depth warning.

New imaging runs are stored under the source's `imaging_runs/` directory.
`OVERRIDE = True` replaces the selected named imaging run; other historical
imaging products remain unchanged. The edited notebook's previous
outputs, including its error, were preserved in
`results/notebook_backups/rotation_only_before_pysersic_20261018_221613.ipynb`.
Its displayed outputs were then cleared so old-source plots cannot be mistaken
for new ID15665 results. The source ID and quick grism settings were preserved.

### Why ID15665 stopped, and what changed

The saved error was a rejected imaging prior, not missing F150W data or a
missing PySersic installation. Its older full-image fit had maximum rank
R-hat 1.336, minimum bulk ESS 10.3, and effective radius 44.98 pixels against
a 45-pixel upper bound. The small 1.2-arcsec fitting region left size and sky
poorly separated. The old runner stopped before attempting the core-masked
alternative and could not retry an existing failed output directory.

The new runner attempts both alternatives and keeps their results visible.
It uses four-chain JAX/NumPyro NUTS with a dense mass matrix. Progress is
reported for each block, with checkpointed adaptation and random-number state.
Atomic report pointers identify committed checkpoint generations, so an
interruption cannot replace the last recoverable state with an unpublished
one. Increasing imaging draws resumes the same target; changing warmup,
aperture, code or inputs requires a fresh imaging run, using override or a
new imaging run name.

An inherited angle-centering error was also corrected in the new worker.
PySersic's renderer defines the major-axis direction as theta + pi/2; the old
moment-to-angle expression reflected this angle. For diagonal sources that
could place the desired fit on the angular seam. Tests render tilted ellipses
and recover the correct convention. Historical imaging files remain unchanged.

The separate real-ID15665 pilot under
`results/real_sources/ID15665/imaging_runs/pysersic_validation_181/` uses
100 warmup steps and 40 draws per chain at depth 8. The full-region run
completed on the GPU with no retained divergences and no radius-boundary
pileup. Its diagnostic median radius is 20.77 pixels (0.623 arcsec), well
inside the old upper limit. Rank R-hat 1.078 and minimum bulk ESS 88 still
fail the acceptance criteria, as expected for such a short pilot. These
numbers are a software and fitting check, not an accepted inclination prior.

The core-excluded variant also finished with zero retained divergences and no
radius-boundary pileup. Its rank R-hat is 1.089 and minimum bulk ESS 30.2,
so this short run also fails acceptance. Its first sampling-block compilation
crashed inside native JAX; restarting the unchanged worker from the committed
checkpoint completed successfully. The notebook wrapper now makes one visibly
reported retry for this native crash when a verified checkpoint exists.
Other failures remain visible; a user interrupt never triggers a retry.

The full-region residual RMS is 4.78 times ERR and the residual image retains
strong spiral structure. Its RMS is not directly comparable to the old fit's
7.64 because the fitted region changed. Removing the radius-boundary problem
does not establish that a single Sérsic model gives an unbiased disk shape.
Inspect both alternatives before interpreting an inclination-dependent mass.

The focused worker, preparation, notebook and plotting tests pass (33 checks).
An executed notebook copy confirms that the real rejected ID15665 fit shows
its table and image panels before the readiness error. The repository notebook
remains unexecuted and ready for the user's run.

## Fresh-start control: 19 October 2026

The notebook now defaults to `OVERRIDE = True`. With fitting enabled, it
deletes and rebuilds only the current source's selected `IMAGING_RUN_NAME`
and `RUN_NAME` output folders, including their logs and checkpoints. This
lets a new run use changed settings without conflicting with a previous
checkpoint. Raw observations, other runs and other sources remain unchanged.
Set `OVERRIDE = False` to reuse or resume a compatible run. Set
`RUN_MCMC = False` to inspect saved grism results without any deletion,
regardless of the override setting.

Only enabled stages are refitted. With imaging fitting disabled,
`prepare_imaging(run=False, override=True)` remains read-only. A fresh imaging
fit ignores any explicit `IMAGING` path and rebuilds the selected named folder
under `imaging_runs/`; it does not delete the supplied external directory.
Before calling grism fit/load, the notebook clears the `result` variable so
a failed rerun cannot leave older source/run results available for later plots.

The first notebook cell reloads the two lightweight notebook API modules,
so rerunning it picks up API fixes in an existing kernel. The user's source
ID15665, grism 1,000/2,000 and imaging 2,000/4,000 warmup/draw counts are
preserved. Stale outputs were cleared after saving an exact notebook backup
outside Git at
`results/notebook_backups/rotation_only_before_override_20261019_095429.ipynb`.
These notebook changes do not start a fit or delete any fit results themselves.
