# A simpler shared gas-emission model: real R+C dispersion checks

This experiment fits a smooth exponential gas-emission template independently from stellar light, as requested. The comparison uses ID21636 and ID3700. It preserves the existing calibrated native-exposure likelihood, detector masks, continuum treatment and rotation model. The initial comparison retains the previous inclination assumptions; a separate ID21636 run uses its newly accepted imaging prior. All **300 fits are complete**. Earlier fits remain unchanged, and their MCMC errors do not apply to this model.

**The single exponential is not yet a reliable replacement for the flexible map.** Its preferred solutions use strongly different light and rotation geometries; ID21636 also reaches the velocity limit. Fixed-width scans show that the flexible model's low-dispersion preference persists when the zero-boundary optimizer problem is bypassed. Numerical refinement and the new ID21636 inclination prior do not resolve the morphology dependence.

Open [real_source_simple_dispersion.ipynb](../notebooks/archive/real_source_simple_dispersion.ipynb), select the source ID, and run with `RUN_FITS=False` to inspect saved results. The notebook displays the shared source map, R/C observed/predicted/residual panels, dispersion profiles and absolute pixel-fit comparisons. Each response/morphology profile has its own zero; those relative zeros cannot assess which morphology describes the pixels better.

## What changed, and what old DINGO actually did

The previous native model uses a positive log-brightness field with 440 adjustable map coefficients, plus a Gaussian envelope. The new model uses five gas-light shape parameters: major-axis effective radius, axis ratio, angle, and two center coordinates. A shared physical line flux sets its amplitude.

The light profile is the n=1 member of the analytic Sérsic family already present in old DINGO:

\[
I(E,N)\propto\exp[-b_1R_{\rm ell}/R_e],\qquad b_1=1.67835,
\]

\[
R_{\rm ell}^2=a^2+(b/q)^2,
\quad a=\cos\phi(E-E_{\rm light})+\sin\phi(N-N_{\rm light}),
\quad b=-\sin\phi(E-E_{\rm light})+\cos\phi(N-N_{\rm light}).
\]

Light center, axis ratio and angle remain independent of the rotation center, inclination and kinematic angle, matching the geometry of the original analytic forward function. **This makes it a phenomenological elliptical emission template, not a self-consistent axisymmetric rotating disk.** Independence from the stars does not require independence from the gas's own rotation geometry. The light radius prior is uniform in log(Re) from 0.05 to 0.8 arcsec; q is uniform from 0.1 to 1. Light-center coordinates have Normal(0,0.15 arcsec) priors truncated at ±0.3 arcsec. The old Gaussian-envelope size penalty is not applied to Re. Flux is normalized within the existing ±0.9 arcsec source square. Re describes the underlying untruncated analytic profile; it is not a measured half-light radius of the truncated map.

The actual old `Archive6/1-fit-kinematics.ipynb` used a different method: `KinematicsFitter.loss` inverse-warped the observed R and C images and minimized their unweighted squared difference. Its nominal `emline_model` was unused in that loss. Thus this experiment uses an analytic profile supplied by old DINGO; it does **not** recreate the notebook's inverse-image likelihood. Original sources: `archive/2026-10-12/yichen_codes/grism-kinematics/dingo/galaxy.py` (Sérsic function), `kinematics.py` (independent light/velocity geometry), and `fitting.py` (actual inverse-image loss).

## How dispersion is tested

One constant intrinsic LOS Gaussian width is shared by all positions and all R/C exposures. Local mean velocities still come from the arctangent rotation field. No extra thermal term or pressure-support correction is introduced.

The scan fixes sigma at 0,10,20,40,80,120,200 km/s and **removes that coordinate from optimization**. Rotation, other geometry, light shape/map, registrations, common flux and relative gains/backgrounds are refitted; positive continuum coefficients are profiled. Two separated stored branches supply upward/downward sweeps. Four fresh kinematic orientations are also attempted at each smooth-model endpoint, with the best endpoint propagated through the scan. Every attempted solution is saved, including alternatives and failed stationarity checks. This reduces local-minimum risk without certifying a global minimum.

Each morphology is tested with nominal LSF widths and with both spectral mixture components scaled by 0.9 and 1.1. These are illustrative sensitivity scenarios, not calibrated uncertainty bounds. Spatial response is held fixed. Each source has 108 fits in the initial comparison.

The objective is J=chi-square/2 plus prior penalties. Its explicit fixed-sigma prior contribution is 0.5(sigma/100 km/s)^2. Profiles are conditional penalized fits, **not confidence intervals, MCMC posteriors or model evidence**. The current simple model includes a matching NumPyro probability definition for future work, but no new kinematic chains are run in this experiment.

## Why the zero-boundary check matters

The symmetric Gaussian renderer is even in sigma, so its derivative at zero vanishes. The sigma prior also has zero derivative there. A first-order optimizer can therefore accept zero even if increasing sigma improves the objective. A regression test demonstrates that counterexample with the actual projector. Fixing sigma outside the optimizer bypasses this particular problem; nuisance-coordinate stationarity still does not guarantee the best basin.

## Numerical and visual validation

Before fitting, the code reproduces each saved flexible baseline's objective and verifies the source, masked SCI/ERR arrays, calibration, code and fitted-artifact hashes. Independent tests compare the exponential with DINGO's Sérsic formula, check light/rotation center independence, finite-difference gradients and the NumPyro density against the profiled joint objective.

Analytic profiles are initially evaluated on the same 61×61 source grid as the flexible model. The smallest allowed exponential minor scale is much narrower than its 0.03 arcsec spacing, so a successful optimizer is insufficient. A separate diagnostic reconstructs calibrated projectors on 61/121/181 grids with 9/17 spectral quadrature nodes, holds physical parameters fixed, reprofiles continuum and compares predicted pixels and objectives. Numerical checks and any resulting refits are recorded separately below.

## Results and conclusions

The first ID21636 scan is complete. With nominal response, the flexible model prefers the tested sigma=0 point with Vrot(0.5 arcsec)=186.88 km/s. The smooth exponential instead prefers the tested sigma=40 point with Vrot=1000 km/s at its upper bound. Its pixel chi-square is 786.42 higher than the flexible model, and structured residuals remain. That positive dispersion is not a credible gas-dispersion detection: the single smooth component changes the inferred dynamics while describing the pixels substantially less well. The response-width variants likewise drive the smooth model to the rotation bound.

ID3700 likewise changes substantially when its emission is restricted: the nominal exponential prefers the tested sigma=0 point with Vrot=661.50 km/s, versus 280.36 km/s for the flexible map. Its pixel chi-square is 935.91 higher. Both shapes prefer the lowest tested width for this source.

The flexible model favors sigma=0 in both sources under all three response scenarios. At sigma=20 km/s, the initial nominal penalized scores above zero are 1.62 for ID21636 and 7.46 for ID3700. Under the narrower/broader LSF they become 1.04/2.76 and 3.33/10.43, respectively. These are score differences, not confidence levels or upper limits. In ID21636 the smooth model's selected width changes from 40 to 80 km/s with the narrower response, while rotation remains at its upper bound.

Raw cross-model chi-square differences alone do **not** reject a model: the exponential has 438 fewer fitted coordinates, and the flexible map is regularized. The stronger warnings are compensating geometry, the velocity boundary, repeated central residual structure and the large change in inferred rotation. Individual example exposures do not by themselves prove that one model is correct. The flexible model also requires scrutiny, particularly ID3700's approximately 7.5% of fitted flux in the outer source-map band after refinement.

### Higher-resolution refits and the new imaging prior

Selected nominal-response points (sigma=0,20,40,80 km/s) were refitted on a 121×121 spatial grid with 17 spectral quadrature nodes, using two saved candidates per width and both continuation directions: 28 additional fits per source. A third 28-fit run repeats ID21636 at the same resolution with its validated full/core-masked F150W inclination distribution. This isolates the inclination change from numerical resolution. No imaging prior was borrowed from another source, and the failed ID15665 imaging fit was not used.

All values below are conditional fitted values at the best **tested** width, not posterior medians or uncertainty intervals. V is deprojected Vrot(0.5 arcsec).

| Source / inclination information | Emission model | Selected sigma (km/s) | V (km/s) | Inclination | Pixel chi-square |
|---|---|---:|---:|---:|---:|
| ID21636 / previous isotropic prior | Flexible | 0 | 187.51 | 69.95° | 123014.72 |
| ID21636 / previous isotropic prior | Exponential template | 40 | 1000, upper bound | 36.18° | 123799.34 |
| ID21636 / new F150W prior | Flexible | 0 | 187.51 | 70.05° | 123014.70 |
| ID21636 / new F150W prior | Exponential template | 40 | 1000, upper bound | 37.73° | 123800.38 |
| ID3700 / existing imaging prior | Flexible | 0 | 283.27 | 78.57° | 136002.94 |
| ID3700 / existing imaging prior | Exponential template | 0 | 661.00 | 38.99° | 136932.18 |

The new ID21636 inclination information has little effect on the preferred flexible fit. It also does not rescue the independent exponential template. The prior remains soft; it cannot enforce physical consistency between the template's freely fitted light ellipse and its rotation field.

![ID21636 refined fits with its own imaging inclination prior: dispersion profiles](../../results/real_sources/ID21636/simple_dispersion_imaging_v1/figures/dispersion_profiles.png)

![ID21636 exponential template: observed R/C, full predictions and residuals at the preferred sampled width](../../results/real_sources/ID21636/simple_dispersion_imaging_v1/figures/exponential_baseline_sigma40_fit_0025_detector.png)

At the higher resolution, the flexible sigma=20-vs-zero scores are 1.81 for ID21636 and 7.25 for ID3700. The broad qualitative low-width preference survives the numerical refinement. This still does not establish a physical zero dispersion, calibrated upper limit, reportable V/sigma, or dynamical mass profile.

### Numerical accuracy actually checked

There are 78 fixed-parameter numerical evaluations across both sources and both shapes, covering 0/20/80 km/s plus the exponential's best width. Rebuilding on 61/121/181 sky grids and 9/17 spectral nodes leaves the tested exponential objectives within 0.1 of their original values. Their largest line-image changes are below 0.02 ERR per pixel. Thus extreme profile undersampling does not explain the poor exponential fits at these particular solutions; it remains a concern elsewhere in the allowed shape parameter space.

The flexible maps show larger absolute objective changes, reaching approximately 1.5 for ID21636 and 4.9 for ID3700 at fixed physical parameters. Much of this cancels in contrasts between widths. Nevertheless, this motivated the actual 121×121 refits above. Differences between 121 and 181 grids remain as large as 0.93 in J for the checked ID3700 flexible candidates. These evaluations do not certify arbitrary precision or validate every high-width/local branch. The 121×121 runs are a refinement check, not a declaration that all numerical/systematic uncertainty has vanished.

### Axisymmetric control and the primary flexible model

For an axisymmetric oblate gas disk with intrinsic thickness q0,

\[
q_{\rm light}^2=\cos^2 i+q_0^2\sin^2 i,
\]

so q_light cannot be smaller than cos(i), and the projected light and rotation major axes should align modulo 180°. The nominal refined exponential fits violate that single-disk interpretation:

- ID21636: q_light=0.256 with i=36.18° (cos(i)≈0.81); light and kinematic axes differ by about 80°.
- ID3700: q_light=0.541 with i=38.99° (cos(i)≈0.78); the axes differ by about 53°.

Such templates could describe intrinsically non-axisymmetric light, but interpreting their fitted speed as a simple disk mass measurement would require additional justification. Merely freeing Sérsic n changes radial concentration and does not resolve this geometric inconsistency.

Following the user's clarification, a gas disk with **shared gas-light and rotation geometry** is being tested as a restrictive diagnostic control only. It is not the primary model or a requirement on the flexible reconstruction. The primary R+C model retains a common positive gas-brightness map fitted jointly with the velocity field, independently of the stellar light. Its clumpy or asymmetric brightness moments need not have the inclination or major axis of the circular velocity field. The relation above applies to an axisymmetric light disk; applying it to an arbitrary emission map would introduce the very morphology restriction the joint R+C approach is intended to relax.

The next comparison therefore tests both this restrictive control and the flexible map's amplitude/correlation priors, keeping the native data and response fixed. The accepted stellar imaging information remains a soft inclination density, not a gas-light template. See [the morphology-prior comparison](morphology_controls_2026-10-18.md) for its separate results and limitations. These changes do not alter any saved fits reported above.

## Completion and validation

- 216 initial fits, 56 higher-resolution fits with retained priors, and 28 ID21636 imaging-prior fits: **300/300 complete**, all passing the specified first-order stationarity threshold. This is numerical stationarity, not global or scientific convergence.
- 78 numerical-resolution evaluations completed. Saved pixel residuals, source identity, input/calibration/code hashes and model artifacts are checked. Model/scan/refinement tests passed, including the zero-width boundary counterexample and matching NumPyro density.
- The initial scans each took about 139 seconds on the RTX 4090, excluding development and subsequent checks. This is optimization timing, not MCMC timing.
- The final imaging-prior worker was interrupted during the disconnection and successfully resumed from verified checkpoints. The requested comparisons are complete; no kinematic MCMC was launched.
- The notebook defaults to loading saved results. Its additional-validation table summarizes the completed finer-grid and imaging-prior runs. Set `RUN_NAME` to `simple_dispersion_fine_v1`, or `simple_dispersion_imaging_v1` for ID21636, to inspect their own panels.
- [Final validation record](../../results/real_sources/comparison_2026-10-18/simple_dispersion_validation.json) records all 300 fitted-artifact checks and five executed notebooks. Each notebook has zero cell errors and three embedded figures; all runs retain their own `review.ipynb` copy.

## Files

- [New exponential model](../dingo/native_simple.py), [checkpointed fixed-sigma worker](../diagnostics/native_simple_dispersion.py), [numerical-resolution diagnostic](../diagnostics/native_simple_numerics.py), [CPU viewer](../dingo/real_simple_dispersion.py).
- [ID21636 initial comparison](../../results/real_sources/ID21636/simple_dispersion_v1/dispersion.json).
- [ID3700 initial comparison](../../results/real_sources/ID3700/simple_dispersion_v1/dispersion.json).
- [ID21636 refined comparison](../../results/real_sources/ID21636/simple_dispersion_fine_v1/dispersion.json), [ID3700 refined comparison](../../results/real_sources/ID3700/simple_dispersion_fine_v1/dispersion.json), and [ID21636 imaging-prior comparison](../../results/real_sources/ID21636/simple_dispersion_imaging_v1/dispersion.json).
- Numerical checks: [ID21636 exponential](../../results/real_sources/ID21636/simple_numerics_v1/numerics.json), [ID21636 flexible](../../results/real_sources/ID21636/flexible_numerics_v1/numerics.json), [ID3700 exponential](../../results/real_sources/ID3700/simple_numerics_v1/numerics.json), [ID3700 flexible](../../results/real_sources/ID3700/flexible_numerics_v1/numerics.json).
- [Higher-resolution / inclination-prior worker](../diagnostics/native_simple_refine.py).
- [Earlier five-source comparison](source_comparison_2026-10-18.md), retained for reference.
