# Repeated-noise and model-error checks: 2026-10-14

**Subsequent wider search:** the [paired R/C comparison](grism_comparison_2026-10-14.md)
finds a second joint branch for matched seed 481: about 499 km/s with turnover
radius at its upper bound, versus the original roughly 211 km/s solution.
The high branch has a training likelihood advantage of only 2 delta NLL = 1.02,
but a very different mass profile and worse held-out predictions. The numbers
below describe the originally explored starts; they do not certify global
optimization. Broader mode searches are required before using these ensembles
to calibrate profile intervals.

This extends the [flexible-map validation](native_flexible_2026-10-14.md).
The question is whether the recovered rotation is repeatable, whether a
nonrotating source can resemble a rotating one, and whether the true velocity
remains compatible with the data after the unknown emission map is re-fitted.
These remain synthetic experiments using the real ID3700 exposure geometry,
ERR and masks. They are not measured properties of ID3700.

## Completed rotating-source ensemble

Six independent noise seeds give the following unpenalized fits to the rotating
reference (injected projected speed 210 km/s at 0.5 arcsec):

| Map controls | Median fitted speed, km/s | Range over six seeds, km/s | Nominal truth compatibility |
|---|---:|---:|---:|
| 13x13 | 184.65 | 147.59--362.60 | 4/6 |
| 17x17 | 225.87 | 157.38--369.35 | 4/6 |

All twelve selected fits satisfy the gradient criterion and none hits a
kinematic bound. These are the same six noise seeds at each resolution, not
twelve independent coverage trials. The 4/6 counts refer to the nominal
one-parameter likelihood reference described below, not validated 95% intervals.

The large range is a scientific warning even with successful optimization.
For example, seed 485 prefers 369.35 km/s and a 1.13-arcsec turnover radius at
17x17 resolution. Its speed near 0.1 arcsec remains close to the truth, while
the speed at 0.5 arcsec is much too large. Constraining the reference speed to
210 km/s raises twice the negative log likelihood by 14.04, despite re-fitting
the map, angle and turnover radius from two starts. This supplies a concrete
case where the nominal likelihood reference excludes the injected value.

The illustrative rotation-only mass ratios span:

| Test radius, arcsec | Range over six 17x17 fits |
|---|---:|
| 0.1 | 0.709--1.252 |
| 0.2 | 0.743--1.343 |
| 0.3 | 0.647--1.823 |
| 0.5 | 0.562--3.093 |

These are realized point-estimate errors, not confidence bounds. The radii are
test coordinates and do not establish the resolved range of any real galaxy.
Changing the reference-radius parameterization alone would not add information
or resolve this outer-profile uncertainty.

The rotating model does nevertheless improve prediction of held-out exposures
over the re-fitted nonrotating model in all six reference realizations. At
17x17 resolution the held-out chi-square reduction is 64.03--152.19, and the
training likelihood statistic is 207.42--240.41. Thus the presence of a
kinematic signal and an accurate outer velocity/mass profile are separate
validation questions.

![Conditional rotating and nonrotating recovery profiles](../../results/2026-10-14_ensemble/ID3700_matched_null/profiles.png)

## Completed nonrotating controls

Six zero-rotation injections give:

| Map controls | Fitted speed range, km/s | Training likelihood statistic | Held-out chi-square improvement | Turnover-bound hits |
|---|---:|---:|---:|---:|
| 13x13 | 16.60--105.00 | 1.22--12.17 | -14.79 to +0.11 | 5/6 |
| 17x17 | 26.08--94.64 | 4.09--14.69 | -17.49 to -1.98 | 6/6 |

The held-out improvement is null chi-square minus rotating chi-square, so a
negative value favors the nonrotating prediction. At 17x17 resolution, all six
true rotating sources improve held-out prediction, while all six nonrotating
sources worsen it when fitted with rotation. The training-statistic ranges are
also clearly separated in this strong-line conditional experiment.

A positive fitted speed alone would misclassify these nonrotators. Boundary
hits and independent-exposure predictions are useful diagnostics, but six null
samples cannot calibrate a rare-false-alarm threshold. These tests also cannot
distinguish a rotating disk from astrophysical alternatives such as noncircular
velocity patterns; only the explicitly simulated zero-rotation null was tested.

## Completed morphology and calibration stress tests

Each scenario uses the same two noise seeds, 481 and 482, as the corresponding
reference runs. These are paired sensitivity checks, not eight independent
noise draws. All eight selected fits satisfy the gradient criterion.

| Scenario | Fitted speeds for seeds 481 / 482, km/s | Mass ratios at 0.5 arcsec, same order | Nominal truth compatibility |
|---|---:|---:|---:|
| Smooth source | 210.37 / 165.06 | 1.004 / 0.618 | 2/2 |
| Two bright clumps | 221.19 / 136.26 | 1.109 / 0.421 | 1/2 |
| Wrong assumed inclination | 234.37 / 166.11 | 0.743 / 0.373 | 2/2 |
| Underestimated blur | 205.50 / 153.84 | 0.958 / 0.537 | 1/2 |

The two-clump seed-482 fit has turnover radius at its upper bound of 2 arcsec;
its inner profile is especially inaccurate. A successful optimizer cannot
establish that spatially sparse emission constrains the whole rotation curve.

In the inclination experiment, projected-speed compatibility passes in both
cases while rotation-only masses are 26--63% too low. Fixing inclination at an
incorrect value can therefore defeat a seemingly reassuring projected-speed
check. Inclination uncertainty must be propagated into mass inference.

The blur change shifts these two recovered speeds by only about -5.2 and
-3.5 km/s relative to their matched-response counterparts. The much larger
absolute errors in the second seed should not all be attributed to blur.
This small paired experiment does not establish robustness to real spatially
varying PSF/LSF errors, contamination or astrometric errors.

![Paired morphology and calibration sensitivity profiles](../../results/2026-10-14_ensemble/ID3700_stress/profiles.png)

## New capability

`NativeMapFitter.fit(..., fixed={...})` can fix physical `v_ref`, `r_turn`,
and/or `theta`. Fixed variables are removed from the optimization vector and
gradient stopping test. The map and continuum are re-fitted at each constraint.
The fitted full state, fixed values and optimizer diagnostics are saved.
The relative-objective stopping tolerance was tightened from 1e-12 to 1e-14
after exact-recovery tests exposed stopping before the gradient tolerance.

The validation CLI now supports independent seeds, nonrotating injections,
alternative source morphologies and different injected/assumed inclination or
blur. `diagnostics/native_recovery_suite.py` assembles the runs and keeps failed
and boundary solutions visible in summaries. Seeds are independent within each
scenario; the same seed is paired across scenarios and map resolutions.

For each map setting, three free fits start at 40, 120 and 300 km/s with
different turnover radii, angles and initial maps. The preceding six-start
experiments already included 600--1300 km/s starts and identified bad local
minima there. Constrained fits use two starts, one from the best free fit and
one from a broad initial map. All these fits learn morphology from the data.

Two additional comparisons are saved when `--profile-reference` is enabled:

- **Nonrotating null:** fix speed to zero. Turnover radius and angle then have
  no effect and are also fixed. Re-fit the map and continuum. Compare its
  likelihood against the free rotating fit.
- **Injected speed:** fix only the speed to its injected value and re-fit
  turnover radius, angle, map and continuum. A shifted best-fit velocity can
  still be consistent with the injected speed if this likelihood penalty is
  small.

A constrained fit cannot beat the true unconstrained optimum. If it beats the
selected free fit, the free optimization is repeated from that constrained
state; such restarts are recorded as `release_null` or `release_truth_speed`.
Likelihood differences are used only within the same data, mask, continuum
model and regularization strength. The experiments below use zero regularization.

This guard fired for matched seed 486 at 17x17 resolution. The three original
starts all stopped near 191 km/s; releasing the injected-speed constraint found
204.44 km/s with an objective lower by only 0.0123. This is a nearly degenerate
optimization result, not statistically strong evidence distinguishing those
speeds. In real data there is no known injected value: a broader constrained
velocity scan or posterior exploration must supply that search coverage.

## Interpretation of the likelihood checks

The statistic is twice the difference in minimized negative log likelihoods.
For the nonzero injected speed, the report records whether the difference is
below 3.84146, a **nominal** one-parameter likelihood-ratio reference. This is a
test of compatibility at one known injected value, not a full interval or
posterior. Nonnegative map boundaries, weak identifiability and model errors
can invalidate nominal coverage. Failed optimization is counted separately.
Six seeds cannot establish 95% coverage even if every check includes the truth.

The zero-rotation statistic is **not** assigned a chi-square detection p-value.
Under zero rotation, angle and turnover radius are unidentifiable; the ordinary
regular likelihood-ratio assumptions cannot simply be assumed. This is the
kind of nonregular testing issue discussed by
[Protassov et al. (2002)](https://arxiv.org/abs/astro-ph/0201547). The small null
ensemble is a diagnostic comparison, not a calibrated rare-false-alarm rate.

## Experimental scope

Both generator and fitter use the explicit smooth cubic deposition response.
The generator samples an analytic map on a 97x97 grid; fits use 49x49 source
quadrature and 13x13 or 17x17 map controls. Continuum is quadratic along each
dispersion strip. Twelve exposures are fitted and six held out. The injected
line has combined signal-to-noise 100 after continuum projection, and Gaussian
noise follows the supplied SCI ERR maps. This is a strong-line conditional test.

The rotating reference injects V(0.5 arcsec) sin(i) = 210 km/s, turnover radius
0.22 arcsec, angle 0.9 radians and inclination 0.9 radians. The null injects
zero rotation with the same intrinsic width of 45 km/s. Center, registration,
width, relative response and the assumed inclination remain fixed in each fit.

The stress-test scenarios change one input at a time:

| Scenario | Injected change | Fitter assumption |
|---|---|---|
| `smooth` | Smooth elliptical brightness; no extra clump | Free nonnegative map |
| `two_clumps` | Two bright clumps plus weak extended emission | Free nonnegative map |
| `inclination` | Inclination 0.65 radians | Inclination fixed at 0.9 radians |
| `blur` | Gaussian-kernel sigma 1.15 detector pixels | Sigma fixed at 1 pixel |

The blur values describe the specified kernel before cubic deposition's added
variance, not a calibrated optical PSF or a complete empirical LSF. The
inclination stress test retains the same injected projected reference speed;
the true deprojected speed is therefore different.

Output profiles are evaluated at 0.1, 0.2, 0.3 and 0.5 arcsec. These are test
radii, not a demonstrated radial range of real-galaxy constraints. The saved
mass ratios are squared ratios of deprojected rotation speeds at fixed radius.
They illustrate error propagation under a rotation-only circular-speed
interpretation; no absolute galaxy masses or pressure corrections are inferred.

## Reproduction and artifacts

Run from `grism-kinematics` in the `grism` environment:

```bash
python -m diagnostics.native_flexible_check \
  --bundle /home/aakavan/EDR_data/magnif_copy/sapphires_kinematics_inputs_2026-10-13 \
  --output /home/aakavan/kinematics/results/2026-10-14_ensemble/ID3700_matched_null \
  --device cuda --precision float64 --deposition cubic \
  --grids 13 17 --strengths 0 --noise-scale 1 --starts 3 --maxiter 1500 \
  --profile-reference --seeds 481 482 483 484 485 486 --scenarios matched null

python -m diagnostics.native_flexible_check \
  --bundle /home/aakavan/EDR_data/magnif_copy/sapphires_kinematics_inputs_2026-10-13 \
  --output /home/aakavan/kinematics/results/2026-10-14_ensemble/ID3700_stress \
  --device cuda --precision float64 --deposition cubic \
  --grids 17 --strengths 0 --noise-scale 1 --starts 3 --maxiter 1500 \
  --profile-reference --seeds 481 482 --scenarios smooth two_clumps inclination blur
```

Use a new output directory for a rerun. Each scenario/seed directory contains
the injection, all fitted maps and continua, optimizer diagnostics, source-code
hashes and complete CLI arguments. `suite.json` summarizes completed runs and
`suite.png` compares recovered speeds, illustrative mass ratios and null
likelihood differences. Summary files are updated as runs finish.
The two suites ran concurrently on the RTX 4090; per-fit runtimes from these
runs should not be interpreted as isolated GPU performance benchmarks.

`python -m diagnostics.plot_native_suite_profiles <suite-directory>` renders
the profile figures from the saved fits. Thin curves are individual noise
realizations and thick curves are pointwise medians; neither is a posterior
uncertainty band. The source-data FITS files remain unchanged.

The full repository suite passes **190 tests** with the three existing warnings,
using `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. New tests verify constrained recovery,
zero-rotation invariance to otherwise unidentifiable angle/turnover radius,
physical deprojection and exclusion of invalid fits from compatibility counts.

## Still required for science use

An empirical response calibration and justified registration/center/width/
inclination priors remain necessary. The free map must be included in uncertainty
propagation. Larger independent-noise ensembles, different sources and weaker
lines are needed to test coverage and false detections. A profile check at the
injected reference speed cannot establish coverage of an entire mass profile.
The generator and fitter also share an arctangent rotation-curve family. Testing
different injected curve shapes is necessary before interpreting the fitted
radial shape as a measurement rather than a consequence of that parameterization.

The next priorities are to propagate line-map and inclination uncertainty in a
joint inference, calibrate the empirical spatial/spectral response, and establish
the radial range supported by the data. Independent-exposure prediction can
check a rotation claim; good image prediction by itself does not validate a
mass profile, as the earlier regularization tests showed.

All **20 synthetic datasets** finished: 97 free fits and 104 constrained fits,
including one restart when a constrained solution beat the original free fits.
All 32 selected fits meet the gradient criterion. Three individual fit attempts
do not, and 37 attempts hit kinematic boundaries; those states remain in the
artifacts. There are six independent noise seeds, reused across the paired
scenarios. Every fit artifact is present and recorded source-code hashes match
the final fitting code. `completion_audit.json` records these checks.
