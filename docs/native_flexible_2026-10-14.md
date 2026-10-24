# Shared line maps and continuum: 2026-10-14

The native-exposure prototype now fits a flexible, nonnegative emission-line
map shared by R and C. It also fits a separate continuum in each exposure.
This advances the [October 13 prototype](native_forward.md); it is still a
conditional validation framework, not a released real-galaxy mass measurement.

The [repeated-noise continuation](native_ensemble_2026-10-14.md) adds nonrotating
controls and constrained likelihood checks to interpret the noisy discrepancies
below. A best estimate displaced from the truth need not exclude that truth.
The [paired R/C comparison](grism_comparison_2026-10-14.md) then tests whether
the joint method adds information relative to either direction alone, including
controls matched in signal-to-noise. The earlier joint-only tests did not answer
that comparative question.

## Implementation

`dingo/native_fitting.py` adds:

- `BilinearLineMap`: a nonnegative control grid interpolated onto finer source
  quadrature. Integrated node flux includes area weights. A fixed flux unit
  sets the parameter scale; the fitted map's total flux remains free.
- `StripContinuumLikelihood`: weighted least-squares continuum coefficients
  eliminated at each trial. Each R detector row, or C column, has a polynomial
  along dispersion. Weighted SVD handles masked and truncated strips. The
  continuum projection is fixed by ERR and masks, not trial velocity.
- `NativeMapFitter`: joint optimization of projected velocity at a reference
  radius, turnover radius, angle and all map coefficients. Positivity uses
  explicit bounds. All derivatives are enabled for every fit; the best full
  state is retained, including the line map and profiled continuum.

The objective is a native-pixel Gaussian likelihood plus an explicitly chosen
brightness-gradient penalty. The map regularization strength is a required
argument, not a hidden default. The penalty approximates the integral of
squared brightness gradient and passes a resolution-invariance test on linear
brightness fields. Optimizer success, projected gradient, boundary hits and
held-out residuals are reported separately.

Continuum elimination carries its uncertainty into the line fit through the
weighted residual projection. It does not treat a subtracted continuum as
known. A separate continuum-prediction variance is provided for pixels held
out of the continuum fit; rank-deficient strips have infinite prediction
variance. That variance must not simply be added to the training-pixel errors,
where continuum estimates and measurement noise are correlated.

The line-map fit is penalized optimization. It is **not** marginalization over
the unknown line map, and its profile score must not be substituted for a
marginal posterior in MCMC. Scores with different regularization strengths or
continuum models cannot be ranked as if they were one common posterior.

## Two numerical issues addressed

**Float32 sky offsets in Astropy.** The new flexible-map diagnostic passed
float32 coordinates into `spherical_offsets_by`. At these tiny angles, single
precision trigonometry shifted source positions by up to about 0.79 detector
pixel in the measured test. `NativeExposure.source_detector_xy` now explicitly
promotes angles to float64 before WCS work. Cached coefficients can subsequently
be placed on a GPU. A real-data regression checks float32/float64 input parity.

The initial float32 pilot had a 3.76% discrepancy in its injected line image
relative to float64. After the repair this is 2.2e-7. The pilot directory is
explicitly marked INVALIDATED. All main resolution/regularization experiments
used float64 and were unaffected. This was a bug in the new diagnostic path;
the student's original fixed-amplitude problem has the separate causes in
the [audit](audit_2026-10-13.md).

**Pixel-crossing derivative discontinuities.** Bilinear deposition is continuous
but has derivative jumps as source samples cross detector pixels. Many noisy
fits stopped with tiny objective changes but did not reach the requested
gradient tolerance. An optional positive cubic B-spline deposition is now
available; it preserves flux and centroid and has continuous first/second
derivatives at crossings. Tests check gradients at the crossings themselves.

Cubic deposition adds exactly 1/3 pixel squared of variance per detector axis.
It is part of the instrumental response and cannot be silently substituted
into a calibrated PSF. The smooth-response validation generates and fits data
with that explicitly specified response. It does not establish that this is
the correct response for the real observations. The generic projector retains
bilinear as its default; the new diagnostic defaults to cubic and float64.

## Recovery with unknown morphology

The generator uses an analytic asymmetric/clumpy brightness distribution on a
97x97 source quadrature. Fits use 49x49 quadrature with 9x9, 13x13 or 17x17
brightness controls. Initial maps include uniform, broad and offset profiles;
none is the injected map. Six velocity starts span 40--1300 km/s and also vary
turnover radius and angle. The last dither of each observation is held out:
12 exposures fit the line map/velocity, and six test its line predictions.
Continuum coefficients are independently refitted in the held-out exposures.

The injected projected speed at 0.5 arcsec is 210 km/s, with turnover radius
0.22 arcsec and angle 0.9 radians. Inclination, center, registration, intrinsic
line width, relative responses and blur are known conditional inputs. These
are simulations using actual exposure geometry/masks/ERR, not galaxy fits.
The instrumental operator and kinematic family are shared by generator/fitter.

The initial bilinear-deposition, noiseless experiments give:

| Geometry | Control map | Smoothness strength | Best V(0.5 arcsec) sin(i), km/s |
|---|---:|---:|---:|
| ID3700 | 9x9 | 0 | 185.01 |
| ID3700 | 13x13 | 0 | 209.87 |
| ID3700 | 17x17 | 0 | 210.01 |
| ID3700 | 17x17 | 0.03 | 199.80 |
| ID3700 | 17x17 | 0.3 | 156.23 |
| ID15665 | 13x13 | 0 | 209.99 |
| ID15665 | 17x17 | 0 | 210.01 |
| ID15665 | 17x17 | 0.03 | 211.83 |
| ID15665 | 17x17 | 0.3 | 225.18 |

Different starts can agree on the **wrong** answer when the source map is too
coarse. Regularization can also shift velocity substantially, and the direction
is not universal across the two geometries. At fixed radius/inclination, the
156 versus 210 km/s case would lower a rotation-only V-squared mass estimator
by about 45%. That is a consequence in this mock, not a revised galaxy mass.

The held-out noiseless squared residual drops from 4.56 for the 9x9 unpenalized
ID3700 map to 0.0217 for 13x13 and 0.0012 for 17x17. This supports refining the
source representation. Bad local minima still exist: some successful optimizer
terminations have large likelihood gaps and turnover radius on its upper bound.
The full states and failed starts are preserved, not discarded from the record.

The noisy bilinear experiment is retained as an optimizer diagnostic. Its best
fits did not satisfy the gradient criterion. Use the separately labeled cubic
runs when assessing smooth-response optimization. Noisy recovery and held-out
prediction do not by themselves establish uncertainty coverage; no noise
ensemble or posterior coverage claim is made here.

### Completed smooth-response checks

With cubic deposition, the unpenalized 17x17 noiseless fit recovers 210.0003
km/s, turnover radius 0.22002 arcsec and angle 0.89999 radians. Five of six
starts reach the gradient tolerance. The most extreme start remains at a bad
boundary solution after the iteration limit; its objective clearly separates
it from the recovered solution. The best held-out squared residual is 0.0000944.

For one Gaussian noise realization (seed 481), the best fits are:

| Control map | Smoothness strength | Best projected speed, km/s | Held-out chi-square |
|---|---:|---:|---:|
| 13x13 | 0 | 165.59 | 37454.49 |
| 13x13 | 0.03 | 170.59 | 37448.00 |
| 13x13 | 0.3 | 143.49 | 37435.49 |
| 17x17 | 0 | 210.72 | 37467.62 |
| 17x17 | 0.03 | 198.88 | 37455.33 |
| 17x17 | 0.3 | 142.28 | 37438.20 |

All six selected fits satisfy the gradient criterion; 34 of all 36 starts do.
Seven starts hit parameter boundaries, including poor stationary local minima.
Thus the smoother operator resolves the gradient-stopping problem for the
selected fits, but does not remove local minima or map/velocity degeneracy.
The noisy estimates also differ between grids. A single realization cannot
establish their statistical bias or uncertainty coverage.

The held-out score favors a strongly smoothed map with the wrong velocity in
this realization. Predictive image quality alone is therefore not a sufficient
criterion for choosing a model for accurate rotation inference. Independent
noise realizations, morphology/response variations and posterior coverage
checks are needed before fixing a regularization prescription.

![Noisy synthetic recovery; inclination and instrumental response fixed](../../results/2026-10-14_flexible/ID3700_cubic_noisy/recovery.png)

## What the actual SCI sidebands show

`diagnostics/native_continuum_audit.py` excludes the central +/-20 dispersion
pixels and holds out every fourth sideband pixel. It fits continuum to the
remaining SCI pixels and evaluates pointwise residuals using ERR squared plus
the continuum prediction variance. For quadratic continua:

| Source / grism | Held-out pixels | Robust residual sigma | Residual RMS |
|---|---:|---:|---:|
| 3700 R | 7854 | 1.013 | 1.032 |
| 3700 C | 7826 | 1.122 | 1.155 |
| 15665 R | 7871 | 1.080 | 1.101 |
| 15665 C | 5360 | 1.103 | 1.131 |

There are also a few >5-sigma residuals. Increasing polynomial degree from 1
to 3 does not remove the excess scatter. These checks do not isolate its cause:
noise calibration, imperfect continuum, contamination and other residual
structure can contribute. The shared continuum fit correlates the held-out
residuals; this table is not an independent-pixel likelihood or detection test.
Do not turn these four numbers into universal error-inflation factors.

## Instrumental-response inputs still need reconciliation

The supplied `calibration/psf/fsun_grisms_4750/4750_f444w_ePSF.fits` has a
0.015-arcsec pixel scale and an FWHM header value of 0.1064. Direct horizontal
and vertical half-maximum cuts through its peak instead give about 0.166 and
0.164 arcsec. The header's measurement definition has not yet been established;
the discrepancy means that its value should not simply become the model's
Gaussian width. The array sums to 0.91324 and includes a small negative pixel
sum of -0.00296. Normalization, sampling, negative pixels and the applicability
of this imaging ePSF to each grism response need explicit treatment. The
current mock tests use a specified synthetic response, not this empirical PSF.

## Reproduce and inspect

Results are under `/home/aakavan/kinematics/results/2026-10-14_flexible/`:

- `ID3700_noiseless/`, `ID15665_noiseless/`: original float64 bilinear grid and
  regularization scans; JSON results, every fitted map/continuum, recovery plots.
- `ID3700_noisy_seed481/`: noisy bilinear failure diagnostics.
- `ID3700_cubic_noisy/`, `ID3700_cubic_noiseless/`: smooth-response validations.
- `ID3700_float32_geometry_fixed/`: repaired float32 geometry comparison.
- `real_continuum/`: SCI sideband predictions, variances, masks and figures.
- `ID3700_pilot/`: INVALIDATED pre-repair float32 experiment.

Each recovery directory includes `injection.npz`, `recovery.json`, individual
fit NPZ files and plots. JSON records the code hashes, split exposure paths,
fixed inputs, response choice and complete optimizer diagnostics. Files made
before the deposition option have no `deposition` field; they used bilinear.

```bash
python -m diagnostics.native_flexible_check \
  --bundle /home/aakavan/EDR_data/magnif_copy/sapphires_kinematics_inputs_2026-10-13 \
  --output /home/aakavan/kinematics/results/2026-10-14_flexible/new_run \
  --device cuda --precision float64 --deposition cubic \
  --grids 13 17 --strengths 0 0.03 0.3 --noise-scale 1 --maxiter 1500

python -m diagnostics.native_continuum_audit \
  --bundle /home/aakavan/EDR_data/magnif_copy/sapphires_kinematics_inputs_2026-10-13 \
  --output /home/aakavan/kinematics/results/2026-10-14_flexible/real_continuum
```

Validation: the full repository suite passes **188 tests** (three existing
warnings), using the `grism` environment with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
Tests include an independent weighted-least-squares continuum reference,
rank-deficient/masked strips, continuum prediction variance, line-map flux and
roughness scaling, full-state fitting, WCS input-precision parity, and cubic
flux/centroid/width/gradient checks. All recovery outputs above were saved;
the cubic GPU fit loop took about 389 seconds for 36 noisy starts. This is a
conditional optimization runtime, not an MCMC or end-to-end survey benchmark.

## Remaining before science posteriors

Calibrate the full spatial/spectral response, including deposition/pixel
integration; model or mask contamination; validate the native noise model;
release center, registration, inclination and intrinsic line width with
justified priors; quantify sensitivity to source support and regularization;
and run independent-noise recovery/coverage experiments. Joint posterior
sampling must include line-map uncertainty. Rotation and mass profiles should
then be reported only over justified observed radii, with the inclination and
pressure/geometry assumptions explicit.
