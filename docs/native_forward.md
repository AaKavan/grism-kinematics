# Native-exposure prototype: 2026-10-13

The [October 14 continuation](native_flexible_2026-10-14.md) adds free line maps,
joint continuum fitting, held-out exposure checks and numerical repairs. This
page records the earlier known-morphology validation and its limits.

The new path reads individual R/C exposures and projects one shared intrinsic
line map onto their detector pixels. It is an experimental foundation, not a
replacement production fitter or a validated measurement of any galaxy's mass.
The existing inverse-rectification fitter is still available unchanged by this
addition. Read the [audit](audit_2026-10-13.md) before interpreting its old results.

## What is implemented

- `dingo/exposures.py`: exact exposure association through each target's R/C
  STATS tables; native SCI, ERR, DQ and fixed masks; provenance and separate
  display-only EMLINE. FITS sections avoid loading entire detector planes.
  Off-detector pixels remain masked. Empty cutouts are explicitly recorded as
  exclusions. No resampling or coaddition is performed on the observations.
- `dingo/native_forward.py`: full position-dependent trace and wavelength
  polynomials, evaluated at each source node in each exposure. Sky offsets use
  the corrected SCI WCS and are anchored at recorded STATS positions. Polynomial
  coefficients are cached in float64 and centered on each native cutout.
- A differentiable line projector batches parameter vectors and exposures in
  PyTorch. It accepts arbitrary shared node fluxes, explicit blur kernels and
  relative responses, intrinsic Gaussian line width, and registration shifts.
  Flux outside the cutout is lost, rather than renormalized back into it.
- A fixed-pixel Gaussian likelihood with proper normalization. Its relative
  version subtracts the zero-model likelihood algebraically, avoiding large
  constant cancellation. Both are the same target up to a fixed constant.
- A velocity parameter defined at a chosen radius, `V(r_ref) sin(i)`, rather
  than requiring a well-measured asymptotic amplitude. Coordinates/radii are in
  arcseconds; angle is counterclockwise from east toward north, not sky PA.
- `diagnostics/native_exposure_check.py`: cutout inventory/figures plus six-start
  conditional synthetic recovery. It scales parameters, profiles a common line
  flux and independent constant backgrounds, retains their fitted values,
  selects the lowest objective, and reports boundary hits and likelihood gaps.

The DQ default rejects DO_NOT_USE and SATURATED (bits 1 and 2). JUMP_DET alone
does not discard a fitted rate. This is a stated starting policy, not a claim
that every remaining pixel is free of detector artifacts or contamination.

## Findings from the actual copied exposures

There are 116 usable source/exposure cutouts across the seven targets (one
exposure can contain several targets). These are not 116 unique detector files.

| Source | R cutouts | C cutouts | Excluded associations |
|---|---:|---:|---:|
| 3700 | 9 | 9 | 0 |
| 6170 | 9 | 9 | 0 |
| 8910 | 3 | 6 | 3 |
| 15665 | 9 | 9 | 0 |
| 16887 | 9 | 9 | 0 |
| 21619 | 9 | 8 | 0 |
| 21636 | 9 | 9 | 0 |

ID8910's three obs132 C associations place the configured Pa-alpha wavelength
at detector y = 2117--2130, outside the detector and the selected cutout. Their
presence in STATS means some spectrum was covered, not that this line was.

ID15665 and ID21619 have partially truncated cutouts in some exposures. These
remain usable with their actual masks; treating all exposures as equally
complete would be incorrect. The largest catalog-WCS versus STATS discrepancy
is about 0.25 native pixel. Registration uncertainty therefore remains a model
ingredient, not something fixed by copying the calibration files.

## What the recovery test establishes

The experiment injects an asymmetric, clumpy line map using actual exposure
geometry, ERR and masks. It fits projected speed at 0.5 arcsec, turnover radius
and angle, with line flux and constant backgrounds profiled. Starts span
40--1300 km/s, 0.08--1 arcsec and substantially different angles. The injected
values are 210 km/s, 0.22 arcsec and 0.9 radians.

For noiseless ID3700 geometry all six starts recover the input. The GPU float32
projector agrees with a CPU float64 image to about 3.4e-7 relative L2 error in
this test. Likelihood arithmetic remains float64: using float32 for both the
model and the likelihood caused premature optimizer stopping near the optimum.

For noiseless ID15665 geometry five starts recover the input. One reports
optimizer success at approximately 872 km/s and the upper turnover-radius
bound (2 arcsec). Its negative log likelihood is worse by about 315. The
multistart selection rejects that solution. This is direct evidence that a
success flag alone is inadequate even after replacing the old objective.

The injected and fitted models are the same, and the morphology, inclination,
center, registration, PSF and line width are known. This is a conditional
numerical check, not an independent physical simulation, uncertainty-coverage
test, or measurement of ID3700/ID15665. A noisy realization can recover a
different optimum; its bias/coverage require an ensemble, not comparison to a
single truth value. The operator also has separate flux, centroid, calibration
polynomial, finite-difference gradient, edge and masking tests.

In the saved noisy ID3700 experiment, the six starts finish at 233.2--234.5 km/s,
with negative log likelihoods separated by less than 0.00042. The injected
speed is 210 km/s. These fits have plainly moved away from their starting
values, but this one realization cannot establish unbiased recovery or correct
error bars. The small differences between starts also warrant checking source
grid/deposition convergence before treating a local optimizer as exact.

GPU gains are chiefly from batching. Timings in the JSON reports cover only the
line projection, including spectral quadrature and blur, not full posterior
sampling. CPU/GPU comparisons should use the same model dtype, batch size and
workload. The prototype does not implement MCMC or checkpoint resumption.

Final float32 projection timings for ID3700 (18 exposures, 625 source nodes,
7 spectral samples, 81x81 pixels; CPU uses four threads) were:

| Parameter vectors per batch | CPU | GPU |
|---|---:|---:|
| 1 | 3.69 ms | 1.54 ms |
| 16 | 41.8 ms | 2.24 ms |
| 64 | 246 ms | 6.09 ms |

That is about 40x for the 64-vector projection workload. It is not a 40x claim
for an entire fit or MCMC chain. Timing variation with host load is noticeable.

## Reproduce

From the repository, using the `grism` environment:

```bash
python -m diagnostics.native_exposure_check prepare \
  --bundle /home/aakavan/EDR_data/magnif_copy/sapphires_kinematics_inputs_2026-10-13 \
  --output /home/aakavan/kinematics/results/2026-10-13_native

python -m diagnostics.native_exposure_check validate \
  --bundle /home/aakavan/EDR_data/magnif_copy/sapphires_kinematics_inputs_2026-10-13 \
  --output /home/aakavan/kinematics/results/2026-10-13_native \
  --source 3700 --device cuda --precision float32 --noise-scale 0
```

Use `--source 15665 --device cpu --precision float64` for the other detector
module; omit `--noise-scale 0` to inject Gaussian noise. All results are written
outside the repository. Input science files and existing chains are untouched.
The NPZ inspection artifacts load with `allow_pickle=False` and contain SCI,
ERR, DQ, valid masks, EMLINE, detector positions and JSON metadata. They are
inspection products; use the loader to reconstruct a model with its WCS and
calibration tables.

The full suite passes 179 tests. One pytest startup encountered an unrelated
auto-loaded Zarr/importlib plugin failure; disabling automatic third-party
plugin loading allowed the complete suite to run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

## Required before a real rotation/mass posterior

1. Fit a shared flexible, nonnegative line morphology under an explicit
   regularization and vary its strength/resolution. The test harness currently
   fixes the morphology to its input map; the projector itself accepts any map.
2. Supply the SCI continuum and contamination model. Reusing SCI errors for
   EMLINE would restore the original uncertainty problem. Validate residual
   noise in processed SCI and compare independent exposure subsets.
3. Calibrate relative flux responses and the spatial/spectral blur. The supplied
   empirical PSF's pixel scale is 0.015 arcsec, not a native detector pixel;
   its `OVERSAMP=2` describes its imaging product. Do not simply apply that
   kernel at native scale or convolve both an optical PSF and an LSF that
   already includes it. Verify spatial and spectral quadrature convergence.
4. Allow registration, center, inclination and intrinsic line width to vary
   with justified priors; test PSF/LSF and astrometric alternatives.
5. Reuse one explicit posterior for optimization and MCMC. Profile nuisance
   parameters for initialization diagnostics, but do not treat a profiled
   nuisance likelihood as a marginalized posterior. Test multiple modes,
   independent chains and injection/recovery coverage with all nuisance
   parameters free.
6. Produce posterior rotation and mass profiles only over justified observed
   radii, including inclination and any pressure/geometry assumptions.
