# Real ID3700: dispersion and instrumental-response comparison

The real R+C pixels prefer the zero-intrinsic-dispersion boundary among the tested values in all five response scenarios. This supports an **upper-limit interpretation**, not a detection of zero gas turbulence. Spectral-response assumptions affect the constraint appreciably more than the tested cross-dispersion spatial widths. No new MCMC or pressure-supported mass inference was run.

Open sections **10–11** of [real_source_kinematics.ipynb](../notebooks/archive/real_source_kinematics.ipynb). Section 10 loads the comparison with `RUN_DISPERSION=False`; section 11 shows an individual fixed-dispersion fit with R above C, observed/predicted/residual columns. The new sections work after a fresh kernel start. Source ID and run name are editable. Existing MCMC samples and scientific code remain unchanged.

## What was fitted

All 18 real native exposures (nine R and nine C) constrain one positive emission map, one inclined arctangent rotation field and one constant intrinsic Gaussian line width. The source is ID3700, using the same Paα identification, redshift 1.1231, F150W inclination prior, masks, ERR maps, finite emission grid and calibration as the completed velocity grid.

For each fixed intrinsic dispersion **0, 10, 20, 40 and 80 km/s**, the optimizer varies deprojected `Vrot(0.5 arcsec)` continuously, turnover, inclination, angle, systemic velocity, center, all emission-map/envelope coordinates and exposure registration. It profiles continuum and nonnegative exposure amplitudes. Center and other regularization match the saved catalog-center velocity grid.

Five response scenarios were tested separately:

- Nominal calibration.
- Spectral LSF widths scaled by 0.9 or 1.1, with spatial response fixed.
- Cross-dispersion spatial PSF width scaled by 0.9 or 1.1, with spectral response fixed.

Both spectral Gaussian components are scaled, retaining their mixture fraction. Widths are changed before removing cubic-deposition variance. The nominal kernel is copied exactly from the saved model. The empirical LSF already includes the optical response along dispersion, so no second optical PSF is added there. These ±10% perturbations are sensitivity scenarios, **not empirically established calibration uncertainty bounds**.

Two sweeps start from the stationary grid solutions at 278 and 483 km/s, respectively. The first follows increasing dispersion; the second starts at 80 km/s and follows decreasing dispersion. They retain their own continuation paths. This explores two branches rather than proving a global minimum. Both paths and every fitted artifact are saved, including higher-score alternatives.

The 50 fits completed on the RTX 4090 in **90.0 seconds of worker time**, including model setup/compilation within the worker; Python process startup is additional. All 50 passed the projected-gradient check; the largest projected gradient was 0.00987 against the acceptance threshold 0.1. Each fit is checkpointed, with progress during compilation and every 100 optimizer iterations. The completed run can be reopened without refitting. Interruptions resume from completed fits when settings, code and inputs agree.

## Results

The displayed score is `2[J(sigma) - min_sigma J]`, where `J` is the penalized, amplitude-profiled objective. Each response uses its own minimum among the five tested dispersions. Connecting lines guide the eye; no intermediate dispersions were fitted. No likelihood-ratio confidence thresholds are assigned. MCMC integrates the amplitudes instead, so this is not the same probability target as a marginalized posterior interval.

| Fixed σ (km/s) | Nominal refitted V(0.5″) (km/s) | Nominal score above best | Pixel Δχ² at that penalized optimum |
|---:|---:|---:|---:|
| 0 | 278.16 | 0.00 | 0.00 |
| 10 | 278.92 | 1.73 | 1.73 |
| 20 | 281.45 | 7.35 | 7.36 |
| 40 | 294.86 | 35.02 | 34.83 |
| 80 | 330.04 | 193.29 | 192.62 |

The score rise comes predominantly from worsening agreement with the pixels, not from the explicit width/map penalty. The aggregate penalty changes from 43.32 at zero dispersion to 43.66 at 80 km/s. Rotation does compensate for increasing dispersion, but cannot maintain the same quality of fit.

| Response scenario | V at σ=0 (km/s) | Score at σ=10 | Score at σ=20 | Score at σ=40 | Score at σ=80 |
|---|---:|---:|---:|---:|---:|
| Nominal | 278.16 | 1.73 | 7.35 | 35.02 | 193.29 |
| Spectral width −10% | 272.22 | 0.64 | 3.22 | 21.60 | 159.96 |
| Spectral width +10% | 292.02 | 2.48 | 10.26 | 45.54 | 220.14 |
| Spatial width −10% | 279.88 | 1.69 | 7.24 | 35.04 | 197.43 |
| Spatial width +10% | 275.83 | 1.70 | 7.20 | 34.04 | 186.30 |

![Joint dispersion profiles and refitted rotation](../../results/real_sources/ID3700/dispersion_v1/figures/dispersion_profile.png)

Narrowing the LSF makes 10–20 km/s more competitive. At 20 km/s the score changes from 7.35 to 3.22 or 10.26 under the spectral perturbations. The tested spatial perturbations have much smaller effects on this score. The 272–292 km/s spread at zero dispersion is a sensitivity result, not a systematic-error interval.

Both grism directions contribute to the worsening data fit, with C contributing more in this joint solution:

| Fixed σ (km/s) | Δχ² from R pixels | Δχ² from C pixels |
|---:|---:|---:|
| 10 | 0.24 | 1.49 |
| 20 | 1.10 | 6.26 |
| 40 | 5.63 | 29.20 |
| 80 | 35.77 | 156.85 |

These are contributions from the **same joint fits**, relative to the nominal zero-dispersion fit. They are not separate R-only and C-only analyses and do not quantify the gain over fitting a single grism direction.

The higher-velocity initialization stays in worse local minima: at σ=0 the difference in penalized score is 97.79 for the nominal response and 25.02 for the narrower LSF. Thus the optimizer remains dependent on initialization, but retaining both branches makes that dependence visible. These optimization branches do not provide posterior mode probabilities. Previous MCMC exploration is a separate check.

## Visual inspection

![Native R/C residuals across the fixed dispersions](../../results/real_sources/ID3700/dispersion_v1/figures/baseline_residuals.png)

![Familiar R/C observed, predicted and residual layout at 20 km/s](../../results/real_sources/ID3700/dispersion_v1/figures/baseline_sigma20.png)

The changes between 0, 10 and 20 km/s are subtle in individual native-exposure residuals. The likelihood accumulates their effects over all exposures and pixels. A persistent positive feature near cutout (x,y) ≈ (29,28) in the displayed C exposure remains across the dispersion choices; changing σ does not explain it. Its origin has not been established. Several exposures and source/contaminant assignments should be examined before interpreting it physically. Gray pixels are the unchanged data mask.

The notebook's brightness scale depends only on the observed exposure, so it remains fixed as the fit choice changes. Residuals use ±5 ERR throughout. The selected R/C exposure numbers can be changed without refitting. The initial pair is illustrative, not the whole dataset.

## Scientific interpretation and next work

1. The present model does not require a resolved positive intrinsic dispersion. It disfavors 40–80 km/s among the tested scenarios. This is conditional evidence, not a model-independent exclusion or a physical zero-dispersion measurement.
2. A posterior median near 6 km/s is compatible with an optimization minimum at zero: a continuous nonnegative distribution can accumulate near its boundary while having a positive median. This test does not replace the posterior with a point estimate.
3. The earlier nominal-response σ upper limit and V/σ lower limit remain conditional. These refits do not validate the numerical limits after including calibration uncertainty. The spectral response is the first calibration uncertainty to investigate more carefully; the flexible emission map, residual structure, continuum/noise treatment, inclination transfer and source assignment remain concerns.
4. Next, inspect the real-exposure calibration/residual patterns and construct a justified response uncertainty model, then propagate it in a new posterior. Comparing alternative emission-map regularization is also needed. Picking the response with the lowest score would not itself establish its calibration accuracy.
5. Pressure-supported profiles need an explicit gas-density assumption and mass geometry. The GEKO correction uses `V_circ² = V_rot² + 2(r/r_s) sigma0²` under its adopted disk assumptions. **The gas-density scale length r_s is not our turnover radius r_t.** Neither the F150W size nor the flexible line-emission envelope automatically supplies a gas-density profile. This diagnostic therefore leaves the existing rotation-only mass proxy labeled as such.

## Artifacts and checks

- [Run report, settings, starts, hashes and all 50 fit records](../../results/real_sources/ID3700/dispersion_v1/dispersion.json).
- [Numerical comparison CSV](../../results/real_sources/ID3700/dispersion_v1/comparison.csv).
- [Streamed progress log](../../results/real_sources/ID3700/dispersion_v1/run.log).
- [Executed standalone notebook sections](../../results/2026-10-16_notebook_validation/ID3700_dispersion_viewer.ipynb).
- [Worker](../diagnostics/native_real_dispersion.py) and [CPU notebook viewer](../dingo/real_dispersion.py).

Thirteen targeted tests passed, including fixed-coordinate optimization, R/C response-axis changes, source identity, artifact integrity, retaining failed lower-score fits visibly, and the existing notebook workflow checks. The new notebook sections executed in a fresh grism kernel with zero errors and three static PNG outputs, without launching fitting or MCMC. Original MCMC scientific-code hashes still match. An initial setup-only check was overly strict about kernel renormalization roundoff (maximum difference 2.8e-17); it was corrected before any fits ran, and that failed setup was preserved separately. It did not change the science model.
