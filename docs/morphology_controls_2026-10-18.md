# Real R+C morphology-prior comparison — 2026-10-18

## Scientific purpose

The primary method fits **one flexible positive gas-brightness map and one velocity field jointly to all R+C exposures**. The gas map does not have to follow the F150W stellar image, a Sérsic profile, or an ellipse aligned with the velocity field. Complementary grism directions are intended to help separate spatial structure from velocity-induced displacement and broadening.

The smooth linked exponential introduced here is a **restrictive diagnostic control only**, not the primary model. A nonaxisymmetric or clumpy brightness map can illuminate a circular velocity field without its brightness moments matching that field's inclination or position angle. Thus the axisymmetric relation `q=cos(i)` must not be imposed on the flexible map. This corrects the emphasis of the preceding smooth-template experiment.

R+C still does not make the inverse problem assumption-free. Finite spatial support and resolution, positivity, map regularization, the circular arctangent velocity law, constant intrinsic dispersion, instrumental response and nuisance treatment remain assumptions. This experiment tests selected **morphology regularization** choices, not all of these at once.

## Controlled experiment

Both ID21636 and ID3700 use the same native pixels, errors, masks, nominal calibrated PSF/LSF, positive continuum, shared flux and calibration nuisance terms as their saved refined runs. The source square remains ±0.9 arcsec, with 21×21 brightness controls, 121×121 integration nodes and 17 spectral quadrature nodes. ID21636 uses its validated F150W inclination density; ID3700 retains its existing imaging density. Each source's exact saved density is copied into the new run. The soft imaging inclination density is **not a stellar-light template for gas emission**.

The flexible map is

\[
\log I_j=g_j(w,q,\phi)+A[B_\ell z]_j,\qquad z_k\sim N(0,1),
\]

with a fitted elliptical Gaussian envelope `g` centered at the catalog origin. The gas map can depart from this envelope through its 440 nonconstant field coefficients. Its ellipse parameters are independent of the velocity geometry.

| Scenario | Field amplitude A | Correlation length ell | Meaning |
|---|---:|---:|---|
| Baseline | 2 | 0.10 arcsec | Existing map prior |
| amplitude1 | 1 | 0.10 arcsec | Greater penalty for the same brightness fluctuations |
| amplitude4 | 4 | 0.10 arcsec | Smaller penalty for the same brightness fluctuations |
| correlation0p05 | 2 | 0.05 arcsec | Different penalty across spatial frequencies |
| correlation0p20 | 2 | 0.20 arcsec | Different penalty across spatial frequencies |

Correlation changes are not uniformly weaker or stronger priors. Removing the fitted envelope is also not automatically a less restrictive choice: it would center the field prior on a uniform square and can heavily penalize an elongated galaxy. That is why the initial tests retain the envelope and its existing size penalty.

The sixth case is an exponential `n=1` disk whose light and rotation share center, position angle and inclination, with a thin-disk light ratio `q=cos(i)`. Only its light scale radius is an additional morphology parameter. It remains independent of the stellar light. Its result must not be adopted simply because its parameters appear more familiar.

### Initialization and objective

For every flexible case, physical starting brightness is preserved exactly. The fixed DCT modes give `B_ell = U diag(s_ell)`, so the transformed coefficients are

\[
z_{{\rm new},k}=\frac{A_{\rm old}s_{{\rm old},k}}{A_{\rm new}s_{{\rm new},k}}z_{{\rm old},k}.
\]

The worker checks the actual source quadrature flux and rendered R/C line images before fitting. The initial objective change must equal precisely the change in map penalty. No data, velocity or nuisance parameter changes in this transport.

At each fixed intrinsic width **0, 20, 40 or 80 km/s**, all remaining parameters are optimized. Sigma is removed from the optimizer coordinates, avoiding its zero-gradient boundary trap. Flexible cases start from the corresponding saved optimum and from the same brightness map with `V=600 km/s` and the kinematic axis shifted by 90 degrees. The linked control has four geometry starts. Every case additionally continues the best candidates in both directions along the sigma grid. This gives **92 fits per source**, with live logs and atomic checkpoints.

The objective is `J = chi-square/2 + prior penalties`. First-order stationarity requires a projected gradient below 0.1. Stationarity does not prove a global minimum. The curve `2[J(sigma)-min J]` resets to zero independently for each scenario. **Cross-prior objectives are not model evidence, and these profiles are not confidence intervals or MCMC samples.** The velocity quoted below is deprojected rotation at **0.5 arcsec**, not asymptotic rotation.

## Results

The first pass completed all **184 fits**, with all saved fits passing the local stationarity threshold. The following conditional preferred values are at each scenario's best tested sigma; they are not posterior measurements or error bars.

| Morphology choice | ID21636 Vrot(0.5 arcsec), km/s | ID3700 Vrot(0.5 arcsec), km/s |
|---|---:|---:|
| Flexible baseline | 187.5 | 283.3 |
| Smaller fluctuations, A=1 | 195.0 | **1000, upper bound** |
| Larger fluctuations, A=4 | 184.0 | 277.7 |
| Shorter correlation, 0.05 arcsec | 186.0 | 278.1 |
| Longer correlation, 0.20 arcsec | 191.5 | 294.7 |
| Linked exponential control | 195.1 | **1000, upper bound** |

Every flexible case prefers the tested sigma=0 point. ID21636's linked control prefers sigma=20, but its improvement over zero is only 2 delta J=1.30; that is not an established dispersion detection. ID3700's linked control prefers sigma=0 and reaches the velocity bound.

For ID21636, the tested flexible morphology changes move Vrot by about 184–195 km/s and projected rotation by about 175–181 km/s. The relative stability is encouraging **within these tested assumptions**. However, the weaker-penalty A=4 case puts the turnover radius at its 0.02-arcsec lower bound. A stable velocity at 0.5 arcsec does not imply a measured inner rotation or mass profile.

For ID3700, the stronger morphology penalty changes the preferred branch, not just a few km/s around the baseline. Its high-speed solution has inclination about 41 degrees and projected speed about 658 km/s; the baseline has inclination about 79 degrees and projected speed about 278 km/s. Thus this is not merely an inclination correction to the same measured projected rotation. The linked control goes to a different high-speed configuration at inclination about 27 degrees.

The linked control's pixel chi-square exceeds the flexible baseline by about 1020 for ID21636 and 1266 for ID3700. This raw difference alone is not a complexity-corrected model-selection statistic. Together with the imposed single-component brightness and the bound-hitting ID3700 solution, it provides no justification for replacing the flexible method with that control.

![ID21636 common gas maps under the different assumptions](../../results/real_sources/ID21636/morphology_controls_v1/figures/source_morphology_comparison.png)

### What this changes about the direction of the project

Keep the flexible common R+C brightness map as the primary approach. The experiments do not support imposing a single axisymmetric light profile as a general repair for the dispersion problem. The physical brightness fluctuations and spatial-correlation assumptions must be visible and tested; a strong penalty can favor extreme velocities even with both grisms present.

The fixed-sigma scans show that the low-width preference is not only the old optimizer's zero-derivative trap. They do **not** prove that these galaxies have no gas dispersion. The instrumental response, continuum treatment, finite map support and limited width sensitivity still need investigation. None of the current sigma=0 points supplies a reportable V/sigma or calibrated mass uncertainty.

Two focused real-data steps should precede another expensive production MCMC run:

1. Test a larger source support and inspect neighboring/contaminating emission for ID3700. Its baseline flexible map assigns about 7.5% of modeled flux to the outer 0.1 arcsec of the current square; the weaker-penalty map assigns about 9.5%. This is sensitivity to a numerical/model boundary, not an observed gas truncation radius.
2. Check predictive fits on withheld exposures, leaving both R and C represented in the training set, alongside the instrumental-response and continuum checks. Freeze the galaxy map and kinematics when assessing held-out predictions. This tests real observations without choosing a morphology prior merely because it returns a familiar velocity.

The present comparisons do not quantify the gain from R+C relative to a single grism. They test whether our **joint R+C** implementation is robust to specific modeling choices.

### Additional starting-solution check for ID3700

The stronger amplitude penalty found a high-velocity branch by continuing down from sigma=80 km/s. At sigma=0 it reaches the 1000 km/s velocity bound and scores better than the approximately 298 km/s local solution **under that same prior**. This is not a reason to accept 1000 km/s or to discard it because it looks unfamiliar. It requires checking whether the other priors missed this branch.

A separate checkpointed follow-up transports the highest- and lowest-velocity prior-specific optima at each sigma into the other priors, checks identical physical starting maps, refits, and repeats continuation in both sigma directions. Each original best point remains available in the follow-up, including the unchanged linked control. These checks separate an observed effect of the prior from a missed starting solution as far as this finite set of basins allows; they do not prove global optimization.

That follow-up is complete: **62 additional optimizations**, plus 24 retained original minima. It preserves the main conclusions above. At fixed sigma=0, the two investigated velocity branches give:

| ID3700 flexible prior | Lower-speed branch, km/s | Higher-speed branch, km/s | 2(J_high − J_low) |
|---|---:|---:|---:|
| Baseline | 283.27 | 988.04 | +79.32 |
| Smaller fluctuations, A=1 | 299.39 | 1000, bound | **−10.60** |
| Larger fluctuations, A=4 | 277.73 | 947.01 | +117.12 |
| Correlation 0.05 arcsec | 278.07 | 967.41 | +70.40 |
| Correlation 0.20 arcsec | 294.70 | 1000, bound | +53.94 |

Positive means the high-speed branch is worse **within that prior**; negative means it is better. These are objective differences, not confidence levels or Bayes factors. The high-speed basin was explicitly tested under the weaker priors, rather than assumed absent. It survives as a worse local solution there. The stronger A=1 penalty reverses which branch is preferred. This is a concrete example of morphology regularization affecting the kinematic conclusion.

![ID3700 velocity sensitivity after cross-initialization](../../results/real_sources/ID3700/morphology_controls_cross_v1/figures/velocity_morphology_comparison.png)

![ID3700 fitted common gas maps after cross-initialization](../../results/real_sources/ID3700/morphology_controls_cross_v1/figures/source_morphology_comparison.png)

The ID3700 first pass had one CUDA `ptxas` compilation failure. It resumed from verified checkpoints without changing saved fits or the scientific code. The follow-up reuses each compiled model within a scenario to avoid the first worker's redundant reference compilations.

## Verification

There are **246 new optimizations** in total: 184 first-pass fits and 62 additional ID3700 cross-checks. All pass the specified local stationarity threshold. The 24 original best points copied into the follow-up are not counted as new optimizations. The new model, transport and existing fixed-sigma tests passed: **28 tests**, including gradients, matching NumPyro density, exact brightness transport and the zero-width boundary test.

Every saved detector artifact is checked against its native SCI/ERR/mask arrays, per-grism residual chi-square, source identity and frozen input/code hashes. The transported real-source starting detector images agree to floating-point precision; the objective difference is checked against the changed map penalty. Each reviewed notebook executes with zero cell errors and five embedded figures. Review imports remain CPU-only and do not import JAX. The [combined validation record](../../results/real_sources/comparison_2026-10-18/morphology_validation.json) records the completed checks.

## Inspect and reproduce

Open [real_source_morphology_controls.ipynb](../notebooks/archive/real_source_morphology_controls.ipynb), choose source ID21636 or ID3700, select the grism kernel and Run All. The default model is flexible, the default scenario is baseline, and `RUN_FITS=False` loads saved results on the CPU. `RUN_NAME=None` selects the completed cross-check when available, otherwise the first pass; the loaded directory is printed. The notebook shows velocities across priors, dispersion profiles, intrinsic map comparisons, actual R/C observed/predicted/residual panels and alternative starting solutions. Setting `RUN_FITS=True` launches a separate GPU worker with logs and writes a new run directory.

- [New comparison worker](../diagnostics/native_morphology_controls.py).
- [Cross-initialization follow-up](../diagnostics/native_morphology_crosscheck.py).
- [Matched-map prior transport](../dingo/native_morphology_controls.py).
- [Linked-disk control](../dingo/native_linked.py).
- [CPU review helper](../dingo/real_morphology_controls.py).
- [ID21636 saved comparison](../../results/real_sources/ID21636/morphology_controls_v1/dispersion.json).
- [ID3700 saved comparison](../../results/real_sources/ID3700/morphology_controls_v1/dispersion.json).
- [ID3700 cross-initialized comparison](../../results/real_sources/ID3700/morphology_controls_cross_v1/dispersion.json).

The linked class includes a matching NumPyro target, tested against its optimization density. This comparison does not launch new kinematic MCMC. Existing science modules and previous result artifacts remain unchanged.
