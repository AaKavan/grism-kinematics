# Real R+C fits: detector mask, shared flux and continuum repair

The extreme ID6170 line/continuum cancellation was mainly triggered by **one persistent bad detector pixel**. A documented mask removes six affected pixels across its cutouts. With this mask, even the previous independent-flux model stops producing the enormous C component canceled by negative continuum. The new calibrated shared-flux model gives sensible relative exposure normalizations and leaves ID3700 close to its previous result.

**ID6170 still does not have a reportable velocity, dispersion, V/σ or mass profile.** Its lowest saved penalized score now occurs near Vrot(0.5″)=863 km/s, with a displaced center at a bound, turnover at its lower bound, and an inclination far from the imaging preference. Numerical stationarity is not validation of that geometry. ID3700's preferred speed changes from about 278 to 280 km/s, an encouraging control check.

Open [real_source_repaired.ipynb](../notebooks/archive/real_source_repaired.ipynb), choose `SOURCE_ID = 6170` or `3700`, and Run All in the **grism** kernel. `RUN_FITS = False` displays saved observations, predictions, residuals, component images, alternative solutions and F150W overlays. The notebook has no new kinematic MCMC run. The older notebooks link to it and retain their historical outputs.

## The detector finding

NRCBLONG detector coordinate **(x,y)=(1817,875), zero based**, has an isolated large negative SCI rate in all 18 associated raw exposures:

- R: approximately −23 to −70 times its quoted ERR, even when the galaxy is elsewhere on the detector.
- C: approximately −93 to −146 times ERR. The artifact falls inside ID6170 cutouts in exposures 10–15, at different cutout coordinates as the dithers move the source.
- Its DQ value is 4. This is `JUMP_DET` in the [official JWST DQ definitions](https://jwst-pipeline.readthedocs.io/en/stable/jwst/references_general/references_general.html); the existing `reject_bits=3` selection did not reject it. A jump flag alone is not the basis for rejecting every such pixel. Here the repeated isolated negative detector feature supplies the independent evidence.

![Repeated detector pixel](../../results/real_sources/ID6170/detector_audit_2026-10-18/detector_pixel.png)

The [audit](../../results/real_sources/ID6170/detector_audit_2026-10-18/audit.json) records 5×5 SCI/ERR/DQ neighborhoods, native filenames and detector coordinates. The [mask manifest](../data/detector_masks/nrcblong_1817_875_2026-10-18.json) is restricted to that coordinate and the 18 audited exposure basenames. Original FITS/DQ files remain unchanged. ID6170 has **116,303** selected pixels after removing six; ID3700 retains all **115,603** previously selected pixels. No exposure is discarded.

This updates the interpretation of the [initial ID6170 failure](real_ID6170_2026-10-18.md). That report correctly identified cancellation in the model decomposition, but did not yet identify the detector artifact that drove it. Simply adding flux/background priors while retaining this pixel did not fix the problem: early trials distorted relative calibration gains and background offsets to accommodate it.

## What the revised model fits

All native R and C exposures still constrain **one emission map, one rotation field and one intrinsic line dispersion**. They are projected through each exposure's trace, astrometry and response. Different predicted detector images are expected. The velocity-law, imaging inclination prior, map prior, registration assumptions and nominal PSF/LSF are retained from the source's previous run.

For exposure e and detector pixel p, the new mean is

\[
\mu_{ep}=F_{17}\,K_e\,g_e\,T_{ep}(v_{\rm rot},i,\sigma_v,\ldots)
           +C_{ep}(a_e)+b_e.
\]

Here T is the unit-flux native line prediction, including the shared intrinsic velocity dispersion. F17 is the common integrated line flux in units of 10⁻¹⁷ erg s⁻¹ cm⁻². The response converts that flux to detector count rate:

\[
K_e=\frac{10^{-17}}{10^{-23}}\,
S_e(\lambda_0)\frac{\lambda_0^2}{c_{\mu\mathrm{m/s}}}
\left|\frac{dp_e}{d\lambda}\right|.
\]

S is the supplied F444W sensitivity in DN/s/Jy. This includes the spectral-pixel frequency width; the R/C point-sensitivity ratio alone is insufficient. SCI already contains rates, so no exposure-time factor is applied. The trace derivative is independently checked against finite differences. For the first ID6170 R/C exposures, K is approximately **0.9124/0.9092 DN/s per F17**. Calibration files and hashes are saved with each run.

The response is evaluated at the nominal line wavelength and catalog position and held fixed within that source. Within-line sensitivity variation at ±1000 km/s is recorded, but not included in the renderer. Absolute calibration uncertainty and a complete spatial/wavelength response uncertainty model are not included.

Relative gains have a zero-sum log constraint, removing their degeneracy with common flux. Their provisional marginal log standard deviation is **0.10**, with Gaussian priors on orthogonal contrasts. This is a soft prior, not a hard 10% bound. It has not been established as the instrument's measured calibration uncertainty.

In the positive-continuum version, C uses three nonnegative quadratic Bernstein coefficients per detector row for R, or per column for C. This is a restricted smooth positive continuum family. Each coefficient has a weak proper HalfNormal prior of scale **1 DN/s**. A separate signed background offset per exposure has a zero-mean Gaussian prior with standard deviation equal to that exposure's median selected ERR. The negative backgrounds permitted by this model matter for background-subtracted data; individual SCI pixels need not be positive. Every selected pixel enters the likelihood once. Sideband fits are not reused as independent priors.

For short optimization, the continuum coefficients are solved exactly by enumerating the eight active sets of each three-coefficient nonnegative least-squares problem. JAX computes the profiled objective gradient; tests compare its derivative to finite differences and its solution to SciPy. The calibration and background assumptions require further validation before adopting this as the definitive science model.

## Controlled comparison on both sources

Each model below used the same documented detector mask and eight starting points. These are continuous-velocity multistart optimizations, not a new 1 km/s grid. All **48** stored fits passed the projected-gradient stationarity threshold of 0.1. They still reach different local minima.

| Source | Model | Best stored Vrot(0.5″), km/s | Inclination | Turnover, arcsec | Center (x0,y0), arcsec | χ² R | χ² C |
|---|---|---:|---:|---:|---|---:|---:|
| 6170 | Previous independent flux + unrestricted continuum | 863.31 | 50.75° | 0.020 | (−0.166, −0.300) | 59,396.3 | 60,386.6 |
| 6170 | Shared calibrated flux + unrestricted continuum | 863.16 | 50.70° | 0.020 | (−0.167, −0.300) | 59,403.5 | 60,380.6 |
| 6170 | Shared calibrated flux + positive continuum/background | 862.63 | 50.78° | 0.020 | (−0.167, −0.300) | 59,405.3 | 60,382.6 |
| 3700 | Previous independent flux + unrestricted continuum | 278.20 | 78.47° | 0.036 | (+0.017, −0.015) | 60,359.2 | 75,630.6 |
| 3700 | Shared calibrated flux + unrestricted continuum | 280.51 | 78.53° | 0.038 | (+0.018, −0.015) | 60,358.2 | 75,632.4 |
| 3700 | Shared calibrated flux + positive continuum/background | 280.36 | 78.53° | 0.038 | (+0.017, −0.015) | 60,361.3 | 75,635.0 |

x0 is east and y0 north of the catalog position. Every preferred solution optimizes intrinsic σ at its zero boundary. That is **not evidence for exactly zero turbulence** and provides no upper limit without a posterior and instrument-response checks. χ² R/C are contributions to the joint fit. Penalized scores across models with different nuisance priors are not model evidence. The earlier unmasked ID6170 χ² also uses a different pixel set, so its reduction is not a calibrated significance statistic.

The masked previous model's median C/R detector amplitude ratio is **0.876**, compared with **324** in the earlier failed solution. Its C line-template power remaining after unrestricted continuum projection is approximately **64.5–65.2%**, compared with **0.19–0.30%** before. This is a noise-weighted squared-norm fraction, not an emission-flux fraction. Together with the images, this establishes that the spectacular cancellation has disappeared without requiring the positive-continuum constraint.

![ID6170 revised components](../../results/real_sources/ID6170/coupled_repair_v3/positive/figures/best_components.png)

The positive model's preferred relative gain ranges are **0.865–1.100** for ID6170 and **0.921–1.098** for ID3700. Common F17 values are 4.791 and 3.415 respectively; these are conditional model normalizations, not flux measurements with calibrated errors.

![ID3700 revised fit](../../results/real_sources/ID3700/coupled_repair_v3/positive/figures/best_fit.png)

## What still prevents an ID6170 measurement

![ID6170 gas/stellar overlay](../../results/real_sources/ID6170/coupled_repair_v3/positive/figures/best_morphology.png)

The preferred ID6170 solution has y0=−0.300″ at its allowed boundary and turnover radius 0.020″ at its lower boundary. Its center is 0.343″ from the catalog position. Inclination is about 51°, whereas the underlying stellar inclination mixture prefers about 76° before the adopted gas/stellar scatter. Systemic offset is about −382 km/s. The gas map is clumpy, with emission at its northern edge; approximately 14% of its fitted integrated flux falls within the outer 0.1″ of the finite ±0.9″ map square. That number describes the fitted map, not an observational coverage or detection criterion.

The positive model also finds local solutions near **526, 521, 258, 256 and 233 km/s**. The 526 km/s solution places the center at the eastern boundary and has a fit score 2Δ(objective)≈154 above the lowest stored result. These are not equally good fits under the current model, but the favored fit's geometry still needs validation. No global optimum is certified by eight starts. Visual inspection should guide model checks, not silently replace the numerical fit with a preferred-looking speed.

ID3700 is much more stable around the preferred branch, but a start at 600 km/s finds a worse local solution near **986 km/s**, with 2Δ(objective)≈83. Thus the optimizer can still find different branches. It is not simply returning its initial speed: for example ID6170 starts at 900 and 100 km/s both move near 526 km/s, and a 300 km/s start finds the 863 km/s branch. This is local-minimum and geometry dependence, rather than a frozen velocity parameter.

## Prior sensitivity checks

Each row below uses two saved starts, the preferred positive-model fit and a velocity-separated local alternative. Other assumptions are held fixed. All **16** additional fits passed the stationarity check; this is a limited local robustness check, not a repeat of the full search.

| Changed assumption | ID6170 preferred speed, km/s | ID3700 preferred speed, km/s |
|---|---:|---:|
| Default: log-gain SD 0.10, background scale 1 | 862.63 | 280.36 |
| Log-gain SD 0.05 | 861.63 | 284.01 |
| Log-gain SD 0.20 | 863.02 | 278.72 |
| Background scale 0.5 | 862.59 | 280.21 |
| Background scale 2 | 862.63 | 280.45 |

ID6170 retains the same center boundary. Its competing branch moves between about 492 and 556 km/s as the calibration prior changes, showing that lesser branches are more sensitive. These checks do not rescue the preferred geometry, and the spread is not an uncertainty interval. Background offsets in the positive fits are mostly negative, so their interpretation and continuum priors should remain visible in further checks.

## MCMC status and next scientific step

The new [NumPyro model](../dingo/native_coupled.py) explicitly samples intrinsic σ, rotation/geometry, the map and registration variables, common line flux, relative calibration gains, background, and **4,374 positive continuum coefficients**. Optimization profiles those coefficients; posterior inference must sample or correctly marginalize them. A profiled optimum is not a marginalized posterior.

The target and initialization interface are implemented, and tests verify density differences against the joint objective at the profiled coefficient solution. **A production runner with fresh warmup, checkpointing and sampler tuning for this larger continuum block is not connected yet. No new kinematic MCMC was launched.** Existing ID3700 chains belong to the earlier model and do not provide errors for these revised fits. Long chains alone cannot validate an incorrect center, response or source assignment.

Next, use the real images to compare stellar-center and freer-center assumptions for ID6170, inspect the neighboring trace/source association, and test position-angle and finite-map support assumptions. Keep the calibrated shared-flux versus positive-continuum comparison: after removing the artifact, adding the positive continuum has little effect on the preferred speed, so its larger sampling cost needs justification. Then repeat dispersion/PSF/LSF checks and run fresh chains for the selected model. Only after those checks should we construct rotation and mass profiles over the radii actually supported by the emission.

## Reproducibility and execution

- Model: [native_coupled.py](../dingo/native_coupled.py); short worker: [native_coupled_fit.py](../diagnostics/native_coupled_fit.py); CPU viewer and GPU launcher: [real_coupled.py](../dingo/real_coupled.py).
- Detector audit: [native_detector_audit.py](../diagnostics/native_detector_audit.py); explicit mask application: [native_quality.py](../dingo/native_quality.py).
- Results: [ID6170](../../results/real_sources/ID6170/coupled_repair_v3) and [ID3700](../../results/real_sources/ID3700/coupled_repair_v3), one immutable settings/code-hashed checkpoint per variant. Each contains native pixels, cached candidate arrays, all starts, logs and figures.
- Prior checks: [ID6170](../../results/real_sources/ID6170/coupled_sensitivity_v1) and [ID3700](../../results/real_sources/ID3700/coupled_sensitivity_v1). [Machine-readable comparison](../../results/real_sources/coupled_repair_comparison_2026-10-18.json).
- **16 targeted tests passed**, covering conversion units, exact continuum solver, profiled gradient, gain gauge, posterior-density consistency, old-model objective consistency, detector mask scope, and existing native posterior tests.
- The 20-cell results notebook was executed in a fresh grism kernel for both source IDs: **zero errors**, nine static images for ID6170 and eight for ID3700. The main notebook retains the ID6170 outputs; the [ID3700 executed copy](../../results/real_sources/ID3700/coupled_repair_v3/inspection.ipynb) is also saved.
- All fitting used **JAX on the RTX 4090**, with SciPy L-BFGS-B driving the optimization. Individual stored optimizations took roughly 0.5–12 seconds; the eight-start workers took about 19–43 seconds including setup/compilation. These timings are not MCMC timings.
- Early `coupled_repair_v1` trials without the mask were deliberately interrupted; they are preserved as failed diagnostics. `coupled_repair_masked_v2` finished its positive-model fits but failed while compiling another model with `std::bad_alloc`. Running one model per worker releases compiler memory; all six `coupled_repair_v3` workers and all sensitivity workers completed. The enclosing earlier tool session reported exit 143 after all six reports were completed; completion here was checked from each worker's checkpoint and artifacts, not inferred from that parent exit code.
- The previous science files, grids, posterior files and original FITS were preserved. New notebook review requires only saved arrays and CPU plotting; all fitting switches default to false.
