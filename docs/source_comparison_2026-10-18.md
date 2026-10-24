# Five more real sources: first joint R+C comparison

The first comparison is complete for **ID8910, ID15665, ID16887, ID21619 and ID21636**. Each has twelve starting points in each of two continuum models: **120 fitted solutions**, all passing the numerical stationarity threshold. These fits use the calibrated shared-flux R+C model and independently checked detector masks.

**ID21636 and ID15665 are the most promising next sources.** Most starts find the same preferred speed, their fitted centers remain close to the catalog positions, and changing the continuum treatment barely changes the result. ID21636 has particularly good spatial correspondence between the inferred gas and F150W structure. This does not establish a posterior error bar or validate a mass profile.

Open [real_source_comparison.ipynb](../notebooks/archive/real_source_comparison.ipynb) in the **grism** kernel, change `SOURCE_ID`, and Run All with `RUN_FITS=False`. It shows the comparison table, real images, model/residual panels, line/continuum decomposition, F150W overlay and a competing solution. ID3700 and ID6170 are also selectable as references.

## Scope and inclination status

All five completed new-source comparisons below use the **same broad isotropic inclination prior**, proportional to sin(i) between 5° and 85°. This makes a consistent initial comparison and avoids attaching another galaxy's imaging prior. The point estimates are **conditional exploratory fits**, not imaging-constrained deprojected velocity measurements. The earlier ID3700 and ID6170 reference fits retain their own imaging priors; the notebook labels this distinction.

The existing ID8910 F150W fit was not accepted: 17 divergences, maximum Rhat 1.0115, and minimum tail ESS about 306. Its apparently narrow axis-ratio distribution is not a reason to bypass those diagnostics.

**At the time these short-fit results were prepared, the independent F150W sampling batch was still running, beginning with ID15665.** It will attempt the existing full/core-masked PySersic workflow for 15665, 16887, 21619 and 21636. These are imaging chains, not kinematic chains. Their live reports are under each source's `imaging/` directory. The [process/status snapshot](../../results/real_sources/comparison_2026-10-18/imaging_progress_snapshot.json) records the active job at review time; it is not a live monitor. The worker writes to [ID15665's imaging log](../../results/real_sources/ID15665/imaging/full_pixel.log), and batch outcomes appear in `comparison_2026-10-18/imaging_status.json` after a source finishes. No unfinished or failed imaging posterior enters the completed comparisons. An accepted imaging-prior refit is a subsequent experiment; it has not been claimed as completed here.

## Inputs and detector checks

![Five sources: F150W and example native R/C images](../../results/real_sources/comparison_2026-10-18/observations.png)

| Source | Redshift | Paα wavelength, μm | Usable R / C exposures | Additional masked cutout pixels |
|---|---:|---:|---:|---:|
| 8910 | 1.4233 | 4.54369 | 3 / 6 | 5 |
| 15665 | 1.1613 | 4.05244 | 9 / 9 | 2 |
| 16887 | 1.3456 | 4.39800 | 9 / 9 | 3 |
| 21619 | 1.3645 | 4.43344 | 9 / 8 | 6 |
| 21636 | 1.1906 | 4.10737 | 9 / 9 | 0 |

ID8910 has nine associated C files, but its systemic-line cutout falls outside the detector in three of them. They have no valid SCI/ERR pixels and are recorded as excluded. ID21619 has eight associated C exposures; all eight are used. ID6084's configuration exists, but its source-specific R/C native association products are missing from the copied bundle, so it was not fitted.

The pre-fit inspection found four more recurrent isolated negative detector features: NRCALONG (542,877) and (827,1242), and NRCBLONG (411,1480) and (424,1475), all zero based. Each was independently inspected in **18 raw exposures at the same detector coordinate**, including exposures with the target elsewhere. Every audited central pixel was below −15 ERR; some were below −200 ERR. The 5×5 neighborhoods demonstrate an isolated detector feature rather than a moving source. The DQ=4 values again survive the existing reject-bits selection.

The [expanded mask manifest](../data/detector_masks/survey_checked_pixels_2026-10-18.json) includes these four coordinates and the previously verified ID6170 coordinate, scoped to the audited exposure filenames. It is applied before optimization. It changes neither the original FITS nor their DQ arrays. The [inventory](../../results/real_sources/comparison_2026-10-18/inventory.json), raw-pixel audits and [per-source mask application records](../../results/real_sources/comparison_2026-10-18/mask_application.json) retain the evidence. This is not an exhaustive detector-quality survey, and negative pixels were not automatically removed based on a threshold or a fit residual.

## Results under the common broad inclination prior

The table shows the lowest stored penalized objective for the **positive continuum + signed background** model. Speeds refer to 0.5 arcsec. V sin(i) is the projected rotation amplitude at that same radius; σ is the intrinsic Gaussian broadening parameter, not an instrumental width.

| Source | Vrot, km/s | V sin(i), km/s | Inclination | Turnover, arcsec | Center (x0,y0), arcsec | Assessment |
|---|---:|---:|---:|---:|---|---|
| 8910 | 493.4 | 171.6 | 20.35° | 0.196 | (−0.300, −0.300) | Both center coordinates at bounds; strong deprojection/geometry dependence |
| 15665 | 277.3 | 195.4 | 44.82° | 0.079 | (−0.050, −0.051) | Eleven starts agree closely; useful follow-up, but map-edge structure needs checking |
| 16887 | 63.5 | 63.3 | 85.00° | 0.020 | (+0.020, −0.001) | Inclination and turnover at bounds; compact source with unresolved geometry |
| 21619 | 173.2 | 148.3 | 58.93° | 0.096 | (+0.152, +0.162) | Competing branches and a 0.22″ center displacement |
| 21636 | 186.9 | 176.2 | 70.52° | 0.030 | (−0.014, −0.041) | Eleven starts agree closely; compact gas structure follows the elongated stellar source |

x0 is east and y0 north of the catalog position. **These numbers are not final measurements.** All five preferred fits place σ at or numerically extremely close to zero (below 0.002 km/s at the stored optima). This is a boundary preference under the model, not a physical assertion of zero turbulence and not an upper limit.

The calibrated shared-flux model with **unrestricted quadratic continuum** gives preferred speeds of **493.4, 277.2, 63.5, 173.2 and 186.9 km/s**, respectively. Thus positive-continuum enforcement is not driving these preferred speeds after the detector exclusions. Penalized objectives from models with different nuisance priors are not compared as model evidence.

Relative calibration gains at the positive-model optima remain modest: approximately 0.95–1.10, 0.96–1.03, 0.94–1.07, 0.92–1.09 and 0.93–1.08, respectively. The response correction, common line flux and provisional 0.10 log-gain prior are unchanged from the [repair experiment](shared_flux_repair_2026-10-18.md).

## Starting-value dependence

Each model uses the same twelve **unfitted** seeds: velocities 50, 250 and 600 km/s crossed with four kinematic position angles. The three speed groups start with intrinsic dispersions 20, 60 and 120 km/s and turnover radii 0.1, 0.3 and 0.7 arcsec. The entire nuisance vector is refitted. No source borrows another source's fitted solution.

| Source | Positive-model starting-point outcome | Competing solution's score above best |
|---|---|---:|
| 8910 | Seven starts near 493; four near 182; one at zero | 182 km/s: 2ΔJ≈23.35 |
| 15665 | Eleven starts near 277.26; one at zero | Zero-speed solution: 2ΔJ≈11,042 |
| 16887 | Eight near 63.5; three worse solutions near 62–66; one at zero with σ≈55 | Zero-speed/σ55 solution: 2ΔJ≈170.89 |
| 21619 | Six near 173; five near 296; one near 203 | 296 km/s: 2ΔJ≈94.46 |
| 21636 | Eleven near 186.88; one near 348.88 | 349 km/s: 2ΔJ≈43.78 |

J is the penalized objective. These score differences identify better and worse stored local solutions, not calibrated confidence levels or posterior mode probabilities. Similar speeds can hide different nuisance solutions, as for ID16887. Twelve starts do not prove a global optimum. The tiny within-branch numerical spread for ID15665 or ID21636 is **not** a velocity error bar.

The optimizer is clearly not simply echoing its starting velocity: starts spanning 50–600 km/s repeatedly move to the same branch in the more stable sources. Remaining initialization dependence is associated with genuine alternative local minima in the current objective. The ID16887 σ≈55 km/s alternative also demonstrates that σ is not fixed to zero in the code; it does not establish the correct physical dispersion.

## Visual conclusions

**ID21636 is my first choice for the next detailed check.** Its preferred center is only about 0.043″ from the catalog position, most starts agree, the two continuum variants agree, and the fitted gas follows the two bright elongated structures in F150W. Very little fitted flux reaches the finite map's outer edge. Turnover is still small, so spatial resolution and response assumptions need attention before a mass profile.

![ID21636 inferred gas over F150W](../../results/real_sources/ID21636/coupled_survey_broad_v1/positive/figures/best_morphology.png)

**ID15665 is a useful second candidate.** The preferred speed and center are stable, with ordinary relative gains. Its larger, clumpier emission map has about 11% of fitted integrated flux in the outer 0.1″ of the ±0.9″ square. Bright structure reaches the northern boundary. Test larger map support and source/contaminant assignment before treating its apparent numerical precision as physical precision.

![ID15665 observed, predicted and residual images](../../results/real_sources/ID15665/coupled_survey_broad_v1/positive/figures/best_fit.png)

**ID21619 needs a center/geometry comparison.** The best stored branch is near 173 km/s, but the center is displaced by about 0.22″ and the image is clumpy. Starts readily find a worse branch near 296 km/s. A close-looking image alone cannot resolve that model dependence.

**ID16887 is useful for response and spatial-resolution tests.** It is compact; the optimizer makes inclination nearly edge-on and turnover as small as allowed. Its low fitted speed is not yet a robust deprojected measurement.

**ID8910 is not a good immediate mass target.** Coverage is weaker, the imaging prior failed diagnostics, its preferred solution uses a low inclination and both center bounds, and a substantially lower-speed branch remains. About 12% of fitted flux reaches the outer map band. Broad-prior fits alone cannot resolve its deprojection.

The component/residual figures are saved for both models under each result directory. Typical full-cutout residual RMS values are around 1.00–1.11 ERR for these new sources; these are raw per-pixel RMS diagnostics, not reduced χ² with a known number of degrees of freedom. Good global residual RMS can coexist with an unsuitable geometry or structured residuals near the source.

## What this says about dispersion

The tendency toward low intrinsic σ is **not unique to ID3700**. It appears in all five preferred new-source fits, across both detector modules, a range of source sizes and two continuum treatments. That makes a shared modeling/calibration contribution worth checking; it does not prove that the instrumental calibration is wrong or that all galaxies have the same physical dispersion.

For these sources, the adopted narrow LSF Gaussian components have velocity standard deviations of roughly **67–75 km/s**, with broad components around **158–189 km/s**. These are the supplied instrumental mixture components, not measured gas dispersions. Small intrinsic widths are inferred through relatively small changes on top of that response. The previously tested [ID3700 response sensitivity](real_dispersion_2026-10-16.md) already showed that its low-σ constraint changes with spectral-width assumptions.

The next useful sequence is: accept and inspect source-specific imaging priors; compare them with these broad-prior results for ID21636 and ID15665; examine response-width and emission-map sensitivity; then use fresh kinematic MCMC for the selected model. Do not calculate a reportable V/σ by dividing these optimum speeds by a near-zero optimizer result. No new dispersion upper limit, pressure correction or dynamical mass interval is established here.

## Artifacts and validation

- [Comparison notebook](../notebooks/archive/real_source_comparison.ipynb), executed successfully for all five new source IDs and ID3700 with fitting disabled. Source-specific executed copies are saved in their result directories. Images and identifiers were checked when switching between nine, seventeen and eighteen exposures.
- [Summary JSON](../../results/real_sources/comparison_2026-10-18/summary.json), [summary CSV](../../results/real_sources/comparison_2026-10-18/summary.csv), and [all-start CSV](../../results/real_sources/comparison_2026-10-18/all_starts.csv). Reference rows retain their original prior settings.
- [ID8910 results](../../results/real_sources/ID8910/coupled_survey_v1), [ID15665](../../results/real_sources/ID15665/coupled_survey_broad_v1), [ID16887](../../results/real_sources/ID16887/coupled_survey_broad_v1), [ID21619](../../results/real_sources/ID21619/coupled_survey_broad_v1), [ID21636](../../results/real_sources/ID21636/coupled_survey_broad_v1).
- [Input/seed preparation](../diagnostics/native_source_prepare.py) constructs actual source-specific data and priors without first running the old velocity grid. Prepared checkpoints explicitly contain **unfitted seeds and no completed velocity scan**. Existing science-model files and earlier result checkpoints are unchanged.
- [Batch runner](../diagnostics/run_source_comparison.py) logs source failures separately and requires an explicit flag or prior selection for exploratory isotropic runs. [Report builder](../diagnostics/native_source_comparison_report.py) and [viewer](../dingo/real_coupled.py) use saved arrays on the CPU. The viewer now chooses R/C examples from the actual exposure list rather than assuming nine of each.
- Each fitted checkpoint records its settings, native-pixel hash, seed hashes, sensitivity hashes and science-code snapshots. Rendered residual χ² is checked against the objective by the worker. All 120 new solutions passed the projected-gradient threshold of 0.1; this is numerical validation, not scientific convergence.
- A separate final verification checked all ten completed runs and 120 candidates: current/snapshotted science-code hashes, input/calibration/seed hashes, source identity, mask counts, unfitted-seed provenance, and residual χ² consistency. The six notebook source selections executed without errors; the main notebook now opens on ID21636 with fitting disabled.

The kinematic work used JAX on the RTX 4090 with SciPy L-BFGS-B and per-fit checkpoints. **No new kinematic MCMC was launched.** The separate F150W imaging sampling is the longer-running stage; the completed comparison remains usable while it runs.
