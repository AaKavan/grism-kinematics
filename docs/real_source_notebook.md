# Real-source notebook — October 16, 2026

Open [real_source_kinematics.ipynb](../notebooks/archive/real_source_kinematics.ipynb) in Jupyter or VS Code and select the **grism** kernel (`/home/aakavan/anaconda3/envs/grism/bin/python`). **Run All** with the defaults displays the completed ID3700 fit. It does not start a new fit or generate simulated science data.

**Sections 10–11 compare dispersion on the real observations.** Section 10 works independently after a kernel restart and defaults to loading `dispersion_v1`. Keep `RUN_DISPERSION=False` to display saved results, or set it to `True` to run/resume fixed-sigma refits with streamed iteration progress and per-fit checkpoints. The default tests 0, 10, 20, 40 and 80 km/s with nominal response and separate ±10% spectral/spatial width changes. Two optimization sweeps start from separated grid solutions; failed checks remain visible. Section 11 shows the selected fit with R above C and observed/predicted/residual columns. These are penalized optimization diagnostics, not MCMC confidence intervals. All three run switches (`RUN_FIT`, `RUN_MCMC`, `RUN_DISPERSION`) default to `False`. The notebook now has 34 cells. See [the dispersion comparison](real_dispersion_2026-10-16.md) for results and scientific limits.

**Real-data MCMC is now a separate section 7.** `RUN_MCMC=False` displays saved real-source chains; `True` starts the named posterior run from the selected grid. See [the posterior report](real_posterior_2026-10-16.md) for the explicit probability target and the first pilot. Corner plots are labeled diagnostic when convergence checks fail; they are not constructed from optimized grid points.

If a kernel opened before the MCMC update reports `cannot import name 'show_posterior'`, rerun the updated section 7 cell. It reloads the notebook helpers and rebuilds the source object with its existing ID, input paths and configuration; restarting the kernel is unnecessary. Leave `RUN_MCMC=False` to display saved results without launching sampling.

**Recovering results after a notebook display problem:** section 8, **Show saved MCMC results**, runs independently after a kernel restart. It loads the saved report and figures without launching any worker. With a fresh kernel it defaults to ID3700; otherwise it respects `SOURCE_ID` from above. `VIEW_POSTERIOR=None` selects the latest saved real posterior for that source. Set an explicit directory to choose another run. Both `RUN_FIT` and `RUN_MCMC` are currently set to `False`. The completed ID3700 `posterior_long_v1` has all four chains' 2,000 retained draws and passes the sampling checks.

**Paper-style contours:** section 9 independently creates a filled contour corner plot using [corner.py](https://corner.readthedocs.io/en/latest/api/). It pools retained draws only when the saved sampling checks pass. Pairwise contours enclose approximately **68% and 95%** probability; diagonal lines and numerical summaries use the **16th/50th/84th percentiles**. These are explicit enclosed probabilities, not Gaussian sigma-radius labels. The plot uses a lightly smoothed 2-D histogram; all 8,000 retained ID3700 draws enter the plot and numerical percentiles, without warmup or tail cuts. `CORNER_PARAMETERS`, `CORNER_BINS` and `CORNER_SMOOTH` control the display. PNG, vector PDF and provenance JSON exports are saved under the posterior's `figures/corner_contours.*`. Center labels are `x0` (east) and `y0` (north), both arcsecond offsets from the catalog sky position. The earlier `center_east`/`center_north` parameter selections remain accepted as aliases. Existing plots colored by chain remain available in sections 7–8. The new cell works from a fresh kernel and launches no sampling worker.

## Prepared longer run — launch from the notebook

The section 7 defaults now prepare `posterior_long_v1`: **4 GPU chains, 1,500 fresh warmup steps and 2,000 retained draws per chain**, maximum tree depth 10, checkpoints every 100 retained draws. Sampling remains disabled until you set `RUN_MCMC=True`. Leave `RUN_FIT=False` and run the preceding cells to load the existing real-data grid first.

`MCMC_INITIAL_POSTERIOR="latest"` selects the latest completed real posterior belonging to the same source and grid. For ID3700 this currently selects `posterior_depth9_v1`, with final velocities 276.55, 276.08, 286.38 and 278.53 km/s. The new runner copies each chain's complete final position, including map, inclination, center and registration. It learns new sampler tuning, excludes previous samples, and retains the original broad velocity prior. It does not narrow the velocity prior around these starts. Starting nearby cannot establish that other solutions carry negligible posterior probability; a subsequent independent run with `MCMC_INITIAL_POSTERIOR=None` and a different tag checks competing grid starts. If no matching completed posterior exists, the notebook falls back to those grid starts.

During compilation, loading and warmup, elapsed-time messages appear every 20 seconds. After each saved sampling block, the fresh-adaptation runner prints a text progress bar, saved draws per chain, block duration, total invocation time, estimated remaining sampling time, and diagnostics. The ETA uses measured checkpoint times including plotting; it excludes final rendering and is approximate. Compilation can make the first checkpoint unusually slow. The full output goes to `../results/real_sources/ID3700/posterior_long_v1/run.log` for ID3700.

Interrupting the cell stops the worker. After warmup completes, rerunning the same settings resumes the saved state; the current unfinished block is lost. Interruption before the warmup checkpoint requires a fresh tag. To extend a completed run, increase `MCMC_SAMPLES` while retaining the other settings. To view it, set `RUN_MCMC=False`. Resume preserves the original initialization source even if a newer posterior appears. The older `MCMC_FROM_POSTERIOR` option reuses adapted tuning; set `MCMC_INITIAL_POSTERIOR=None` before using it.

The new [fresh-adaptation runner](../diagnostics/native_real_posterior_rewarm.py) leaves the earlier drivers and posterior target unchanged. Preparation checks passed: 21 targeted tests; notebook schema/code validation and saved-result viewing; launch routing captured without executing a worker; archived posterior code hashes unchanged. **The longer science run has not been launched during preparation.**

## Use it

1. Set `SOURCE_ID` in the first code cell.
2. Leave `RUN_FIT=False` to inspect the most recent saved fit, or set it to `True` to run/resume the chosen source.
3. Use `VELOCITY_STEP=25` for an exploratory scan, or `1` for the full 0–1000 km/s grid. A new grid spacing has its own automatically generated run name. The zero-rotation control is always included.
4. Select `CENTER="catalog"`, `"stellar"`, or `"free"`. Choose `INCLINATION="imaging"` for the source's own F150W PySersic constraint. `"isotropic"` explicitly selects a broad orientation prior; no other source's imaging is substituted.
5. Run the cells in order. If needed, the imaging cell first fits this source's real F150W image. Progress appears in the notebook and is saved to log files. Interrupting the running cell terminates its worker process.
6. Compare velocity trials using the slider and separate R/C exposure menus. The two rows show R and C together; all exposures belong to the same **joint R+C fit**. An uncached trial needs **Prepare selected view**; this renders its stored model rather than rerunning optimization. A manual plotting cell works without a widget front end.
7. Record the reason for retaining or questioning a trial with **Save inspection note**. These notes preserve the fit and score rather than replacing the optimizer's winner.

Changing source ID, run tag, velocity spacing or center/inclination choice selects a different run directory. Other changes to settings, data or fitting code require a fresh `RUN_TAG`; resume rejects incompatible checkpoints. `SAVED_RUN` can explicitly select an earlier directory. To compare center treatments, run the desired cases with their distinct names, then select each saved run for inspection.

## Available inputs

These are counts of STATS-associated files present locally; the loader separately checks valid SCI/ERR/DQ coverage at the selected line and records excluded exposures.

| Source ID | R associations | C associations | Native files present? |
|---:|---:|---:|---|
| 3700 | 9 | 9 | Yes |
| 6170 | 9 | 9 | Yes |
| 8910 | 3 | 9 | Yes |
| 15665 | 9 | 9 | Yes |
| 16887 | 9 | 9 | Yes |
| 21619 | 9 | 8 | Yes |
| 21636 | 9 | 9 | Yes |
| 6084 | 0 | 0 | R/C association files missing |

The current data adapter is for the staged M0416/F444W bundle and its supplied calibration, not arbitrary survey pointings or filters. New IDs need their own configuration, R/C STATS files and every associated native exposure. The notebook reports missing paths explicitly. F150W cutouts come from the 4750 mosaic and must pass coverage checks. The displayed availability table does not certify disk morphology or sufficient kinematic signal.

## What is fitted

The underlying model is the same native-exposure real-data model used for ID3700: a shared positive flexible emission map and arctangent velocity field, projected into each original R/C exposure. It adjusts inclination, turnover, kinematic angle, center, systemic velocity, intrinsic width, map shape and exposure registration. Positive line amplitudes and quadratic continua are profiled analytically. The trial coordinate is **deprojected rotation at 0.5 arcsec**.

The generic runner tries multiple starting angles, scans velocities in both directions, and follows the best branch outward. Each fit is checkpointed atomically. A local gradient threshold is recorded separately from the optimizer's success message; the viewer displays failed checks. Completing the loop does not certify global optimization or physical model adequacy.

Imaging uses full and central-0.18-arcsec-excluded PySersic fits. The angle interval is centered on this source's measured image major axis, replacing the ID3700-specific angle cut. Source identity and imaging diagnostics are checked before transferring an orientation prior. The two fits describe the same image and enter as alternatives, not independent measurements. The provisional thickness is uniform from 0.05 to 0.30 before rejecting impossible `q < q0` pairs; the rejection fraction is recorded. Additional inclination scatter is 10 degrees. Highly irregular or poorly fit sources can fail this gate and need inspection rather than an automatic inclination claim.

New runs are saved under `../results/real_sources/ID<source>/`. Imaging and each grism run have separate folders. ID3700's original October 15 results remain in their existing directory and are discovered automatically. Those archived mathematical-code hashes and original results are preserved. GPU workers run in separate processes so notebook viewing remains lightweight, and interrupts stop the worker instead of leaving an invisible fitting job.

## Interpretation during visual inspection

The velocity-search plot asks **which trial speed fits best**, not how velocity changes with radius. Its vertical axis is now labeled **Fit score above best — lower is better**, with a star at the minimum. This is the same quantity previously called “2 Δ penalized objective”: `2 * (objective_at_trial - minimum_objective)`. Zero is the best saved fit by construction, not zero residuals. The objective combines error-weighted pixel mismatch and prior/regularization penalties. The factor of two does not change the winner. The search-direction diagnostic is optional (`SHOW_SEARCH_DETAILS=True`); scores above 100 are outside the default vertical zoom (`score_limit=None` shows all scores).

The main fit figure follows the old notebook's two-row layout: R above C, with **observed image / predicted image / residual** across the first three columns. The last column shows the **shared intrinsic emission map / shared line-of-sight velocity field**. These last two panels have sky-offset axes; the first three columns have native detector-cutout axes. The image predictions include line and continuum. Positive/red residuals mean the model is too faint; negative/blue mean it is too bright. Velocity red/blue means receding/approaching relative to the fitted systemic velocity. The ±500 km/s velocity scale is fixed between trials; the 1%-of-peak map mask only controls display, not detection significance.

Below this is the real F150W image with inferred gas contours. Their coordinates come from the saved image WCS and source coordinates, without forcing the centers to coincide. The contours describe an intrinsic model while the stellar image is PSF blurred, so compare morphology with that difference in mind. The gas brightness display removes the numerical quadrature weights from the saved integrated node fluxes. Contours are fractions of peak brightness, not significance levels.

The separate rotation/mass figure now shows a smooth best-fit rotation curve and a logarithmic enclosed-mass proxy, with the 0.5-arcsec reference radius marked. `COMPARE_VELOCITIES` adds explicitly selected alternatives. A zero-rotation curve has zero mass and is omitted from the logarithmic mass axis, but remains in the table. None of these display changes refits the source or changes saved scores.

Focus on real-source residuals, contaminating features, consistent R/C structure and stability between exposures. The notebook preserves native SCI and fixed DQ/error masks; EMLINE is not fitted with SCI uncertainties. The image selector only changes the displayed exposure within the joint fit. Model comparisons use fixed data-based stretches and residual limits of ±5 ERR.

The score is a penalized profile, not a marginalized posterior. Keep multiple plausible velocities rather than choosing a single attractive image. Inspection notes are append-only JSON lines with source ID, velocity, score, fitted-artifact checksum and the written reason.

The rotation/mass curves use each source's own redshift and Planck18 distance. Their radius grid is explicitly illustrative until radial information has been established. `r Vrot²/G` is a spherical-equivalent rotation-only proxy, not a pressure-corrected mass measurement. Faint edge emission, unresolved turnover, response calibration and contamination remain scientific limitations. Sections 4–6 show conditional grids; section 7 optionally launches real-data posterior sampling. It does not manufacture credible intervals from grid scores.

Real sources and visual inspection are the focus of this interface. Injection/recovery studies remain a separate way of checking bias and interval coverage; they are not a prerequisite cell or a substitute for inspecting the actual observations here.

## How this relates to your old DINGO notebook

Inspected `/home/aakavan/kinematics/archive/2026-10-12/yichen_codes/Archive6/1-fit-kinematics.ipynb`, including its saved figures, and the adjacent archived `grism-kinematics/dingo/{fitting,kinematics,plot}.py`. That saved notebook run is **ID15665**, whereas the current notebook defaults to **ID3700**.

Both methods fit an inclined arctangent rotating disk using the different Doppler distortions in grism R and C. The common line-of-sight relation is `v_los = V(r) sin(i) cos(phi)` (plus a fitted systemic offset in the new model). The fitting objective and treatment of observations changed:

| | Archived DINGO run | Current real-source model |
|---|---|---|
| Observations | Coadded R and C emission-line images | Individual native R/C SCI exposures, with ERR and masks |
| Model comparison | Inverse-warp both observations into velocity-corrected images; minimize `sum((corrected_R - corrected_C)**2)` | Forward-project a shared positive emission map and disk velocity field into each exposure; compare observed SCI with predicted line + continuum |
| Residual plot | Corrected R minus corrected C (and its opposite) | `(observed SCI - predicted SCI) / ERR` for each selected exposure |
| Optimization | Staged local optimization from configured initial values | Trial-velocity grid, multiple starts, both scan directions, continuation, and local stationarity checks |
| Inclination | Parameter of the disk fit | Parameter constrained by the source's imaging prior when selected |
| Velocity normalization | Asymptotic `V_rot = V_inf` | Deprojected `V_ref = V(0.5 arcsec)` |

The rotation-curve normalizations convert as

`V(r) = (2/pi) V_inf atan(r/r_turn) = V_ref atan(r/r_turn) / atan(0.5 arcsec/r_turn)`.

Thus a new speed cannot be compared directly to an old `V_rot` without accounting for turnover radius, even for the same galaxy. The grid is a conditional optimized profile, **not an MCMC posterior**. The PySersic imaging step uses posterior sampling separately. The new section 7 also samples the real kinematic posterior, with analytically integrated line amplitudes and explicit diagnostics.

The old visual layout is useful, but its residuals and “velocity corrected” panels cannot simply be relabeled as the new predictions. The notebook now preserves that distinction. The new mass plots also retain source-specific redshift and calibrated angular units: the old saved mass cell hardcodes `z=1.123` even though its selected source has `z=1.161`, and its mass/velocity-map cells use different nominal pixel scales. Those assumptions were not copied into the new display.

## Validation

The notebook now has **30 cells**, including the independent saved-results viewer in section 8 and the contour corner plot in section 9. The contour cell was checked on all 8,000 retained ID3700 draws, its PNG/PDF exports were generated, and its source-identity and convergence checks were exercised. That viewer was executed from an empty namespace on the completed longer run: seven saved images displayed and no worker launched. The earlier full-notebook validation below used 26 cells including section 7. Its real-data MCMC view was executed successfully on the completed ID3700 chains: zero errors and 11 static PNG outputs. The posterior implementation has 17 passing targeted tests; the pilot results and scientific limitations are in [the real posterior report](real_posterior_2026-10-16.md). The earlier 23-cell validation records below predate the MCMC section.

October 16 plotting refresh: executed all 23 cells again for both saved ID3700 and ID8910 runs, with zero errors (eight and seven static PNGs respectively). The new R/C layout, F150W contour overlay, optional search diagnostics, and smooth/logarithmic radial curves were checked. The overlay catalog pixel agrees with the original mosaic location, its sky/pixel coordinate round trip is accurate to better than one microarcsecond numerically (not a claim about absolute astrometric accuracy), and the brightness conversion matches the model's integration weights. The zero-rotation mass display and wrong-grism selection error were also exercised. Seven workflow tests passed; the archived ID3700 fitting-code hashes still match. Updated executed notebooks are `../results/2026-10-16_notebook_validation/ID3700_familiar_plots_executed.ipynb` and `ID8910_familiar_plots_executed.ipynb`. These are display checks using existing fits, not new science fits.

- Executed all 23 notebook cells on the saved real ID3700 run: zero errors, eight static PNG outputs plus widget controls, including both imaging-fit residual panels.
- Executed all 23 cells after switching to a real ID8910 workflow-test run: zero errors, seven static PNG outputs; its redshift and native exposure set differ from ID3700, and the failed imaging prior is clearly labeled.
- Ran the new GPU fitting path on real ID8910 pixels with a deliberately reduced 7-by-7 map, 21-by-21 quadrature, 500 km/s grid spacing and an explicit isotropic inclination prior. It completed checkpointing, bidirectional/continuation searches and candidate rendering. Two coarse trial solutions remain flagged as nonstationary. This is a software exercise, **not a scientific velocity result for ID8910**.
- Rendered the previously uncached 301 km/s ID3700 trial through the new worker using the original frozen model/input hashes.
- Ran the generalized full-image F150W fit on real ID8910. It reached `r_eff` near the 45-pixel upper bound, had 17 divergences and maximum Rhat 1.0115, and **failed** the imaging-prior gate. Its model/residuals are saved and displayed, but this result is not used as an inclination prior. ID8910 currently needs an improved imaging model or an explicitly selected isotropic prior for exploratory grism fitting. The existing accepted ID3700 imaging results remain usable.
- **15 targeted tests passed**, with two existing library warnings. These cover source identity, missing inputs, refusing another source's imaging prior, invalid thickness pairs, stale candidate caches, inspection-note provenance, stopping interrupted workers, and the underlying real-model response/gradient tests.

Executed validation notebooks and logs are in `../results/2026-10-16_notebook_validation/`. The small ID8910 workflow run records its exact earlier code snapshot; later candidate-provenance metadata changes mean it is intended for saved-result inspection, not resuming with the current source. A production ID8910 run uses a new directory and the full default map.

One automated ID3700 execution launched alongside another notebook kernel stalled and was interrupted. A subsequent sequential execution completed all cells. The validation helper now prints per-cell progress and has a 120-second cell timeout; the stall's cause was not established.

Implementation: [real_workflow.py](../dingo/real_workflow.py) handles source discovery and workers, [real_review.py](../dingo/real_review.py) handles viewing, and [native_source_grid.py](../diagnostics/native_source_grid.py) runs generalized fits. [validate_real_notebook.py](../diagnostics/validate_real_notebook.py) executes the notebook with fitting disabled for reproducible UI checks.
