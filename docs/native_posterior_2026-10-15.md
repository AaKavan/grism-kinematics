# Joint R+C posterior pilot

**October 16 notebook:** [real_source_kinematics.ipynb](../notebooks/archive/real_source_kinematics.ipynb)
provides a source-ID entry, real F150W/native-exposure inspection, resumable joint
velocity fitting, interactive model/residual comparisons and saved visual notes.
Use the **grism** kernel; the defaults load ID3700's existing results. See the
[notebook guide](real_source_notebook.md) for running another source.

**October 16 real-data sampler:** the notebook now also has a separate section 7 for
real-source kinematic MCMC, chain diagnostics and corner plots. Its implementation
and first ID3700 pilot are in [the real posterior report](real_posterior_2026-10-16.md).
The earlier posterior results below remain synthetic tests and are not substituted
for real-source chains.

The subsequent application to **real ID3700 observations**, using a velocity grid and F150W PySersic orientation constraint, is documented in [real_ID3700_grid_2026-10-15.md](real_ID3700_grid_2026-10-15.md). The posterior results below remain synthetic-data tests.

**Real-data update:** the requested soft stellar-center and free-center comparisons prefer deprojected `Vrot(0.5 arcsec)` of **277.95 and 278.15 km/s**, respectively, using all 18 R+C exposures jointly. Both preferred centers lie approximately 0.02 arcsec from the F150W stellar centroid. Worse local solutions remain near 431 and 525 km/s, so the initial-condition concern is real. The linked report includes their scores, residual panels, illustrative rotation/mass-proxy curves and response/morphology sensitivity checks. These are conditional fits, not real-data posterior intervals. Faint off-source/edge emission prevents treating the fitted outer flux radius as a measured disk extent; mass profiles still require source-assignment and radial-information validation.

This implements the next step proposed in the [R/C comparison](grism_comparison_2026-10-14.md): sample a shared positive emission map, its hyperparameters, rotation curve and inclination together. Four NumPyro NUTS chains begin in the two competing kinematic branches. This is an experiment on **synthetic data**, using real ID3700 detector geometry, uncertainty maps and masks. It is not a mass measurement of ID3700.

The experiment began on October 14; results are stored under `../../results/2026-10-14_posterior/`. The source injection is unchanged from `2026-10-14_grism_comparison/nonlinear_RC_seed481`: projected speed at 0.5 arcsec = 210 km/s, turnover radius = 0.22 arcsec, inclination = 0.9 rad. Twelve exposures (six R and six C) enter the likelihood; six remain outside this posterior fit. Those six were already inspected in earlier model exploration, so they are not a pristine final validation set.

**Result:** the final chains pass the convergence gate and do not retain their starting velocities. However, their median outer mass is low in this realization. The completed October 15 controls below localize much of the discrepancy to inferring the unknown emission shape: changing its correlation prior gives outer mass medians of **0.50–0.66 times the injection**, versus **1.01** when the exact shape is supplied. Grid interpolation contributes a smaller shift. This is a successful sampler pilot and a diagnosis of model sensitivity, not yet a validation of accurate mass recovery.

## Completed result

The final run, `conditional12_uniform_depth11`, has four chains with 500 retained draws each, following the original 800-step adaptation and 1,000-draw trajectory-limit experiment. Only the new 500 draws per chain enter the following summaries. It has maximum rank R-hat **1.00457**, minimum bulk ESS **1,564**, minimum tail ESS **1,298**, zero divergences, zero trajectory-limit hits and BFMI 0.85–1.04. The additional sampling took **453 seconds**. The two low starts were 211 and 211 km/s; the two high starts were 497 and 503 km/s.

The projected reference speed is **157 km/s**, with a central 90% interval of **123–204 km/s**. The injected value is 210 km/s. Turnover radius is 0.117 arcsec (0.050–0.244), compared with the injected 0.22 arcsec. No retained draw has turnover radius above 0.8 arcsec. Under these specified priors, the old high-speed likelihood branch does not retain the chains. This does not prove that every possible posterior mode was found.

![Movement away from starting branches](../../results/2026-10-14_posterior/conditional12_uniform_depth11/initialization.png)

The profiles below include joint inclination uncertainty. Parentheses are central **90% pointwise posterior intervals**. The last column divides the rotation-only mass by the injected rotation-only mass at the same radius.

| Test radius, arcsec | Injected deprojected speed, km/s | Posterior deprojected speed, km/s | Mass / injected mass |
|---|---:|---:|---:|
| 0.10 | 98.9 | 107.7 (80.5–139.4) | 1.19 (0.66–1.99) |
| 0.15 | 138.7 | 137.1 (110.5–168.0) | 0.98 (0.63–1.47) |
| 0.20 | 171.1 | 156.7 (130.5–190.9) | 0.84 (0.58–1.25) |
| 0.30 | 217.5 | 181.4 (149.0–222.4) | 0.70 (0.47–1.05) |
| 0.50 | 268.1 | 204.6 (159.6–264.6) | 0.58 (0.35–0.97) |

At 0.5 arcsec, the absolute rotation-only proxy is `4.10e10 Msun` (`2.49e10–6.86e10`), compared with injected `7.04e10 Msun`. The injected mass lies at the 95.6th percentile, just outside the central 90% interval. **One such draw does not establish systematic bias or a coverage failure rate.** It does establish that agreement among chains should not be mistaken for accurate outer point estimates. The 0.15–0.2 arcsec region is recovered more closely in this particular realization; it is not yet a generally certified radial range.

Reweighting the same draws to a log-uniform turnover prior changes the median projected reference speed from 157 to **149 km/s** and the median outer mass from `4.10e10` to **`3.70e10 Msun`**, about 10% lower. The reweighted outer central 90% interval is `2.28e10–6.16e10 Msun`. Kish weight ESS is 1,558 out of 2,000 draws, the largest normalized weight is 0.00243, and no chain dominates. This comparison does not resolve the low outer mass; it motivated the emission-map investigation below.

![Profiles and turnover-prior sensitivity](../../results/2026-10-14_posterior/conditional12_uniform_depth11/turnover_prior_sensitivity.png)

For 16 selected draws per chain, held-out chi-square medians range from **37,439 to 37,446**. The injected line gives 37,418 on those same noisy pixels; the projected noise rank is 37,020. The posterior predictions are close in image space even though the outer mass medians differ considerably from the injection. Total pixel chi-square is dominated by noise and cannot by itself validate the velocity profile. The Gaussian replicated-residual fractions in `predictive.json` use only 16 replicates per chain and should not be treated as precise tail probabilities or detection significances.

![Examples of prior and posterior emission maps](../../results/2026-10-14_posterior/conditional12_uniform_depth11/map_draws.png)

## Emission-map prior investigation, October 15

This follow-up keeps **joint R+C inference throughout**: the same twelve training exposures, same noisy SCI arrays, same kinematic and inclination priors, same continuum projection and same fixed instrument response. There are no single-direction fits in this investigation. Artifacts are in `../../results/2026-10-15_map_prior/`.

The comparison brackets the baseline log-field correlation length of 0.10 arcsec with fresh four-chain NUTS fits at **0.05 and 0.20 arcsec**, each with its own 800-step adaptation from the original low/high branches. A known-map diagnostic uses the exact injected brightness shape on the 97x97 injection quadrature, while still sampling flux, speed, turnover, angle and inclination. Knowing the true shape is unavailable in real data; this control isolates the combined effect of learning the map and representing it on a coarser control grid.

Scalar hyperprior sensitivity uses the existing baseline draws, changing one prior at a time: map-amplitude median 1 or 4 instead of 2, map-amplitude log standard deviation 1 instead of 0.5, and envelope-width median 0.125 or 0.5 arcsec instead of 0.25. These are importance-reweighting estimates, not fresh MCMC fits. In contrast, reweighting the spatial correlation changes has effectively only 1–2 weighted draws out of 2,000; those estimates are rejected; the correlation comparison requires fresh fits.

The scalar checks are complete: the outer mass median changes by at most about **4%**, with Kish weight ESS between 1,500 and 1,907. These specific hyperprior changes do not explain the deficit. Each row changes only the stated prior; these weight ESS values ignore chain autocorrelation.

| Scalar hyperprior change | Outer mass / injected mass, median (central 90%) | Kish weight ESS / 2,000 |
|---|---:|---:|
| Field-amplitude prior median 1 instead of 2 | 0.562 (0.342–0.941) | 1,602 |
| Field-amplitude prior median 4 instead of 2 | 0.602 (0.369–1.035) | 1,537 |
| Field-amplitude log standard deviation 1 instead of 0.5 | 0.562 (0.341–0.939) | 1,500 |
| Envelope-width prior median 0.125 instead of 0.25 arcsec | 0.585 (0.355–0.976) | 1,907 |
| Envelope-width prior median 0.5 instead of 0.25 arcsec | 0.580 (0.352–0.972) | 1,895 |

The **exact known-map control is complete and converged**: projected reference speed is **212.4 km/s (180.7–253.9)** and outer rotation-only mass is **1.007 times the injected mass (0.739–1.405)**. It uses the identical noisy R+C pixels and kinematic priors. Its maximum R-hat is 1.00228, minimum bulk/tail ESS 4,144/2,491, with no divergences or trajectory-limit hits. Thus this noise realization can yield an accurate outer mass when the brightness shape is supplied. This localizes the discrepancy to learning/representing the unknown map and its interaction with the velocity field; it does not yet isolate prior preference from marginalization or grid error.

The baseline sampled emission map has second-moment axis ratio **0.718 (0.672–0.761)**, versus **0.616** for the injected brightness. Its axis ratio and projected rotation have correlation **-0.578**: rounder inferred maps tend to have lower inferred rotation. These are brightness moments including faint wings within the finite field, not an inclination measurement. Sampling the true brightness on the 17x17 controls and interpolating it on 49x49 quadrature produces axis ratio **0.626**, much closer to the injection. At the true kinematics its total continuum-projected squared image error is **4.24** over the training exposures, versus **0.276** for a 33x33 / 97x97 representation. These are constructive representations, not optimized fits. The known-map posterior on the same 17/49 grid below measures the resulting kinematic shift directly.

The known-map interpolation control is also complete and converged (maximum R-hat 1.00270, minimum bulk/tail ESS 4,944/2,456, no divergences or trajectory-limit hits). Its projected reference speed is **201.5 km/s (171.0–239.6)**, and outer mass ratio is **0.888 (0.649–1.249)**. This is a measurable discretization effect, but is much smaller than the free-map deficit:

| Same noisy joint R+C pixels | Projected speed median, km/s | Outer mass / injected mass, median (central 90%) |
|---|---:|---:|
| Exact injected shape known, 97 grid | 212.4 | 1.007 (0.739–1.405) |
| Injected shape sampled onto 17 controls / 49 quadrature | 201.5 | 0.888 (0.649–1.249) |
| Unknown map, 0.05 arcsec prior | 145.6 | 0.502 (0.317–0.826) |
| Unknown map, baseline 0.10 arcsec prior | 156.7 | 0.582 (0.354–0.974) |
| Unknown map, 0.20 arcsec prior, smaller-step follow-up | 167.7 | 0.659 (0.409–1.109) |

Here “outer” means the 0.5 arcsec test radius. The medians are comparisons between different conditional posteriors, not an additive decomposition of bias. Nonetheless, interpolation alone does not explain the large shift: supplying the shape on the **same** 17/49 grid recovers substantially more of the true rotation.

The **0.05 arcsec fit has converged**: maximum R-hat 1.00531, minimum bulk/tail ESS 1,226/889, no divergences or trajectory-limit hits, and BFMI 0.77–1.12. Its outer mass median is 14% below the baseline, and its brightness axis ratio is rounder: **0.744 (0.700–0.785)**, with axis-ratio/speed correlation **-0.524**. The shorter correlation length therefore does not repair mass recovery. Its four chain speed medians are 145.7, 143.9, 146.8 and 146.3 km/s, despite initial speeds of 211.5, 496.9, 211.0 and 503.3 km/s. This directly distinguishes the current low estimate from simply retaining the starting velocity.

The original 0.20 arcsec fit completed 500 draws per chain with maximum R-hat 1.00589, minimum bulk/tail ESS 1,025/1,226 and no trajectory-limit hits, but **one divergent transition**. It therefore failed the reporting gate and supplies no pooled intervals. The separate **smaller-step follow-up is complete and passes** after 1,000 new draws per chain: maximum R-hat **1.00233**, minimum bulk/tail ESS **2,047/2,301**, zero divergences or trajectory-limit hits, and BFMI 0.90–1.00. It retains the same posterior and adapted covariance and halves only the integration step sizes. The original divergent run is preserved without deleting any transitions and is excluded from pooled comparison curves.

For this passing 0.20 arcsec fit, projected reference speed is **167.7 km/s (132.5–218.5)** and outer mass is **`4.64e10 Msun` (`2.88e10–7.81e10`)**, or **0.659 (0.409–1.109)** times the injection. The mass median is 13% above the baseline but still 34% below the injected value; the truth is at the 90.75th percentile and inside the central 90% interval. Its brightness axis ratio is **0.692 (0.647–0.736)**, with axis-ratio/speed correlation **-0.569**. Longer correlation permits a more elongated inferred map and raises rotation, but this test does not establish that 0.20 arcsec is the correct prior. Across 0.05–0.20 arcsec, changing this one prior setting changes the outer mass median by 31%. The four chain speed medians in the passing follow-up are 168.3, 165.9, 167.9 and 169.2 km/s.

The withheld-image chi-square medians are 37,449–37,452 across chains for 0.05 arcsec, 37,439–37,446 for the baseline, and 37,432–37,438 for the passing 0.20 arcsec fit; the injected line gives 37,418. These checks use only 16 selected draws per chain and the same previously inspected withheld exposures. The image predictions change modestly while the mass changes substantially; neither total pixel chi-square nor these small predictive samples validate mass accuracy.

A separate **numerical representation check** compares two ways to interpolate the same known brightness controls, at the true kinematics. Interpolating log brightness and then exponentiating reduces the 17/49 training-image squared error from **4.24 to 1.31**, and changes the represented axis ratio from **0.626 to 0.617**, close to the injected 0.616. Refining only the quadrature to 97 with the same 17 log-brightness controls gives error 1.57, so quadrature refinement alone is not a monotonic remedy. A finer 33/97 brightness representation gives error 0.276. The exact 97/97 representation reproduces the saved injection to `1.1e-14` of a pixel's quoted error. These are deterministic image checks, not posterior fits or evidence that a mass bias has been fixed. The log-brightness option is a useful next candidate because it adds no map coefficients.

**Interpretation and next changes:** the completed controls identify a coupling between emission shape and rotation. The present circular envelope plus residual field can represent an elongated source, but the free-map posterior favors rounder brightness and lower rotation in this realization. Changing only the scalar amplitude/width hyperpriors is insufficient. Shortening the spatial correlation worsens the outer point estimate; lengthening it helps, but the median remains low. All three free-map fits favor rounder brightness than the injection. Knowing the brightness restores much more rotation even on the same coarse grid. This supports testing an independently sampled elliptical sky-plane envelope with a flexible residual field, alongside improved interpolation. A useful implementation would evaluate the smooth elliptical envelope directly on the integration grid and interpolate the residual log field before exponentiation, keeping the number of residual coefficients fixed. These are proposed changes that still require posterior recovery tests; a prior should not be chosen because it reproduces this one injection.

The brightness ellipse must retain its own angle and axis ratio. In this mock the brightness second-moment angle is about -0.079 rad, while the kinematic angle is 0.9 rad, in the same east-to-north convention. Tying them together, or converting the brightness axis ratio directly into inclination, would impose a false constraint on this test. Any disk-alignment prior in a science application needs separate justification.

The next validation should first compare the new representation and envelope on this fixed dataset, then repeat over independent noise realizations and a range of brightness shapes and inclinations. Priors should be specified before evaluating recovery, and accuracy and interval coverage should be measured over that ensemble. These results remain consistent with joint R+C containing useful rotation information; they do not yet establish accurate unknown-map mass profiles or a precision advantage over a single grism direction.

![Joint R+C profiles under different map assumptions](../../results/2026-10-15_map_prior/comparison/map_prior_profiles.png)

![Emission-map shape and projected rotation](../../results/2026-10-15_map_prior/comparison/map_shape_rotation.png)

## What is sampled

The detector likelihood uses the same native-pixel forward projection as the preceding Torch experiments. The new JAX implementation copies the calibration coefficients, quadrature, deposition, kernels and fixed continuum projection. It does not resample observed images. All calculations use float64.

The old free-map optimization has been replaced by an explicit, proper generative prior. A circular Gaussian envelope plus a smooth log-brightness field defines a positive 17x17 control map. Bilinear integration on the 49x49 source quadrature normalizes this shape; total line flux is a separate parameter. All 288 nonconstant field coefficients are sampled, along with the envelope width and field amplitude. No emission map is held at its best-fit value inside MCMC.

| Quantity | Prior in the baseline run |
|---|---|
| Projected speed at 0.5 arcsec | Uniform(0, 1500) km/s |
| Turnover radius | Uniform(0.02, 2) arcsec |
| Kinematic position angle | Uniform(-pi, pi) rad |
| Inclination | Normal(0.9 rad, 5 degrees), truncated to 5–85 degrees |
| Log total flux relative to the injection's flux unit | Normal(0, 2) |
| Log field amplitude | Normal(log(2), 0.5) |
| Log envelope width in arcsec | Normal(log(0.25), 0.5) |
| Whitened field coefficients | 288 independent standard normals |

The finite-grid field uses nonconstant DCT modes, with variance proportional to `(1 + ell^2 lambda)^-2` for discrete Laplacian eigenvalue `lambda`, normalized to unit mean pointwise variance. The correlation length `ell = 0.10 arcsec` is fixed for this pilot. This is a specified finite-grid prior, not a claim of exact continuum Gaussian-process equivalence or grid invariance.

The inclination prior is **imposed external information for a toy problem**. It has not been inferred from the galaxy's direct image, and no additional isotropic-orientation factor is silently multiplied into it. An isotropic alternative, uniform in cos(i), is available through `--inclination-error 0`. The broad flux scale is also tied to this injection; real-source flux and inclination priors need their own justification.

Continuum coefficients are removed with a fixed weighted projection. With fixed errors and masks, the remaining likelihood has the same dependence on line parameters as integrating linear continuum coefficients with a flat measure; equivalently, it is a Gaussian likelihood on the continuum-orthogonal data. Constants omitted here prohibit using this score as a model evidence across different continuum models.

## Starting conditions and diagnostics

Two chains start near the previously found 211 km/s branch and two near the 499 km/s, large-turnover branch. Initial zero map pixels are floored only when constructing finite log-field starting values. That floor is not a constraint on posterior predictions.

The sampler uses four vectorized chains on one GPU, 800 adaptation iterations, and a full covariance matrix across kinematic and map parameters. Code snapshots, input hashes, prior draws, warmup draws, retained draws, sampler statistics and adapted states are saved. Checkpoints include random keys and all retained arrays. `--resume` continues an identical target without repeating adaptation. `--from-checkpoint` starts a separate retained sample set from an identical target's adapted state, permitting a larger trajectory limit while preserving the old experiment.

The reporting gate requires all sampled parameters **and derived radial profiles** to have rank-normalized split R-hat < 1.01, bulk and tail ESS >= 400, no divergences, fewer than 1% of transitions at the trajectory limit, and BFMI > 0.3 in every chain. Per-chain summaries are always saved; pooled intervals are withheld unless the gate passes. These checks are necessary for this pilot, not proof of correct mode probabilities or frequentist interval coverage. See [rank-normalized convergence diagnostics](https://arxiv.org/abs/1903.08008) and the [NumPyro MCMC interface](https://num.pyro.ai/en/stable/mcmc.html).

## Numerical validation and runtime

JAX and Torch agree at both old fitted branches to better than `3e-15` of a pixel's quoted error, with zero reported log-likelihood difference at the saved precision. Tests also compare gradients, asymmetric kernels, dispersion signs, batch evaluation and masked continuum strips. Additional checks cover map normalization, proper field variance, inclination sampling, propagation into mass and rejection of separated chains.

The first full run, `conditional12_uniform_full`, completed 1,000 retained draws per chain. Maximum R-hat was 1.00327, minimum bulk ESS 3,143, minimum tail ESS 3,138, with zero divergences. All chains moved away from their starting branches into the same lower-speed region. However, 23.525% of transitions hit tree depth 9, so the reporting gate correctly failed. The completed follow-up `conditional12_uniform_depth11` kept the same target and adapted covariance, raised the limit to 11, and collected a fresh sample set. Its original depth-9 draws remain separate. The resulting distributions are consistent across these two trajectory settings.

The depth-9 run took 807 seconds for warmup and 889 seconds for 1,000 retained draws per chain (about 28 minutes total, excluding setup). A compiled single-chain potential/gradient evaluation takes roughly 3 milliseconds on the local GPU. A NUTS draw needs many such evaluations; this gradient benchmark is not the time per posterior draw.

An initial attempt stalled in XLA GEMM-fusion compilation with an almost idle GPU and no retained draws. The driver now defaults to `--xla_gpu_enable_triton_gemm=false`, retaining float64 arithmetic while using the alternative GEMM backend. The setting is recorded and explicit user-supplied XLA flags are respected. Persistent compilation caching is enabled. The failed attempt is preserved as `conditional12_uniform_compile_stalled`; it is excluded from posterior results. See [JAX compilation caching](https://docs.jax.dev/en/latest/persistent_compilation_cache.html).

The full repository suite now passes **207 tests**, with three existing warnings (`JAX_PLATFORMS=cpu PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q`). New checks cover the known-map prior and flux scaling, hyperprior density ratios, correlation-field transport, brightness moments and preservation of the target/state when reducing integration steps. The earlier small smoke run exercised checkpoint continuation. The baseline final run exercised reuse of the full adapted covariance with a different trajectory limit.

The driver now accepts `--warmup-tree-depth` to limit adaptation trajectories separately from retained sampling, while transferring the adapted state and retaining the same convergence gate. A four-chain workflow check (50 warmup steps at depth 5, then 16 draws at depth 6) completed and exercised the larger sampling limit. Its deliberately inadequate chains failed the reporting gate, as expected; it contributes no scientific intervals. Two smaller two-chain checks terminated in native code with exit 139, on CPU and GPU; their cause is unresolved. An initial full-suite invocation also crashed in native JAX compilation, while subsequent complete runs passed. The failed artifacts are preserved and excluded. The four-chain smoke check validates the state-transfer workflow, not a speedup or scientific convergence.

The 0.05 arcsec scientific fit used depth 11 for both warmup and sampling: adaptation took 5,649 seconds, followed by only 117 seconds for 500 draws per chain. This makes adaptation the main runtime target in that fit. Concurrent GPU work and a recorded temporary pause of the 0.20 arcsec run prevent treating these wall times as clean throughput benchmarks. The original 0.20 arcsec adaptation recorded 7,673 seconds, including the documented 2,018-second pause. The optional smaller warmup limit is a candidate for future full runs; it did not change the targets or settings of the correlation experiments reported here. A new `--from-checkpoint --step-size-scale 0.5` option addresses divergent integration by reducing fixed step sizes after adaptation, without changing the covariance or posterior target. Its unit check verifies state preservation and that subsequent transitions retain the reduced step. The passing 0.20 arcsec follow-up collected 1,000 draws per chain in 474 seconds using this option, with no repeated adaptation.

## Rotation and mass interpretation

Each retained draw supplies a projected curve, a deprojected curve obtained with that draw's inclination, and the spherical-equivalent rotation-only quantity `M_rot(r) = r V_rot(r)^2 / G`. Inclination and turnover covariances therefore propagate into the entire profile. Conversion to solar masses explicitly uses Planck18 and the geometry configuration's redshift, 1.1231; neither redshift nor distance uncertainty is sampled.

These are **rotation-only mass proxies**, not a full pressure-corrected dynamical mass model. A flattened mass distribution need not obey the spherical enclosed-mass formula. Intrinsic line width is fixed at 45 km/s; source center, registration, instrument response and errors are also fixed, and the mock has no contamination. The listed radii (0.05–0.5 arcsec) are diagnostic coordinates, not an established resolved radial range for a real galaxy. Bands are pointwise, not simultaneous confidence regions for an entire curve.

Changing the turnover prior from uniform to log-uniform on identical support can be checked by bounded importance weights proportional to `1/r_turn`. The sensitivity helper records weight concentration and per-chain results. Its Kish weight ESS does not account for chain autocorrelation, and reweighted summaries are not an independent posterior-sampling validation. This check leaves the emission-map and inclination assumptions unchanged.

The priority after this investigation is to improve and validate emission-map inference over multiple noise draws and source shapes. All current fits remain joint R+C. A controlled comparison against individual directions can follow once the joint model passes recovery checks. A narrow, well-mixed posterior can still give inaccurate mass estimates under its morphology assumptions. The present experiment does not establish a mass-precision advantage over GEKO or measure population-level coverage.

## Reproduction and artifacts

The optional dependency group is `native-posterior` (JAX, NumPyro and ArviZ Stats). The local run uses JAX 0.11.2 and NumPyro 0.22.0. A GPU-enabled JAX installation is needed for `--device cuda`.

```bash
python -m diagnostics.native_posterior_check \
  --run-dir ../results/2026-10-14_grism_comparison/nonlinear_RC_seed481 \
  --mode-dir ../results/2026-10-14_grism_comparison/RC_released_modes \
  --output ../results/2026-10-14_posterior/new_run \
  --warmup 800 --samples 1000 --chunk 250 --dense-mass full --max-tree-depth 11
```

The exact absolute paths and arguments for the completed experiments are in their `posterior.json` files. Resume requires the recorded input path strings and target settings. Use a fresh output directory for a new experiment. `--all-exposures` includes all eighteen exposures and consumes the remaining holdout; it is a different target and requires new adaptation.

After a run finishes, repeat its `--run-dir`, `--mode-dir` and `--output` with `--predictive-only` to render selected draws against all exposures. This writes per-chain projected-residual checks and example emission maps. After the convergence gate passes, `python -m diagnostics.native_posterior_sensitivity OUTPUT` writes the turnover-prior reweighting comparison.

The October 15 controls can be reproduced with the helpers below, using a new output directory for each fresh run. The initial two correlation fits used the main posterior command above with `--correlation 0.05` or `--correlation 0.20`, `--warmup 800 --samples 500 --chunk 250 --dense-mass full --max-tree-depth 11`, and the same source/mode directories. They did not use the new warmup-depth option. The 0.20 arcsec follow-up adds `--from-checkpoint ../results/2026-10-15_map_prior/correlation020 --step-size-scale 0.5`, uses a fresh `correlation020_halfstep` output directory, and collects `--samples 1000`; the source/mode paths and target settings must exactly match the saved checkpoint. For the local artifacts those input paths are absolute, as recorded in `posterior.json`.

```bash
python -m diagnostics.native_oracle_check \
  --baseline ../results/2026-10-14_posterior/conditional12_uniform_depth11 \
  --output ../results/2026-10-15_map_prior/new_known_map \
  --warmup 500 --samples 1000 --max-tree-depth 10
# Add --interpolated-map for the known-shape 17/49 control.

python -m diagnostics.native_map_representation \
  --baseline ../results/2026-10-14_posterior/conditional12_uniform_depth11 \
  --output ../results/2026-10-15_map_prior/representation
```

The comparison helper `diagnostics.native_map_prior_comparison` accepts `--baseline`, a `--runs` list and `--output`. It checks matching source data and priors, preserves each run's convergence status and pools only passing fits. Its `comparison.json` stores the full radial profiles, brightness moments, scalar reweightings and rejected correlation-weight overlap diagnostics. Each helper saves its own source snapshot and hashes. All five scientific posterior runs from this follow-up completed: `correlation005`, `correlation020` (failed gate), `correlation020_halfstep`, `known_map_noisy`, and `known_map_interpolated`. The input hashes and snapshots were verified, and the final comparison plots were visually checked. The three new helper modules and `runtime_scheduling.json` document the October 15 controls and execution details.

Main files: `dingo/native_jax.py`, `dingo/native_posterior.py`, `diagnostics/native_posterior_check.py`, `diagnostics/native_posterior_predictive.py`, `diagnostics/native_posterior_sensitivity.py`, and `tests/test_native_posterior.py`. This remains a separate experimental path; the legacy `dingo.run` interface has not been switched to it.
