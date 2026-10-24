# Real-source joint R+C posterior — October 16, 2026

**Follow-up on intrinsic dispersion:** [the real-data fixed-dispersion comparison](real_dispersion_2026-10-16.md) refits all nuisance parameters at five widths and tests separate spectral/spatial response changes. All 50 optimizations passed their stationarity checks; each scenario favors the zero-width boundary among tested points. This supports treating dispersion as an upper limit, conditional on calibration and morphology assumptions. Notebook sections 10–11 display these results. The completed MCMC samples and their original probability target are unchanged.

The real-source notebook now has a separate **section 7: kinematic MCMC**. Sections 4–6 still inspect the optimized velocity grid. This implementation uses real SCI/ERR/DQ pixels from the saved source's native exposures; it does not borrow synthetic chains or another galaxy's morphology prior.

## Model and probability target

One shared positive emission map and one inclined arctangent velocity field predict every R/C exposure. The physical projection, finite grid, log-brightness interpolation, response kernels, masks and calibration match the saved grid run. The posterior samples deprojected `Vrot(0.5 arcsec)`, turnover, inclination, kinematic angle, systemic velocity, intrinsic line width, kinematic center, independent emission-envelope width/shape/angle, all smooth map coefficients, and per-exposure registration. The ID3700 model has 440 map coefficients, 36 registration coordinates and 11 other coordinates: **487 sampled dimensions**.

The emission map is not held fixed inside MCMC. Its smoothness length (0.10 arcsec) and field amplitude (2.0), the response, and the quoted noise/masks remain conditional assumptions inherited from the current grid. Their uncertainty is not included in this first posterior.

The finite-support priors follow the grid's parameter coordinates and penalties: log-uniform turnover on 0.02–2 arcsec; uniform deprojected speed on 0–1000 km/s; the exact saved imaging-informed inclination density; bounded Gaussian systemic velocity, width and center; a bounded Gaussian on log envelope width; uniform envelope axis ratio; normal field coefficients and registration offsets. Physical angles have uniform priors over a complete cycle. The adopted center treatment is read from the saved grid, including catalog, stellar or free options. All prior definitions are recorded in `posterior.json`.

**The likelihood is not simply the negative grid score.** The grid profiles each exposure's nonnegative line amplitude. The posterior integrates each amplitude under an explicit independent `HalfNormal(100 DN/s)` prior, where amplitude scales the source's unit-normalized line template. This is broad compared with the approximately 2.8–3.7 DN/s fitted amplitudes of the two ID3700 starting branches; it remains an assumption to test. The model-dependent normalization term must be included.

For a continuum-projected, error-whitened unit template `t` and data `d`, let `a = sum(t²) + 1/s²`, `b = sum(t*d)`, and `s = 100 DN/s`. The exact log integral relative to a zero-line likelihood is

`log(2/s) - 0.5 log(a) + 0.5 b²/a + log Phi(b/sqrt(a))`.

This expression is tested against direct numerical integration for positive, negative and zero amplitude evidence. With no template information it integrates to one. The conditional amplitude distribution is `Normal(b/a, 1/sqrt(a))` truncated below zero. The source map has a fixed unit integral, which makes the amplitude prior's units and normalization explicit.

The quadratic continuum still uses fixed-design flat-measure integration. Its determinant is independent of the sampled parameters because errors, masks and continuum design are fixed; it therefore contributes only a constant to this posterior. This construction must not be used as an absolute model evidence.

## Sampling and saved outputs

The runner uses [NumPyro NUTS and adapted-state continuation](https://num.pyro.ai/en/stable/mcmc.html), four vectorized GPU chains, and a dense adaptation block for the low-dimensional physical parameters. Map and registration dimensions use diagonal adaptation. Saved grid fits seed the chains; they do not constrain subsequent draws. Initialization selects the best stationary grid branch and a separated stationary alternative, preferring a displaced center when available. For ID3700 these are 278 and 483 km/s, each used by two independently jittered chains.

Outputs live in `../results/real_sources/ID3700/posterior_pilot_v1/` for the first pilot:

- `posterior.json`: source, priors, settings, checksums, starts and diagnostics.
- `warmup.npz`: adaptation draws, excluded from inference.
- `samples.npz`, `sampler_stats.npz`: retained chains and NUTS diagnostics.
- `sampler_state.pkl`: atomic local adapted-state checkpoint, including random keys and retained arrays.
- `trace.png`, `corner.png`: chain movement and parameter correlations, colored separately by chain.
- `profiles.npz`, `profiles.png`: per-draw rotation/mass profiles at explicitly illustrative radii.
- `chain_*_last_prediction.*`: one actual final draw per chain, using conditional-mean amplitude and continuum for visual residual inspection. These are not posterior predictive uncertainty bands.

Code, input and library versions are checked before resuming. Run directories are locked against concurrent workers. Checkpoints are written after warmup and each retained block. A notebook interrupt stops the worker. An interruption during the first warmup requires a new run, because no adapted state has yet been saved. Increasing the retained draw target can resume the same settings; changing tree depth or warmup requires a fresh run/tag.

`MCMC_FROM_POSTERIOR` in the notebook optionally reuses a completed pilot's adapted states with a longer trajectory limit, in a fresh output directory. The dedicated [continuation driver](../diagnostics/native_real_posterior_extend.py) checks code, inputs, priors and library versions, then verifies the reconstructed potential **and its gradient** against the saved state before transitioning. It retains the adapted covariance and step sizes and starts new retained arrays; the pilot is not pooled into the follow-up. Warmup and chain count are inherited. This avoids repeating adaptation just to investigate a tree-depth cap. It does not by itself repair multimodality or bad adaptation.

Sampling checks include rank-normalized Rhat, bulk/tail effective sample sizes, divergences, energy behavior and tree-limit saturation, following the concerns described in [Stan's diagnostic guide](https://mc-stan.org/learn-stan/diagnostics-warnings.html). The gate checks **all map and registration coefficients as well as physical parameters and radial profiles**. It requires Rhat below 1.01, bulk/tail ESS at least 400, zero divergences, less than 1% tree-limit saturation, and BFMI above 0.3. The corner plot is available even when checks fail, explicitly labeled diagnostic. Pooled quantiles and credible bands are withheld on failure; per-chain medians remain visible to expose disagreement. Passing diagnostics is necessary, not proof that disconnected modes have correct posterior weights.

## Pilot and validation

The first real ID3700 pilot completed 4 chains, 250 warmup steps and 100 retained draws per chain, with 50-draw checkpoints and maximum tree depth 6. It **failed convergence**: maximum rank Rhat 2.421, minimum bulk ESS 5.33, minimum tail ESS 12.05, zero divergences, and 100% of retained draws at the 63-step trajectory limit. Warmup took 150.7 seconds; the two retained blocks took 33.0 and 25.7 seconds, excluding setup/plotting. Chains moved within their starting families but did not mix between the roughly 280 and 470 km/s regions. The latter also has a displaced northward coordinate near the negative center bound. These chain populations are not valid posterior mode weights, and no pooled velocity/mass interval is reported.

The baseline is specifically the saved grid's **catalog-center prior**, Gaussian scale 0.15 arcsec and ±0.3-arcsec coordinate bounds. It is not the separate soft-stellar-center or free-center sensitivity run. Center-prior comparisons remain important before a mass claim.

A follow-up in `posterior_depth9_v1/` starts from the pilot's adapted states with maximum tree depth 9 and a fresh set of 100 draws per chain. Its reconstructed potential agreed with the saved state to `9.1e-13`, and its gradient to `1.5e-12`. This tests whether longer trajectories reduce the limitation; it does not presume convergence.

**Completed follow-up:** both chains that originally started at 483 km/s left the high-speed/displaced-center region and joined the approximately 280 km/s, 78-degree, centrally located region. Their transition occurs around draws 40–60 of the new sample set. Across the last 20 draws, the four chain median speeds are 279.5, 276.1, 286.3 and 279.5 km/s. These short-tail medians only describe the chain movement; they are not a selected posterior estimate or uncertainty interval.

| Diagnostic | Short pilot, depth 6 | Fresh follow-up, depth 9 |
|---|---:|---:|
| Retained draws per chain | 100 | 100 |
| Maximum rank Rhat | 2.421 | 1.582 |
| Minimum bulk ESS | 5.33 | 7.37 |
| Minimum tail ESS | 12.05 | 17.30 |
| Divergences | 0 | 0 |
| Fraction at trajectory limit | 100% | 23.5% |
| Mean leapfrog steps per draw | 63 | 315 |
| Sampling checks passed? | No | No |

The deeper run's cap hits are concentrated in chain 1 (94% of its draws); the other three chains do not hit that cap. Its retained blocks took 215.4 and 195.3 seconds, including first-block compilation. All four final states have speeds around 276–286 km/s and centers close to the central solution. The complete retained sample set still contains substantial transient movement, so its global diagnostics correctly fail and **pooled mass/velocity credible intervals remain absent**.

This is evidence that the sampler can escape the supplied high-speed starting solution on the actual observations. It is not proof that all modes have been found, that mode weights are correct, or that the scientific model is adequate. The proper next sampling stage is a fresh, longer retained sample set after adequate adaptation/exploration, with enough trajectory length for the slow chain. Reassess convergence for every sampled map/registration coordinate as well as physical/radial outputs. Then compare the stellar-center and freer-center assumptions and response/morphology sensitivities. Do not discard an inconvenient chain merely to make a corner plot look converged.

**Longer-run setup:** section 7 now specifies four chains, 1,500 fresh warmup steps and 2,000 retained draws per chain, tree depth 10 and 100-draw checkpoints in `posterior_long_v1`. The new `native_real_posterior_rewarm` driver initializes from each chain's complete final position in `posterior_depth9_v1` (speeds 276.55, 276.08, 286.38 and 278.53 km/s), then learns fresh tuning. It preserves the posterior, including the 0–1000 km/s velocity prior, and excludes previous adaptation and samples. Progress includes elapsed time, checkpoint durations and an approximate sampling ETA. The completed longer-run results are recorded below. Starting in the same region motivates an independent repeat from alternative grid solutions before trusting an interval. Setup and use are documented in [the notebook guide](real_source_notebook.md#prepared-longer-run--launch-from-the-notebook).

## Completed longer run

`posterior_long_v1` completed on October 16 (Phoenix time) in **2 hours 25 minutes 30 seconds**. Fresh warmup took **78.0 minutes**. All four chains retained **2,000 draws each**, with the same probability target and broad velocity prior as the earlier runs. The saved results survived the notebook display issue; section 8 of the notebook reloads them independently after a kernel restart.

The complete saved chain set **passes the implemented sampling checks**: maximum rank Rhat **1.00503**, minimum bulk ESS **1,424.96**, minimum tail ESS **2,200.13**, zero divergences, zero trajectory-limit saturation, and BFMI **0.945–1.027**. Conditional pooled intervals are therefore available in this run's report and radial-profile plot. These are sampling diagnostics for this model; starts were all near the central solution, so an independent repeat from alternative solutions and the center/model sensitivity comparisons remain necessary before a robust mass claim.

The figures below remain the earlier, unconverged depth-9 diagnostic run for comparison. The completed longer-run figures are [trace](../../results/real_sources/ID3700/posterior_long_v1/trace.png), [corner](../../results/real_sources/ID3700/posterior_long_v1/corner.png), and [rotation/mass profiles](../../results/real_sources/ID3700/posterior_long_v1/profiles.png).

![Real-data chain movement with longer trajectories](../../results/real_sources/ID3700/posterior_depth9_v1/trace.png)

![Real-data kinematic corner plot, not yet converged](../../results/real_sources/ID3700/posterior_depth9_v1/corner.png)

Before launch, **16 targeted tests passed**: exact positive-amplitude marginalization, zero-information normalization, finite-difference kinematic/map/registration gradients, forward-model agreement, consistency between explicit priors and grid penalties, branch selection, and existing source/provenance/cache/worker behavior. On the real 18-exposure dataset, the forward audit passed at both starting branches, with a single-chain gradient time of approximately 2.5 ms on the local RTX 4090. A NUTS draw involves many gradient evaluations, so this is not the time per draw.

The full targeted set passed after the workflow changes: **17 tests**. The expanded notebook executed all **26 cells with zero errors and 11 static PNG outputs**, using real saved chains with both optimization and sampling disabled during validation. The executed copy is `../results/2026-10-16_notebook_validation/ID3700_MCMC_executed.ipynb`. Checkpoint blocks contain distinct retained draws; all original grid mathematical-code hashes remain unchanged. The GPU reported an allocation warning during initial warmup compilation, recovered, and completed the pilot; it did not silently fall back to CPU. Both sampling workers exited normally.

The radius grid is 0.2–0.8 arcsec for illustration, not a measured disk boundary or proof of radial resolution. Mass is `r Vrot(r)²/G`, a rotation-only spherical-equivalent proxy, without pressure-support or lensing correction. The model still needs scrutiny for contamination, faint edge emission, unresolved turnover, response approximations, noise covariance and the stellar-to-gas inclination transfer. Sampling alone does not remove those limitations.

Implementation: [posterior model](../dingo/native_real_posterior.py), [runner](../diagnostics/native_real_posterior.py), [diagnostic plotting](../diagnostics/native_real_posterior_plot.py), and [notebook](../notebooks/archive/real_source_kinematics.ipynb).
