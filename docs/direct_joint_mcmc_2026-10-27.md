# Direct R+C joint MCMC

## Current ID6170 alignment test

The notebook is now configured with `fixed_rdy=0.271` in `direct.prepare`,
selecting the positive R y-offset branch from the previous free-offset run.
The setting applies to both the profile initializer and MCMC. Velocity, turnover
radius, orientation, R x-offset, inclination and the shared emission map remain
free. C registration and the source center retain their previous fixed values.

The new run is named `direct_joint_i75s5_fixed_dyR_p0271_v1`, preserving
`direct_joint_i75s5_v450_v1`. Starting V_inf is 189.25 km/s, turnover radius
3.22 pixels, PA 306.53 degrees, R dx 1.406 pixels and inclination 80.22 degrees.
These are starting values near the selected branch; the inclination prior is
still 75 +/- 5 degrees, and velocity remains free over 0–1000 km/s.

The five-parameter corner omits fixed R dy. The summary marks it as fixed,
without a measured uncertainty, and diagnostics verify its constant value
before excluding it. Resulting velocity intervals are conditional on the chosen
alignment and do not account for its uncertainty or the other alignment branch.
Set `fixed_rdy=None` with another run name to restore the free-offset model.

The executed notebook was backed up as
`docs/notebook_backups/direct_source_before_fixed_rdy_20261027_033625.ipynb`.
Fifteen focused code checks passed, including model traces, fixed-parameter
gradients, optimizer argument inspection, reopen metadata and saved plotting.
No new source fit, simulation or MCMC was run while configuring this test.

## Joint MCMC implementation

`direct_source_fit.ipynb` now has six code cells: setup, source/input display,
starting fit, starting-fit inspection, MCMC, and posterior inspection. The old
notebook is preserved under `docs/notebook_backups/direct_source_before_joint_mcmc_20261027_024944.ipynb`.

## What is sampled

[NumPyro NUTS](https://num.pyro.ai/en/stable/mcmc.html#nuts) samples deprojected asymptotic velocity, turnover radius in pixels,
cutout orientation, R x/y registration, inclination, and every positive source
brightness pixel. One shared map predicts R and C. The source center, C
registration and systemic velocity remain fixed. ID6170 defaults retain the
center (40.25,41.0) and inclination prior 75 +/- 5 degrees. Velocity remains
bounded at 1000 km/s. No input FITS were edited.

For source map s and parameters theta, the target is

```
p(theta,s | R,C) ∝ p(theta) × product_j HalfNormal(s_j | scale)
                  × exp[-0.5 × sum_pupil,pixel ((data - A(theta)s)/noise)^2].
```

The map is sampled jointly, not optimized and substituted into an MCMC target,
and not frozen to the starting fit. NumPyro accounts for the transformation of
positive parameters. A dense mass matrix covers only the six kinematic
parameters; pixel coordinates use a diagonal mass matrix.

`emission_prior_scale` is in FITS intensity units. None chooses ten times the
larger of the positive observed peak and fixed input noise. This is an empirical
fixed scale, not an inferred hyperparameter. It imposes neither a Sérsic shape
nor spatial smoothing, but independence, positivity, pixelization and brightness
scale still affect the posterior. In particular, independent positive pixels
can imply nonzero background emission. Prior sensitivity and injection recovery
must be examined before treating errors as science results.

The JAX model uses float64 arithmetic and the same rounded calibration tables,
reference wavelength, cutout conventions and bilinear deposition as the Torch
profile model. Float32 rounding means their predictions are not bit-identical;
coordinate/prediction parity and finite-difference derivatives are tested.

There is still no explicit PSF, intrinsic dispersion, noise covariance or center
uncertainty in this prototype. Intervals are conditional on those choices.

## Running, inspecting and continuing

```
fit_result = direct.fit(run, max_evaluations=150)
posterior_path = direct.sample(run, warmup=500, draws=1000, chains=4,
                               checkpoint_every=25, overwrite=True)
views = direct.show_posterior(run)
```

Warmup is excluded from `draws`, which is a count per independent chain, not an
emcee walker count. The actual JAX CPU/GPU device is printed. Progress bars count
steps; heartbeat messages cover compilation or slow chunks. Warmup adaptation
continues across checkpoints without restarting.

After losing the notebook kernel, run setup and reopen without preparation:

```
run = direct.load_run(OUTPUT_ROOT / "6170" / "direct_joint_i75s5_v450_v1")
direct.sample(run, warmup=500, draws=1000, chains=4,
              checkpoint_every=25, resume=True, overwrite=False)
```

Keep the original sampling settings and priors. `draws` can be increased to
extend a run; it means the desired total, not an additional count. The saved
random state and adaptation are restored. Changed data, calibration, target
code, versions or sampler settings reject continuation. `load_run` preserves
the original fixed noise estimates. Re-running preparation with OVERRIDE=True
intentionally deletes that named run, so do not do it when resuming.

For inspection alone, pass the directory to `direct.show_posterior`; source FITS
and calibration are unnecessary. Partial checkpoints containing retained draws
also have a saved representative prediction. A warmup-only checkpoint is not
a posterior and cannot supply uncertainty plots yet.

## Saved products

Each source/run folder contains:

- `fit.npz` and the original profile-fit figures.
- `posterior.npz`: kinematic and full emission draws, NUTS statistics, a coherent
  representative prediction, and exact resumable state; no pickle required.
  All draws use [chain,step,...] order. Atomic replacement prevents a partial
  write from replacing the last usable checkpoint.
- `posterior_prediction.npz`: one actual joint parameter/map draw and its R/C
  predictions, observations and projected centers, for coherent residual plots.
- `posterior_diagnostics.json`, `posterior_summary.txt`, PNG plots, and the
  black-contour corner plot as both PNG and PDF.
- `run_info.txt`: source/settings, fixed geometry, brightness prior, input/code
  hashes, versions, timestamps, actual backend, requested/completed steps and
  sampling diagnostics.

The corner plot pools all retained chains and prints median and 16th/84th
percentiles. The residual panels use one actual draw near the pooled kinematic
median and label its own velocity; independently combining median parameters
and a median map would not necessarily describe a sampled galaxy. Rotation
bands propagate joint V_inf and turnover-radius draws. The displayed radius
range is not a measurement of the emission extent.

Diagnostics include rank R-hat, bulk/tail ESS, divergences, tree-depth saturation,
BFMI, all source pixels, and source-map flux/spatial summaries. Failed checks
do not hide plots; intervals and corners are marked provisional. Passing checks
does not establish mode completeness, correct noise, or scientific coverage.

## Validation so far

57 focused tests passed, including an interruption/reopening/resume integration
test. They cover probability-density
normalization and Jacobians, JAX/Torch forward agreement, derivatives, masks,
warmup exclusion, exact chunk/restart trajectories, rejection of damaged
checkpoints and stale handles, diagnostic failures and coherent saved plots.
An independent one-pixel problem with analytically marginalized brightness
checks the sampled velocity distribution against its known posterior.

A 41x41 galaxy injection with true V_inf=200 km/s and inclination=60 degrees
was fitted with its map unknown and an inclination prior of 60 +/- 5 degrees.
Two chains, 200 warmup and 200 retained draws, pixel-prior scale 10, completed
in 50.8 seconds on four CPU cores, using about 1.8 GiB peak resident memory.
The pooled V_inf percentiles were 157.9,168.6,188.4 km/s (16th/50th/84th),
and its 95% interval was 149.2–217.1 km/s. There were no retained divergences,
but velocity R-hat=1.026, bulk ESS=60 and tail ESS=21; about half the trajectories
hit the depth-7 limit. This is a functioning implementation test, not converged
or coverage-calibrated uncertainty recovery. The injected velocity lies inside
the provisional 95% interval; one realization cannot establish coverage.

Changing the fixed brightness scale by factors of two gave the following short
run results with the same synthetic observations and sampling settings:

| Half-normal scale | V_inf 16/50/84 percentiles [km/s] | Velocity rank R-hat | Bulk ESS |
|---|---|---:|---:|
| 5 | 157.5 / 168.0 / 183.2 | 1.031 | 60 |
| 10 | 157.9 / 168.6 / 188.4 | 1.026 | 60 |
| 20 | 161.8 / 179.6 / 213.8 | 1.210 | 8 |

All had zero retained divergences, but none passed convergence checks. The
scale-20 chains had medians 171.5 and 193.9 km/s. These differences cannot yet
be assigned solely to prior sensitivity because sampling error/mixing remains
substantial. Increasing prior width is not automatically an improvement.

All six notebook code cells also executed on the real ID6170 inputs with a
five-evaluation starting fit and deliberately tiny 2-chain, 20-warmup/20-draw
settings. That interface/plot test took 33.8 seconds on CPU. Reopening its handle
and extending to 25 draws preserved the first 20 draws exactly. The tiny real
run is not a velocity measurement; its corner plot explicitly fails the
convergence checks. Normal notebook settings remain four chains, 500 warmup
and 1000 retained draws. No long real-source MCMC was launched in this task.

Validation artifacts and logs were isolated under `/tmp/dingo_joint_validation/`;
the source FITS and existing science run folders were not changed.
