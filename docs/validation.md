# What the validation suite says, and what it does not

Measured on **ID15665** (the V_sini config), 2026-10-09, with
`diagnostics/validate_mcmc.py`. Reproduce with:

```bash
python diagnostics/validate_mcmc.py floor    --id 15665
python diagnostics/validate_mcmc.py ensemble --id 15665 -K 20 --nproc 20
```

Before any of these numbers are read, note that **the suite had not been
running**. Three independent faults, each silent:

| | |
|---|---|
| amplitude displacement | the ensemble displaced `velocity.V_rot` and read it back through `offsets.get(name, 0.0)`, so under the V_sini parameterisation it displaced the amplitude by **zero** — it stopped testing the parameter the fit exists to measure |
| floor test | never re-froze the noise model after building its mock, so it raised on the first `log_likelihood` call |
| ensemble mocks | ran short chains against configs carrying `resume: true`, so all 20 tried to **resume the production chain** and died |

So no number below was available before 2026-10-09, and nothing that was
quoted previously rested on them.

## Ensemble calibration (K = 20)

`inflation = std(medians) / mean(posterior sigma)` — the ratio of the true
frequentist error to the error the likelihood claims. `coverage` is the
fraction of realisations whose 16–84 interval contains the truth; it should be
≈ 0.68.

| parameter | bias | scatter | claimed σ | inflation | coverage |
|---|---|---|---|---|---|
| velocity.V_sini | +3.65 | 11.04 | 3.44 | **3.21** | 0.35 |
| velocity.R_v | +0.29 | 0.29 | 0.11 | 2.59 | 0.25 |
| velocity.x0_v | +0.36 | 0.34 | 0.16 | 2.12 | 0.15 |
| velocity.y0_v | +0.81 | 0.71 | 0.22 | 3.22 | 0.15 |
| velocity.theta_v | −0.008 | 0.008 | 0.008 | **1.03** | 0.55 |
| velocity.inc_v | +0.019 | 0.052 | 0.056 | 0.93 | 0.80 |
| image.R.dx | −0.25 | 0.17 | 0.023 | **7.24** | **0.00** |
| image.R.dy | +0.03 | 0.014 | 0.015 | 0.95 | 0.20 |

### The amplitude is over-confident by ~3×

`V_sini`'s posterior width is 3.44 km/s; the realisation-to-realisation scatter
is 11.04. The production chain quotes ±2.8 km/s, so **the honest uncertainty is
roughly ±11 km/s**, and coverage of 0.35 says the 68% interval contains the
truth about a third of the time.

The floor test agrees independently: refitting a **noiseless** mock from a 2%
displaced start recovers `V_sini` biased by +1.05 km/s with the offsets at zero
and +6.88 km/s at the MAP — a systematic of order the quoted error, with no
noise in the data at all.

### Position angle is the one well-calibrated parameter

`theta_v` has inflation 1.03 and the best coverage in the table. This is the
quantity the two-direction argument rests on — a single dispersion direction
cannot separate a velocity gradient from an elongation along its own axis — and
it is exactly the parameter that comes out honest. The over-confidence is in
the amplitude, not the geometry.

### The registration offset is shrunk toward zero

`image.R.dx` has inflation 7.24 and coverage **0.00**: across 20 realisations
the posterior never contained the truth. It is a bias, not just a narrow
interval, and the floor test shows it scales with the offset itself:

| true dx | recovered | bias |
|---|---|---|
| 1.4541 px | 1.1275 | −0.327 (−22%) |
| 0.0000 px | 0.0263 | +0.026 |

This is the forward/backward asymmetry the floor test was written to measure:
the mock scatters source → grism while the fit rectifies grism → source, and the
double bilinear pass is a smoothing operator. A smoothed image best-fits a
smaller offset. It is a property of the estimator, not a coding error.

**It matters more than a nuisance parameter should.** A relative R-versus-C
registration error maps into an apparent systemic velocity at 71.5 km/s per
pixel (`diagnostics/vsys_separability.py`), so a 0.25 px shrinkage is ~18 km/s
of systemic velocity with nowhere to go — `velocity.v_sys` is frozen in every
shipped config. That is an argument for switching it on for this source and
seeing whether it absorbs the bias.

## Does freeing v_sys absorb the registration bias?

Tested directly, because the shrinkage above predicted it should: 0.25 px at
71.5 km/s per pixel is ~18 km/s with nowhere to go while `velocity.v_sys` is
frozen. `data/ID15665/config_kinematics_vsys.yaml` is the production config with
that one parameter freed; the floor test gives a known truth to measure against.

| quantity (floor test, at MAP) | v_sys frozen | v_sys free | |
|---|---|---|---|
| V_sini bias (km/s) | 6.88 | **4.10** | −40% |
| SSE_floor / (N·var_base) | 0.191 | **0.119** | −38% |
| y0_v bias (px) | 1.20 | 0.84 | −30% |
| image.R.dx bias (px) | −0.327 | −0.256 | −22% |

**The prediction held in sign and magnitude.** Fitted on the real data,
`v_sys = -23.4 km/s` against ~18 predicted. Taken at face value that says
`summary.z` is high by 1.7e-4, i.e. 1.16130 would become 1.16113.

**It does not absorb the shrinkage.** `image.R.dx` is still biased −0.26 px.
What moved is the offset itself: ~0.27 px of it migrated into `v_sys` (≈19 km/s,
consistent with the fitted value), while the smoothing-induced shrinkage simply
follows whichever parameter now carries the offset. That is what the 45° angle
in `vsys_separability.py` implies — the two share the direction rather than
separating it.

**And v_sys is poorly recovered in its own right:** true −23.44, fitted −35.02,
a −49% bias. So freeing it buys a 40% reduction in the headline parameter's bias
at the price of a new parameter that cannot be quoted. Promising, not settled,
which is why it lives in a variant config rather than the production one.

## The caveat on the inflation factor

The ensemble mocks run **32 walkers × 1200 steps after 400 burn**, against a
production chain of 256 × 8000. At τ ≈ 35 that is n/τ ≈ 34, below the 50 the
pipeline treats as quotable. An under-explored chain reports a posterior that is
too *narrow*, which inflates the ratio.

So the `scatter` column is a frequentist quantity and robust, while `claimed σ`
may be biased low. **The direction is unambiguous — the errors are understated —
but 3.21 is an upper bound on the factor, not a measured constant.** Pinning it
down needs an ensemble at production chain length, roughly 3–4 hours on a
28-core machine.

## Floor: the resampling systematic

`SSE_floor / (N · var_base)` = 0.124 with the offsets at zero, 0.191 at the MAP.
The script treats < 0.1 as acceptable, so the smoothing systematic is 12–19% of
the noise variance rather than the assumed < 10%. Not fatal — the docstring's
failure threshold is 1.0, where the systematic would rival the noise — but it is
above the level the design assumed.

## Running it

Each mock is ~36 min idle and ~60 min with 20 workers contending; `fit_all`
alone goes 38 s → 104 s under load. Workers pin torch to one thread, so set
`--nproc` near the core count. Note that with `K == nproc` every worker runs
exactly one mock, so progress appears all at once at the end rather than
incrementally.

**Sizing a longer run: `n/tau` is `nsteps/tau`, and walkers do not enter it.**
The short run's confound is chain length, so it is fixed by lengthening, not by
widening:

| | n/tau | cost |
|---|---|---|
| 32 walkers x 1200 steps | 34 | 1.0x |
| 256 walkers x 4000 steps | 114 | 25.0x |
| 32 walkers x 4000 steps | 114 | **3.1x** |

Raising walkers to 256 reaches the same `n/tau` for eight times the price —
18 h per mock against 2. `bash run_long_ensemble.sh [config]` uses the third
row.
