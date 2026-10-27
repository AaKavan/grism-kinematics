# Current real-source pipeline on a matched mock — 2026-10-27

The first full recovery **failed**. This is a reproducible failure of the current
direct inference procedure on simple simulated data, not a measurement of a
high-velocity galaxy. Numerical convergence failed, so the saved draw quantiles
must not be reported as scientific uncertainty intervals.

## Reproduce or inspect

Use [simulation_real_pipeline.ipynb](../simulation_real_pipeline.ipynb).
To view the completed results without refitting, run its setup cell and final
results cell only. Run All starts a fresh fit of the named run when
`OVERRIDE=True`; change `RUN_NAME` to preserve another configuration.

Outputs are in:

`results/simulations/disk_200_joint_i60s5/runs/9000001/unknown_map_i60s5_start350/`

The folder contains the posterior NPZ, run record, diagnostics, corner/traces,
residuals, recovered emission, rotation curve, `truth_comparison.csv`,
`recovery_comparison.png`, and `recovery_info.json`.

## What was tested

The notebook calls the same `direct.prepare`, `direct.fit`, `direct.sample`, and
`direct.show_posterior` used for real sources. No alternative science likelihood
or sampler was introduced for this test. All 6,561 positive emission pixels and
six kinematic/registration parameters are inferred under that existing model.

The R and C FITS are byte-for-byte identical to both earlier
`disk_200_forward` and `disk_200_forward_i60s5` experiments. Truth is V_inf=200
km/s, R_t=5 pixels, PA=35 degrees, inclination=60 degrees, center=(40.3,40.7),
and R/C registration offsets zero. The image is 81 by 81, with exponential
emission Re=6 pixels, peak=1, independent Gaussian noise sigma=.02, seed=42.

The recovery does not use the injected brightness map. `image.fits` is only a
display reference; the initial emission map is inferred from R+C by the
nonnegative profile solve. There are no true morphology parameters in the
generated source configuration. The known center and noise are deliberately
retained for this first stage. Inclination has a 60 +/- 5 degree Gaussian prior;
R dx/dy are free and C defines the registration reference. Starting guesses are
V=350, R_t=3, PA=20 degrees, R offsets=(.2,-.2), and i=60 degrees.

NUTS used 4 chains, 500 warmup steps and 1,000 retained draws per chain,
target_accept=.9, maximum tree depth=8, seed=42. The unchanged default
HalfNormal brightness scale was 9.94244, determined from observed R/C pixels.
The run used CPU because CUDA was inaccessible in the agent's tool environment.
MCMC took 1,106 seconds (18.4 minutes). Real-source inputs and runs were untouched.

## Results

| Quantity | Injected | Saved draw median | 16th–84th percentiles, **not reliable errors** |
|---|---:|---:|---:|
| V_inf (km/s) | 200 | 997.38 | 984.61–999.47 |
| R_t (pixel) | 5 | 17.70 | 17.13–19.25 |
| PA (degrees) | 35 | 99.80 | 93.95–103.11 |
| R dx (pixel) | 0 | .0609 | .0127–.1153 |
| R dy (pixel) | 0 | .1962 | .1811–.2143 |
| Inclination (degrees) | 60 | 53.46 | 51.27–54.75 |
| V at the same radius of 5 pixels (km/s) | 100 | 174.86 | 158.57–180.40 |

The discrepancy is not merely an unconstrained asymptotic velocity: the
rotation speed at the same finite radius is also wrong. Approximately 97.85%
of saved V_inf draws exceed 950 km/s. The four chain medians are 999.56,
990.85, 998.77 and 985.68 km/s.

There are 339 retained divergences out of 4,000 transitions; 91.525% hit maximum
tree depth. Velocity rank R-hat is 2.811 and bulk ESS is 4.64. The other
kinematic parameters also fail mixing checks. These results do not establish
where the correctly explored marginal posterior would concentrate.

For comparison, the user's preceding known-map, Gaussian-inclination-prior
emcee run gives V_inf=207.22 (-9.90,+11.78) km/s. That comparison demonstrates
different outcomes, but it changes the sampler and other kinematic priors as
well as the emission treatment; it does not isolate a single cause.

## Specific concerns to diagnose next

The profile optimizer already starts in an inferior local basin. Its current
solution is approximately V=291, PA=9 degrees and R dy=.735 pixels. In the
preceding saved free-map inclination-prior experiment, the similar solution
has chi2=10795.20, while an already evaluated point at the true kinematics has
chi2=10603.28. An older free-map solution with V=215.32 and i fixed at 60 degrees
has chi2=10355.76 and is feasible under today's inclination prior. This is
evidence of an initialization/search problem; it does not prove initialization
alone causes the later drift to the velocity bound.

A forward-only boundary check also shows substantial changes in how source
pixels are constrained. At true kinematics, zero source pixels are completely
outside both R and C, and 6,464 have some footprint in both. At an actual
high-velocity draw (chain 3, retained step 207, one-based), 47 source pixels are
outside both images and only 5,488 have a footprint in both. Pixels with combined
R+C deposition weight below 1.5 increase from 248 to 1,220. This is consistent
with a possible interaction between finite image boundaries and the large
number of independent positive brightness variables. These counts alone do
not prove that effect causes the high velocity.

The next controlled diagnostic is the same current JAX likelihood, priors and
NUTS, with only the brightness map fixed to injected truth. If that succeeds,
compare the unchanged unknown-map posterior from the lower-score solution and
the high-velocity branch. These would separate sampler/likelihood differences
from unknown-morphology and initialization effects. No such additional MCMC
runs were performed in this task.

PSF effects, fainter sources and correlated noise have not been added: the
simple baseline already fails. This does not show that joint R+C inference is
impossible, nor identify NumPyro or the brightness prior alone as the cause.
It shows that this implementation has not earned reportable real-source errors.

## Software checks

The new comparison helper passed 12 lightweight tests of provenance, parameter
ordering, retained samples, angular units, fixed coordinates, plotting and
separation of truth inclusion from convergence. Notebook schema and code-cell
syntax checks passed. These software checks are distinct from the failed
scientific recovery above.
