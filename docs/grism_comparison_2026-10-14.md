# Does combining R and C improve the mass constraints?

**The controlled comparison supports the scientific motivation for joint R+C.**
The earlier repeated-noise tests assessed the absolute reliability of the
joint fits. They did not compare against one grism, and should not be read as
evidence that one direction would perform better.

The wider nonlinear search also finds a second joint-fit branch missed by the
three original starts. The information advantage therefore does not yet give
us a unique or calibrated outer mass profile.

**Follow-up:** the [joint posterior pilot](native_posterior_2026-10-15.md) now
samples the shared emission map and inclination, with chains initialized in
both branches. It uses an explicit morphology prior, so it is a different
inferential calculation from the free-map profile likelihood below.

These new experiments use synthetic sources with the real ID3700 exposure
geometry, ERR and masks. They are not measurements of ID3700 or a direct
benchmark against GEKO. Results live in
`../../results/2026-10-14_grism_comparison/`.

## Why two directions help

A velocity change moves line emission along dispersion. With one direction,
an unknown brightness map can compensate for much of that change. With two
directions, the *same* sky map must account for two differently dispersed
images. The compensating changes generally disagree. The gain can therefore
exceed the gain from collecting more photons along one direction.

The appropriate likelihood shares the emission map and velocity field across
R and C, with separate instrument operators and exposure nuisance parameters.
Fitting R and C with unrelated maps and then combining their velocity errors
throws away an important part of this information.

## Local information experiment

`diagnostics/grism_information.py` differentiates the native forward model
around a clumpy rotating source. It removes the continuum and all free
brightness-map directions using weighted projections, then computes the
remaining information about speed, turnover radius, angle and inclination.
The joint calculation stacks R and C *before* profiling the shared map.

The injected reference speed is V(0.5 arcsec) sin(i) = 210 km/s, turnover
radius 0.22 arcsec, inclination 0.9 rad and kinematic angle 0.9 rad. Intrinsic
width is 45 km/s. The response uses the same specified Gaussian kernel and
cubic deposition as the preceding validation experiments; it is not yet an
empirical PSF/LSF calibration. Center, registration and response are known.

There are six R training exposures and six C training exposures. Their line
signal-to-noise after continuum projection is 46.8 and 67.2, respectively;
joint training S/N is 81.9. All eighteen exposures, including six held out
from fitting, define the injection normalization of S/N 100. Selecting one
grism does **not** silently renormalize its source to S/N 100.

Controls compare:

- R only, C only, and R+C with a shared free map.
- R only and C only with errors ideally reduced to match the joint training
  line S/N. This isolates geometry from total S/N, not observing-time efficiency.
- Both directions with independent maps, obtained by adding the separately
  profiled information matrices. This deliberately removes the shared-map
  constraint while retaining both sets of exposures.

At fixed inclination, the 17x17 shared-map calculation has local uncertainty
38.1 km/s in projected reference speed. Either direction by itself has extremely
weak local curvature: the formal widths exceed 3,000 km/s and even the allowed
parameter range. Matching total S/N does not resolve that weakness. Permitting
independent maps for R and C also loses most of the joint constraint.

**Those enormous single-grism widths are a diagnosis of local degeneracy,
not usable error bars or a defensible numerical improvement factor.** The local
calculation permits infinitesimal signed changes about a positive map. It does
not impose positivity on large excursions or integrate a morphology prior.
Finite nonlinear likelihood profiles and proper priors can produce narrower
single-grism constraints. GEKO supplies such additional morphology assumptions.

For the joint fit, the local errors in natural log of the rotation-only mass
proxy are:

| Radius, arcsec | Known inclination | 5-degree inclination prior | Free inclination |
|---|---:|---:|---:|
| 0.1 | 0.268 | 0.315 | 0.942 |
| 0.2 | 0.228 | 0.259 | 0.702 |
| 0.3 | 0.267 | 0.281 | 0.544 |
| 0.5 | 0.362 | 0.365 | 0.444 |

Here the proxy is proportional to r[v_projected(r)/sin(i)]^2, or r times
deprojected rotation speed squared. The differential uncertainty
includes covariance between speed, turnover radius and inclination; inclination
errors cannot simply be added to independent velocity errors afterward.
For small errors, sigma(log M) approximates fractional mass uncertainty.
These are local diagnostics, not posterior intervals or pressure-corrected
dynamical masses. The listed radii are test coordinates, not a demonstrated
resolved radial range for a real galaxy.

The joint constraint changes only modestly between 13x13 and 17x17 map controls:
fixed-inclination reference-speed uncertainty is 37.1 versus 38.1 km/s.
Single-direction curvature becomes still weaker with the finer map. All map
columns remain numerically resolved; SVD rank thresholds 1e-6, 1e-8 and 1e-10
give the same answer. Halving finite-difference steps changes the full
kinematic derivative norm by only about 3.3e-7. The shared-map information gain
over independent maps has positive eigenvalues in all four kinematic directions.

![Local information comparison](../../results/2026-10-14_grism_comparison/local_information/information.png)

## What the nonlinear comparison tests

The companion experiment uses one paired Gaussian noise draw (seed 481), an
analytic clumpy source on 97x97 quadrature, and a 17x17 nonnegative fitted map
on 49x49 quadrature. This repeats the earlier injection, now fitting R only,
C only, or both. Inclination remains fixed at its true value. Three independent
free starts are followed by two-start constrained fits at projected reference
speeds 0, 75, 150, 210, 300, 450 and 750 km/s.

These scans test finite nonlinear changes and positivity, which the local
information calculation omits. In particular, the enormous single-direction
local widths must not be interpreted as guaranteed broad nonlinear intervals.
Profiling hundreds of nonnegative brightness parameters can favor a distorted
map and wrong velocity in a noisy image. Weak curvature at the noiseless truth
and a displaced, apparently preferred noisy profile are compatible findings.
The conditional profile likelihood is not a morphology-marginalized posterior.

The nominal horizontal reference of 3.84 in the plot is a visual diagnostic
for a nonzero fixed speed, not established 95% coverage. It is not a valid
chi-square detection threshold for zero rotation, where angle and turnover
radius are unidentifiable. One paired draw cannot establish coverage or a
population-level improvement over single-direction inference.

### Completed paired result

After constrained scans, releases of competitive constraints, and a longer
continuation of the initially unfinished C-only fit:

| Data used to fit | Best projected V(0.5 arcsec), km/s | Turnover radius, arcsec | Common held-out chi-square |
|---|---:|---:|---:|
| R | 767.78 | 2.00 (upper bound) | 38184.14 |
| C | 1160.25 | 2.00 (upper bound) | 50325.09 |
| R+C, best training objective | 498.95 | 2.00 (upper bound) | 37490.38 |
| R+C, competitive second mode | 210.76 | 0.2004 | 37467.60 |

All four listed solutions meet the gradient criterion. They use the same six
held-out exposures, with the predicted line fixed and continuum re-fitted.
The injected projected speed is 210 km/s and turnover radius 0.22 arcsec.
Raw training chi-squares across R, C and R+C are not compared because those
training datasets differ.

The joint fit predicts the held-out data substantially better than either
single-direction fit. It also retains a near-truth mode, whereas the selected
single-direction fits strongly favor distorted high-speed solutions in this
free-map model. This supports the value of the second direction under the same
weak morphology assumptions. It is not a comparison with GEKO's morphology priors.

**The near-truth joint mode is not the best training objective found.** Releasing
the 450 km/s constraint converges to 498.95 km/s and the maximum allowed turnover
radius. Its twice-negative-log-likelihood advantage over the 210.76 km/s mode is
only 1.02. The near-truth mode predicts held-out data better by chi-square 22.78.
Held-out performance is a useful diagnostic; selecting a mode after inspecting
these exposures would use them for model selection and consume their independence
as a final validation set.

Relative to the best joint objective, the sampled fixed-speed penalties are:

| Fixed projected speed, km/s | Twice delta negative log likelihood |
|---|---:|
| 0 | 225.61 |
| 75 | 33.45 |
| 150 | 4.16 |
| 210 | 1.02 |
| 300 | 4.03 |
| 450 | 1.82 |
| 750 | 12.20 |

The nonmonotonic profile exposes the second branch. These sparse points do not
define calibrated interval endpoints. In particular, the local 38 km/s width
around the injected source misses this distant branch.

For the high-speed joint mode, the rotation-only mass ratios relative to the
injection are 1.72, 2.29, 3.17 and 5.65 at 0.1, 0.2, 0.3 and 0.5 arcsec. Thus
even the joint maximum-likelihood point estimate can be substantially wrong.
The two joint branches must both be explored in posterior inference. Restricting
the turnover radius would remove the high branch, but such a restriction needs
physical justification and a sensitivity test; it cannot simply be chosen
because it restores the injected answer.

![Paired nonlinear fits and competing joint modes](../../results/2026-10-14_grism_comparison/nonlinear_comparison.png)

### Consequence for the initial-condition concern

The three original joint starts all converged near 211 km/s. A constraint at
450 km/s had a slightly *worse* objective, so the earlier guard that restarts
only when a constrained fit beats the free fit did not fire. Releasing that
competitive constraint nevertheless found a better free solution. The new
`diagnostics/native_profile_modes.py` explicitly supports this search.

This is a remaining source of starting-point dependence: distinct likelihood
branches with very different mass profiles. Parameter updates and convergence
checks can work correctly while a limited set of starts misses a branch.
Future inference must search competitive profile solutions, run posterior chains
from distinct modes and assess their relative probability with a specified
morphology prior. Agreement between several nearby optimizer starts is insufficient.

The comparison contains 57 optimization attempts, including four additional
free-mode continuations. Two C attempts initially failed the gradient criterion;
the final selected C solution converged after continuation. All selected free
and constrained solutions now meet the criterion. An interrupted first R run
is preserved separately and excluded. The earlier ensemble remains a record
of its explored starts, not a certification that all its global modes were found.

### Exploratory fit using all eighteen exposures

The common holdout contains information that distinguishes the two branches.
As a separate experiment, both joint branches were used to initialize fits to
all eighteen exposures, with no independent holdout remaining:

| Starting branch | Final projected speed, km/s | Turnover radius, arcsec | Twice delta NLL within the full dataset |
|---|---:|---:|---:|
| Near injected speed | 189.17 | 0.1712 | 0 |
| High-speed / large-turnover | 363.59 | 2.00 (upper bound) | 10.76 |

Both fits meet the gradient criterion. Including the remaining exposures now
favors the near-truth branch substantially. The best full-data speed is about
10% below the injected 210 km/s; its rotation-only mass ratio at 0.5 arcsec is
0.811. This is encouraging evidence that additional joint data help resolve
the ambiguity in this realization. It does not establish posterior coverage,
exclude unsearched modes, or provide an independent validation after all data
have been used to fit. Raw likelihoods from the twelve- and eighteen-exposure
fits are not compared. These two additional fits are stored separately in
`RC_all_exposures_modes/` and excluded from the common-holdout comparison.

![Full-data rotation and mass profiles](../../results/2026-10-14_grism_comparison/all_exposure_profiles.png)

## Relation to the local GEKO implementation

The inspected parametric fitting path in
[GEKO fitting.py](../../geko/geko/fitting.py) accepts one
observed map, error map and grism object. Preprocessing supports choosing R or
C. It uses NumPyro/NUTS MCMC; it should not be characterized as having no MCMC.
Its model learns a Sersic profile with imaging-informed morphology priors,
links apparent axis ratio to inclination with finite disk thickness, and fits
an arctan velocity curve. In [GEKO models.py](../../geko/geko/models.py), the sampled turnover
radius is restricted to 0 < r_turn < r_eff. The inspected local code also
places a prior on kinematic angle around the morphology angle.

Our current free-map experiment instead allows independent brightness controls,
does not tie the map to inclination, and allows turnover radii up to 2 arcsec.
Its nonlinear tests fix inclination while GEKO samples it. These differences
alone can substantially alter mass constraints. A direct comparison of their
reported interval widths would confound data geometry with model assumptions.
The [GEKO methods paper](https://arxiv.org/abs/2510.07369) describes its
parametric morphology and Bayesian inference approach.

The fair first benchmark is **the same forward model and priors with R, C,
and R+C**. A subsequent GEKO-style Sersic-plus-imaging-prior version should be
compared with a flexible-map version under each data selection. That will test
whether adding the second direction permits weaker morphology assumptions
while retaining useful, correctly calibrated mass constraints.

## Implications for the next science model

1. Keep the shared-map joint native-exposure likelihood. The new calculation
   supports its purpose, rather than suggesting a return to inverse rectification.
2. Infer inclination with justified imaging information and propagate its joint
   uncertainty into the rotation and mass profiles. Permit uncertainty in disk
   thickness and alignment between stellar morphology and line-emitting gas.
3. Use a flexible but controlled emission model, and compare its prior sensitivity
   with a Sersic model. A free map is a useful stress test, not a complete prior
   specification for a posterior sampler.
4. Repeat paired R/C/joint recovery across noise draws, viewing angles, line
   strengths and morphology mismatch. Compare calibrated profile coverage and
   held-out predictions, rather than selecting the smallest quoted error bar.
5. Fit or propagate registration, PSF/LSF and contamination uncertainty, then
   address pressure support and the relation between rotation and enclosed mass.
   Restrict science profiles to radii shown to carry information in these tests.

## Reproduction

From the repository root, using the grism Python environment:

```bash
python -m diagnostics.grism_information \
  --bundle /home/aakavan/EDR_data/magnif_copy/sapphires_kinematics_inputs_2026-10-13 \
  --output /home/aakavan/kinematics/results/2026-10-14_grism_comparison/local_information \
  --device cuda
```

`information.json` records matrices, rank checks, inclination conditions,
exposure paths, arguments and code hashes. Derivative arrays and a plot are saved.
The information calculation takes seconds on the available GPU. The full
nonlinear fits are more expensive because they repeatedly optimize the map
and kinematic parameters.

For each of R, C and RC, use a distinct output directory and substitute the
selection in `--grisms`:

```bash
python -m diagnostics.native_flexible_check \
  --bundle /home/aakavan/EDR_data/magnif_copy/sapphires_kinematics_inputs_2026-10-13 \
  --output /home/aakavan/kinematics/results/2026-10-14_grism_comparison/nonlinear_RC_seed481 \
  --device cuda --grids 17 --strengths 0 --starts 3 --maxiter 1600 \
  --noise-scale 1 --seed 481 --grisms RC \
  --profile-reference --profile-speeds 75 150 300 450 750
```

Each fixed-speed point re-fits the nonnegative map, continuum, turnover radius
and angle from two starts. The zero-speed null fixes its unidentifiable angle
and turnover radius as well. The same six held-out exposures are used for every
selection; unused training exposures from the other direction stay excluded.
The new explicit `held_indices` option prevents accidental changes to that
validation set. A constraint that beats the selected free fit triggers a free
restart from the constrained state.

After all three runs finish, the following command verifies paired data,
injection brightness, masks, held-out exposures and code versions before
writing `comparison.json` and `nonlinear_comparison.png`:

```bash
python -m diagnostics.plot_grism_comparison \
  /home/aakavan/kinematics/results/2026-10-14_grism_comparison
```

GPU scatter summation can change synthetic SCI values at floating-point
roundoff. The comparison requires identical ERR, masks and continuum arrays
and agreement of line/SCI arrays to 1e-12 in ERR units; exact array hashes and
actual maximum differences are retained for the audit.

To search the competitive joint branches explicitly:

```bash
python -m diagnostics.native_profile_modes \
  --run-dir /home/aakavan/kinematics/results/2026-10-14_grism_comparison/nonlinear_RC_seed481 \
  --output /home/aakavan/kinematics/results/2026-10-14_grism_comparison/RC_released_modes \
  --references speed_150 speed_300 speed_450 --device cuda
```

The C continuation uses `--references free_best --maxiter 6000`, with the C
source and output directories. The full-data exploration uses
`--references truth_speed speed_450 --all-exposures --maxiter 3000` and a fresh
output directory. Mode searches preserve the original reports and record the
source and operator hashes; the plotter rejects mixing full-data refinements
into the common-holdout comparison.

The numerical tests check the shared-nuisance information against explicit
linear regression, verify null-space uncertainties and mass gradients, and
prevent training/holdout overlap. The full repository suite passes **195 tests**
with three existing warnings. Plot and mode-search helpers were also exercised
on the saved experiment, and the plots were inspected.
