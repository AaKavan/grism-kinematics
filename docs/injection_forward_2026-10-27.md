# Direct R+C injection recovery, October 27, 2026

`simulation_recovery.ipynb` now compares forward predictions with the original
simulated R and C pixels. The previous notebook, including its executed outputs,
is preserved in `notebook_backups/simulation_recovery_before_forward_20261027_085036.ipynb`.
The original `results/simulations/disk_200` experiment remains available.

## What changed

The earlier objective dispersed the source, rectified the two resulting images,
and minimized their difference. Pixel mixing in the two successive resampling
operations is not invertible. In addition, changing the velocity/registration
changes the remapped noise, while the likelihood used a fixed variance.

The new experiment instead uses

\[
y_R=A_R(\theta)s+\epsilon_R,\qquad
y_C=A_C(\theta)s+\epsilon_C,
\]

with the calibrated, flux-conserving forward operators and one shared source
map. Independent Gaussian noise is added in the original detector pixels, and
the fit compares predictions with those same fixed pixels. No inverse
rectification is used in the new likelihood. The arctangent velocity field and
R/C calibration convention are unchanged.

There are two explicitly separate tests:

1. **Known map:** fix the injected emission map, center and inclination. Fit
   V_inf, turnover radius, PA and two R registration offsets. Emcee samples this
   five-parameter conditional posterior with the PyTorch prediction on the chosen
   CPU/GPU device. These intervals omit morphology and fixed-geometry uncertainty.
2. **Free map:** fit every nonnegative source-pixel intensity jointly to R+C.
   No generating emission values, Sérsic profile or smoothing penalty enter the
   solver. The conditional velocity grid refits the map at each velocity; a
   subsequent fit varies all five parameters. Center/inclination stay fixed.
   These are nuisance-profile fits, not marginalized posterior samples.

Sparse free-map solves use CPU linear algebra with a temporary one-thread BLAS
limit. The caller's thread setting is restored afterward. `threadpoolctl` is a
declared dependency. Mapping and envelope gradients use the chosen torch device.

## Measured results

Default injection: V_inf=200 km/s, R_t=5 pixels, i=60 degrees, PA=35 degrees,
center=(40.3,40.7), exponential emission Re=6 pixels, independent noise sigma=.02,
seed=42. No PSF or intrinsic velocity dispersion.

| Check | Result |
|---|---|
| Earlier rectification conditional noiseless grid | Minimum at 150 km/s |
| Earlier noisy recovery | V_inf=105.5 km/s |
| Direct known-map conditional noiseless grid | Minimum at 200; exactly zero residual at truth |
| Direct known-map noisy starts 75/350/800 | V_inf=205.90/208.23/198.68 km/s |
| Known-map MCMC, 100 warmup + 500 draws, 32 walkers | V_inf=206.18 −10.14/+10.09 km/s; R_t=5.132 −.314/+.326 pixels |
| Free-map conditional noiseless/noisy grids, 25 km/s spacing | Both minima at 200 km/s |
| Free-map noiseless five-parameter recovery from 350 | Approximately 200 km/s and 5 pixels |
| Free-map noisy five-parameter fit from 350 | V_inf=216.09, R_t=5.699 pixels, PA=27.59 degrees |

All seven notebook code cells executed with their default settings in 55.9 seconds
on CPU. The known-map sampler took 7.8 seconds; the free-map block 38.8 seconds.
GPU execution was not available in this verification environment. Scalar and
batched model calculations agree in tests. The focused suite passed 39 tests.

Outputs are under `results/simulations/disk_200_forward/`, with separate
`direct_known_map/` and `direct_free_map/` directories, arrays, plots and run records.
The notebook deliberately contains no embedded execution outputs; users can run
the cells themselves or reopen the saved known-map posterior.

## Interpretation and remaining work

This removes the previously demonstrated failure for this controlled experiment.
It is a consistency test using the same forward operator for generation and
recovery; it does not validate a physical instrument model, realistic coadd noise,
real-source velocities, or mass uncertainties.

The known-map chain spans approximately 16–19 estimated autocorrelation times.
It is an exploratory run, not a coverage study. Repeated realizations and longer
independent runs are required before claiming calibrated intervals.

The free-map noisy optimizer stops on small relative score changes. Its strict
projected-gradient criterion is not met (maximum scaled gradient ~6.34); the recorded
216 km/s value is the best local point found, not a certified global optimum.
The inner nonnegative solve succeeds with optimality ~1.22e−8. The differing PA and
turnover radius need to be assessed with morphology uncertainty included, rather
than attaching the known-map corner errors to this result.

The next statistical step is to include the unknown emission map in uncertainty
inference and test sensitivity to its support, resolution and positivity
constraint. Then add PSF/pixel-integration mismatch, independent simulation
generation, and observational noise/registration tests. The real-source notebook
still uses its existing method; this experiment does not silently replace it.

## Optional inclination prior added after the runs above

`ForwardExperiment(mock, inclination_mean=60., inclination_sigma=5.)` now adds
inclination as the sixth fitted and sampled parameter. Both arguments are in
degrees. The prior is Gaussian, truncated to 5–85 degrees; `inclination_mean=None`
preserves the original fixed-true-inclination experiment. Inclination changes
the velocity-field projection but does not reshape the supplied brightness map.

The notebook defaults to this 60 ± 5 degree prior and saves the new experiment
in `results/simulations/disk_200_forward_i60s5/`, preserving the earlier results.
The prior and whether inclination is sampled are recorded in `run_info.txt`.
Tables, traces and corner plots include inclination in degrees. Loading results
with a different prior or fixed/sampled mode is rejected to prevent mislabeling.

The optional free-map optimizer uses the same inclination prior for noisy data.
Its conditional velocity scan still holds every other kinematic parameter,
including inclination, at injected truth. Noiseless fits omit prior penalties:
their raw flux SSE has no measurement-noise scale for weighting a Gaussian prior.

The updated notebook has not been executed, and no new recovery fits or MCMC
chains were started for this change. The known-map MCMC still conditions on
the true emission map and center; it does not include morphology uncertainty.
