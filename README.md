<p align="center">
  <img src="fig/logo.png" alt="DINGO Logo" width="300"/>
</p>

# DINGO

**DINGO** — *Dispersion-based INference for Galaxy Observations* — fits galaxy
kinematics to JWST/NIRCam WFSS data taken in **both grism dispersion directions
at once**.

## What it actually does

The R and C grisms disperse along perpendicular detector axes. A single-direction
fit cannot separate a velocity gradient from an elongation along its own
dispersion axis, so it has to break that degeneracy with a light-profile prior.
Two perpendicular directions break it with data.

DINGO is **not a forward model**. It takes each observed grism and rectifies it
*back* to the direct-image frame through the velocity field, then asks the two
reconstructions to agree:

```
loss = sum((image_R - image_C)**2)
```

Both `image_R` and `image_C` are data. There is no model image and no Sérsic
prior, which is why every parameter DINGO reports is a measurement rather than a
restatement of an assumption. The one informative prior in a typical fit is on
inclination, and only because `V_rot` and `inc` enter the line-of-sight field
almost entirely through the product `V_rot·sin(i)` — a degeneracy no amount of
R+C data breaks.

Two consequences worth knowing before trusting a result:

- The objective constrains no flux, so it is globally minimised at **exactly 0**
  by a solution that rectifies both grisms off the cutout. `fit_report()` checks
  for it, and the shipped configs freeze `image.C.dx/dy`, which makes it
  unreachable.
- A systemic velocity shift and a velocity gradient both lower `R − C`.
  `compare_null_model()` is the statistic that says which the data prefer.

## Where everything lives

```
~/kinematics/
  grism-kinematics/   this repo
  geko/               `import geko` resolves here -- AND it is what grism_cal/ points at
  grism_cal/          GRISM_CAL_DIR: trace + dispersion calibration
  stpsf_data/         PSF reference data (STPSF_PATH)
  results/            run logs land here
  archive/2026-10-12/ superseded material; see its WHAT_IS_THIS.md

~/EDR_data/           the SAPPHIRES release: 1d/2d spectra, catalogues, mosaics
~/backup_kinematics/  a Sept 23 snapshot, deliberately frozen
```

**`grism_cal/` holds almost no real files** -- it is ~60 symlinks into
`geko/nircam_grism/FS_grism_config/`. Moving or renaming `geko/` breaks
`grism_cal/`, which breaks `dingo/grism.py` and the whole test suite without any
obvious error pointing at the cause. If the tests suddenly fail on calibration
loading, check `find $GRISM_CAL_DIR -xtype l` for broken links first.

Everything else the pipeline needs is inside this repo, including
`psf/webbPSF_F444W.fits` -- it used to be read out of an old run directory,
so archiving that run silently broke the morphology fit.

Results and chains go to `results/<ID>/` and are gitignored; the repo holds
code and configs only. A config is the artifact worth versioning, since it
records what was fitted and why.

### If your commits do not appear on GitHub

Almost certainly the branch lost its upstream. Diagnose in one line:

```bash
git log origin/main..main      # non-empty = commits exist only on this machine
git push -u origin main        # -u sets tracking so plain `git push` works after
```

## Install

```bash
git clone https://github.com/AaKavan/grism-kinematics.git
cd grism-kinematics
pip install -e ".[mcmc,test]"      # mcmc = emcee + corner; test = pytest
```

Python ≥ 3.12 is a hard requirement (the code uses nested same-quote f-strings).

### Calibration files

DINGO needs the NIRCam WFSS trace and dispersion tables. They are not bundled.

```bash
export GRISM_CAL_DIR=/path/to/grism_cal
```

It must contain `dy_dx_tracing_model/` and `dx_wave_dispersion_model/`.
Supported filters: F277W, F322W2, F335M, F356W, F360M, F410M, F444W, F480M.

### Catalogues (optional)

`dingo.catalogs` generates configs from the SAPPHIRES EDR catalogues:

```bash
export SAPPHIRES_CAT_DIR=/path/to/sapphires_edr_catalogs
```

## One environment for geko and DINGO

`bash env/make_grism_env.sh` builds a conda env called `grism` that runs both,
sharing the GPU, and registers a Jupyter kernel for it.

The split into `dingo` (3.12, torch) and `geko_clean` (3.11, jax) was never
forced: geko's `requires-python` is `>=3.8` with no upper bound and DINGO needs
`>=3.12`, so 3.12 satisfies both. The real question was CUDA — torch ships
CUDA 13 wheels, jax ships CUDA 12 — and those are *differently named* packages
(`nvidia-cublas` vs `nvidia-cublas-cu12`), so pip installs both and the driver
serves both. Verified: torch and jax each run a GPU matmul in the same process.

Two settings the script bakes in, both of which bite if missed:

- `XLA_PYTHON_CLIENT_PREALLOCATE=false` — jax otherwise reserves 75% of the GPU
  on first use and starves torch.
- `setuptools<81` — `jax_cosmo` imports `pkg_resources`, which setuptools 81
  removed and python 3.12 does not bundle.

## Running a source

```bash
python -m dingo.run 21636                 # adam -> report -> mcmc -> maps
python -m dingo.run 21636 --stage adam    # optimiser only, with the fit report
python -m dingo.run 21636 --stage mcmc    # sample; resumes from the checkpoint
python -m dingo.run 21636 --stage null    # rotation vs no rotation
python -m dingo.run 21636 --strict        # stop on a railed or unconverged fit
```

Everything — walkers, steps, priors, noise model, checkpoint path — comes from
the config. Nothing on the command line overrides it, so a run is reproducible
from a file you can commit.

In Python:

```python
from dingo import fitting

f = fitting.KinematicsFitter(config_path='ID21636/config_kinematics.yaml')
f.fit_all()
f.fit_report(strict=True)        # railed parameters, convergence, flux retention
f.prepare_mcmc()                 # stage, free params, frozen noise model, priors
f.fit_MCMC()
f.save_mcmc('../results/ID21636_chain.npz')

# later, in a fresh process:
f.fit_all(); f.prepare_mcmc()
f.load_chain('../results/ID21636_chain.npz')
f.get_mcmc_results()             # {param: (16th, 50th, 84th)}
f.get_mcmc_diagnostics()         # tau, n/tau, acceptance
f.compare_null_model()           # dlnL, Wilks p, dBIC
```

`prepare_mcmc()` puts the fitter in exactly the state `fit_MCMC` samples from
without sampling, and does not import emcee — so scoring a saved chain needs
neither a new run nor the sampler installed.

## Configs

One YAML per source under `data/ID*/config_kinematics.yaml`, holding the
summary (position, redshift, filter), the staged optimiser strategy, and the
`mcmc:` block. Generate one with `dingo.catalogs.make_config(ID)` rather than
copying a neighbour — the header records why, and the configs are checked
against the catalogue by the test suite.

Notes on the fitting stages:

- Bounds (`min`/`max`) are **inherited** by later stages; `lr` and `fit` are
  per-stage, because changing those is the point of staging.
- An `override:` key that matches no parameter is an **error**, not a warning.
  `velocity.inc` instead of `velocity.inc_v` used to silently do nothing.
- `velocity.v_sys` exists in every fit and is frozen unless a config overrides
  it. It is identifiable only because `image.C.dx/dy` are frozen; with all four
  offsets free it is an exactly flat direction, and `_setup_data` refuses that.

## Tests

```bash
pytest -q                 # ~100 tests, a few seconds
pytest -q -m "not data"   # pure tests only; no FITS, no calibration
```

Data-backed tests pin theta to a known MAP rather than fitting, so they stay
fast and reproducible. `diagnostics/regression_snapshot.py record|check` freezes
the end-to-end numbers a refactor must not move.

## Use the GPU

`BaseFitter` selects CUDA when it is available and the batched likelihood is
~17x faster on it (measured: 1.1877 s to 0.0685 s per 32-walker step, so a
production chain goes from days to hours). **Do not set `CUDA_VISIBLE_DEVICES=`
on a production run** — that belongs in `tests/`, where determinism matters.

GPU and CPU agree to ~4e-7 relative, which is float32 rounding, so keep
`diagnostics/regression_snapshot.py` on CPU where its baseline was recorded, and
do not switch device partway through a chain or across a resume.

## Before quoting an uncertainty

Read [docs/validation.md](docs/validation.md). The short version, measured on
ID15665: the posterior widths are **understated by roughly 3x** for the rotation
amplitude, position angle is the one well-calibrated parameter, and the grism
registration offset is shrunk toward zero by ~20% with coverage 0.00. Multiply
by the per-parameter inflation factor before quoting anything.

## Diagnostics

| | |
|---|---|
| `regression_snapshot.py` | freeze / verify the numbers a refactor must not change |
| `validate_mcmc.py` | mock-recovery and K-realisation ensemble; reports the error-bar inflation factor |
| `measure_inclination.py` | PSF-convolved Sérsic fit → inclination prior |
| `scan_inclination.py` | conditional and profile Δχ² scans over inclination |
| `vsys_separability.py` | is a systemic velocity separable from a pointing error |
| `run_mcmc.py` | standalone Adam + MCMC driver with derived quantities |

Cosmology is Planck18 throughout.
