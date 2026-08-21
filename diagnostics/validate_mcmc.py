#!/usr/bin/env python
'''
Validation suite for the DINGO kinematics MCMC.

Three tests, in increasing cost, all built on KinematicsFitter.make_mock_data():

  1. floor    -- noiseless mock at theta_true. The fit rectifies grism -> source
                 while the mock scatters source -> grism, so the double bilinear
                 pass is a smoothing operator and the residual at theta_true is
                 not zero. Reports SSE_floor/(N*var_base); if that approaches 1
                 the systematic rivals the noise and the likelihood is misspecified.
                 Also runs Adam on the noiseless mock to measure the systematic
                 BIAS of the estimator, which the floor number alone does not show.

  2. single   -- one full-noise mock at a theta_true displaced from the MAP.
                 Confirms nothing is wired backwards: the posterior should cover
                 theta_true and nothing should rail against a prior bound.

  3. ensemble -- K independent noise realisations at one theta_true, each with its
                 own short chain. This is the test that makes the error bars
                 quotable. Per parameter it compares the scatter of the K medians
                 (the true frequentist error) with the mean posterior width (what
                 the likelihood claims):

                     inflation = std(medians) / mean(posterior sigma)

                 The R-C residuals are spatially correlated (correlation area
                 measured at 3-4 versus 1.0 for the raw data), so the diagonal
                 Gaussian likelihood overcounts independent pixels and inflation
                 is expected to be well above 1. Multiply the production error
                 bars by these factors before quoting them.

Usage:
    python validate_mcmc.py floor
    python validate_mcmc.py single [--nsteps N]
    python validate_mcmc.py ensemble [-K 20] [--nproc 6] [--nsteps N]

Ensemble workers are separate processes that each rebuild their fitter from the
config path: the forward models are closures and cannot be pickled, and macOS
spawns rather than forks.
'''

import argparse
import logging
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))   # pipeline/
ROOT = os.path.dirname(HERE)                        # DINGO_development/
GK   = os.path.dirname(ROOT)                        # Galaxy_Kinematics/
os.environ.setdefault('GRISM_CAL_DIR', os.path.join(GK, 'grism_cal'))

import numpy as np
import torch

from dingo import fitting

LOG = logging.getLogger('validate')

DEFAULT_ID = '15665'
DEFAULT_WORKDIR = os.path.join(ROOT, 'data')
DEFAULT_OUTDIR = os.path.join(ROOT, 'results')


def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger('dingo.kinematics').setLevel(logging.ERROR)
    logging.getLogger('dingo.fitting').setLevel(logging.WARNING)
    torch.set_num_threads(1)


def build_fitter(config=DEFAULT_CONFIG, fit=True):
    '''Fresh fitter at its Adam MAP, with the MCMC scaffolding in place.'''
    f = fitting.KinematicsFitter(config_path=config)
    if fit:
        f.fit_all()
    mc = f.config.get('mcmc') or {}
    stage = int(mc.get('stage', f.current_stage))
    f.current_stage = stage
    f._assign_cfgs_for_stage(stage)
    f.mcmc_xy_iters = int(mc.get('xy_iters', 40))
    f.mcmc_xy_tol = float(mc.get('xy_tol', 5e-3))
    f.mcmc_ln_f_fixed = 0.0
    f.setup_noise_model()
    names, cfgs, _ = fitting.collect_free_scalar_params(f._mcmc_stage_cfg())
    f._mcmc_names, f._mcmc_cfgs = names, cfgs
    f.mcmc_param_names = list(names) + list(f.mcmc_extra_param_names)
    f._mcmc_priors = f._build_mcmc_priors()
    return f, names


def current_theta(f, names):
    all_cfg = f._mcmc_stage_cfg()
    return {n: float(all_cfg[n].tensor.detach().cpu()) for n in names}


# ── test 1: resampling floor and estimator bias ────────────────────────────────

def cmd_floor(args):
    f, names = build_fitter(args.config)
    theta_true = current_theta(f, names)
    n_pix, var_base = f.n_mcmc_pix, f.mcmc_var_base
    LOG.info(f'N_pix={n_pix}  var_base={var_base:.6g}  N*var_base={n_pix*var_base:.6g}')

    f._reset_state()
    with torch.no_grad():
        f.loss()
    source = f.image_R.detach().clone()

    for tag, override in [('dx=dy=0', {k: 0.0 for k in names if k.startswith('image.')}),
                          ('dx,dy at MAP', {})]:
        truth = dict(theta_true)
        truth.update(override)
        f.make_mock_data(intensity=source, theta=truth, sigma_R=0.0, sigma_C=0.0,
                         inplace=True)
        f.set_params_dict(truth)
        f.log_likelihood(**{'noise.ln_f': 0.0})
        sse = float(torch.sum(((f.image_R - f.image_C)[f.mcmc_mask])**2))
        LOG.info(f'[{tag}] SSE_floor={sse:.6g}  ratio to N*var_base = {sse/(n_pix*var_base):.4f}'
                 + ('  <-- OK (<0.1)' if sse/(n_pix*var_base) < 0.1 else '  <-- HIGH'))

        # Systematic bias: re-fit the noiseless mock from a displaced start and see
        # where the estimator lands relative to truth. The floor SSE says how big
        # the systematic is; this says which way it pushes each parameter.
        f.set_params_dict({k: v*1.02 for k, v in truth.items()})
        f.current_stage = 0
        f._assign_cfgs_for_stage(0)
        f.fit_gradient()
        fitted = current_theta(f, names)
        LOG.info(f'[{tag}] noiseless-mock refit bias (fitted - true):')
        for n in names:
            LOG.info(f'    {n:20s} true={truth[n]:10.4f}  fitted={fitted[n]:10.4f}  '
                     f'bias={fitted[n]-truth[n]:+.4f}')
        f.set_params_dict(theta_true)
        f.restore_true_data()
    return 0


# ── test 2: single-mock recovery ───────────────────────────────────────────────

def cmd_single(args):
    f, names = build_fitter(args.config)
    theta_map = current_theta(f, names)
    f._reset_state()
    with torch.no_grad():
        f.loss()
    source = f.image_R.detach().clone()

    # displace theta_true a few sigma from the MAP so recovery is a real test
    offsets = {'velocity.V_rot': 25.0, 'velocity.R_v': 0.25, 'velocity.x0_v': 0.6,
               'velocity.y0_v': -0.6, 'velocity.theta_v': 0.06, 'velocity.inc_v': -0.04,
               'image.R.dx': 0.15, 'image.R.dy': -0.15,
               'image.C.dx': 0.15, 'image.C.dy': -0.15}
    theta_true = {n: theta_map[n] + offsets.get(n, 0.0) for n in names}

    f.make_mock_data(intensity=source, theta=theta_true, seed=args.seed, inplace=True)
    # re-fit the mock: seeding the walkers at theta_true would flatter the recovery
    f.fit_all()
    f.current_stage = 0
    f._assign_cfgs_for_stage(0)
    f.setup_noise_model()            # refreeze mask/footprint on the mock
    chain, _ = f.fit_MCMC(nwalkers=args.nwalkers, nsteps=args.nsteps, nburn=args.nburn)

    LOG.info('--- single-mock recovery ---')
    ok = True
    for j, pname in enumerate(f.mcmc_param_names):
        lo, med, hi = np.percentile(chain[:, j], [16, 50, 84])
        if pname in f.mcmc_extra_param_names:
            LOG.info(f'  {pname:20s} {med:10.4f} [{lo:.4f}, {hi:.4f}]  (no truth)')
            continue
        true = theta_true[pname]
        sigma = 0.5*(hi - lo)
        pull = (med - true)/sigma if sigma > 0 else np.nan
        flag = '' if abs(pull) < 3 else '  <-- >3 sigma'
        if abs(pull) >= 3:
            ok = False
        LOG.info(f'  {pname:20s} true={true:10.4f} med={med:10.4f} '
                 f'sigma={sigma:8.4f} pull={pull:+6.2f}{flag}')
    LOG.info('single-mock recovery: ' + ('PASS' if ok else 'FAIL (see >3 sigma above)'))
    f.restore_true_data()
    np.savez_compressed(os.path.join(args.outdir, 'validate_single.npz'),
                        chain=chain, param_names=np.array(f.mcmc_param_names, dtype=object),
                        theta_true=np.array([theta_true.get(n, np.nan)
                                             for n in f.mcmc_param_names]))
    return 0 if ok else 1


# ── test 3: ensemble coverage ──────────────────────────────────────────────────

def _ensemble_worker(job):
    '''
    One mock + one short chain, in its own process.

    Everything here is rebuilt from picklable arguments: the fitter holds grism
    forward-model closures, which cannot cross a process boundary, and macOS
    spawns rather than forks.
    '''
    idx, seed, theta_true, config, workdir, nwalkers, nsteps, nburn = job
    os.chdir(workdir)
    logging.getLogger('dingo.kinematics').setLevel(logging.ERROR)
    logging.getLogger('dingo.fitting').setLevel(logging.ERROR)
    torch.set_num_threads(1)
    try:
        f, names = build_fitter(config)
        f._reset_state()
        with torch.no_grad():
            f.loss()
        source = f.image_R.detach().clone()
        f.make_mock_data(intensity=source, theta=theta_true, seed=seed, inplace=True)
        # Re-fit the mock rather than starting the walkers at theta_true: seeding
        # the chain at the truth would flatter both the coverage and the bias.
        f.fit_all()
        f.current_stage = 0
        f._assign_cfgs_for_stage(0)
        f.setup_noise_model()
        chain, _ = f.fit_MCMC(nwalkers=nwalkers, nsteps=nsteps, nburn=nburn, seed=seed)
        lo, med, hi = np.percentile(chain, [16, 50, 84], axis=0)
        return idx, med, 0.5*(hi - lo), lo, hi, list(f.mcmc_param_names)
    except Exception as exc:                       # a dead worker must not kill the run
        logging.getLogger('validate').error(f'mock {idx} failed: {exc!r}')
        return idx, None, None, None, None, None


def cmd_ensemble(args):
    import multiprocessing as mp

    f, names = build_fitter(args.config)
    theta_true = current_theta(f, names)
    param_names = list(f.mcmc_param_names)

    # theta_true comes from Adam, which ignores the priors. Where a prior is tight
    # enough to actually pull the posterior, truth must be moved onto it or the
    # calibration is meaningless: the prior would drag every mock away from its own
    # truth, driving coverage to zero for reasons that say nothing about the
    # estimator. Only inc_v is in that regime here (prior sigma 0.044 against a
    # posterior width of 0.029); the centres and offsets carry deliberately weak
    # priors (sigma 2 px against a posterior width of ~0.1 px) and pull nothing, so
    # they are left at their fitted values. Hence this is explicit rather than
    # automatic -- see --truth-override.
    for key, value in (args.truth_override or {}).items():
        if key not in theta_true:
            raise KeyError(f'--truth-override {key}: not a sampled parameter')
        LOG.info(f'theta_true[{key}] = {value} (was {theta_true[key]:.4f})')
        theta_true[key] = float(value)

    for name, spec in f._mcmc_priors:
        if spec.get('type') == 'gaussian' and name in theta_true:
            pull = abs(theta_true[name] - spec['mu'])/spec['sigma']
            if pull > 1.0:
                LOG.warning(
                    f'theta_true[{name}] is {pull:.1f} prior sigma from its prior mean '
                    f'({theta_true[name]:.4f} vs {spec["mu"]:.4f} +- {spec["sigma"]:.4f}). '
                    f'If that prior is informative, coverage for it will be biased low; '
                    f'consider --truth-override {name}={spec["mu"]}'
                )

    LOG.info(f'theta_true = { {k: round(v,4) for k,v in theta_true.items()} }')
    del f

    jobs = [(i, 1000 + i, theta_true, args.config, args.workdir,
             args.nwalkers, args.nsteps, args.nburn) for i in range(args.K)]
    ctx = mp.get_context('spawn')
    LOG.info(f'running {args.K} mocks on {args.nproc} processes '
             f'({args.nwalkers}x{args.nsteps} after {args.nburn} burn each)')
    with ctx.Pool(args.nproc) as pool:
        results = pool.map(_ensemble_worker, jobs)

    good = [r for r in results if r[1] is not None]
    LOG.info(f'{len(good)}/{args.K} mocks completed')
    if len(good) < 3:
        LOG.error('too few successful mocks to calibrate')
        return 1

    meds = np.array([r[1] for r in good])
    sigmas = np.array([r[2] for r in good])
    los = np.array([r[3] for r in good])
    his = np.array([r[4] for r in good])
    truth_vec = np.array([theta_true.get(n, np.nan) for n in param_names])

    LOG.info('--- ensemble calibration ---')
    LOG.info(f'{"parameter":22s} {"bias":>10s} {"scatter":>10s} {"width":>10s} '
             f'{"inflation":>10s} {"coverage":>9s}')
    table = {}
    for j, pname in enumerate(param_names):
        scatter = float(np.std(meds[:, j], ddof=1))
        width = float(np.mean(sigmas[:, j]))
        inflation = scatter/width if width > 0 else np.nan
        if np.isnan(truth_vec[j]):
            bias, coverage = np.nan, np.nan
        else:
            bias = float(np.mean(meds[:, j]) - truth_vec[j])
            coverage = float(np.mean((los[:, j] <= truth_vec[j]) & (truth_vec[j] <= his[:, j])))
        table[pname] = dict(bias=bias, scatter=scatter, width=width,
                            inflation=inflation, coverage=coverage)
        LOG.info(f'{pname:22s} {bias:10.4f} {scatter:10.4f} {width:10.4f} '
                 f'{inflation:10.2f} {coverage:9.2f}')

    finite = [v['inflation'] for v in table.values() if np.isfinite(v['inflation'])]
    LOG.info(f'median inflation factor = {np.median(finite):.2f}')
    LOG.info('Multiply the production posterior widths by the per-parameter '
             'inflation factor above before quoting any uncertainty.')

    np.savez_compressed(
        os.path.join(args.outdir, f'ID{args.id}_ensemble.npz'),
        param_names=np.array(param_names, dtype=object),
        theta_true=truth_vec, medians=meds, sigmas=sigmas, lo=los, hi=his,
        inflation=np.array([table[n]['inflation'] for n in param_names]),
        coverage=np.array([table[n]['coverage'] for n in param_names]),
    )
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('test', choices=['floor', 'single', 'ensemble'])
    ap.add_argument('--id', default=DEFAULT_ID)
    ap.add_argument('--config', default=None,
                    help='defaults to ID<id>/config_kinematics.yaml under --workdir')
    ap.add_argument('--workdir', default=DEFAULT_WORKDIR)
    ap.add_argument('--outdir', default=DEFAULT_OUTDIR)
    ap.add_argument('-K', type=int, default=20, help='number of mocks (ensemble)')
    ap.add_argument('--nproc', type=int, default=6)
    ap.add_argument('--nwalkers', type=int, default=32)
    ap.add_argument('--nsteps', type=int, default=1200)
    ap.add_argument('--nburn', type=int, default=400)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--truth-override', nargs='*', metavar='KEY=VALUE',
                    help='force theta_true entries, e.g. velocity.inc_v=0.6412')
    args = ap.parse_args()
    if args.config is None:
        args.config = f'ID{args.id}/config_kinematics.yaml'
    # absolute before the chdir below, and one folder per source
    args.outdir = os.path.join(os.path.abspath(args.outdir), f'ID{args.id}')
    os.makedirs(args.outdir, exist_ok=True)
    args.truth_override = dict(
        (kv.split('=', 1)[0], kv.split('=', 1)[1]) for kv in (args.truth_override or []))

    setup_logging()
    os.chdir(args.workdir)
    os.makedirs(args.outdir, exist_ok=True)
    return {'floor': cmd_floor, 'single': cmd_single, 'ensemble': cmd_ensemble}[args.test](args)


if __name__ == '__main__':
    sys.exit(main())
