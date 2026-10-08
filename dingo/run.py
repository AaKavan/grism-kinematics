"""Run one source end to end.

    python -m dingo.run 21636                  # adam -> report -> mcmc -> maps
    python -m dingo.run 21636 --stage adam     # just the optimiser, with the report
    python -m dingo.run 21636 --stage mcmc     # sample (resumes if a checkpoint exists)
    python -m dingo.run 21636 --stage maps     # rectified images at the saved chain's MAP
    python -m dingo.run 21636 --stage null     # rotation-vs-no-rotation test
    python -m dingo.run 21636 --strict         # refuse to continue past a bad Adam fit

This exists because there was no single entry point. The closest thing,
geko_dingo_compare.run_dingo(), lives outside the repo and HARDCODES
nwalkers/nsteps/nburn into the fit_MCMC call, silently overriding whatever the
config asks for -- which is why scripts/run_21636_showcase.py had to exist as a
separate copy that calls bare fit_MCMC() just to let ID21636's own 30000/5000
take effect. Everything here reads the config and nothing overrides it.

The MCMC settings, the priors, the noise model and the checkpoint path all come
from the config's `mcmc:` block. If you want different ones, edit the config --
that way the run is reproducible from a file that can be committed.
"""
import argparse
import logging
import os
import sys

LOG = logging.getLogger('dingo.run')


def _setup_logging(verbose):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s %(levelname)-7s %(name)s | %(message)s',
        datefmt='%H:%M:%S')
    for noisy in ('matplotlib', 'PIL'):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def resolve(source, data_dir=None, config=None):
    """(config path, data dir, source tag) from an ID, a tag or a config path."""
    if config:
        config = os.path.abspath(config)
        return config, data_dir or os.path.dirname(os.path.dirname(config)), \
            os.path.basename(os.path.dirname(config))
    tag = source if str(source).startswith('ID') else f'ID{source}'
    data_dir = data_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    return os.path.join(data_dir, tag, 'config_kinematics.yaml'), data_dir, tag


def chain_path(data_dir, tag):
    """Where save_mcmc writes, matching the layout already on disk."""
    results = os.path.join(os.path.dirname(data_dir), 'results')
    for cand in (os.path.join(results, f'{tag}_chain.npz'),
                 os.path.join(results, tag, f'{tag}_chain.npz')):
        if os.path.exists(cand):
            return cand
    return os.path.join(results, f'{tag}_chain.npz')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('source', help='source id (21636), tag (ID21636), or --config path')
    ap.add_argument('--stage', default='all',
                    choices=['all', 'adam', 'mcmc', 'maps', 'null', 'report'])
    ap.add_argument('--config', default=None, help='explicit config path')
    ap.add_argument('--data-dir', default=None)
    ap.add_argument('--out', default=None, help='npz for --stage maps')
    ap.add_argument('--strict', action='store_true',
                    help='stop if fit_report() finds a railed or unconverged parameter')
    ap.add_argument('--device', default=None)
    ap.add_argument('-v', '--verbose', action='store_true')
    args = ap.parse_args(argv)
    _setup_logging(args.verbose)

    cfg_path, data_dir, tag = resolve(args.source, args.data_dir, args.config)
    if not os.path.exists(cfg_path):
        ap.error(f'no config at {cfg_path}')

    import torch
    from dingo import fitting

    # The configs' `path:` entries are relative to data/.
    os.chdir(data_dir)
    device = torch.device(args.device) if args.device else None
    f = fitting.KinematicsFitter(config_path=os.path.relpath(cfg_path, data_dir),
                                 device=device)
    LOG.info(f'{tag}: {cfg_path}')

    stage = args.stage
    did_fit = False
    if stage in ('all', 'adam', 'mcmc', 'maps', 'null', 'report'):
        # Everything needs the optimiser first: the noise model is frozen at the
        # Adam point, and the sampler starts from it.
        f.fit_all()
        did_fit = True
        f.fit_report(strict=args.strict)
    if stage == 'adam' or stage == 'report':
        return 0

    if stage in ('all', 'mcmc'):
        # No overrides: nwalkers/nsteps/nburn/priors/checkpoint all come from the
        # config, which is the whole point of this entry point.
        f.fit_MCMC()
        out = chain_path(data_dir, tag)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        f.save_mcmc(out)
        LOG.info(f'chain -> {out}')
        diag = f.get_mcmc_diagnostics()
        worst = min(diag['n_steps_over_tau'].values())
        LOG.info(f'acceptance {diag["acceptance_fraction"][0]:.3f} | worst n/tau {worst:.1f} '
                 f'| max tau {max(diag["tau"].values()):.1f}')
        if worst < 50:
            LOG.warning(f'worst n/tau = {worst:.1f} < 50: the chain is too short to quote. '
                        f'Raise mcmc.nsteps and re-run; it will resume.')
        if stage == 'mcmc':
            return 0

    if stage in ('all', 'maps', 'null'):
        f.prepare_mcmc()
        f.load_chain(chain_path(data_dir, tag), prepare=False)

    if stage in ('all', 'null'):
        f.set_params_to_mcmc('map')
        res = f.compare_null_model()
        LOG.info(f'rotation vs no rotation: dlnL={res["dlnL"]:+.1f} '
                 f'({res["sigma_equiv"]:.1f} sigma, dBIC={res["dBIC"]:+.1f})')
        if stage == 'null':
            return 0

    if stage in ('all', 'maps'):
        import numpy as np
        theta = f.set_params_to_mcmc('map')
        R, C, vz_R, vz_C = f.get_fitting_results()
        tR, tC = f.get_true_images()
        out = args.out or os.path.join(os.path.dirname(data_dir), 'maps', f'dingo_{tag}.npz')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(
            out, R=R, C=C, vz_R=vz_R, vz_C=vz_C, true_R=tR, true_C=tC,
            mask=f.mcmc_mask.detach().cpu().numpy(),
            var_map=f.mcmc_var_map.detach().cpu().numpy(),
            theta=theta, param_names=np.array(f.mcmc_param_names, dtype=object))
        LOG.info(f'maps -> {out}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
