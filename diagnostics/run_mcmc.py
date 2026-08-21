#!/usr/bin/env python
'''
MCMC posterior estimation for a DINGO kinematics fit.

Runs the standard Adam fit to find the MAP, freezes a noise model, samples the
posterior with emcee, and writes the chain, diagnostics and plots.

    /opt/anaconda3/envs/DINGO/bin/python run_mcmc_ID15665.py [--quick]

Must be run from a directory where the config's relative `path:` entries resolve
(Archive_6/ for the ID15665 config); the script chdir's there itself.
'''

import argparse
import logging
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))   # pipeline/
ROOT = os.path.dirname(HERE)                        # DINGO_development/
GK   = os.path.dirname(ROOT)                        # Galaxy_Kinematics/

# Must be set before dingo.grism is touched. The archived notebook hardcodes
# /data/grism_cal, which only exists on the magnif server.
os.environ.setdefault(
    'GRISM_CAL_DIR',
    os.path.join(GK, 'grism_cal')
)

import numpy as np
import torch

from dingo import fitting, plot

LOG = logging.getLogger('run_mcmc')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='ID15665/config_kinematics.yaml')
    ap.add_argument('--workdir', default=os.path.join(ROOT, 'data'))
    ap.add_argument('--outdir', default=os.path.join(ROOT, 'results'))
    ap.add_argument('--quick', action='store_true',
                    help='short chain for a smoke test (32x400 after 100 burn)')
    ap.add_argument('--nwalkers', type=int, default=None)
    ap.add_argument('--nsteps', type=int, default=None)
    ap.add_argument('--nburn', type=int, default=None)
    ap.add_argument('--no-plots', action='store_true')
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    # iteratively_find_xy is called with tol=0 during sampling, so it reports
    # "not converged" on every call by construction. fit_MCMC has its own,
    # footprint-restricted convergence guard.
    logging.getLogger('dingo.kinematics').setLevel(logging.ERROR)

    # 81x81 tensors are far below torch's intra-op parallel threshold, so extra
    # threads are pure overhead and oversubscribe if several galaxies run at once.
    torch.set_num_threads(1)

    os.chdir(args.workdir)
    os.makedirs(args.outdir, exist_ok=True)

    if args.quick:
        args.nwalkers = args.nwalkers or 32
        args.nsteps = args.nsteps or 400
        args.nburn = args.nburn or 100

    # ── 1. MAP via the existing gradient fit ────────────────────────────────
    fitter = fitting.KinematicsFitter(config_path=args.config)
    LOG.info(f'cutouts: R={fitter.cutout_R} C={fitter.cutout_C}')
    fitter.fit_all()

    velocity, disp_R, disp_C = fitter.get_params()
    LOG.info('--- Adam MAP ---')
    for k, v in velocity.items():
        LOG.info(f'  {k:10s} = {float(v):.4f}')
    LOG.info(f'  R  dx,dy  = {float(disp_R["dx"]):.4f}, {float(disp_R["dy"]):.4f}')
    LOG.info(f'  C  dx,dy  = {float(disp_C["dx"]):.4f}, {float(disp_C["dy"]):.4f}')

    # ── 2. Sample ───────────────────────────────────────────────────────────
    # fit_MCMC freezes the noise model itself (mcmc.stage selects which stage's
    # fit flags define the sampled set).
    chain, log_prob = fitter.fit_MCMC(
        nwalkers=args.nwalkers, nsteps=args.nsteps, nburn=args.nburn
    )
    if len(chain) == 0:
        LOG.error('no parameters were sampled; check the config')
        return 1

    name = fitter.name
    chain_path = os.path.join(args.outdir, f'{name}_chain.npz')
    fitter.save_mcmc(chain_path)

    # ── 3. Posterior summary ────────────────────────────────────────────────
    results = fitter.get_mcmc_results()
    LOG.info('--- posterior: median, -16th, +84th ---')
    for pname, (lo, med, hi) in results.items():
        LOG.info(f'  {pname:20s} {med:12.4f}  -{med-lo:.4f} +{hi-med:.4f}')

    diag = fitter.get_mcmc_diagnostics()
    acc_mean, acc_min, acc_max = diag['acceptance_fraction']
    LOG.info('--- diagnostics ---')
    LOG.info(f'  acceptance   mean={acc_mean:.3f} min={acc_min:.3f} max={acc_max:.3f} '
             f'(want 0.2-0.5)')
    LOG.info(f'  -inf fraction  {diag["frac_neg_inf"]:.3g}')
    worst = min(diag['n_steps_over_tau'].items(), key=lambda kv: kv[1])
    LOG.info(f'  n_steps/tau  worst = {worst[1]:.1f} for {worst[0]} (want > 50)')
    for pname in fitter.mcmc_param_names:
        LOG.info(f'    {pname:20s} tau={diag["tau"][pname]:8.1f}  '
                 f'n/tau={diag["n_steps_over_tau"][pname]:7.1f}  '
                 f'split-half shift={diag["split_half_shift"][pname]:.3f} sigma')
    if worst[1] < 50:
        LOG.warning('  chain is too short for a reliable autocorrelation time; '
                    'increase mcmc.nsteps')

    # ── 4. MAP cross-check ──────────────────────────────────────────────────
    # log_likelihood() caches the model images of whatever sample it saw last, so
    # the fitter must be put back on a meaningful point before anything is plotted.
    theta_map = fitter.set_params_to_mcmc('map')
    LOG.info('--- posterior MAP vs Adam MAP (in posterior sigma) ---')
    for j, pname in enumerate(fitter.mcmc_param_names):
        # the nuisance dims never had an Adam value to compare against
        if pname in fitter.mcmc_extra_param_names:
            continue
        lo, med, hi = results[pname]
        sigma = 0.5*(hi - lo)
        shift = (theta_map[j] - fitter.mcmc_theta0[j])/sigma if sigma > 0 else np.nan
        LOG.info(f'  {pname:20s} {shift:+.2f} sigma')

    # ── 5. Derived posteriors ───────────────────────────────────────────────
    kpc_per_pix = kpc_per_pixel(fitter.z)
    names = list(fitter.mcmc_param_names)
    v_rot = chain[:, names.index('velocity.V_rot')]
    r_v = chain[:, names.index('velocity.R_v')]
    LOG.info(f'--- derived (kpc/pixel = {kpc_per_pix:.3f} at z={fitter.z}) ---')
    lo, med, hi = np.percentile(r_v*kpc_per_pix, [16, 50, 84])
    LOG.info(f'  R_v [kpc]            {med:12.4f}  -{med-lo:.4f} +{hi-med:.4f}')
    for radius_kpc in (2.0, 5.0):
        m = arctangent_dynamical_mass(radius_kpc, v_rot, r_v*kpc_per_pix)
        lo, med, hi = np.percentile(np.log10(m), [16, 50, 84])
        LOG.info(f'  log10 M_dyn(<{radius_kpc:.0f} kpc) {med:12.4f}  '
                 f'-{med-lo:.4f} +{hi-med:.4f}')

    np.savez_compressed(
        os.path.join(args.outdir, f'{name}_derived.npz'),
        param_names=np.array(names, dtype=object),
        kpc_per_pix=kpc_per_pix,
        R_v_kpc=r_v*kpc_per_pix,
        Vsini=v_rot*np.sin(chain[:, names.index('velocity.inc_v')]),
    )

    # ── 6. Plots ────────────────────────────────────────────────────────────
    if not args.no_plots:
        import matplotlib
        matplotlib.use('Agg')
        full_chain = fitter.mcmc_sampler.get_chain()
        plot.plot_mcmc_chains(
            full_chain, names, nburn=fitter.mcmc_nburn, show=False,
            filename=os.path.join(args.outdir, f'{name}_chains.png'))
        plot.plot_mcmc_corner(
            chain, names, truths=list(fitter.mcmc_theta0), show=False,
            filename=os.path.join(args.outdir, f'{name}_corner.png'))
        LOG.info(f'plots written to {args.outdir}')

    LOG.info('IMPORTANT: these posterior widths are the raw likelihood widths. The '
             'R-C residuals are spatially correlated, so they must be multiplied by '
             'the inflation factors measured by validate_mcmc.py before being quoted.')
    return 0


def kpc_per_pixel(z, arcsec_per_pixel=0.063):
    '''Angular scale of a NIRCam long-wavelength pixel at redshift z.'''
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    d_a = cosmo.angular_diameter_distance(z).to(u.kpc).value
    return arcsec_per_pixel*d_a/206265.0


def arctangent_dynamical_mass(r_kpc, V_rot, R_v_kpc):
    '''
    M(<r) = V(r)^2 r / G for the arctangent rotation curve, in solar masses.
    Vectorised over chain samples. Same form as cell 24 of 1-fit-kinematics.ipynb.
    '''
    G = 4.302e-6  # kpc (km/s)^2 / Msun
    V = (2.0/np.pi)*V_rot*np.arctan(r_kpc/R_v_kpc)
    return V**2*r_kpc/G


if __name__ == '__main__':
    sys.exit(main())
