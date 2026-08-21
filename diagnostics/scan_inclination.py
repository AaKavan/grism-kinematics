'''
Does the R-C objective actually constrain inclination?

DINGO differences two grisms dispersed in PERPENDICULAR directions, so unlike a
single-roll code it should not need a Sersic light model to separate spatial
extent from velocity -- the two views constrain the intrinsic map between them.
geko says as much about R+C data (arXiv:2510.07369), and lists joint two-direction
fitting as future work it has not implemented.

Yet the fitted posterior gives corr(V_rot, inc_v) = -0.99. Either the information
is present and the sampler is not using it, or the objective genuinely does not
respond to inclination. This measures which.

Two scans, both in delta chi2 = delta SSE / var_base over the frozen MCMC mask:

  conditional  vary inc with V_rot = Vsini_MAP / sin(inc) and everything else
               pinned at the MAP. Fast, but pessimistic: nuisance parameters
               cannot absorb anything, so structure here is a lower bound on
               what is available.

  profile      at each inc, FREEZE inc and re-fit every other parameter. This is
               the honest test. A flat profile means inclination is genuinely
               unconstrained; a minimum means the information is there and the
               fit should be finding it.

delta chi2 = 1 is the 1-sigma interval, 4 is 2-sigma, 9 is 3-sigma.
'''

import os
import sys
import argparse
import logging

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GK = os.path.dirname(ROOT)
os.environ.setdefault('GRISM_CAL_DIR', os.path.join(GK, 'grism_cal'))
sys.path.insert(0, os.path.join(ROOT, 'grism-kinematics'))

from dingo import fitting  # noqa: E402

LOG = logging.getLogger('scan_inc')


def masked_sse(f):
    '''SSE over the frozen mask, from the same cold-start fixed-iteration path
    the MCMC likelihood uses. Returns np.inf when the fixed point fails.'''
    ll = f.log_likelihood(**{'noise.ln_f': 0.0})
    if not np.isfinite(ll):
        return np.inf
    var = f.mcmc_var_base
    n = f.n_mcmc_pix
    # ll = -0.5*sse/var - 0.5*n*ln(2 pi var)
    return -2.0*var*(ll + 0.5*n*np.log(2.0*np.pi*var))


def scan(source_id, n_inc=21, inc_lo=0.15, inc_hi=1.40, profile_maxiter=600,
         workdir=None, outdir=None):
    workdir = workdir or os.path.join(ROOT, 'data')
    outdir = outdir or os.path.join(ROOT, 'results', f'ID{source_id}')
    os.makedirs(outdir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        f = fitting.KinematicsFitter(
            config_path=f'ID{source_id}/config_kinematics.yaml')
        f.fit_all()
        f.setup_noise_model()
        f.current_stage = 0
        f._assign_cfgs_for_stage(0)

        all_cfg = f._mcmc_stage_cfg()
        names, cfgs, _ = fitting.collect_free_scalar_params(all_cfg)
        f._mcmc_names, f._mcmc_cfgs = names, cfgs
        map_vals = {n: float(c.tensor.detach()) for n, c in zip(names, cfgs)}
        i_map = map_vals['velocity.inc_v']
        v_map = map_vals['velocity.V_rot']
        vsini = v_map*np.sin(i_map)
        LOG.info(f'ID{source_id} MAP: V_rot={v_map:.2f} inc={i_map:.4f} '
                 f'({np.rad2deg(i_map):.2f} deg) Vsini={vsini:.2f} '
                 f'R_v={map_vals["velocity.R_v"]:.3f} px')

        def restore():
            f.set_params_dict(map_vals)

        incs = np.linspace(inc_lo, inc_hi, n_inc)

        # ---- conditional: everything else pinned at the MAP --------------
        cond = np.full(n_inc, np.inf)
        for k, inc in enumerate(incs):
            restore()
            f.set_params_dict({'velocity.inc_v': float(inc),
                               'velocity.V_rot': float(vsini/np.sin(inc))})
            cond[k] = masked_sse(f)
        restore()

        # ---- profile: freeze inc, re-fit everything else ------------------
        inc_cfg = dict(zip(names, cfgs))['velocity.inc_v']
        stash = f.config['fitting'][0]['maxiter']
        f.config['fitting'][0]['maxiter'] = profile_maxiter
        prof = np.full(n_inc, np.inf)
        prof_pars = []
        for k, inc in enumerate(incs):
            restore()
            f.set_params_dict({'velocity.inc_v': float(inc),
                               'velocity.V_rot': float(vsini/np.sin(inc))})
            inc_cfg.fit = False              # propagates to tensor.requires_grad
            try:
                f.current_stage = 0
                f._assign_cfgs_for_stage(0)
                f.fit_gradient()
                prof[k] = masked_sse(f)
                prof_pars.append({n: float(c.tensor.detach())
                                  for n, c in zip(names, cfgs)})
            except Exception as exc:
                LOG.warning(f'  inc={inc:.3f} profile fit failed: {exc}')
                prof_pars.append({})
            finally:
                inc_cfg.fit = True
        f.config['fitting'][0]['maxiter'] = stash
        restore()

        var = f.mcmc_var_base
        d_cond = (cond - np.nanmin(cond[np.isfinite(cond)]))/var
        d_prof = (prof - np.nanmin(prof[np.isfinite(prof)]))/var

        LOG.info(f'\n{"inc rad":>9}{"inc deg":>9}{"V_rot@Vsini":>13}'
                 f'{"dchi2 cond":>12}{"dchi2 prof":>12}{"V_rot fitted":>14}')
        for k, inc in enumerate(incs):
            vfit = prof_pars[k].get('velocity.V_rot', np.nan) if k < len(prof_pars) else np.nan
            LOG.info(f'{inc:>9.4f}{np.rad2deg(inc):>9.2f}{vsini/np.sin(inc):>13.1f}'
                     f'{d_cond[k]:>12.2f}{d_prof[k]:>12.2f}{vfit:>14.1f}')

        good = np.isfinite(d_prof)
        best = incs[good][np.argmin(d_prof[good])]
        within1 = incs[good][d_prof[good] < 1.0]
        LOG.info(f'\nprofile minimum at inc = {best:.4f} rad ({np.rad2deg(best):.2f} deg)')
        if len(within1):
            LOG.info(f'delta chi2 < 1 spans {within1.min():.4f}-{within1.max():.4f} rad '
                     f'({np.rad2deg(within1.min()):.1f}-{np.rad2deg(within1.max()):.1f} deg)')
        LOG.info(f'profile range: dchi2 = {np.nanmin(d_prof[good]):.2f} to '
                 f'{np.nanmax(d_prof[good]):.2f}')
        LOG.info(f'VERDICT: the profile is '
                 f'{"FLAT -- inclination is not constrained by the R-C objective"
                    if np.nanmax(d_prof[good]) < 4 else
                    "CURVED -- inclination IS constrained; the information is there"}')

        np.savez_compressed(
            os.path.join(outdir, f'ID{source_id}_inc_scan.npz'),
            inc=incs, sse_cond=cond, sse_prof=prof,
            dchi2_cond=d_cond, dchi2_prof=d_prof,
            var_base=var, n_pix=f.n_mcmc_pix, vsini_map=vsini,
            inc_map=i_map, V_rot_map=v_map,
            R_v_map=map_vals['velocity.R_v'],
            param_names=np.array(names, dtype=object))
        return incs, d_cond, d_prof, vsini, i_map
    finally:
        os.chdir(cwd)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('ids', nargs='*', type=int, default=[15665, 8910])
    ap.add_argument('--n-inc', type=int, default=21)
    ap.add_argument('--profile-maxiter', type=int, default=600)
    ap.add_argument('--plot', action='store_true')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.getLogger('dingo.fitting').setLevel(logging.ERROR)
    logging.getLogger('dingo.kinematics').setLevel(logging.ERROR)
    torch.set_num_threads(4)

    results = {}
    for sid in (args.ids or [15665, 8910]):
        LOG.info(f'\n{"="*72}\nID {sid}\n{"="*72}')
        results[sid] = scan(sid, n_inc=args.n_inc,
                            profile_maxiter=args.profile_maxiter)

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, len(results), figsize=(6*len(results), 4.5),
                                squeeze=False)
        for ax, (sid, (incs, dc, dp, vsini, imap)) in zip(axs[0], results.items()):
            ax.plot(np.rad2deg(incs), dc, 'o-', color='0.6',
                    label='conditional (others at MAP)')
            ax.plot(np.rad2deg(incs), dp, 'o-', color='crimson',
                    label='profile (others re-fit)')
            ax.axvline(np.rad2deg(imap), color='navy', ls='--', label='MAP')
            for lv, ls in [(1, ':'), (4, '-.'), (9, '--')]:
                ax.axhline(lv, color='k', lw=0.6, ls=ls)
            ax.set_xlabel('inclination [deg]')
            ax.set_ylabel(r'$\Delta\chi^2$')
            ax.set_ylim(-1, 60)
            ax.set_title(f'ID {sid}')
            ax.grid(False)
        axs[0][0].legend(frameon=True, fontsize=8)
        plt.tight_layout()
        out = os.path.join(ROOT, 'results', 'inclination_scan.png')
        plt.savefig(out, dpi=130, bbox_inches='tight')
        LOG.info(f'\nplot -> {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
