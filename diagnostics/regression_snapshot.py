#!/usr/bin/env python
'''
Record, and later re-check, the numbers a refactor must not move.

    python diagnostics/regression_snapshot.py record        # before changing anything
    python diagnostics/regression_snapshot.py check         # after

The point of a refactor is that it changes how the code is organised and not
what it computes. That is a claim, and this is how it gets tested: freeze an
end-to-end evaluation of every source that has a converged chain, then assert
the same numbers come back.

Four groups are snapshotted per source, deliberately separated so a failure
says WHICH layer moved:

  adam    the parameter values fit_all() lands on. Moves if the optimiser, its
          bounds or its staging change -- which Phase 1.2 (bound inheritance)
          is specifically meant to do, so an intended change shows up here and
          nowhere else.
  noise   sigma_R/sigma_C, the mask size, N_eff and the mean variance, i.e.
          everything setup_noise_model() freezes at the Adam point.
  logp    log_probability at the SAVED CHAIN MAP under that frozen noise model.
          This is the end-to-end number: priors, likelihood, rectification and
          noise model all feed it.
  chi2    the raw chi2 at the same point, so a change can be attributed to the
          likelihood's normalisation rather than to the fit.

Tolerance is exact for logp/chi2 (float equality on a deterministic CPU path)
and 1e-10 relative for the rest. If a change is intended, re-record and say in
the commit message which numbers moved and why.
'''
import json, os, sys

os.environ.setdefault('GRISM_CAL_DIR', '/home/aakavan/kinematics/grism_cal')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')   # CPU: deterministic
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import logging, warnings
warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

import numpy as np
import torch
from dingo import fitting

SNAP = os.path.join(REPO, 'tests', 'regression_baseline.json')
# Only sources with a converged, current-generation chain on disk.
SOURCES = {
    'ID21636': '../results/ID21636_chain.npz',
    'ID15665': '../results/ID15665_chain.npz',
}


def snapshot(sid, chain_path):
    torch.set_num_threads(2)
    os.chdir(os.path.join(REPO, 'data'))
    f = fitting.KinematicsFitter(config_path=f'{sid}/config_kinematics.yaml')
    f.fit_all()

    # Reproduce fit_MCMC's preamble well enough to evaluate the likelihood.
    # NOTE: this hand-rolled block is exactly what Phase 2.1 replaces with
    # prepare_mcmc(); when that lands, this becomes two lines and the snapshot
    # must still match.
    mc = f.config.get('mcmc') or {}
    stage = mc.get('stage')
    if stage is not None and int(stage) != f.current_stage:
        f.current_stage = int(stage)
        f._assign_cfgs_for_stage(f.current_stage)
    f.mcmc_xy_iters = int(mc.get('xy_iters', 40))
    f.mcmc_xy_tol = float(mc.get('xy_tol', 5e-3))
    f._mcmc_names, f._mcmc_cfgs, _ = fitting.collect_free_scalar_params(f._mcmc_stage_cfg())
    f.mcmc_param_names = list(f._mcmc_names) + list(f.mcmc_extra_param_names)
    f._mcmc_prepare()
    f._mcmc_priors = f._build_mcmc_priors()

    adam = {n: float(c.tensor.detach().cpu()) for n, c in zip(f._mcmc_names, f._mcmc_cfgs)}
    noise = {
        'sigma_R': float(f.sigma_R), 'sigma_C': float(f.sigma_C),
        'n_pix': int(f.n_mcmc_pix), 'n_eff': float(f.mcmc_n_eff),
        'corr_area': float(f.mcmc_corr_area), 'var_base': float(f.mcmc_var_base),
    }

    # The saved chain's MAP: a FIXED theta, so logp isolates the likelihood from
    # wherever Adam happened to land.
    d = fitting.BaseFitter.load_mcmc(chain_path)
    chain, lp = np.asarray(d['chain']), np.asarray(d['log_prob'])
    names = [str(n) for n in d['param_names']]
    flat, flat_lp = chain.reshape(-1, chain.shape[-1]), lp.reshape(-1)
    theta_map = flat[int(np.nanargmax(flat_lp))]
    assert names == list(f.mcmc_param_names), (names, list(f.mcmc_param_names))

    logp = float(f.log_probability(theta_map))
    chi2 = float(f._mcmc_last_chi2)
    return {'adam': adam, 'noise': noise, 'theta_map': [float(v) for v in theta_map],
            'param_names': names, 'logp': logp, 'chi2': chi2}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'check'
    new = {sid: snapshot(sid, p) for sid, p in SOURCES.items()}

    if mode == 'record':
        os.makedirs(os.path.dirname(SNAP), exist_ok=True)
        with open(SNAP, 'w') as fh:
            json.dump(new, fh, indent=2, sort_keys=True)
        print(f'recorded {len(new)} source(s) -> {SNAP}')
        for sid, s in new.items():
            print(f'  {sid}: logp={s["logp"]:.10g} chi2={s["chi2"]:.10g} '
                  f'n_pix={s["noise"]["n_pix"]} n_eff={s["noise"]["n_eff"]:.1f}')
        return 0

    with open(SNAP) as fh:
        old = json.load(fh)
    bad = []
    for sid, s in new.items():
        o = old.get(sid)
        if o is None:
            bad.append(f'{sid}: not in baseline'); continue
        for k in ('logp', 'chi2'):
            if s[k] != o[k]:
                bad.append(f'{sid}.{k}: {o[k]!r} -> {s[k]!r}  (delta {s[k]-o[k]:+.6g})')
        for grp in ('adam', 'noise'):
            for k, v in s[grp].items():
                ov = o[grp][k]
                if ov != v and abs(v - ov) > 1e-10*max(1.0, abs(ov)):
                    bad.append(f'{sid}.{grp}.{k}: {ov!r} -> {v!r}  (delta {v-ov:+.6g})')
        print(f'  {sid}: logp={s["logp"]:.10g} chi2={s["chi2"]:.10g}')
    if bad:
        print('\nREGRESSION -- these numbers moved:')
        for b in bad:
            print('  ' + b)
        print('\nIf a change was intended, re-record and name the moved numbers in the commit.')
        return 1
    print('\nOK: every snapshotted number is unchanged.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
