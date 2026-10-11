#!/usr/bin/env python
'''Is the Adam fit the right minimum, or just a minimum?

    python diagnostics/multistart.py 15665 [-N 16] [--device cpu]

The optimiser's answer matters twice over: fit_MCMC starts its walkers there,
and setup_noise_model freezes the mask and per-pixel variance AT those
parameters. A fit in the wrong basin therefore gives a posterior that explores
the wrong place, weighted by the wrong noise model -- and nothing downstream
would say so.

fit_report() answers "did Adam converge", not "did it converge to the right
place". This answers the second question the only way it can be answered: run
the optimiser from many independent starts drawn from the prior and see whether
they agree.

The sample is already known to need this. data/ID15665/config_kinematics.yaml
records three different starts reaching loss 0.142019, 0.133961 and 0.4722 --
the surface is multimodal and the comment says so, but nothing checks it per
source.

Reports the distinct minima found, how often each is reached, and where the
config's own starting point lands. A config that lands outside the best basin
is the failure this is looking for.
'''
import argparse
import os
import sys

os.environ.setdefault('GRISM_CAL_DIR', '/home/aakavan/kinematics/grism_cal')
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import logging
import warnings

import numpy as np
import torch

warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

from dingo import fitting                                   # noqa: E402


def one_start(cfg, device, theta0=None):
    '''Run the full staged fit once. theta0=None uses the config's own values.'''
    f = fitting.KinematicsFitter(config_path=cfg, device=device)
    if theta0 is not None:
        f.set_params_dict(theta0)
    f.fit_all()
    all_cfg = f._mcmc_stage_cfg()
    names, cfgs, _ = fitting.collect_free_scalar_params(all_cfg)
    theta = {n: float(c.tensor.detach().cpu()) for n, c in zip(names, cfgs)}
    return float(f.losses[-1]), theta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('source')
    ap.add_argument('-N', type=int, default=16, help='random starts (default 16)')
    ap.add_argument('--device', default='cpu',
                    help="cpu (default) leaves the GPU free for chains; Adam is "
                         "faster on CPU anyway -- it drives the scalar path")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--rtol', type=float, default=1e-3,
                    help='two losses within this relative tolerance are one basin')
    args = ap.parse_args()

    tag = args.source if str(args.source).startswith('ID') else f'ID{args.source}'
    os.chdir(os.path.join(REPO, 'data'))
    cfg = f'{tag}/config_kinematics.yaml'
    device = torch.device(args.device)
    torch.set_num_threads(2)

    # The config's own start, which is what every run so far has used.
    loss_cfg, theta_cfg = one_start(cfg, device)
    print(f'{tag}: config start -> loss {loss_cfg:.6g}')

    # Independent starts drawn from the SAMPLED priors. prepare_mcmc gives the
    # prior objects; _draw_mcmc_prior is what fit_MCMC's `init: prior` uses.
    probe = fitting.KinematicsFitter(config_path=cfg, device=device)
    setup = probe.prepare_mcmc()
    rng = np.random.default_rng(args.seed)

    print(f'running {args.N} starts from the prior on {args.device} '
          f'({len(setup["names"])} free parameters)...')
    out = []
    for i in range(args.N):
        draw = probe._draw_mcmc_prior(rng)
        theta = {n: float(v) for n, v in zip(setup['param_names'], draw)
                 if not n.startswith('noise.')}
        try:
            loss, th = one_start(cfg, device, theta)
            out.append((loss, th))
            print(f'  start {i+1:>3}/{args.N}  loss {loss:.6g}')
        except Exception as exc:
            print(f'  start {i+1:>3}/{args.N}  FAILED {type(exc).__name__}: {str(exc)[:70]}')

    if not out:
        print('no start converged'); return 1

    losses = np.array([l for l, _ in out])
    order = np.argsort(losses)

    # Cluster by loss: successive values within rtol are the same basin.
    basins, cur = [], [order[0]]
    for a, b in zip(order, order[1:]):
        if abs(losses[b] - losses[a]) <= args.rtol*max(abs(losses[a]), 1e-30):
            cur.append(b)
        else:
            basins.append(cur); cur = [b]
    basins.append(cur)

    best = losses[order[0]]
    print(f'\n{len(basins)} distinct minimum/minima among {len(out)} starts '
          f'(same basin = loss within {args.rtol:g} relative)\n')
    print('%6s %12s %9s %10s   %s' % ('basin', 'loss', 'share', 'vs best', 'V amplitude'))
    amp = fitting.amplitude_key(setup['param_names'])
    for k, idx in enumerate(basins):
        ls = losses[idx]
        vs = [out[j][1].get(amp, np.nan) for j in idx]
        print('%6d %12.6g %8.0f%% %9.3f%%   %.1f .. %.1f'
              % (k + 1, ls.mean(), 100*len(idx)/len(out),
                 100*(ls.mean() - best)/best, np.nanmin(vs), np.nanmax(vs)))

    cfg_basin = next((k + 1 for k, idx in enumerate(basins)
                      if abs(loss_cfg - losses[idx].mean()) <= args.rtol*abs(losses[idx].mean())), None)
    print()
    if cfg_basin == 1:
        print(f'The config start lands in the BEST basin, reached by '
              f'{100*len(basins[0])/len(out):.0f}% of random starts.')
    elif cfg_basin:
        print(f'*** The config start lands in basin {cfg_basin}, NOT the best one. '
              f'Its loss is {100*(loss_cfg-best)/best:.2f}% above the best found. ***')
    elif loss_cfg < best:
        print(f'The config start is BETTER than every random start '
              f'({loss_cfg:.6g} vs {best:.6g}) -- it was tuned by hand.')
    else:
        print(f'The config start ({loss_cfg:.6g}) matches no basin found here.')

    if len(basins) == 1:
        print('One basin: the optimiser is reliable for this source.')
    else:
        print(f'{len(basins)} basins: a single Adam run is a coin flip for this source, '
              f'and both init=adam and the frozen noise model inherit whichever it hits.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
