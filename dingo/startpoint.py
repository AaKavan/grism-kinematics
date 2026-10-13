'''Choose where the kinematics fit starts, without hand-tuning each source.

The R-C objective is periodic in the velocity amplitude. Shifting the velocity
field by one dispersion pixel re-registers R against C onto the neighbouring
pixel and reproduces nearly the same residual, so the loss carries a row of
near-degenerate minima across the whole plausible velocity range. For ID16887
they sit 88 km/s apart against a predicted c*D_WAVE/lambda_obs/sin(i) of 94.5.

A gradient optimiser started once falls into whichever tooth it was dropped in
and reports convergence from there -- zero gradient, flat loss, and downstream a
chain that mixes perfectly while exploring one tooth. `multistart.py` found 16
distinct minima from 16 random starts on ID16887, and the config's own start
beat all 16 because it had been tuned by hand.

That hand-tuning is what this replaces. `data/*/config_kinematics.yaml` carried
per-source literals -- a sign table, `R_v: 4 if ID==3700 else ...`, an x0/y0 for
one source, and a "do not tidy these" warning -- which is one person having
found the right tooth for seven galaxies by trial. It does not scale to the 95
sources that have Pa-alpha in both dispersion directions.

    scan   evaluate the loss on an (amplitude, theta_v) grid   -- no gradients
    refine run the staged fit from each distinct minimum
    pick   keep the lowest final loss

theta_v is scanned alongside the amplitude because (V, theta) and (-V, theta+pi)
give an identical line-of-sight field; scanning theta over its full range is
what makes the sign table unnecessary rather than replacing one guess with
another. Stage A needs one forward pass per node (~40-100 ms), so the number of
staged fits stays small and fixed instead of being chosen per source by eye.
'''
import logging

import numpy as np
import torch

from .fitting import KinematicsFitter, AMPLITUDE_KEYS

LOG = logging.getLogger(__name__)

__all__ = ['amplitude_name', 'scan_grid', 'distinct_minima', 'find_start']


def amplitude_name(config):
    '''"V_rot" or "V_sini" -- whichever this config defines. Never both.'''
    vel = config.get('velocity', {})
    # AMPLITUDE_KEYS are fully dotted ('velocity.V_rot'); a config's velocity
    # block holds the bare names.
    bare = [k.split('.', 1)[1] for k in AMPLITUDE_KEYS]
    found = [k for k in bare if k in vel]
    if len(found) != 1:
        raise ValueError(
            f'velocity: must define exactly one of {sorted(bare)}, found {found}')
    return found[0]


def scan_grid(fitter, amp_key, amplitudes, thetas):
    '''Loss on the (amplitude, theta_v) grid, shape (len(amplitudes), len(thetas)).

    Forward evaluations only -- no optimiser, no gradients. Every other
    parameter stays where the config put it, so this maps the teeth rather
    than fitting anything.
    '''
    stage = fitter.param_config_lists['velocity'][0]
    amp, theta = stage[f'velocity.{amp_key}'], stage['velocity.theta_v']
    out = np.empty((len(amplitudes), len(thetas)))
    for i, v in enumerate(amplitudes):
        for j, th in enumerate(thetas):
            with torch.no_grad():
                amp.tensor.data.fill_(float(v))
                theta.tensor.data.fill_(float(th))
            fitter._reset_state()
            with torch.no_grad():
                out[i, j] = float(fitter.loss())
    return out


def distinct_minima(loss, amplitudes, thetas, n_keep=6, amp_tol=12.0, theta_tol=1.0):
    '''Local minima of the grid, best first, with duplicates removed.

    theta_v wraps, so the neighbourhood wraps with it. At amplitude ~ 0 the
    velocity field is zero everywhere and theta_v is entirely degenerate -- every
    theta node returns the same loss, and treating those as distinct minima
    spends the whole refinement budget re-fitting a single point.
    '''
    nv, nt = loss.shape
    found = []
    for i in range(nv):
        for j in range(nt):
            nb = [loss[a, b % nt]
                  for a in range(max(0, i-1), min(nv, i+2)) for b in range(j-1, j+2)
                  if not (a == i and b % nt == j)]
            if loss[i, j] <= min(nb):
                found.append((float(loss[i, j]), float(amplitudes[i]), float(thetas[j])))
    found.sort()
    amp_step = float(amplitudes[1] - amplitudes[0]) if len(amplitudes) > 1 else amp_tol

    def same(a, b):
        _, v1, t1 = a
        _, v2, t2 = b
        if v1 < amp_step and v2 < amp_step:      # theta meaningless at V ~ 0
            return True
        dt = min(abs(t1 - t2), 2*np.pi - abs(t1 - t2))
        return abs(v1 - v2) <= amp_tol and dt <= theta_tol

    # Cover the comb before refining the position angle. Spending the budget on
    # several theta values at one amplitude is how a run ends up testing one
    # tooth repeatedly while leaving the others unvisited, which is the exact
    # failure this module exists to prevent.
    keep = []
    for m in found:                                   # pass 1: one per tooth
        if all(abs(m[1] - k[1]) > amp_tol for k in keep) and not any(same(m, k) for k in keep):
            keep.append(m)
        if len(keep) >= n_keep:
            return keep
    for m in found:                                   # pass 2: other position angles
        if not any(same(m, k) for k in keep):
            keep.append(m)
        if len(keep) >= n_keep:
            break
    return keep


def find_start(config_path, device=None, amp_max=1400.0, amp_step=25.0,
               n_theta=16, n_keep=6, make_fitter=None):
    '''Scan, refine, and return the best starting point for this source.

    `amp_max` is 1400, not the 625 this first shipped with: ID3700's best fit
    sits near 516 and the comb continues past it, so a ceiling anywhere near the
    answer hides the real minimum and the scan reports the highest tooth it was
    allowed to see. The grid costs one forward pass per node, so raising it is
    cheap; getting it wrong is not. The MCMC prior allows 0-2500.

    Returns (params, loss, trials): the winning parameter dict, its final loss,
    and every (start, loss, params) tried, so a caller can see whether the teeth
    were close together -- which is the honest measure of how much the answer
    depended on the choice.

    `make_fitter(config_path)` lets a caller apply a fixed inclination (or any
    other per-run edit) to each fitter the same way; the default just builds one.
    '''
    build = make_fitter or (lambda p: KinematicsFitter(config_path=p, device=device))
    probe = build(config_path)
    amp_key = amplitude_name(probe.config)

    amplitudes = np.arange(0.0, amp_max, amp_step)
    thetas = np.arange(0.0, 2*np.pi, 2*np.pi/n_theta)
    loss = scan_grid(probe, amp_key, amplitudes, thetas)
    seeds = distinct_minima(loss, amplitudes, thetas, n_keep=n_keep, amp_tol=amp_step/2)

    trials = []
    for _, v0, th0 in seeds:
        f = build(config_path)
        stage = f.param_config_lists['velocity'][0]
        with torch.no_grad():
            stage[f'velocity.{amp_key}'].tensor.data.fill_(v0)
            stage['velocity.theta_v'].tensor.data.fill_(th0)
        f.fit_all()
        params, _, _ = f.get_params()
        trials.append({'start': (v0, th0), 'loss': float(f.losses[-1]), 'params': params})

    if not trials:
        raise RuntimeError('the grid found no local minimum -- check the config loads')
    best = min(trials, key=lambda t: t['loss'])

    # A winner against the ceiling means the scan stopped before the loss did, so
    # the "best tooth" is just the last one it was allowed to look at. ID3700
    # returned 631 under the old amp_max of 625 and the true minimum was 516 --
    # in the other direction, which is why this warns rather than assuming high.
    if best['params'][amp_key] > amp_max - 2*amp_step:
        LOG.warning(
            f'the winning {amp_key} ({best["params"][amp_key]:.1f}) is within two '
            f'grid steps of amp_max ({amp_max:.0f}). The scan may have been cut off '
            f'before the loss turned over; re-run with a higher amp_max to check.')
    return best['params'], best['loss'], trials
