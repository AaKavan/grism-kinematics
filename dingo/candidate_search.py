"""Batched, multistart posterior search. No MCMC is run by this module."""
from __future__ import annotations
import logging
import numpy as np

LOG = logging.getLogger(__name__)


class PosteriorSearch:
    """Profile only the uniform-prior log noise scale; retain physical priors.

    Coordinates remain the sampler's coordinates, with either V_rot or V_sini.
    A fixed-radius velocity profile is a constrained optimization in these same
    coordinates, NOT a change of prior measure or a marginalized posterior.
    """
    def __init__(self, fitter, batch_size=128, radius_pixels=None):
        self.fitter = fitter
        info = fitter.prepare_mcmc()
        self.names = info['param_names']
        self.theta0 = info['theta0']
        self.batch_size = batch_size
        self.radius_pixels = radius_pixels
        self.index = {n: i for i, n in enumerate(self.names)}
        self.noise = self.index['noise.ln_f']
        amplitudes = [n for n in ('velocity.V_sini', 'velocity.V_rot') if n in self.index]
        if len(amplitudes) != 1:
            raise ValueError('Search requires exactly one sampled amplitude: V_rot or V_sini.')
        self.amplitude_name = amplitudes[0]
        self.projected_amplitude = self.amplitude_name == 'velocity.V_sini'
        self.amp = self.index[self.amplitude_name]
        self.inc = self.index['velocity.inc_v']
        self.rt = self.index['velocity.R_v']
        self.angle = self.index['velocity.theta_v']
        self.bounds = np.array([[s.get('min', -np.inf), s.get('max', np.inf)]
                                for _, s in fitter._mcmc_priors])
        scales = {'velocity.V_sini': 25., 'velocity.V_rot': 25., 'velocity.R_v': .5,
                  'velocity.inc_v': np.deg2rad(3), 'velocity.theta_v': np.deg2rad(5),
                  'velocity.x0_v': .5, 'velocity.y0_v': .5,
                  'image.R.dx': .25, 'image.R.dy': .25, 'noise.ln_f': 0.}
        self.scales = np.array([scales[n] for n in self.names])

    def constrain(self, points, velocities=None):
        points = np.array(points, dtype=float, copy=True)
        points[:, self.angle] %= 2*np.pi
        if velocities is not None:
            # Polls can step outside radius support (including exactly zero).
            # evaluate() rejects these proposals before any model evaluation.
            with np.errstate(divide='ignore', invalid='ignore'):
                amplitude = (np.asarray(velocities) /
                    (2/np.pi*np.arctan(self.radius_pixels/points[:, self.rt])))
                if self.projected_amplitude:
                    amplitude = amplitude*np.sin(points[:, self.inc])
                points[:, self.amp] = amplitude
        return points

    def evaluate(self, points, velocities=None):
        points = self.constrain(points, velocities)
        points[:, self.noise] = 0.
        scores = np.full(len(points), -np.inf)
        # Bounds checked BEFORE the GPU model, including amplitude derived by a profile.
        good = np.isfinite(points).all(axis=1) & ((points >= self.bounds[:, 0]) &
                                                  (points <= self.bounds[:, 1])).all(axis=1)
        indices = np.flatnonzero(good)
        for start in range(0, len(indices), self.batch_size):
            ix = indices[start:start+self.batch_size]
            ll0 = self.fitter.log_likelihood_batch(points[ix])
            chi2 = -2*np.asarray(ll0)/self.fitter.mcmc_temper-self.fitter._mcmc_lnvar_const
            valid = np.isfinite(chi2) & (chi2 >= 0)
            ii, cc = ix[valid], chi2[valid]
            points[ii, self.noise] = np.clip(.5*np.log(np.maximum(cc, 1e-300)/self.fitter.n_mcmc_pix),
                                                   *self.bounds[self.noise])
            lp = np.array([self.fitter.log_prior(t) for t in points[ii]])
            scores[ii] = self.fitter._chi2_to_lnL(cc, points[ii, self.noise])+lp
        return points, scores

    def optimize(self, starts, *, iterations=100, velocities=None, label='Refine', callback=None):
        """Parallel coordinate/pattern search with per-start step sizes.

        Each poll checks both directions of every free coordinate in batches.
        Extrapolation along successful moves helps traverse correlated valleys.
        No derivatives through the legacy inverse solver are assumed.
        """
        points, scores = self.evaluate(starts, velocities)
        widths = np.ones(len(points))
        free = [j for j in range(len(self.names)) if j != self.noise and
                not (velocities is not None and j == self.amp)]
        directions = np.eye(len(self.names))[free]*self.scales
        directions = np.concatenate([directions, -directions])
        for step in range(iterations):
            active = np.flatnonzero(widths >= 1/128)
            if not len(active):
                break
            proposals = points[active, None, :]+widths[active, None, None]*directions[None, :, :]
            speeds = None if velocities is None else np.repeat(np.asarray(velocities)[active], len(directions))
            flat, trial_scores = self.evaluate(proposals.reshape(-1, points.shape[1]), speeds)
            flat = flat.reshape(proposals.shape)
            trial_scores = trial_scores.reshape(len(active), -1)
            best = trial_scores.argmax(axis=1)
            new, ns = flat[np.arange(len(active)), best], trial_scores[np.arange(len(active)), best]
            improved = ns > scores[active]+1e-3
            accepted = active[improved]
            old = points[accepted].copy()
            points[accepted], scores[accepted] = new[improved], ns[improved]
            widths[active[~improved]] *= .5
            if len(accepted):
                extrapolated, es = self.evaluate(points[accepted]+(points[accepted]-old),
                    None if velocities is None else np.asarray(velocities)[accepted])
                better = es > scores[accepted]+1e-3
                points[accepted[better]], scores[accepted[better]] = extrapolated[better], es[better]
            if step == 0 or (step+1) % 10 == 0 or step+1 == iterations:
                LOG.info('%s: iteration %d/%d, %d starts active, best lnP %.3f',
                         label, step+1, iterations, len(active), scores.max())
                if callback:
                    callback(points, scores, step+1)
        return points, scores

    def velocity(self, points):
        p = np.atleast_2d(points)
        amplitude = p[:, self.amp]
        if self.projected_amplitude:
            amplitude = amplitude/np.sin(p[:, self.inc])
        return amplitude*2/np.pi*np.arctan(self.radius_pixels/p[:, self.rt])


def diverse_indices(points, scores, search, count):
    """Keep distinct parameter solutions; do not merge by velocity alone."""
    kept = []
    scale = search.scales.copy()
    scale[search.noise] = 1.
    for i in np.argsort(-scores):
        if not np.isfinite(scores[i]):
            continue
        distinct = True
        for k in kept:
            delta = np.abs(points[i]-points[k])
            delta[search.angle] = min(delta[search.angle], 2*np.pi-delta[search.angle])
            delta[search.noise] = 0
            if np.max(delta/scale) < 1.:
                distinct = False
                break
        if distinct:
            kept.append(int(i))
        if len(kept) == count:
            break
    return kept


def grid_seeds(search, step=25., n_angles=16, checkpoint=None):
    """Cover inclination/turnover and amplitude bands before local refinement."""
    spec = dict(search.fitter._mcmc_priors)['velocity.inc_v']
    if spec['type'] == 'gaussian':
        incs = np.unique(np.clip(spec['mu']+spec['sigma']*np.array([-2., -1., 0., 1., 2.]),
                                  spec['min']+.001, spec['max']-.001))
    else:
        incs = np.deg2rad([15., 30., 45., 60., 75.])
    rts = np.unique(np.clip([1., 3., 8.], *search.bounds[search.rt]))
    amps = np.arange(step, search.bounds[search.amp, 1], step)
    angles = np.arange(n_angles)*2*np.pi/n_angles
    bands = [0., 100., 250., 500., 1000., np.inf]
    seeds, all_scores = [], []
    for j, (inc, rt) in enumerate((i, r) for i in incs for r in rts):
        p = np.tile(search.theta0, (len(amps)*len(angles), 1))
        p[:, search.amp] = np.repeat(amps, len(angles))
        p[:, search.angle] = np.tile(angles, len(amps))
        p[:, search.inc], p[:, search.rt] = inc, rt
        p, scores = search.evaluate(p)
        for lo, hi in zip(bands[:-1], bands[1:]):
            ids = np.flatnonzero((p[:, search.amp] >= lo) & (p[:, search.amp] < hi) & np.isfinite(scores))
            if len(ids):
                best = ids[np.argmax(scores[ids])]
                seeds.append(p[best]); all_scores.append(scores[best])
        LOG.info('Geometry grid %d/%d: inclination %.1f deg, turnover %.1f pixels',
                 j+1, len(incs)*len(rts), np.rad2deg(inc), rt)
        if checkpoint:
            checkpoint(np.asarray(seeds), np.asarray(all_scores), j+1)
    if not seeds:
        raise RuntimeError('No finite broad-search starts; inspect input coverage and priors.')
    return np.asarray(seeds), np.asarray(all_scores)


def profile(search, starts, velocities, *, iterations=80, checkpoint=None):
    """Optimize all nuisance parameters from multiple starts at EVERY velocity."""
    velocities = np.asarray(velocities, dtype=float)
    points = np.tile(starts, (len(velocities), 1))
    fixed = np.repeat(velocities, len(starts))
    points, scores = search.optimize(points, iterations=iterations, velocities=fixed,
                                    label='Velocity profile', callback=checkpoint)
    scores = scores.reshape(len(velocities), len(starts))
    points = points.reshape(len(velocities), len(starts), -1)
    best = scores.argmax(axis=1)
    return points[np.arange(len(velocities)), best], scores[np.arange(len(velocities)), best]


def polish_profile(search, points, scores, velocities, anchors, *, iterations=80, passes=2, checkpoint=None):
    """Try both neighbouring branches AND globally competitive free solutions.

    Keep the existing solution at every point so polishing cannot worsen a
    profile. This is finite multistart optimization, not a proof of the global
    envelope; report the last pass's improvement to expose unresolved changes.
    """
    points, scores = points.copy(), scores.copy()
    gain = 0.
    for turn in range(passes):
        starts = np.stack([points, np.vstack([points[:1], points[:-1]]),
                          np.vstack([points[1:], points[-1:]])], axis=1)
        starts = np.concatenate([starts, np.broadcast_to(anchors[None, :, :],
                                    (len(points), *anchors.shape))], axis=1)
        n = starts.shape[1]
        trial, values = search.optimize(starts.reshape(-1, points.shape[1]),
            velocities=np.repeat(velocities, n), iterations=iterations,
            label=f'Profile cross-check {turn+1}/{passes}', callback=checkpoint)
        trial = trial.reshape(len(points), n, -1); values = values.reshape(len(points), n)
        best = values.argmax(axis=1)
        new = values[np.arange(len(points)), best]
        improved = new > scores
        finite = np.isfinite(scores) & np.isfinite(new)
        gain = float(np.max(np.maximum(new[finite]-scores[finite], 0.))) if finite.any() else 0.
        if (np.isfinite(new) & ~np.isfinite(scores)).any():
            gain = float('inf')
        points[improved] = trial[np.arange(len(points)), best][improved]
        scores[improved] = new[improved]
        LOG.info('Profile cross-check %d: largest lnP improvement %.4f', turn+1, gain)
        if gain < .05:
            break
    return points, scores, gain
