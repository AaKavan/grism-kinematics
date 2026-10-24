"""Reusable real-source joint R+C scan and candidate rendering for the notebook."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import time

from diagnostics.native_real_grid import (
    CODE as LEGACY_CODE, REPO, imaging_prior, response_kernels, projected_gradient)
import jax
import jax.numpy as jnp
import numpy as np
import torch
from scipy.optimize import minimize

from dingo.exposures import load_native_target
from dingo.native_fitting import BilinearLineMap, StripContinuumLikelihood
from dingo.native_forward import NativeLineProjector
from dingo.native_jax import JaxNativeProjector, JaxLineMap, JaxStripLikelihood
from dingo.native_real import RealVelocityProfile, oblate_inclination, broadened_inclination_density
from dingo.real_workflow import RealSource, read_json, write_json, sha256

CODE = LEGACY_CODE + ['diagnostics/native_source_grid.py']


def inclination_prior(source_id, imaging, kind, scatter=10.):
    if kind == 'isotropic':
        grid = np.deg2rad(np.linspace(5, 85, 1601))
        pdf = np.sin(grid); pdf /= np.trapezoid(pdf, grid)
        return grid, np.log(pdf), {'kind': kind, 'source_id': source_id,
                                  'description': 'Broad isotropic orientation, truncated to 5–85 degrees; no imaging constraint.'}
    rng = np.random.default_rng(source_id)
    angles, sources, rejected = [], [], []
    for name in ('full_pixel', 'core_pixel_clean'):
        path = Path(imaging) / name
        report = read_json(path / 'imaging.json')
        if report['source_id'] != source_id:
            raise ValueError(f'Imaging fit for a different source: {path}')
        if not (report['completed'] and report['diagnostics']['convergence_gate_passed'] and report['orientation_parameterization_ok']):
            raise ValueError(f'Imaging fit failed its diagnostics: {path}')
        for input_path, digest in report['input_sha256'].items():
            if sha256(input_path) != digest:
                raise ValueError(f'Imaging input changed: {input_path}')
        with np.load(path / 'samples.npz') as saved:
            q = saved['q'].ravel()[::8]
        thickness = rng.uniform(.05, .30, len(q))
        keep = (q >= thickness) & (q <= 1.) & np.isfinite(q)
        if not keep.any():
            raise ValueError('No valid axis-ratio/thickness pairs; inspect imaging model.')
        # Equal weights for the two alternative fits of the same image, even
        # if their physically valid sample counts differ.
        sample = oblate_inclination(q[keep], thickness[keep])
        angles.append(sample)
        rejected.append(float(1 - keep.mean()))
        sources.append({'path': str(path), 'samples_sha256': sha256(path / 'samples.npz')})
    n = min(map(len, angles))
    stellar = np.concatenate([x[np.linspace(0, len(x)-1, n, dtype=int)] for x in angles])
    grid, logp = broadened_inclination_density(stellar, scatter)
    return grid, logp, {'kind': 'imaging', 'source_id': source_id, 'sources': sources,
        'q0_uniform_before_rejection': [.05, .30], 'invalid_q_q0_pair_fraction': rejected,
        'additional_scatter_deg': scatter, 'alternative_fits_not_independent_data': True,
        'stellar_inclination_5_50_95_deg': np.rad2deg(np.quantile(stellar, [.05, .5, .95])).tolist()}


def stellar_center(target, imaging):
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    with np.load(Path(imaging) / 'cutout.npz') as a:
        wcs = WCS(fits.Header.fromstring(str(a['wcs_header'])), fix=False)
    origin = SkyCoord(target.summary['ra'], target.summary['dec'], unit='deg')
    centers = []
    for name in ('full_pixel', 'core_pixel_clean'):
        r = read_json(Path(imaging) / name / 'imaging.json')
        if r['source_id'] != target.source_id or not (r['completed'] and r['diagnostics']['convergence_gate_passed']):
            raise ValueError('Stellar-center fit unavailable or for another source.')
        with np.load(Path(imaging) / name / 'samples.npz') as a:
            sky = wcs.pixel_to_world(float(np.median(a['xc'])), float(np.median(a['yc'])))
        e, n = origin.spherical_offsets_to(sky)
        centers.append([e.to_value(u.arcsec), n.to_value(u.arcsec)])
    return np.mean(centers, axis=0)


def build_engine(report):
    """Rebuild exactly the saved source's likelihood, or stop on mismatches."""
    s = report['settings']; sid = report['source_id']
    for p, digest in report['code_sha256'].items():
        if sha256(REPO / p) != digest:
            raise ValueError(f'Fit implementation changed: {p}. Use a new run; saved pixel views still work.')
    target = load_native_target(s['bundle'], s.get('config', REPO / f'data/ID{sid}/config_kinematics.yaml'), s['size'])
    if target.source_id != sid:
        raise ValueError('Saved run/source configuration mismatch.')
    science, error, valid = [np.stack([getattr(e, k) for e in target.exposures]) for k in ('science', 'error', 'valid')]
    if report.get('format') == 'native_source_grid_v1':
        ig, ip, ir = inclination_prior(sid, s['imaging'], s['inclination'], s['inclination_scatter'])
    else:
        # Preserve the exact October 15 target when rendering archived ID3700 fits.
        if sid != 3700:
            raise ValueError('Unknown legacy run format.')
        ig, ip, ir = imaging_prior(Path(s['imaging']), s['inclination_scatter'])
    kernels, kr = response_kernels(target, Path(s['bundle']), s['response_scale'])
    grid = BilinearLineMap(s['controls'], s['quadrature'], half_extent=.9, dtype=torch.float64)
    op = NativeLineProjector.from_target(target, grid.east.numpy(), grid.north.numpy(), kernels,
        np.ones(len(target.exposures)), spectral_samples=9, dtype=torch.float64, deposition='cubic')
    arrays = {'science': science, 'error': error, 'valid': valid, 'projection_coefficients': op.coefficients.numpy(),
              'kernels': kernels, 'inclination_grid': ig, 'inclination_logpdf': ip}
    hashes = {k: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() for k, a in arrays.items()}
    if report.get('array_sha256') and hashes != report['array_sha256']:
        raise ValueError('Real pixels, calibration or imaging prior changed. Use a new run.')
    likelihood = StripContinuumLikelihood(science, error, valid, [e.pupil for e in target.exposures], degree=2)
    model = RealVelocityProfile(JaxNativeProjector(op), JaxLineMap(grid), JaxStripLikelihood(likelihood), ig, ip,
                                correlation=s['correlation'], map_sigma=s['map_sigma'])
    bounds = model.bounds.copy()
    center_kind = s.get('center', 'catalog')
    mu = stellar_center(target, s['imaging']) if center_kind == 'stellar' else np.zeros(2)
    if center_kind == 'free':
        bounds[5] = bounds[6] = (-6., 6.)
    def objective(x, speed):
        value = model.objective(x, speed)
        center = .1*x[5:7]
        if center_kind != 'catalog':
            value -= .5*jnp.sum((center/.15)**2)
            if center_kind == 'stellar':
                value += .5*jnp.sum(((center-jnp.asarray(mu))/.05)**2)
        return value
    report.update(array_sha256=hashes, inclination_prior=ir, instrument_response=kr,
                  source_summary=target.summary, exposures=[e.metadata for e in target.exposures],
                  excluded_exposures=target.excluded_exposures,
                  nominal_wavelength_um=target.wavelength_um, n_valid_pixels=int(valid.sum()),
                  continuum_rank=likelihood.continuum_rank, center_mean_east_north_arcsec=mu.tolist())
    return target, model, likelihood, bounds, jax.jit(jax.value_and_grad(objective)), jax.jit(model.components)


def save_candidate(output, fit, model, likelihood, components, target):
    with np.load(output / fit['artifact']) as a:
        x = a['vector']
    speed = fit['vrot_0p5arcsec_kms']
    _, parts = components(jnp.asarray(x), jnp.asarray(float(speed)))
    _, _, _, amplitudes, line, flux, velocity = map(np.asarray, parts)
    continuum = likelihood.continuum(torch.from_numpy(line.copy())).numpy()
    science, error, valid = [np.stack([getattr(e, k) for e in target.exposures]) for k in ('science', 'error', 'valid')]
    residual = np.where(valid, (science-line-continuum)/error, np.nan)
    np.savez_compressed(output / f'candidate_{speed:g}.npz', vector=x, line=line, continuum=continuum,
                        residual=residual, source_flux=flux, source_velocity=velocity,
                        source_id=target.source_id, fit_artifact=fit['artifact'],
                        fit_sha256=sha256(output / fit['artifact']))


def run(args):
    torch.set_num_threads(4)
    if args.device == 'cuda' and jax.default_backend() != 'gpu':
        raise RuntimeError('GPU requested but not available. Select the grism kernel or explicitly choose CPU.')
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    # Do not let two notebook cells mutate the same checkpoint concurrently.
    with (output / '.workflow.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('This run is already active in another process.') from exc
        run_locked(args, output)


def run_locked(args, output):
    path = output / 'grid.json'
    if args.stage == 'render':
        report = read_json(path)
        if args.source_id is not None and args.source_id != report['source_id']:
            raise ValueError('Requested source differs from saved run.')
    else:
        if args.source_id is None or args.config is None:
            raise ValueError('A source ID and its configuration are required.')
        if args.step < 1 or args.vmax < 1 or args.controls < 3 or args.quadrature < args.controls:
            raise ValueError('Require positive integer step/vmax, controls >= 3 and quadrature >= controls.')
        source = RealSource(args.source_id, args.bundle, config=args.config)
        source.require_data()
        settings = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
                    if k not in ('stage', 'velocities', 'device', 'source_id', 'output')}
        hashes = {p: sha256(REPO / p) for p in CODE}
        if path.exists():
            report = read_json(path)
            if (report.get('format') != 'native_source_grid_v1' or report['source_id'] != args.source_id
                    or report['settings'] != settings or report['code_sha256'] != hashes):
                raise ValueError('Settings/source/code differ from this run. Choose a new RUN_NAME.')
        else:
            report = {'format': 'native_source_grid_v1', 'source_id': args.source_id,
                      'real_observations': True, 'completed': False, 'fits': [], 'best_by_velocity': [],
                      'settings': settings, 'code_sha256': hashes,
                      'interpretation': 'Conditional penalized joint R+C fits, not posterior intervals.'}
            for p in CODE:
                dest = output / 'code' / p; dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes((REPO / p).read_bytes())
    print(f"ID{report['source_id']}: loading real R+C data and building shared model…", flush=True)
    target, model, likelihood, bounds, objective, components = build_engine(report)
    best = {f['vrot_0p5arcsec_kms']: f for f in report['best_by_velocity']}
    if args.stage == 'render':
        for speed in args.velocities:
            if speed not in best:
                raise ValueError(f'{speed:g} km/s has no fitted trial. Run the grid first.')
            save_candidate(output, best[speed], model, likelihood, components, target)
        print('Saved requested native-pixel views.', flush=True)
        return
    target.save(output / 'native_data.npz')
    np.savez_compressed(output / 'inclination_prior.npz', angle_rad=np.asarray(model.inclination_grid),
                        log_density=np.asarray(model.inclination_logpdf))
    write_json(path, report)
    zero = float(np.sum(np.asarray(model.likelihood.residual_data)**2))
    done = {f['start']: f for f in report['fits']}
    def vector(f):
        with np.load(output / f['artifact']) as a:
            return a['vector']
    def fit(speed, start, label, maxiter):
        if label in done:
            return vector(done[label])
        begin = time.perf_counter()
        def fun(x):
            value, gradient = objective(jnp.asarray(x), jnp.asarray(float(speed)))
            return float(value), np.asarray(gradient)
        opt = minimize(fun, start, jac=True, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': maxiter, 'gtol': .02, 'ftol': 1e-12, 'maxls': 40, 'maxcor': 40})
        if not label.startswith('coarse') and projected_gradient(opt.x, opt.jac, bounds) >= .1:
            opt = minimize(fun, opt.x, jac=True, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 1600, 'gtol': .002, 'ftol': 1e-12, 'maxls': 40, 'maxcor': 40})
        if not np.isfinite(opt.fun) or not np.isfinite(opt.jac).all():
            raise RuntimeError(f'Nonfinite fit at {speed} km/s; checkpoint retained.')
        _, parts = components(jnp.asarray(opt.x), jnp.asarray(float(speed)))
        relative, _, _, amplitudes, _, flux, velocity = map(np.asarray, parts)
        grad = projected_gradient(opt.x, opt.jac, bounds)
        tag = f"fit_{len(report['fits']):05d}.npz"
        entry = model.describe(opt.x, speed) | {'objective': float(opt.fun), 'chi_square': float(zero-2*relative),
             'start': label, 'stage': label.split('_')[0], 'seconds': time.perf_counter()-begin,
             'iterations': int(opt.nit), 'optimizer_success': bool(opt.success), 'message': str(opt.message),
             'projected_gradient_inf': grad, 'stationarity_passed': grad < .1, 'artifact': tag}
        np.savez_compressed(output / tag, vector=opt.x, amplitudes=amplitudes, flux=flux, velocity=velocity)
        report['fits'].append(entry); done[label] = entry
        if speed not in best or entry['objective'] < best[speed]['objective']:
            best[speed] = entry
        report['best_by_velocity'] = [best[v] for v in sorted(best)]
        write_json(path, report)
        print(f"{label}: V={speed:g}, score={opt.fun:.3f}, gradient={grad:.3g}, {entry['seconds']:.1f}s", flush=True)
        return opt.x
    grid = np.unique(np.r_[np.arange(0, args.vmax+1, args.step), args.vmax]).astype(int)
    incgrid, pdf = np.asarray(model.inclination_grid), np.exp(np.asarray(model.inclination_logpdf))
    inc = float(np.interp(.5, np.cumsum(pdf)/np.sum(pdf), incgrid))
    for speed in sorted(set(min(v, args.vmax) for v in (0, 100, 250, 500, 750, 1000))):
        for j, theta in enumerate(np.deg2rad([0, 90, 180, 270])):
            fit(speed, model.initial(theta, inc, rturn=.25 if j % 2 == 0 else .7),
                f'coarse_{speed}_{j}', min(600, args.maxiter*4))
    if args.stage == 'coarse':
        print('Coarse preview saved; the full grid is not complete.', flush=True)
        return
    for direction, values in [('up', grid), ('down', grid[::-1])]:
        previous = None
        for speed in values:
            starts = [vector(best[v]) for v in sorted(best, key=lambda v: abs(v-speed))[:3]]
            if previous is not None:
                starts.append(previous)
            scores = [float(objective(jnp.asarray(x), jnp.asarray(float(speed)))[0]) for x in starts]
            previous = fit(int(speed), starts[int(np.argmin(scores))], f'{direction}_{speed}', args.maxiter)
    # Fix the anchor in the checkpoint so an interrupted outward sweep resumes
    # the same branch instead of silently changing its sequence.
    if 'outward_anchor_artifact' not in report:
        winner = min(best.values(), key=lambda f: f['objective'])
        report['outward_anchor_artifact'] = winner['artifact']
        report['outward_anchor_velocity'] = winner['vrot_0p5arcsec_kms']
        write_json(path, report)
    anchor = report['outward_anchor_velocity']
    with np.load(output / report['outward_anchor_artifact']) as a:
        start = a['vector']
    for values in (grid[grid > anchor], grid[grid < anchor][::-1]):
        previous = start
        for speed in values:
            previous = fit(int(speed), previous, f'outward_{speed}', args.maxiter)
    for speed in sorted(best):
        if not best[speed]['stationarity_passed']:
            fit(speed, vector(best[speed]), f'retry_{speed}', 1600)
    report['completed'] = True
    report['grid_velocities_kms'] = grid.tolist()
    report['stationarity_failures'] = [v for v, f in best.items() if not f['stationarity_passed']]
    write_json(path, report)
    winner = min(best.values(), key=lambda f: f['objective'])
    selected = set(grid[np.linspace(0, len(grid)-1, min(12, len(grid)), dtype=int)].tolist())
    selected.add(winner['vrot_0p5arcsec_kms'])
    print('Saving native-pixel comparison candidates…', flush=True)
    for speed in sorted(selected):
        save_candidate(output, best[speed], model, likelihood, components, target)
    print(f"Complete: best conditional Vrot(0.5 arcsec)={winner['vrot_0p5arcsec_kms']:g} km/s; "
          f"{len(report['stationarity_failures'])} trials need stationarity review. {output}", flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-id', type=int)
    p.add_argument('--config', type=Path)
    p.add_argument('--bundle', type=Path)
    p.add_argument('--imaging', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=['fit', 'coarse', 'render'], default='fit')
    p.add_argument('--velocities', type=float, nargs='+', default=[])
    p.add_argument('--step', type=int, default=25)
    p.add_argument('--vmax', type=int, default=1000)
    p.add_argument('--center', choices=['catalog', 'stellar', 'free'], default='catalog')
    p.add_argument('--inclination', choices=['imaging', 'isotropic'], default='imaging')
    p.add_argument('--inclination-scatter', type=float, default=10.)
    p.add_argument('--response-scale', type=float, default=1.)
    p.add_argument('--correlation', type=float, default=.1)
    p.add_argument('--map-sigma', type=float, default=2.)
    p.add_argument('--size', type=int, default=81)
    p.add_argument('--controls', type=int, default=21)
    p.add_argument('--quadrature', type=int, default=61)
    p.add_argument('--maxiter', type=int, default=160)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    run(p.parse_args())
