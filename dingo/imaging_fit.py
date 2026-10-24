"""Checkpointed F150W PySersic fitting for a source-specific inclination prior.

The full and core-masked fits are alternative descriptions of the same image.
They constrain stellar shape; disk thickness and stellar/gas misalignment are
propagated separately by the rotation loader. Historical fits are untouched.
"""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import time

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_enable_triton_gemm=false')

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
from numpyro.infer import NUTS
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize

from .real_workflow import read_json, sha256, write_json
from .rotation_fit import atomic_npz, clean_json, heartbeat


def resample_effective_psf(psf, input_scale, output_scale):
    """Sample the centered empirical pixel response, without integrating twice."""
    psf = np.asarray(psf, dtype=float)
    if (psf.ndim != 2 or not np.isfinite(psf).all()
            or not np.isfinite([input_scale, output_scale]).all() or min(input_scale, output_scale) <= 0):
        raise ValueError('Invalid PSF or pixel scale')
    positive = np.maximum(psf, 0.)
    if positive.sum() <= 0:
        raise ValueError('PSF must have positive mass')
    yy, xx = np.indices(psf.shape)
    peak = np.unravel_index(np.argmax(positive), psf.shape)
    core = positive*((xx-peak[1])**2+(yy-peak[0])**2 <= (.12/input_scale)**2)
    cx, cy = (core*xx).sum()/core.sum(), (core*yy).sum()/core.sum()
    half = int(np.floor(min(cx, cy, psf.shape[1]-1-cx, psf.shape[0]-1-cy)*input_scale/output_scale))
    if half < 1:
        raise ValueError('Insufficient PSF support around the measured core')
    axis = np.arange(-half, half+1)*output_scale/input_scale
    y, x = np.meshgrid(axis+cy, axis+cx, indexing='ij')
    result = np.maximum(map_coordinates(positive, [y, x], order=3, mode='constant'), 0.)
    result /= result.sum()
    return result, dict(input_scale_arcsec=input_scale, output_scale_arcsec=output_scale,
        core_centroid_xy=[float(cx), float(cy)], original_sum=float(psf.sum()),
        clipped_negative_fraction=float(-psf[psf < 0].sum()/positive.sum()),
        output_shape=list(result.shape), method='centered cubic sampling of effective PSF, then unit normalization')


class ImagingSampler:
    """Four-chain NUTS with global adaptation counters and exact chunk restarts."""
    def __init__(self, model, initial, *, warmup, seed, max_tree_depth=10,
                 target_accept=.95, state=None):
        self.warmup = int(warmup)
        if self.warmup < 0 or not .5 < target_accept < 1 or not 1 <= max_tree_depth <= 12:
            raise ValueError('Invalid NUTS settings')
        self.chains = next(iter(initial.values())).shape[0]
        if any(np.shape(v)[0] != self.chains or not np.isfinite(v).all() for v in initial.values()):
            raise ValueError('Initial chain coordinates must be finite and consistently shaped')
        self.kernel = NUTS(model, dense_mass=True, target_accept_prob=target_accept,
                           max_tree_depth=max_tree_depth)
        keys = jax.random.split(jax.random.key(seed), self.chains)
        initialized = jax.vmap(lambda key, z: self.kernel.init(key, self.warmup, z, (), {}))(keys, initial)
        jax.block_until_ready(initialized)
        if state is not None:
            if jax.tree.structure(state) != jax.tree.structure(initialized):
                raise ValueError('Incompatible saved NUTS state structure')
            for old, new in zip(jax.tree.leaves(state), jax.tree.leaves(initialized)):
                if np.shape(old) != np.shape(new) or old.dtype != new.dtype:
                    raise ValueError('Incompatible saved NUTS state shapes or dtypes')
        self.state = initialized if state is None else jax.device_put(state)
        self.constrain = self.kernel.postprocess_fn((), {})
        self.functions = {}
        _ = self.iterations

    @property
    def iterations(self):
        counts = np.asarray(self.state.i)
        if counts.shape != (self.chains,) or np.any(counts != counts[0]) or counts[0] < 0:
            raise ValueError('NUTS chains have inconsistent iteration counts')
        return int(counts[0])

    def advance(self, count, collect=True):
        if int(count) != count or count < 1:
            raise ValueError('A positive integer number of transitions is required')
        key = (int(count), bool(collect))
        if key not in self.functions:
            def transition(state, _):
                state = jax.vmap(lambda s: self.kernel.sample(s, (), {}))(state)
                stats = {name: getattr(state, name) for name in
                         ('energy', 'potential_energy', 'num_steps', 'accept_prob', 'diverging')}
                stats.update(iteration=state.i, is_warmup=state.i <= self.warmup)
                samples = jax.vmap(self.constrain)(state.z) if collect else {}
                return state, (samples, stats)
            self.functions[key] = jax.jit(lambda state: jax.lax.scan(transition, state, xs=None, length=int(count)))
        before = self.iterations
        self.state, outputs = self.functions[key](self.state)
        jax.block_until_ready(self.state)
        if self.iterations != before+count:
            raise RuntimeError('NUTS iteration counter did not advance correctly')
        return jax.tree.map(lambda x: np.asarray(x).swapaxes(0, 1), outputs)


def diagnostics(samples, stats, max_depth):
    """Rank diagnostics for every physical/base coordinate and standard NUTS BFMI."""
    import arviz_stats as azs
    count = next(iter(samples.values())).shape[:2]
    report = dict(convergence_gate_passed=False, chains=count[0], draws_per_chain=count[1], parameters={},
        gate='Four chains; every rank Rhat < 1.01, bulk/tail ESS >= 400; no divergences; BFMI >= .3. '
             'Maximum tree depth is an efficiency warning. Passing is necessary, not proof of scientific adequacy.')
    required = {'diverging', 'num_steps', 'accept_prob', 'energy', 'is_warmup'}
    finite = (required.issubset(stats) and
              all(np.shape(v) == count and np.isfinite(v).all() for v in stats.values()) and
              all(np.shape(v)[:2] == count and np.isfinite(v).all() for v in samples.values()))
    report['all_diagnostics_finite'] = bool(finite)
    if not finite or np.any(stats.get('is_warmup', False)):
        report['sampling_input_error'] = 'Nonfinite/inconsistent arrays, missing statistics, or warmup in retained draws'
        return report
    report.update(divergences=int(np.sum(stats['diverging'])),
        max_depth_fraction=float(np.mean(stats['num_steps'] >= 2**max_depth-1)),
        mean_acceptance=float(np.mean(stats['accept_prob'])), mean_leapfrog_steps=float(np.mean(stats['num_steps'])))
    report['sampling_warnings'] = ([] if report['max_depth_fraction'] < .01 else
        ['Maximum tree-depth saturation exceeds 1%; inspect sampling efficiency and convergence.'])
    if count[0] < 2 or count[1] < 4:
        return report
    for name, values in samples.items():
        report['parameters'][name] = dict(max_rank_rhat=float(np.max(azs.rhat(values, method='rank'))),
            min_bulk_ess=float(np.min(azs.ess(values, method='bulk'))),
            min_tail_ess=float(np.min(azs.ess(values, method='tail', prob=(.05, .95)))))
    values = list(report['parameters'].values())
    with np.errstate(divide='ignore', invalid='ignore'):
        bfmi = np.mean(np.diff(stats['energy'], axis=1)**2, axis=1)/np.var(stats['energy'], axis=1)
    report.update(max_rank_rhat=max(v['max_rank_rhat'] for v in values),
        min_bulk_ess=min(v['min_bulk_ess'] for v in values), min_tail_ess=min(v['min_tail_ess'] for v in values),
        bfmi_by_chain=bfmi.tolist())
    finite = np.isfinite([list(v.values()) for v in values]).all() and np.isfinite(bfmi).all()
    report['all_diagnostics_finite'] = bool(finite)
    report['convergence_gate_passed'] = bool(count[0] >= 4 and finite and report['max_rank_rhat'] < 1.01
        and report['min_bulk_ess'] >= 400 and report['min_tail_ess'] >= 400 and report['divergences'] == 0 and np.min(bfmi) >= .3)
    return clean_json(report)


def boundary_diagnostics(samples, theta_interval):
    theta = np.asarray(samples['theta'])
    radius = np.asarray(samples['r_eff'])
    angle_fraction = float(np.mean(np.minimum(theta-theta_interval[0], theta_interval[1]-theta) < .01))
    radius_fraction = float(np.mean(np.minimum(radius-1., 45.-radius) < .1))
    finite = np.isfinite(theta).all() and np.isfinite(radius).all()
    return dict(angle_boundary_fraction=angle_fraction, radius_boundary_fraction=radius_fraction,
        radius_bounds_pixels=[1., 45.], radius_boundary_margin_pixels=.1,
        orientation_parameterization_ok=bool(finite and angle_fraction < .01 and radius_fraction < .01))


def moment_theta(weight, xx, yy, center):
    """Center the angular chart using PySersic's major-axis angle theta + pi/2."""
    if weight.sum() <= 0:
        raise ValueError('No positive source signal for the orientation moments')
    dx, dy = xx-center[0], yy-center[1]
    covariance = np.array([[(weight*dx*dx).sum(), (weight*dx*dy).sum()],
                           [(weight*dx*dy).sum(), (weight*dy*dy).sum()]])/weight.sum()
    axis = np.linalg.eigh(covariance)[1][:, -1]
    return float(np.arctan2(-axis[0], axis[1]) % np.pi)


def build_model(args):
    from astropy.io import fits
    from pysersic import FitSingle
    from pysersic.priors import SourceProperties
    from pysersic.loss import gaussian_loss
    from pysersic.rendering import PixelRenderer
    provenance = read_json(args.inputs/'input_provenance.json')
    with np.load(args.inputs/'cutout.npz') as arrays:
        image, error = arrays['image'], arrays['error']
        scale, center = float(arrays['pixel_scale_arcsec']), arrays['center_xy']
    if image.shape != error.shape or image.ndim != 2:
        raise ValueError('Imaging science and error arrays must have the same two-dimensional shape')
    support = min(center[0], center[1], image.shape[1]-1-center[0], image.shape[0]-1-center[1])*scale
    if args.fit_radius > support:
        raise ValueError(f'Fitting radius {args.fit_radius:g} arcsec exceeds cutout support {support:.3f}; prepare a larger cutout')
    with fits.open(args.inputs/'F150W_ePSF_original.fits') as hdus:
        psf, psf_report = resample_effective_psf(hdus[0].data, hdus[0].header['PIXSCALE'], scale)
    yy, xx = np.indices(image.shape)
    radius = np.hypot(xx-center[0], yy-center[1])*scale
    mask = (~np.isfinite(image) | ~np.isfinite(error) | (error <= 0)
            | (radius > args.fit_radius) | (radius < args.mask_core))
    if np.sum(~mask) < 50:
        raise ValueError('Insufficient valid F150W pixels in fitting aperture')
    unit = float(np.median(error[~mask]))
    data = np.where(np.isfinite(image), image/unit, 0.)
    rms = np.where(np.isfinite(error) & (error > 0), error/unit, 1.)
    prior = SourceProperties(data, mask=mask).generate_prior('sersic', sky_type='flat')
    flux_scale = float(np.maximum(data[(radius <= args.fit_radius) & np.isfinite(data)], 0.).sum())
    if flux_scale <= 0:
        raise ValueError('No positive source signal in the F150W fitting aperture')
    prior.set_uniform_prior('flux', 0., 5*flux_scale)
    prior.sky_prior.update_prior('sky_back', 0., 3.)
    prior.set_uniform_prior('ellip', .01, .95)
    weight = np.where((radius < .8) & ~mask, np.maximum(data, 0.), 0.)
    if weight.sum() <= 0:
        raise ValueError('No positive source signal within 0.8 arcsec')
    theta_center = moment_theta(weight, xx, yy, center)
    interval = [theta_center-np.pi/2, theta_center+np.pi/2]
    prior.set_uniform_prior('theta', *interval)
    prior.set_uniform_prior('n', .3, 6.)
    prior.set_truncated_gaussian_prior('r_eff', 10., 15., low=1., high=45.)
    prior.set_gaussian_prior('xc', float(center[0]), 5.)
    prior.set_gaussian_prior('yc', float(center[1]), 5.)
    fitter = FitSingle(data=data, rms=rms, psf=psf, mask=mask, prior=prior,
                       loss_func=gaussian_loss, renderer=PixelRenderer)
    pixels = dict(image=image, error=error, psf=psf, mask=mask)
    metadata = dict(source_id=int(provenance['source_id']), band='F150W', real_imaging=True,
        theta_interval_rad=interval, renderer='PySersic PixelRenderer', pixel_unit_scale=unit,
        pixel_scale_arcsec=scale, psf=psf_report, prior=repr(prior), valid_pixels=int(np.sum(~mask)))
    return fitter, fitter.build_model(return_model=False), pixels, metadata


def initialize(model, seed):
    info = initialize_model(jax.random.key(seed), model, init_strategy=init_to_value(values={
        'theta_base': .5, 'ellip_base': .65, 'n_base': .12, 'r_eff_base': .2,
        'flux_base': .2, 'xc_base': .013, 'yc_base': .017, 'sky_back': 0.}))
    initial, unravel = ravel_pytree(info.param_info.z)
    objective = jax.jit(jax.value_and_grad(lambda x: info.potential_fn(unravel(x))))
    tracked = dict(nonfinite_trials_rejected=0, best_value=np.inf, best_vector=None, calls=0)
    def fun(x):
        value, grad = objective(jnp.asarray(x))
        value, grad = float(value), np.asarray(grad)
        tracked['calls'] += 1
        if tracked['calls'] % 25 == 0:
            print(f'  Imaging initialization: {tracked["calls"]} evaluations; best objective {tracked["best_value"]:.3f}', flush=True)
        if not np.isfinite(value) or not np.isfinite(grad).all():
            tracked['nonfinite_trials_rejected'] += 1
            return 1e100, np.zeros_like(x)
        if value < tracked['best_value']:
            tracked.update(best_value=value, best_vector=np.array(x, copy=True))
        return value, grad
    hessian = jax.jit(jax.hessian(lambda x: info.potential_fn(unravel(x))))
    with heartbeat('Compiling and optimizing PySersic initialization'):
        opt = minimize(fun, np.asarray(initial), jac=True, hess=lambda x: np.asarray(hessian(jnp.asarray(x))),
            method='trust-exact', options=dict(maxiter=500, gtol=1e-4, initial_trust_radius=.1, max_trust_radius=1.))
    best = tracked.pop('best_vector')
    if best is None:
        raise RuntimeError('No finite PySersic initialization; inspect F150W pixels, PSF and fitting aperture')
    value, grad = fun(best)
    tracked.pop('best_vector', None)
    if np.max(np.abs(grad)) > .1:
        raise RuntimeError(f'PySersic initialization is not stationary: maximum gradient {np.max(np.abs(grad)):.3g}. Inspect the imaging model and aperture.')
    tracked.update(optimizer_success=bool(opt.success), message=str(opt.message), gradient_inf=float(np.max(np.abs(grad))))
    rng = np.random.default_rng(seed)
    starts = [unravel(jnp.asarray(best+rng.normal(0, .01, len(best)))) for _ in range(4)]
    return jax.tree.map(lambda *x: jnp.stack(x), *starts), info.postprocess_fn(unravel(jnp.asarray(best))), tracked


def save_prediction(output, fitter, parameters, pixels, report, stage):
    unit = report['pixel_unit_scale']
    model = np.asarray(fitter.renderer.render_source(parameters, 'sersic'))*unit + float(parameters['sky_back'])*unit
    with np.errstate(divide='ignore', invalid='ignore'):
        residual = (pixels['image']-model)/pixels['error']
    report.update(residual_rms_inside_mask=float(np.sqrt(np.mean(residual[~pixels['mask']]**2))), prediction_stage=stage)
    atomic_npz(output/'fit.npz', **pixels, model=model, residual=residual)
    report['fit_sha256'] = sha256(output/'fit.npz')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    high = float(np.nanpercentile(np.arcsinh(pixels['image']/.05), 99.5))
    for ax, values, title in zip(axes, [np.arcsinh(pixels['image']/.05), np.arcsinh(model/.05), residual],
                                ['F150W data', 'PySersic model', '(data-model)/ERR']):
        options = dict(cmap='RdBu_r', vmin=-5, vmax=5) if title.startswith('(') else dict(cmap='magma', vmin=0, vmax=high)
        ax.imshow(values, origin='lower', **options)
        ax.contour(~pixels['mask'], levels=[.5], colors='lime', linewidths=.6)
        ax.set_title(title)
    fig.suptitle(f'ID{report["source_id"]} F150W; {stage}; central exclusion {report["arguments"]["mask_core"]:.2f} arcsec')
    fig.tight_layout(); fig.savefig(output/'fit.png', dpi=150); plt.close(fig)


def signature(args):
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                if k not in ('samples', 'chunk', 'resume', 'output')}
    inputs = {str(args.inputs/name): sha256(args.inputs/name) for name in
              ('cutout.npz', 'F150W_ePSF_original.fits', 'input_provenance.json')}
    codes = {str(Path(__file__).parent/name): sha256(Path(__file__).parent/name)
             for name in ('imaging_fit.py', 'rotation_fit.py', 'real_workflow.py')}
    versions = {name: importlib.metadata.version(name) for name in ('jax', 'jaxlib', 'numpyro', 'pysersic', 'numpy', 'scipy')}
    return dict(settings=settings, input_sha256=inputs, code_sha256=codes, versions=versions)


def read_checkpoint(output, report, identity):
    """Load only the checkpoint generation committed by the atomic report."""
    if report.get('identity') != identity:
        raise ValueError('Imaging inputs, code or warmup settings changed; choose a new imaging run name')
    name = report.get('checkpoint_file')
    if name is None:
        if not report.get('completed') and report.get('phase') == 'initialization':
            return None  # Interrupted initialization/NUTS compilation can restart safely.
        raise ValueError('Imaging report has no committed checkpoint; choose a fresh run directory')
    if not isinstance(name, str) or Path(name).name != name or not name.startswith('checkpoint_'):
        raise ValueError('Invalid imaging checkpoint generation name')
    path = output/name
    if not path.is_file() or report.get('checkpoint_sha256') != sha256(path):
        raise ValueError('Imaging checkpoint checksum mismatch; saved state is not safe to resume')
    with path.open('rb') as stream:
        checkpoint = pickle.load(stream)
    if checkpoint.get('identity') != identity:
        raise ValueError('Imaging checkpoint target does not match the requested target')
    return checkpoint


def export_samples(output, report, samples, stats):
    """Public viewing files are exports; the committed checkpoint owns the draws."""
    if samples:
        atomic_npz(output/'samples.npz', **samples)
        atomic_npz(output/'sampler_stats.npz', **stats)
        report.update(samples_sha256=sha256(output/'samples.npz'),
                      sampler_stats_sha256=sha256(output/'sampler_stats.npz'))


def commit_checkpoint(output, report, checkpoint):
    """Commit new state without invalidating the last good checkpoint on interruption.

    A crash before the final atomic JSON replacement leaves the old pointer
    intact. Uncommitted generations and partially updated public exports are
    ignored on restart. Keep the preceding committed generation as well.
    """
    previous = report.get('checkpoint_file')
    name = f'checkpoint_{time.time_ns()}.pkl'
    path = output/name
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        pickle.dump(checkpoint, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)
    next_report = dict(report, checkpoint_file=name, checkpoint_sha256=sha256(path))
    export_samples(output, next_report, checkpoint['samples'], checkpoint['stats'])
    write_json(output/'imaging.json', clean_json(next_report))
    report.clear(); report.update(next_report)
    for old in output.glob('checkpoint_*.pkl'):
        if old.name not in (name, previous):
            old.unlink()


def verify_or_restore_exports(output, report, checkpoint):
    """Repair an interrupted export from verified committed draws, never the reverse."""
    valid = all((output/name).is_file() and sha256(output/name) == report.get(key)
                for name, key in (('samples.npz', 'samples_sha256'), ('sampler_stats.npz', 'sampler_stats_sha256')))
    if not valid:
        print('Restoring imaging sample exports from the verified committed checkpoint.', flush=True)
        export_samples(output, report, checkpoint['samples'], checkpoint['stats'])
        write_json(output/'imaging.json', clean_json(report))


def run(args):
    if args.device == 'cuda' and jax.default_backend() != 'gpu':
        raise RuntimeError('GPU requested but unavailable; choose device="cpu" explicitly if intended')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'.fit.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another imaging worker is using {args.output}') from exc
        return _run_locked(args)


def _run_locked(args):
    identity = signature(args)
    report_path = args.output/'imaging.json'
    report, checkpoint = {}, None
    if report_path.exists():
        if not args.resume:
            raise FileExistsError('Imaging output exists; enable resume or choose a new imaging run name')
        report = read_json(report_path)
        checkpoint = read_checkpoint(args.output, report, identity)
        if checkpoint is not None and report.get('completed') and report.get('retained_draws_per_chain', 0) >= args.samples:
            verify_or_restore_exports(args.output, report, checkpoint)
            print(f'Using completed PySersic run: {args.output}', flush=True)
            return report
    fitter, model, pixels, metadata = build_model(args)
    if checkpoint is None:
        print(f'Initializing PySersic for ID{metadata["source_id"]}: {metadata["valid_pixels"]} pixels; radius {args.fit_radius:g} arcsec', flush=True)
        initial, best, initialization = initialize(model, args.seed)
        report = dict(metadata, format='dingo_imaging_v1', completed=False, phase='initialization', identity=identity,
            arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            initialization=initialization, input_sha256=identity['input_sha256'], code_sha256=identity['code_sha256'],
            diagnostics=dict(convergence_gate_passed=False), orientation_parameterization_ok=False,
            limitations=['Single Sersic approximation to stellar morphology', 'Mosaic pixel covariance is not modeled',
                         'Conditional statistical errors do not include gas/stellar misalignment or disk thickness'])
        save_prediction(args.output, fitter, best, pixels, report, 'MAP initialization')
        write_json(report_path, clean_json(report))
        (args.output/'imaging_code.py').write_bytes(Path(__file__).read_bytes())
        saved, stats, state = {}, {}, None
        jax.clear_caches()
    else:
        initial, state, saved, stats = (checkpoint[k] for k in ('initial', 'state', 'samples', 'stats'))
    with heartbeat('Initializing PySersic NUTS transition'):
        sampler = ImagingSampler(model, initial, warmup=args.warmup, seed=args.seed+1,
            max_tree_depth=args.max_tree_depth, target_accept=args.target_accept, state=state)

    def persist(completed=False):
        report.update(completed=completed, phase=('complete' if completed else 'warmup' if sampler.iterations < args.warmup else 'sampling'),
            warmup_completed=min(sampler.iterations, args.warmup),
            retained_draws_per_chain=max(0, sampler.iterations-args.warmup), requested_draws_per_chain=args.samples)
        commit_checkpoint(args.output, report, dict(identity=identity, initial=jax.device_get(initial),
            state=jax.device_get(sampler.state), samples=saved, stats=stats))

    persist()
    while sampler.iterations < args.warmup+args.samples:
        warming = sampler.iterations < args.warmup
        stop = args.warmup if warming else args.warmup+args.samples
        count = min(args.chunk, stop-sampler.iterations)
        before = sampler.iterations
        print(f'PySersic {"warmup" if warming else "draws"}: starting {before if warming else before-args.warmup} + {count} per chain', flush=True)
        start = time.perf_counter()
        with heartbeat('PySersic transition block, including compilation on first use'):
            block, extra = sampler.advance(count, collect=not warming)
        if not warming:
            block['q'] = 1-block['ellip']
            saved = {k: np.concatenate((saved[k], v), axis=1) if k in saved else v for k, v in block.items()}
            stats = {k: np.concatenate((stats[k], v), axis=1) if k in stats else v for k, v in extra.items()}
        persist()
        done = sampler.iterations if warming else sampler.iterations-args.warmup
        total = args.warmup if warming else args.samples
        print(f'PySersic {"warmup" if warming else "draws"}: {done}/{total} per chain; {time.perf_counter()-start:.1f}s; checkpoint saved', flush=True)
    report['diagnostics'] = diagnostics(saved, stats, args.max_tree_depth)
    report.update(boundary_diagnostics(saved, report['theta_interval_rad']))
    report['quantiles_5_50_95'] = {k: np.quantile(saved[k], [.05, .5, .95]).tolist() for k in ('q', 'theta', 'r_eff', 'n', 'xc', 'yc')}
    parameters = {k: jnp.asarray(np.median(v)) for k, v in saved.items()}
    save_prediction(args.output, fitter, parameters, pixels, report, 'posterior medians (check convergence)')
    persist(completed=True)
    print(json.dumps(clean_json(dict(output=str(args.output), diagnostics={k:v for k,v in report['diagnostics'].items() if k != 'parameters'},
        orientation_parameterization_ok=report['orientation_parameterization_ok'], radius_boundary_fraction=report['radius_boundary_fraction'],
        residual_rms=report['residual_rms_inside_mask']))), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--mask-core', type=float, default=0.)
    parser.add_argument('--fit-radius', type=float, default=2.)
    parser.add_argument('--warmup', type=int, default=1000)
    parser.add_argument('--samples', type=int, default=2000)
    parser.add_argument('--chunk', type=int, default=50)
    parser.add_argument('--seed', type=int, default=735)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--max-tree-depth', type=int, default=10)
    parser.add_argument('--target-accept', type=float, default=.95)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.inputs, args.output = args.inputs.expanduser().resolve(), args.output.expanduser().resolve()
    if not (0 <= args.mask_core < args.fit_radius and args.warmup >= 1 and args.samples >= 1 and args.chunk >= 1):
        parser.error('Require 0 <= mask-core < fit-radius and positive warmup, samples and chunk')
    run(args)


if __name__ == '__main__':
    main()
