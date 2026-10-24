"""Short real-source multistart fits with calibrated shared flux/continuum.

The coupled ablation retains the old unrestricted polynomial continuum. The
positive model adds a nonnegative continuum plus a signed background offset.
These are conditional optimization diagnostics, never posterior intervals.
"""
import argparse
import fcntl
from pathlib import Path
import time

from diagnostics.native_source_grid import build_engine, CODE as BASE_CODE
from diagnostics.native_real_grid import projected_gradient
from diagnostics.native_real_posterior import heartbeat, save_npz
from dingo.native_coupled import CoupledNativeModel, calibration_response
from dingo.native_quality import apply_detector_mask
from dingo.native_fitting import StripContinuumLikelihood
from dingo.native_jax import JaxStripLikelihood
from dingo.real_workflow import REPO, read_json, write_json, sha256
import jax
import jax.numpy as jnp
import numpy as np
import torch
from scipy.optimize import minimize

CODE = list(dict.fromkeys(BASE_CODE+['dingo/native_coupled.py',
    'dingo/native_real_posterior.py', 'diagnostics/native_coupled_fit.py',
    'diagnostics/native_real_posterior.py', 'diagnostics/native_posterior_check.py', 'dingo/native_quality.py']))


def run_locked(args):
    grid_path, output = args.grid.resolve(), args.output.resolve()
    grid = read_json(grid_path/'grid.json')
    if grid.get('real_observations') is not True or grid['source_id'] != args.source_id:
        raise ValueError('Select the requested source\'s real-data grid, including a preserved partial diagnostic.')
    if args.device == 'cuda' and jax.default_backend() != 'gpu':
        raise RuntimeError('Requested GPU is unavailable.')
    settings = {k: [str(x.resolve()) for x in v] if k == 'initial_fit' else
        str(v.resolve()) if isinstance(v, Path) else v
        for k,v in vars(args).items() if k not in ('output', 'device')}
    inputs = {str(grid_path/p): sha256(grid_path/p) for p in ('grid.json', 'native_data.npz', 'inclination_prior.npz')}
    for p in args.initial_fit:
        inputs[str(p.resolve())] = sha256(p)
    if args.detector_mask:
        inputs[str(args.detector_mask.resolve())] = sha256(args.detector_mask)
    hashes = {p: sha256(REPO/p) for p in CODE}
    path = output/'coupled.json'
    report = read_json(path) if path.exists() else dict(format='native_coupled_v1', source_id=args.source_id,
        real_observations=True, completed=False, settings=settings, input_sha256=inputs,
        code_sha256=hashes, fits=[], started_unix=time.time(),
        interpretation='Conditional multistart optimization. No credible intervals or model evidence.',
        baseline_grid_completed=grid['completed'])
    for k,v in (('settings', settings), ('input_sha256', inputs), ('code_sha256', hashes)):
        if report[k] != v:
            raise ValueError(f'Cannot resume changed {k}; use a new run directory.')
    for f in report['fits']:
        if sha256(output/f['artifact']) != f['artifact_sha256']:
            raise ValueError('Saved fit artifact changed.')
    if report['completed']:
        print('Already complete:', output, flush=True); return
    for p in CODE:
        dest = output/'code'/p; dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes((REPO/p).read_bytes())
    write_json(path, report)
    with heartbeat('Loading native data and response calibration'):
        target, engine, old_like, _, _, _ = build_engine(grid)
        if args.detector_mask:
            report['detector_mask'] = apply_detector_mask(target,args.detector_mask)
            arrays = [np.stack([getattr(e,k) for e in target.exposures]) for k in ('science','error','valid')]
            old_like = StripContinuumLikelihood(*arrays,[e.pupil for e in target.exposures],degree=2)
            engine.likelihood = JaxStripLikelihood(old_like)
        response, calibration = calibration_response(target, engine.projector, grid['settings']['bundle'])
    calibration_hashes = {r['sensitivity_path']: r['sensitivity_sha256'] for r in calibration['records']}
    if report.get('calibration_sha256', calibration_hashes) != calibration_hashes:
        raise ValueError('Sensitivity calibration changed.')
    target.save(output/'native_data.npz')
    report.update(calibration=calibration, calibration_sha256=calibration_hashes,
        native_data_sha256=sha256(output/'native_data.npz'), source_summary=target.summary,
        exposures=[e.metadata for e in target.exposures], grid_settings=grid['settings'],
        inclination_prior=grid['inclination_prior'], n_valid_pixels=sum(int(e.valid.sum()) for e in target.exposures))
    pupils = np.array([e.pupil for e in target.exposures])
    data, error, valid = [np.stack([getattr(e,k) for e in target.exposures]) for k in ('science','error','valid')]
    seeds = []
    if not args.no_default_starts:
        best = min(grid['best_by_velocity'], key=lambda f:f['objective'])
        other = min(grid['best_by_velocity'], key=lambda f:abs(f['vrot_0p5arcsec_kms']-500.))
        for label, f in [('old_best',best), ('old_500',other)]:
            with np.load(grid_path/f['artifact']) as a:
                seeds.append((label, a['vector'].copy(), f['vrot_0p5arcsec_kms'],
                              max(.1, float(np.median(a['amplitudes'][pupils=='R']/response[pupils=='R'])))))
        angles, pdf = np.asarray(engine.inclination_grid), np.exp(np.asarray(engine.inclination_logpdf))
        inc = float(np.interp(.5, np.cumsum(pdf)/pdf.sum(), angles))
        for speed, angle in [(300.,0), (300.,90), (300.,180), (300.,270), (100.,90), (600.,270)]:
            seeds.append((f'fresh_{speed:g}_{angle}', engine.initial(np.deg2rad(angle), inc), speed, 3.))
    if not seeds and not args.initial_fit:
        raise ValueError('At least one initial fit is required.')
    done = {(f['mode'], f['start']) for f in report['fits']}
    for mode in args.modes:
        model = CoupledNativeModel(engine, target, response, mode=mode,
            calibration_sigma=args.calibration_sigma, background_scale=args.background_scale,
            continuum_scale=args.continuum_scale, center_kind=grid['settings'].get('center','catalog'),
            center_mean=grid['center_mean_east_north_arcsec'])
        report.update(nonlinear_parameters=len(model.bounds),
            continuum_coefficient_count=int(np.prod(model.continuum_model.coefficient_shape)),
            background_sigma_DN_s=np.asarray(model.background_sigma).tolist(),
            prior=dict(log_gain_sd=args.calibration_sigma, gain_gauge='sum(log_gain)=0',
                calibration_sd_is_provisional=True, flux_halfnormal_scale_1e17=model.flux_scale,
                continuum_coefficient_halfnormal_scale_DN_s=args.continuum_scale,
                background='Independent zero-mean Gaussian; scale times median valid ERR per exposure',
                old_dynamics_and_map_priors_retained=True))
        starts = [(label, model.initial(x,v,f)) for label,x,v,f in seeds]
        for i,p in enumerate(args.initial_fit):
            with np.load(p) as a:
                if int(a['source_id']) != args.source_id:
                    raise ValueError('Initialization belongs to another source.')
                parameters = a['parameters'].copy()
                if parameters.shape != (len(model.bounds),):
                    raise ValueError('Initialization uses a different parameter layout.')
                starts.append((f'saved_{i}', parameters))
        compiled = jax.jit(jax.value_and_grad(model.objective)); components = jax.jit(model.components)
        with heartbeat(f'{mode}: compiling value and gradient'):
            v,g = compiled(jnp.asarray(starts[0][1])); jax.block_until_ready(g)
        if not np.isfinite([float(v), *np.asarray(g)]).all():
            raise RuntimeError('Nonfinite initial target.')
        for label,start in starts:
            if (mode,label) in done:
                continue
            begin = time.perf_counter(); count = [0]
            print(f'ID{args.source_id} {mode} {label}: initial V={100*start[0]:.1f}',flush=True)
            def fun(p):
                value, grad = compiled(jnp.asarray(p))
                return float(value), np.asarray(grad)
            def callback(p):
                count[0] += 1
                if count[0] % 200 == 0:
                    print(f'  iteration {count[0]}: V={100*p[0]:.2f}; elapsed {time.perf_counter()-begin:.0f}s',flush=True)
            with heartbeat(f'{mode} {label}'):
                opt = minimize(fun,start,jac=True,method='L-BFGS-B',bounds=model.bounds,callback=callback,
                    options=dict(maxiter=args.maxiter,gtol=.01,ftol=1e-12,maxls=40,maxcor=40))
                pg = projected_gradient(opt.x,opt.jac,model.bounds)
                if pg >= .1:
                    opt = minimize(fun,opt.x,jac=True,method='L-BFGS-B',bounds=model.bounds,callback=callback,
                        options=dict(maxiter=args.refine,gtol=.002,ftol=1e-13,maxls=50,maxcor=60))
                    pg = projected_gradient(opt.x,opt.jac,model.bounds)
            if not np.isfinite(opt.fun) or not np.isfinite(opt.jac).all():
                raise RuntimeError('Nonfinite optimization result; previous checkpoints preserved.')
            value, parts = components(jnp.asarray(opt.x))
            parts = {k:np.asarray(v) for k,v in parts.items()}
            if mode in ('coupled','legacy'):
                parts['continuum'] = old_like.continuum(torch.from_numpy(parts['line'].copy())).numpy()
            total = parts['line']+parts['continuum']+parts['background'][:,None,None]
            residual = np.divide(data-total,error,out=np.full_like(data,np.nan),where=valid)
            chi2 = float(np.nansum(residual**2))
            np.testing.assert_allclose(chi2,parts['chi2'],rtol=1e-10,atol=1e-6)
            x,speed,flux,gz,bz = model.split(opt.x)
            tag = f'fit_{len(report["fits"]):03d}.npz'
            save_npz(output/tag, source_id=np.asarray(args.source_id), parameters=opt.x, vector=np.asarray(x),
                     residual=residual, **parts)
            entry = engine.describe(x,speed) | dict(mode=mode,start=label,objective=float(value),
                chi_square=chi2, chi_square_R=float(np.nansum(residual[pupils=='R']**2)),
                chi_square_C=float(np.nansum(residual[pupils=='C']**2)),penalty=float(parts['penalty']),
                line_flux_1e17=None if mode=='legacy' else float(flux), gain_min_max=[float(parts['gains'].min()),float(parts['gains'].max())],
                background_standardized=np.asarray(bz).tolist(),
                projected_gradient_inf=float(pg),stationarity_passed=bool(pg < .1),
                iterations=count[0],optimizer_success=bool(opt.success),message=str(opt.message),
                seconds=time.perf_counter()-begin,artifact=tag,artifact_sha256=sha256(output/tag))
            report['fits'].append(entry)
            report['best_by_mode'] = [min([f for f in report['fits'] if f['mode']==m],key=lambda f:f['objective'])
                for m in args.modes if any(f['mode']==m for f in report['fits'])]
            write_json(path,report)
            print(f'  saved V={speed:.2f}, sigma={entry["intrinsic_sigma_kms"]:.2f}, '
                  f'chi2={chi2:.2f}, gradient={pg:.4g}, {entry["seconds"]:.1f}s',flush=True)
    report['completed']=True; report['finished_unix']=time.time()
    write_json(path,report)
    print('Completed optimization diagnostic:',output,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-id',type=int,required=True)
    p.add_argument('--grid',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--modes',nargs='+',choices=['positive','coupled','legacy'],default=['positive','coupled','legacy'])
    p.add_argument('--calibration-sigma',type=float,default=.1)
    p.add_argument('--background-scale',type=float,default=1.)
    p.add_argument('--continuum-scale',type=float,default=1.)
    p.add_argument('--maxiter',type=int,default=1200)
    p.add_argument('--refine',type=int,default=4000)
    p.add_argument('--initial-fit',type=Path,nargs='*',default=[])
    p.add_argument('--detector-mask',type=Path)
    p.add_argument('--no-default-starts',action='store_true')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=p.parse_args()
    if min(args.calibration_sigma,args.background_scale,args.continuum_scale,args.maxiter,args.refine)<=0:
        raise ValueError('Positive scales/iteration limits required.')
    torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.workflow.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run_locked(args)


if __name__=='__main__':
    main()
