"""Retrospective real-exposure prediction check; no MCMC or model selection.

One declared dither split, rotation only, existing flexible calibrated renderer.
Galaxy parameters learn only from training exposures. Held-out continuum uses
sidebands only; withheld line pixels cannot tune flux, geometry or registration.
"""
import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import time

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from scipy.linalg import helmert
from scipy.optimize import minimize
import torch

from diagnostics.native_simple_numerics import rebuild_saved_target, refined_engine
from diagnostics.native_simple_dispersion import boundary_parameters, verify_hashes
from diagnostics.native_real_grid import projected_gradient
from diagnostics.native_real_posterior import heartbeat, save_npz
from dingo.native_coupled import CoupledNativeModel, PositiveStripContinuum
from dingo.real_workflow import read_json, write_json, sha256, REPO


def utc():
    return datetime.now(timezone.utc).isoformat()


def metrics(residual, mask, pupils):
    def one(m):
        values = residual[m]
        return dict(n_pixels=int(m.sum()), chi_square=float(values @ values),
                    rms=float(np.sqrt(np.mean(values**2))), mean=float(values.mean()),
                    n_abs_gt5=int(np.sum(np.abs(values)>5)))
    return dict(all=one(mask), **{g: one(mask & (pupils==g)[:,None,None]) for g in ('R','C')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--maxiter', type=int, default=1600)
    parser.add_argument('--refine', type=int, default=2400)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output/'holdout.json'
    if path.exists():
        raise ValueError('Use a new output directory; never overwrite a diagnostic.')
    if jax.default_backend() != 'gpu':
        raise RuntimeError('This run requests GPU execution.')
    torch.set_num_threads(4)
    source_dir = args.source.resolve()
    source = read_json(source_dir/'dispersion.json')
    origin_dir = Path(source['settings']['run'])
    origin = read_json(origin_dir/'dispersion.json')
    seeds_dir = Path(source['settings']['seeds'])
    seeds = read_json(seeds_dir/'dispersion.json')
    if source['source_id'] != 21636 or not source['completed']:
        raise ValueError('This predeclared diagnostic is for completed real ID21636 fits.')
    low = min((f for f in source['fits'] if f['model']=='flexible' and
        f['variant']=='baseline' and f['fixed_sigma_kms']==0), key=lambda f:f['objective'])
    high = max((f for f in seeds['fits'] if f['model']=='flexible' and
        f['variant']=='baseline' and f['fixed_sigma_kms']==0), key=lambda f:f['vrot_0p5arcsec_kms'])
    train = np.array([0,1,3,4,6,7,9,10,12,13,15,16])
    held = np.array([2,5,8,11,14,17])
    report = dict(format='native_real_holdout_v1', source_id=21636, completed=False,
        started_utc=utc(), device=str(jax.devices()), fits=[],
        train_indices=train.tolist(), held_indices=held.tolist(),
        split_policy='Third dither in each activity group withheld; 6 R + 6 C training; 3 R + 3 C withheld.',
        settings=dict(spatial_nodes=121, spectral_nodes=1, fixed_sigma_kms=0.,
            maxiter=args.maxiter, refine=args.refine, stationarity_threshold=.1,
            sideband_guard_pixels=28., assessment_core_radius_pixels=15.,
            starts=['saved_lower', 'saved_higher', 'smooth_200', 'smooth_800']),
        holdout_policy='Galaxy map, kinematics and physical flux frozen after training. '
            'Held exposure registration zero, gain one. Continuum/background estimated '
            'only at dispersion offsets >=28 pixels, identically for every candidate. '
            'Line-window scores exclude those sideband pixels.',
        limitations=['Retrospective experiment: previous full-data fits informed model design and saved starts.',
            'Two additional smooth starts do not use saved galaxy fits.',
            'One split and local optimization do not establish a unique posterior or validated masses.',
            'Prediction variance conditions on active continuum coefficients and fixed calibration.',
            'The existing velocity prior bounds Vrot(0.5 arcsec) to 0–1000 km/s; bound hits are failures, not detections.'],
        source_report=str(source_dir/'dispersion.json'), source_sha256=sha256(source_dir/'dispersion.json'),
        worker_sha256=sha256(__file__))
    write_json(path, report)
    with heartbeat('Reconstructing ID21636 native exposures and calibration'):
        target, engine, old_like, response, grid = rebuild_saved_target(origin, origin_dir)
        with np.load(seeds_dir/'inclination_prior.npz', allow_pickle=False) as a:
            engine.inclination_grid = jnp.asarray(a['inclination_grid'])
            engine.inclination_logpdf = jnp.asarray(a['inclination_logpdf'])
        engine = refined_engine(engine, target, 21, 121, 1, np.asarray(engine.projector.kernels))
    data, error, valid = [np.stack([getattr(e,k) for e in target.exposures]) for k in ('science','error','valid')]
    pupils = np.array([e.pupil for e in target.exposures])
    if len(data)!=18 or list(pupils)!=['R']*9+['C']*9:
        raise ValueError('Exposure ordering differs from the declared split.')
    centers = np.stack([np.asarray(e.line_xy)-np.asarray(e.origin_xy) for e in target.exposures])
    train_valid = valid.copy(); train_valid[held] = False
    held_valid = valid.copy(); held_valid[train] = False
    yy, xx = np.indices(data.shape[1:])
    core = (xx-centers[:,0,None,None])**2+(yy-centers[:,1,None,None])**2 <= 15**2
    save_npz(output/'observations.npz', science=data, error=error, valid=valid,
        pupils=pupils, line_centers_xy=centers, train_indices=train, held_indices=held, core=core)
    report.update(exposures=[e.metadata for e in target.exposures],
        observation_sha256=sha256(output/'observations.npz'),
        native_data_sha256=source['native_data_sha256'], nuisance_settings=source['nuisance_settings'],
        inclination_prior=source.get('inclination_prior'),
        training_pixels=int(train_valid.sum()), held_pixels=int(held_valid.sum()))
    model = CoupledNativeModel(engine, target, response, **source['nuisance_settings'])
    arrays = []
    for directory, fit in ((source_dir, low), (seeds_dir, high)):
        verify_hashes({str(directory/fit['artifact']):fit['artifact_sha256']}, 'Starting fit')
        with np.load(directory/fit['artifact'], allow_pickle=False) as a:
            arrays.append(a['parameters'].copy())
    # At zero dispersion one spectral node must reproduce the saved 17-node
    # prediction. This avoids redundant work without changing the model.
    with heartbeat('Verifying exact zero-dispersion renderer against saved prediction'):
        value, parts = jax.jit(model.components)(jnp.asarray(arrays[0]))
        with np.load(source_dir/low['artifact'], allow_pickle=False) as a:
            np.testing.assert_allclose(np.asarray(parts['line']), a['line'], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(float(value), low['objective'], rtol=1e-10, atol=1e-6)
    report['zero_dispersion_renderer_check_passed'] = True
    model.continuum_model = PositiveStripContinuum(data, error, train_valid, pupils,
        coefficient_scale=source['nuisance_settings']['continuum_scale'])
    # Training-only zero-sum gains: heldout gains remain exactly one.
    basis = np.zeros((18,17))
    basis[train,:11] = helmert(12).T*np.sqrt(12/11)
    model.gain_basis = jnp.asarray(basis)
    stop = 1+model.base_size
    regstart = 11+engine.nmap
    fixed = [5, *(regstart+2*held), *(regstart+2*held+1),
        *range(stop+1+11, stop+18), *(stop+18+held)]
    fixed = np.array(sorted(set(fixed)), dtype=int)
    free = np.setdiff1d(np.arange(len(model.bounds)), fixed)
    bounds = [model.bounds[i] for i in free]
    template = jnp.zeros(len(model.bounds))
    def unpack(q):
        return template.at[free].set(q)
    compiled = jax.jit(jax.value_and_grad(lambda q:model.objective(unpack(q))))
    components = jax.jit(model.components)
    np.testing.assert_array_equal(np.asarray(model.continuum_model.inv)[held], 0.)
    np.testing.assert_array_equal(np.asarray(model.continuum_model.data)[held], 0.)
    report['held_pixels_zero_weight_verified'] = True
    # Reset exposure nuisances for every start; old full-data estimates cannot
    # be carried over into heldout registration, gain or background.
    for p in arrays:
        p[5] = 0.; p[regstart:stop] = 0.; p[stop+1:] = 0.
    cold = np.zeros_like(arrays[0])
    cold[0] = 2.; cold[1] = np.log(.2); cold[2] = 0.
    cold[3] = float(engine.inclination_grid[jnp.argmax(engine.inclination_logpdf)])
    cold[8] = np.log(.35); cold[9] = .7; cold[10] = 0.; cold[stop] = 3.
    other = cold.copy(); other[0] = 8.; other[2] = np.pi/2
    starts = dict(saved_lower=arrays[0], saved_higher=arrays[1], smooth_200=cold, smooth_800=other)
    report['fixed_parameter_indices'] = fixed.tolist()
    report['n_free_parameters'] = len(free)
    write_json(path, report)
    for name, start in starts.items():
        begin = time.perf_counter(); counter=[0]; last=[0.]
        print(f'Start {name}: V={start[0]*100:.2f} km/s', flush=True)
        report['active'] = dict(start=name, iteration=0, started_utc=utc())
        write_json(path, report)
        def fun(q):
            value, grad = compiled(jnp.asarray(q))
            return float(value), np.asarray(grad)
        def progress(q):
            counter[0] += 1
            elapsed=time.perf_counter()-begin
            if elapsed-last[0] >=20:
                p=np.asarray(unpack(jnp.asarray(q)))
                print(f'  {name} iteration {counter[0]} elapsed {elapsed:.0f}s V={p[0]*100:.2f}',flush=True)
                save_npz(output/f'checkpoint_{name}.npz', parameters=p, iteration=np.array(counter[0]))
                report['active'].update(iteration=counter[0], seconds=elapsed, velocity_kms=float(p[0]*100))
                write_json(path, report);last[0]=elapsed
        with heartbeat(f'Training fit {name}'):
            opt = minimize(fun, start[free], jac=True, method='L-BFGS-B', bounds=bounds,
                callback=progress, options=dict(maxiter=args.maxiter, gtol=.01, ftol=1e-12, maxls=40,maxcor=40))
            pg=projected_gradient(opt.x,opt.jac,bounds)
            if pg >= .1:
                retry=minimize(fun,opt.x,jac=True,method='L-BFGS-B',bounds=bounds,callback=progress,
                    options=dict(maxiter=args.refine,gtol=.002,ftol=1e-13,maxls=50,maxcor=60))
                if np.isfinite(retry.fun) and retry.fun<=opt.fun:
                    opt=retry
                pg=projected_gradient(opt.x,opt.jac,bounds)
        p=np.asarray(unpack(jnp.asarray(opt.x)))
        if not np.isfinite(opt.fun) or not np.isfinite(p).all():
            raise ValueError('Nonfinite training result.')
        value, parts = components(jnp.asarray(p));parts={k:np.asarray(v) for k,v in parts.items()}
        np.testing.assert_array_equal(p[fixed],0.)
        np.testing.assert_allclose(parts['gains'][held],1.,rtol=0,atol=1e-14)
        total=parts['line']+parts['continuum']+parts['background'][:,None,None]
        residual=np.divide(data-total,error,out=np.zeros_like(data),where=valid)
        train_metrics=metrics(residual,train_valid,pupils)
        np.testing.assert_allclose(train_metrics['all']['chi_square'],parts['chi2'],rtol=1e-11,atol=1e-6)
        desc=engine.describe(p[1:stop],100*p[0])
        record=desc|dict(start=name,objective=float(value),penalty=float(parts['penalty']),
            projected_gradient_inf=float(pg),stationarity_passed=bool(pg<.1),
            optimizer_success=bool(opt.success),message=str(opt.message),iterations=counter[0],
            seconds=time.perf_counter()-begin,training=train_metrics,
            training_core=metrics(residual,train_valid&core,pupils),
            boundary_parameters=boundary_parameters(p,model,'flexible'),artifact=f'{name}.npz')
        east,north=np.asarray(engine.line_map.east),np.asarray(engine.line_map.north)
        edge=(np.abs(east)>.8)|(np.abs(north)>.8)
        record['source_edge_flux_fraction']=float(parts['source_flux'][edge].sum())
        save_npz(output/record['artifact'],parameters=p,initial_parameters=start,total=total,residual=residual,
            source_east=east,source_north=north,**parts)
        record['artifact_sha256']=sha256(output/record['artifact'])
        report['fits'].append(record);write_json(path,report)
        print(f'Finished {name}: V={100*p[0]:.2f}, i={desc["inclination_deg"]:.2f}, J={float(value):.3f}, grad={pg:.4g}',flush=True)
    report.pop('active',None)
    report.update(training_completed=True,training_finished_utc=utc())
    write_json(path,report)
    print('Training complete; run the CPU heldout assessment next:',output,flush=True)


if __name__=='__main__':
    main()
