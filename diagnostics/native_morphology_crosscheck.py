"""Cross-initialize flexible prior cases from newly found high/low velocity basins.

Retains each original best point and the linked control unchanged. All new
optimizations use exactly the parent's physical target and numerical sampling.
"""
import argparse
import copy
import fcntl
from pathlib import Path
import time

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
import torch

from diagnostics.native_morphology_controls import CODE as PARENT_CODE, boundaries, light_moments
from diagnostics.native_simple_dispersion import pack_free, unpack_free, choose_best, verify_hashes, snapshot_inputs
from diagnostics.native_simple_numerics import rebuild_saved_target, refined_engine
from diagnostics.native_real_grid import projected_gradient
from diagnostics.native_real_posterior import heartbeat, save_npz
from dingo.native_coupled import CoupledNativeModel
from dingo.native_morphology_controls import morphology_engine, transport_parameters
from dingo.real_workflow import REPO, read_json, write_json, sha256

CODE = list(dict.fromkeys(PARENT_CODE+['diagnostics/native_morphology_crosscheck.py']))


def run(args):
    parent_dir, output = args.parent.resolve(), args.output.resolve()
    if output == parent_dir:
        raise ValueError('Cross-check requires a new directory.')
    parent = read_json(parent_dir/'dispersion.json')
    if not parent.get('completed') or not parent.get('morphology_controls'):
        raise ValueError('Select a completed morphology comparison.')
    if jax.default_backend() != 'gpu':
        raise RuntimeError('This real-source cross-check requests the GPU explicitly.')
    sid = parent['source_id']
    original_dir = Path(parent['settings']['run'])
    original = read_json(original_dir/'dispersion.json')
    inputs = dict(parent['input_sha256'])
    inputs[str(parent_dir/'dispersion.json')] = sha256(parent_dir/'dispersion.json')
    for filename, key in [('native_data.npz', 'native_data_sha256'), ('inclination_prior.npz', 'inclination_prior_sha256')]:
        inputs[str(parent_dir/filename)] = parent[key]
    points = choose_best(parent['fits'])
    for fit in points:
        inputs[str(parent_dir/fit['artifact'])] = fit['artifact_sha256']
    verify_hashes(inputs, 'Cross-check input')
    verify_hashes({str(REPO/p): h for p,h in parent['code_sha256'].items()}, 'Parent code')
    hashes = {p: sha256(REPO/p) for p in CODE}
    settings = dict(parent=str(parent_dir), run=str(original_dir), baseline=parent['settings']['baseline'],
                    seed_policy='Highest and lowest velocity of the prior-specific best solutions at each sigma; '
                                'matched brightness transport, then bidirectional sigma continuation.',
                    maxiter=1200, refine=4000, stationarity_threshold=.1)
    path = output/'dispersion.json'
    if path.exists():
        report = read_json(path)
        for key,value in [('input_sha256',inputs),('code_sha256',hashes),('settings',settings)]:
            if report[key] != value:
                raise ValueError(f'Cannot resume changed {key}.')
        for fit in report['fits']:
            verify_hashes({str(output/fit['artifact']):fit['artifact_sha256']}, 'Cross-check checkpoint')
    else:
        report = {k:copy.deepcopy(v) for k,v in parent.items() if k not in
            ('fits','best_by_sigma','input_snapshot_paths','input_snapshot_sha256','active',
             'started_unix','finished_unix','updated_unix','initial_map_checks')}
        report.update(cross_initialization=True, completed=False, settings=settings, input_sha256=inputs,
            code_sha256=hashes, fits=[], best_by_sigma=[], initial_map_checks=[], started_unix=time.time(),
            device=str(jax.devices()), interpretation=parent['interpretation']+' Includes retained parent minima '
            'and additional cross-prior basin checks; linked control is copied, not refitted.')
    for name,digest in hashes.items():
        dest=output/'code'/name; dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists(): dest.write_bytes((REPO/name).read_bytes())
        verify_hashes({str(dest):digest},'Frozen cross-check code')
    snapshots,snapshot_hashes=snapshot_inputs(output,inputs)
    report.update(input_snapshot_paths=snapshots,input_snapshot_sha256=snapshot_hashes)
    for filename in ('native_data.npz','inclination_prior.npz'):
        dest=output/filename
        if not dest.exists(): dest.write_bytes((parent_dir/filename).read_bytes())
        verify_hashes({str(dest):sha256(parent_dir/filename)}, 'Frozen arrays')
    if report['completed']:
        print('Already complete:',output,flush=True); return
    inherited={f.get('inherited_artifact') for f in report['fits'] if f.get('inherited')}
    for fit in points:
        if fit['artifact'] in inherited: continue
        tag=f'fit_{len(report["fits"]):04d}.npz'
        (output/tag).write_bytes((parent_dir/fit['artifact']).read_bytes())
        row=copy.deepcopy(fit)
        row.update(start='inherited_'+fit['start'],inherited=True,inherited_artifact=fit['artifact'],
                   artifact=tag,artifact_sha256=sha256(output/tag))
        report['fits'].append(row)
    sigmas=parent['settings']['sigmas']; variants=parent['settings']['variants']
    source_starts={}
    for sigma in sigmas:
        available=[f for f in points if f['model']=='flexible' and f['fixed_sigma_kms']==sigma]
        source_starts[sigma]=[(label,fn(available,key=lambda f:f['vrot_0p5arcsec_kms']))
                             for label,fn in [('low',min),('high',max)]]
    new_count=sum(f['variant']!=variant for variant in variants for starts in source_starts.values() for _,f in starts)
    report['expected_fit_count']=len(points)+new_count+len(variants)*2*(len(sigmas)-1)
    report['best_by_sigma']=choose_best(report['fits']);write_json(path,report)
    with heartbeat('Rebuilding the exact parent target for cross-initialization'):
        target,engine,old_like,response,_=rebuild_saved_target(original,original_dir)
        engine=copy.copy(engine)
        with np.load(parent_dir/'inclination_prior.npz') as prior:
            if int(prior['source_id'])!=sid: raise ValueError('Prior source mismatch')
            engine.inclination_grid=jnp.asarray(prior['inclination_grid'])
            engine.inclination_logpdf=jnp.asarray(prior['inclination_logpdf'])
        engine=refined_engine(engine,target,parent['grid_settings']['controls'],
            parent['source_grid']['shape'][0],parent['source_grid']['spectral_quadrature_nodes'],
            np.asarray(engine.projector.kernels))
    nuisance=parent['nuisance_settings']
    if nuisance['mode']!='positive': raise ValueError('This follow-up retains the positive-continuum target.')
    data,error,valid=[np.stack([getattr(e,k) for e in target.exposures]) for k in ('science','error','valid')]
    pupils=np.asarray([e.pupil for e in target.exposures])
    done={(f['variant'],f['fixed_sigma_kms'],f['start']):f for f in report['fits']}
    def read(directory,fit):
        with np.load(directory/fit['artifact']) as a:
            if int(a['source_id'])!=sid: raise ValueError('Artifact source mismatch')
            return {k:a[k].copy() for k in ('parameters','source_flux','line')}
    for variant in variants:
        trial=morphology_engine(engine,variant)
        model=CoupledNativeModel(trial,target,response,**nuisance)
        bounds=[b for i,b in enumerate(model.bounds) if i!=5]
        objective=jax.jit(jax.value_and_grad(lambda free,sigma:model.objective(unpack_free(free,sigma))))
        components=jax.jit(model.components)
        def fit_one(start,sigma,label,origin):
            if (variant,sigma,label) in done: return
            start=np.asarray(start).copy();start[5]=sigma/100
            print(f'[{len(report["fits"])+1}/{report["expected_fit_count"]}] '
                  f'ID{sid} {variant}, sigma={sigma:g}, {label}, Vstart={100*start[0]:.2f}',flush=True)
            begin=time.perf_counter();count=[0]
            report['active']=dict(variant=variant,sigma=sigma,start=label,started_unix=time.time())
            write_json(path,report)
            def fun(free):
                value,gradient=objective(jnp.asarray(free),jnp.asarray(sigma))
                return float(value),np.asarray(gradient)
            def progress(free): count[0]+=1
            with heartbeat(f'Cross-checking {variant}/{sigma:g}'):
                opt=minimize(fun,pack_free(start),jac=True,method='L-BFGS-B',bounds=bounds,callback=progress,
                    options=dict(maxiter=1200,gtol=.01,ftol=1e-12,maxls=40,maxcor=40))
                pg=projected_gradient(opt.x,opt.jac,bounds)
                if pg>=.1:
                    retry=minimize(fun,opt.x,jac=True,method='L-BFGS-B',bounds=bounds,callback=progress,
                        options=dict(maxiter=4000,gtol=.002,ftol=1e-13,maxls=50,maxcor=60))
                    if np.isfinite(retry.fun) and retry.fun<=opt.fun: opt=retry
                    pg=projected_gradient(opt.x,opt.jac,bounds)
            if not np.isfinite(opt.fun) or not np.isfinite(opt.x).all() or not np.isfinite(opt.jac).all():
                raise RuntimeError('Nonfinite cross-check fit')
            p=np.asarray(unpack_free(jnp.asarray(opt.x),jnp.asarray(sigma)))
            value,parts=components(jnp.asarray(p));parts={k:np.asarray(v) for k,v in parts.items()}
            residual=np.divide(data-parts['line']-parts['continuum']-parts['background'][:,None,None],
                error,out=np.full_like(data,np.nan),where=valid)
            chi2=float(np.nansum(residual**2))
            np.testing.assert_allclose(chi2,parts['chi2'],rtol=1e-10,atol=1e-6)
            np.testing.assert_allclose(value,.5*chi2+parts['penalty'],rtol=1e-10,atol=1e-6)
            x,speed,flux,gz,bz=model.split(p)
            east,north=np.asarray(trial.line_map.east),np.asarray(trial.line_map.north)
            tag=f'fit_{len(report["fits"]):04d}.npz'
            save_npz(output/tag,source_id=np.asarray(sid),parameters=p,initial_parameters=start,
                     vector=np.asarray(x),residual=residual,source_east=east,source_north=north,**parts)
            edge=(abs(east)>.8)|(abs(north)>.8)
            row=trial.describe(np.asarray(x),float(speed))|light_moments(parts['source_flux'],east,north)|dict(
                model='flexible',variant=variant,fixed_sigma_kms=sigma,start=label,sweep='crosscheck',
                initial_artifact=origin,mode=nuisance['mode'],spectral_scale=1.,spatial_scale=1.,
                objective=float(value),chi_square=chi2,penalty=float(parts['penalty']),
                map_penalty=float(.5*np.sum(np.asarray(x[10:10+trial.nmap])**2)),fixed_sigma_prior_penalty=.5*(sigma/100)**2,
                chi_square_R=float(np.nansum(residual[pupils=='R']**2)),chi_square_C=float(np.nansum(residual[pupils=='C']**2)),
                line_flux_1e17=float(flux),source_edge_flux_fraction=float(parts['source_flux'][edge].sum()),
                gain_min_max=[float(parts['gains'].min()),float(parts['gains'].max())],
                background_standardized=np.asarray(bz).tolist(),projected_gradient_inf=float(pg),
                stationarity_passed=bool(pg<.1),optimizer_success=bool(opt.success),message=str(opt.message),
                iterations=count[0],seconds=time.perf_counter()-begin,boundary_parameters=boundaries(p,model,'flexible'),
                artifact=tag,artifact_sha256=sha256(output/tag),inherited=False)
            report['fits'].append(row);report['best_by_sigma']=choose_best(report['fits'])
            report['updated_unix']=time.time();report.pop('active',None);write_json(path,report)
            done[variant,sigma,label]=row
            print(f'  V={float(speed):.2f}, J={float(value):.3f}, gradient={pg:.3g}',flush=True)
        for sigma,starts in source_starts.items():
            for label,seed in starts:
                if seed['variant']==variant: continue
                saved=read(parent_dir,seed)
                old_engine=morphology_engine(engine,seed['variant'])
                p=transport_parameters(saved['parameters'],old_engine,trial)
                value,parts=components(jnp.asarray(p))
                np.testing.assert_allclose(parts['source_flux'],saved['source_flux'],rtol=1e-9,atol=1e-12)
                np.testing.assert_allclose(parts['line'],saved['line'],rtol=1e-9,atol=1e-12)
                delta=.5*(np.sum(p[11:11+trial.nmap]**2)-np.sum(saved['parameters'][11:11+trial.nmap]**2))
                np.testing.assert_allclose(float(value)-seed['objective'],delta,rtol=1e-8,atol=1e-6)
                key=dict(variant=variant,sigma=sigma,seed_variant=seed['variant'],seed_label=label)
                if not any(all(c.get(k)==v for k,v in key.items()) for c in report['initial_map_checks']):
                    report['initial_map_checks'].append(key|dict(max_line_difference=float(np.max(abs(np.asarray(parts['line'])-saved['line'])))))
                fit_one(p,sigma,f'cross_{label}_{seed["variant"]}',str(parent_dir/seed['artifact']))
        def best(sigma):
            return min((f for f in report['fits'] if f['model']=='flexible' and f['variant']==variant
                        and f['fixed_sigma_kms']==sigma),key=lambda f:f['objective'])
        for direction,values in [('up',sigmas),('down',sigmas[::-1])]:
            previous=best(values[0])
            for sigma in values[1:]:
                fit_one(read(output,previous)['parameters'],sigma,f'cross_continuation_{direction}',str(output/previous['artifact']))
                previous=best(sigma)
        jax.clear_caches()
    report.update(completed=True,finished_unix=time.time(),
                  all_selected_stationary=all(f['stationarity_passed'] for f in report['best_by_sigma']))
    report.pop('active',None);write_json(path,report)
    print('Cross-check complete:',output,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    with (a.output/'.workflow.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(a)


if __name__=='__main__': main()
