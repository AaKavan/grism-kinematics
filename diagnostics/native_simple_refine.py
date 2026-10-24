"""Refit selected real-source dispersion points with finer numerical quadrature.

Every nuisance coordinate is refitted at each fixed width. Two stored starts
and continuation in both sigma directions are retained. An optional validated
imaging inclination density is a separate physical-prior experiment, explicitly
recorded in the output; the default preserves the source run's prior exactly.
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

from diagnostics.native_simple_dispersion import (
    VARIANTS, SIGMA_INDEX, pack_free, unpack_free, choose_best, boundary_parameters,
    snapshot_inputs, verify_hashes)
from diagnostics.native_simple_numerics import (
    CODE as NUMERICS_CODE, rebuild_saved_target, refined_engine)
from diagnostics.native_source_grid import inclination_prior
from diagnostics.native_real_dispersion import scaled_kernels
from diagnostics.native_real_grid import projected_gradient
from diagnostics.native_real_posterior import heartbeat, save_npz
from dingo.native_coupled import CoupledNativeModel
from dingo.native_simple import ExponentialCoupledModel
from dingo.real_workflow import REPO, read_json, write_json, sha256

CODE = list(dict.fromkeys(NUMERICS_CODE+['diagnostics/native_simple_refine.py']))


def selected_starts(fits, model, variant, sigma):
    """Retain the best stored candidate and a distinct basin when available."""
    pool = [f for f in fits if f['model']==model and f['variant']==variant
            and np.isclose(f['fixed_sigma_kms'],sigma,rtol=0,atol=1e-10)
            and np.isfinite(f['objective'])]
    if len(pool) < 2:
        raise ValueError(f'At least two saved starts required for {model}/{variant}/sigma={sigma:g}.')
    best = min(pool,key=lambda f:f['objective'])
    others = [f for f in pool if f['artifact'] != best['artifact']]
    def separation(fit):
        angle = abs((fit['theta_east_to_north_deg']-best['theta_east_to_north_deg']+180)%360-180)
        center = np.linalg.norm(np.asarray(fit['center_east_north_arcsec'])-
                                best['center_east_north_arcsec'])
        return abs(fit['vrot_0p5arcsec_kms']-best['vrot_0p5arcsec_kms'])/50.+angle/60.+center/.1
    distinct = [f for f in others if separation(f)>=1.]
    other = min(distinct or others,key=lambda f:f['objective'])
    return (best,other), bool(distinct)


def run_locked(args):
    run_dir, output = args.run.resolve(), args.output.resolve()
    if run_dir == output:
        raise ValueError('Refinement must use a new output directory.')
    source = read_json(run_dir/'dispersion.json')
    if source.get('format')!='native_simple_dispersion_v1' or source.get('real_observations') is not True:
        raise ValueError('Select a real-source native_simple_dispersion_v1 scan.')
    if not source.get('completed'):
        raise ValueError('Complete the original scan before refining its fixed snapshot.')
    if source.get('refinement'):
        raise ValueError('Select the original scan; repeated refinement must declare its target from that baseline.')
    sid=source['source_id']
    if args.source_id is not None and args.source_id != sid:
        raise ValueError('Source ID disagrees with selected scan.')
    if args.device=='cuda' and jax.default_backend()!='gpu':
        raise RuntimeError('GPU requested but unavailable.')
    if args.device=='cpu' and jax.default_backend()!='cpu':
        raise RuntimeError('Set JAX_PLATFORMS=cpu for an explicitly requested CPU run.')
    sigmas=sorted(set(float(x) for x in args.sigmas))
    controls=int(source['grid_settings']['controls'])
    if not sigmas or not np.isfinite(sigmas).all() or min(sigmas)<0 or max(sigmas)>200:
        raise ValueError('Fixed dispersions must be finite and within 0–200 km/s.')
    if args.spatial < controls or args.spectral < 1 or min(args.maxiter,args.refine)<1:
        raise ValueError('Require spatial>=controls, spectral>=1, and positive iteration limits.')
    if args.inclination=='imaging' and args.imaging is None:
        raise ValueError('--inclination imaging requires --imaging with validated fits of this source.')
    selections={}
    for name in args.models:
        for variant in args.variants:
            for sigma in sigmas:
                selections[name,variant,sigma]=selected_starts(source['fits'],name,variant,sigma)
    inputs=dict(source['input_sha256'])
    inputs[str(run_dir/'dispersion.json')]=sha256(run_dir/'dispersion.json')
    inputs[str(run_dir/'native_data.npz')]=source['native_data_sha256']
    for selected,_ in selections.values():
        for fit in selected:
            inputs[str(run_dir/fit['artifact'])]=fit['artifact_sha256']
    verify_hashes(inputs,'Refinement input')
    image_prior=None
    if args.inclination=='imaging':
        image_prior=inclination_prior(sid,args.imaging,'imaging',args.inclination_scatter)
        for record in image_prior[2]['sources']:
            directory=Path(record['path'])
            for filename in ('imaging.json','samples.npz'):
                inputs[str(directory/filename)]=sha256(directory/filename)
    settings=dict(run=str(run_dir),baseline=source['settings']['baseline'],source_id=sid,
        models=list(args.models),variants=list(args.variants),sigmas=sigmas,
        spatial=args.spatial,spectral=args.spectral,maxiter=args.maxiter,refine=args.refine,
        inclination=args.inclination,imaging=str(args.imaging.resolve()) if args.imaging else None,
        inclination_scatter=args.inclination_scatter,
        stationarity_threshold=.1,continuum_mode=source['nuisance_settings']['mode'],
        source_report_sha256=inputs[str(run_dir/'dispersion.json')],
        seed_policy='Two stored starts at each sigma, then bidirectional continuation of the best endpoint basins.')
    hashes={name:sha256(REPO/name) for name in CODE}
    path=output/'dispersion.json'
    report=read_json(path) if path.exists() else dict(format='native_simple_dispersion_v1',
        refinement=True,source_id=sid,real_observations=True,completed=False,settings=settings,
        input_sha256=inputs,code_sha256=hashes,fits=[],best_by_sigma=[],started_unix=time.time(),
        device=str(jax.devices()),interpretation='J = chi-square/2 + prior penalties. '
        'Conditional fixed-dispersion nuisance refits at finer quadrature; not MCMC, confidence '
        'intervals, or model evidence. Imaging-prior overrides, if requested, change the physical target.',
        limitations=['Numerical convergence requires comparison against another finer quadrature.',
            'Finite spatial support and empirical instrumental response remain assumptions.',
            'First-order stationarity and multiple starts do not prove a global minimum.',
            'Raw objectives between morphology/prior choices are not model-selection evidence.'])
    for key,value in [('settings',settings),('input_sha256',inputs),('code_sha256',hashes)]:
        if report[key] != value:
            raise ValueError(f'Cannot resume changed {key}; use a new output directory.')
    for fit in report['fits']:
        verify_hashes({str(output/fit['artifact']):fit['artifact_sha256']},'Refinement checkpoint')
    for relative,digest in report.get('input_snapshot_sha256',{}).items():
        verify_hashes({str(output/relative):digest},'Input snapshot')
    for name,digest in hashes.items():
        destination=output/'code'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        if not destination.exists():
            destination.write_bytes((REPO/name).read_bytes())
        verify_hashes({str(destination):digest},'Code snapshot')
    if report.get('native_data_sha256'):
        verify_hashes({str(output/'native_data.npz'):report['native_data_sha256']},'Saved native data')
    if report.get('inclination_prior_sha256'):
        verify_hashes({str(output/'inclination_prior.npz'):report['inclination_prior_sha256']},'Saved inclination density')
    if report['completed']:
        print('Already complete:',output,flush=True)
        return
    snapshots,snapshot_hashes=snapshot_inputs(output,inputs)
    report.update(input_snapshot_paths=snapshots,input_snapshot_sha256=snapshot_hashes)
    write_json(path,report)
    with heartbeat('Rebuilding exact saved source before refinement'):
        target,engine,old_like,response,grid=rebuild_saved_target(source,run_dir)
    if image_prior:
        engine=copy.copy(engine)
        engine.inclination_grid=jnp.asarray(image_prior[0])
        engine.inclination_logpdf=jnp.asarray(image_prior[1])
        prior_description=image_prior[2]
    else:
        prior_description=source['inclination_prior']
    density=output/'inclination_prior.npz'
    prior_arrays=dict(source_id=np.asarray(sid),inclination_grid=np.asarray(engine.inclination_grid),
                      inclination_logpdf=np.asarray(engine.inclination_logpdf))
    if density.exists():
        with np.load(density) as saved:
            for key,value in prior_arrays.items():
                np.testing.assert_array_equal(saved[key],value,err_msg='Frozen inclination density changed')
    else:
        save_npz(density,**prior_arrays)
    native_dest=output/'native_data.npz'
    if not native_dest.exists():
        native_dest.write_bytes((run_dir/'native_data.npz').read_bytes())
    data,error,valid=[np.stack([getattr(e,k) for e in target.exposures]) for k in ('science','error','valid')]
    pupils=np.asarray([e.pupil for e in target.exposures])
    if set(pupils)!={'R','C'}:
        raise ValueError('Both R and C observations are required.')
    nuisance=source['nuisance_settings']
    grid_settings=dict(source['grid_settings'])
    grid_settings['quadrature']=args.spatial
    if image_prior:
        grid_settings.update(inclination='imaging',imaging=str(args.imaging.resolve()),
                             inclination_scatter=args.inclination_scatter)
    report.update(native_data_sha256=sha256(native_dest),source_summary=target.summary,
        exposures=[e.metadata for e in target.exposures],inclination_prior=prior_description,
        inclination_prior_sha256=sha256(density),grid_settings=grid_settings,
        instrument_response=source['instrument_response'],calibration=source['calibration'],
        calibration_sha256=source['calibration_sha256'],nuisance_settings=nuisance,
        n_valid_pixels=int(valid.sum()),n_exposures=len(pupils),
        source_grid=dict(shape=[args.spatial,args.spatial],half_extent_arcsec=float(engine.half_extent),
                         spectral_quadrature_nodes=args.spectral),
        original_source_grid=source['source_grid'],model_settings={},
        expected_fit_count=len(args.models)*len(args.variants)*(4*len(sigmas)-2),
        starts=[dict(model=name,variant=variant,fixed_sigma_kms=sigma,
                     distinct_stored_basin_found=distinct,artifacts=[f['artifact'] for f in selected])
                for (name,variant,sigma),(selected,distinct) in selections.items()])
    done={(f['model'],f['variant'],f['fixed_sigma_kms'],f['start']):f for f in report['fits']}
    write_json(path,report)

    for variant in args.variants:
        spectral_scale=VARIANTS[variant]
        kernels=np.asarray(engine.projector.kernels)
        if variant!='baseline':
            kernels=scaled_kernels(source['instrument_response']['exposures'],spectral_scale,1.)
            kernels/=kernels.sum(axis=(-2,-1),keepdims=True)
        with heartbeat(f'Constructing {args.spatial} sky and {args.spectral} spectral nodes'):
            refined=refined_engine(engine,target,controls,args.spatial,args.spectral,kernels)
        for name in args.models:
            cls=CoupledNativeModel if name=='flexible' else ExponentialCoupledModel
            model=cls(refined,target,response,**nuisance)
            free_bounds=[bound for i,bound in enumerate(model.bounds) if i!=SIGMA_INDEX]
            compiled=jax.jit(jax.value_and_grad(lambda free,sigma:model.objective(unpack_free(free,sigma))))
            components=jax.jit(model.components)
            report['model_settings'][name]=copy.deepcopy(source['model_settings'][name])
            report['model_settings'][name].update(bounds=[list(x) for x in model.bounds],
                parameter_count=len(model.bounds),free_parameters_per_fixed_sigma=len(free_bounds))

            def read_parameters(directory,fit):
                with np.load(directory/fit['artifact']) as saved:
                    if int(saved['source_id'])!=sid:
                        raise ValueError('Starting fit belongs to another source.')
                    parameters=saved['parameters'].copy()
                if parameters.shape!=(len(model.bounds),):
                    raise ValueError('Starting fit parameter layout does not match refined model.')
                return parameters

            def fit_one(start,sigma,label,sweep,initial_artifact):
                key=name,variant,sigma,label
                if key in done:
                    return read_parameters(output,done[key])
                number=len(report['fits'])+1
                print(f'[{number}/{report["expected_fit_count"]}] ID{sid} {name}/{variant} '
                      f'sigma={sigma:g}, {label}, Vstart={100*start[0]:.1f}',flush=True)
                begin=time.perf_counter();count=[0];last=[0.]
                report['active']=dict(model=name,variant=variant,fixed_sigma_kms=sigma,
                                      start=label,iteration=0,started_unix=time.time())
                write_json(path,report)
                def fun(free):
                    value,gradient=compiled(jnp.asarray(free),jnp.asarray(sigma))
                    return float(value),np.asarray(gradient)
                def progress(free):
                    count[0]+=1
                    elapsed=time.perf_counter()-begin
                    if elapsed-last[0]>=20:
                        print(f'  iteration {count[0]}, {elapsed:.0f}s, V={100*free[0]:.2f}',flush=True)
                        report['active'].update(iteration=count[0],seconds=elapsed,
                                                vrot_0p5arcsec_kms=float(100*free[0]))
                        write_json(path,report);last[0]=elapsed
                with heartbeat(f'Refining {name}/{variant}/sigma={sigma:g}'):
                    opt=minimize(fun,pack_free(start),jac=True,method='L-BFGS-B',bounds=free_bounds,
                        callback=progress,options=dict(maxiter=args.maxiter,gtol=.01,ftol=1e-12,maxls=40,maxcor=40))
                    pg=projected_gradient(opt.x,opt.jac,free_bounds)
                    if pg>=.1:
                        second=minimize(fun,opt.x,jac=True,method='L-BFGS-B',bounds=free_bounds,
                            callback=progress,options=dict(maxiter=args.refine,gtol=.002,ftol=1e-13,maxls=50,maxcor=60))
                        # A failed retry must never conceal a lower stored score.
                        if np.isfinite(second.fun) and second.fun<=opt.fun:
                            opt=second
                        pg=projected_gradient(opt.x,opt.jac,free_bounds)
                if not np.isfinite(opt.fun) or not np.isfinite(opt.x).all() or not np.isfinite(opt.jac).all():
                    report['active']['error']='Nonfinite optimizer result; previous checkpoints preserved.'
                    write_json(path,report)
                    raise RuntimeError(report['active']['error'])
                parameters=np.asarray(unpack_free(jnp.asarray(opt.x),jnp.asarray(sigma)))
                value,parts=components(jnp.asarray(parameters))
                parts={k:np.asarray(v) for k,v in parts.items()}
                if nuisance['mode']=='coupled':
                    parts['continuum']=old_like.continuum(torch.from_numpy(parts['line'].copy())).numpy()
                total=parts['line']+parts['continuum']+parts['background'][:,None,None]
                residual=np.divide(data-total,error,out=np.full_like(data,np.nan),where=valid)
                chi2=float(np.nansum(residual**2))
                np.testing.assert_allclose(chi2,parts['chi2'],rtol=1e-10,atol=1e-6)
                np.testing.assert_allclose(float(value),.5*chi2+float(parts['penalty']),rtol=1e-10,atol=1e-6)
                x,speed,flux,gz,bz=model.split(parameters)
                description=refined.describe(np.asarray(x),float(speed)) if name=='flexible' else model.describe(parameters)
                tag=f'fit_{len(report["fits"]):04d}.npz'
                east,north=np.asarray(refined.line_map.east),np.asarray(refined.line_map.north)
                save_npz(output/tag,source_id=np.asarray(sid),parameters=parameters,
                    initial_parameters=np.asarray(start),vector=np.asarray(x),residual=residual,
                    source_east=east,source_north=north,**parts)
                edge=(np.abs(east)>.8)|(np.abs(north)>.8)
                entry=description | dict(model=name,mode=nuisance['mode'],variant=variant,fixed_sigma_kms=sigma,
                    start=label,sweep=sweep,initial_artifact=initial_artifact,
                    spectral_scale=spectral_scale,spatial_scale=1.,objective=float(value),
                    chi_square=chi2,penalty=float(parts['penalty']),fixed_sigma_prior_penalty=.5*(sigma/100.)**2,
                    chi_square_R=float(np.nansum(residual[pupils=='R']**2)),
                    chi_square_C=float(np.nansum(residual[pupils=='C']**2)),line_flux_1e17=float(flux),
                    source_edge_flux_fraction=float(parts['source_flux'][edge].sum()),
                    gain_min_max=[float(parts['gains'].min()),float(parts['gains'].max())],
                    background_standardized=np.asarray(bz).tolist(),projected_gradient_inf=float(pg),
                    stationarity_passed=bool(pg<.1),optimizer_success=bool(opt.success),
                    message=str(opt.message),iterations=count[0],seconds=time.perf_counter()-begin,
                    boundary_parameters=boundary_parameters(parameters,model,name),artifact=tag,
                    artifact_sha256=sha256(output/tag))
                report['fits'].append(entry);report['best_by_sigma']=choose_best(report['fits'])
                report['updated_unix']=time.time();report.pop('active',None)
                write_json(path,report);done[key]=entry
                print(f'  saved V={float(speed):.2f}, J={float(value):.3f}, gradient={pg:.3g}, '
                      f'{entry["seconds"]:.1f}s',flush=True)
                return parameters

            for sigma in sigmas:
                selected,_=selections[name,variant,sigma]
                for index,fit in enumerate(selected):
                    fit_one(read_parameters(run_dir,fit),sigma,f'stored_{index}','stored',str(run_dir/fit['artifact']))
            def best_at(sigma):
                return min((f for f in report['fits'] if f['model']==name and f['variant']==variant
                    and f['fixed_sigma_kms']==sigma),key=lambda f:f['objective'])
            for sweep,values in [('up',sigmas),('down',sigmas[::-1])]:
                previous=best_at(values[0])
                for sigma in values[1:]:
                    fit_one(read_parameters(output,previous),sigma,f'continuation_{sweep}',sweep,
                            str(output/previous['artifact']))
                    previous=best_at(sigma)
            jax.clear_caches()
    report.update(completed=True,finished_unix=time.time(),
                  all_selected_stationary=all(f['stationarity_passed'] for f in report['best_by_sigma']))
    report.pop('active',None);write_json(path,report)
    print(f'Completed {len(report["fits"])} refined fits; results: {output}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-id',type=int)
    parser.add_argument('--models',nargs='+',choices=['flexible','exponential'],default=['exponential'])
    parser.add_argument('--variants',nargs='+',choices=list(VARIANTS),default=['baseline'])
    parser.add_argument('--sigmas',nargs='+',type=float,default=[0,10,20,40,80,120,200])
    parser.add_argument('--spatial',type=int,default=121)
    parser.add_argument('--spectral',type=int,default=17)
    parser.add_argument('--inclination',choices=['saved','imaging'],default='saved')
    parser.add_argument('--imaging',type=Path)
    parser.add_argument('--inclination-scatter',type=float,default=10.)
    parser.add_argument('--maxiter',type=int,default=1200)
    parser.add_argument('--refine',type=int,default=4000)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args()
    if len(set(args.models))!=len(args.models) or len(set(args.variants))!=len(args.variants):
        raise ValueError('Duplicate model/response scenarios are not allowed.')
    torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'.workflow.lock').open('w') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('This refinement run is already active.') from exc
        run_locked(args)


if __name__=='__main__':
    main()
