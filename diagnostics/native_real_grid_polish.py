"""Refit failed stationary points without changing the real grid's target."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from diagnostics.native_real_grid import REPO,CODE,imaging_prior,response_kernels,projected_gradient
import jax
import jax.numpy as jnp
import numpy as np
import torch
from scipy.optimize import minimize
from dingo.exposures import load_native_target
from dingo.native_fitting import BilinearLineMap,StripContinuumLikelihood
from dingo.native_forward import NativeLineProjector
from dingo.native_jax import JaxNativeProjector,JaxLineMap,JaxStripLikelihood
from dingo.native_real import RealVelocityProfile


def run(args):
    if jax.default_backend()!='gpu':raise RuntimeError('GPU required')
    torch.set_num_threads(4)
    path=args.grid/'grid.json';report=json.loads(path.read_text());s=report['settings']
    if not report.get('completed'):raise ValueError('Complete the grid first')
    for p in CODE:
        if hashlib.sha256((REPO/p).read_bytes()).hexdigest()!=report['code_sha256'][p]:
            raise ValueError(f'Grid target changed: {p}')
    target=load_native_target(s['bundle'],REPO/'data/ID3700/config_kinematics.yaml',s['size'])
    science,error,valid=[np.stack([getattr(e,k) for e in target.exposures]) for k in ('science','error','valid')]
    kernels,_=response_kernels(target,Path(s['bundle']),s['response_scale'])
    ig,ip,_=imaging_prior(Path(s['imaging']),s['inclination_scatter'])
    grid=BilinearLineMap(s['controls'],s['quadrature'],half_extent=.9,dtype=torch.float64)
    op=NativeLineProjector.from_target(target,grid.east.numpy(),grid.north.numpy(),kernels,
        np.ones(len(target.exposures)),spectral_samples=9,dtype=torch.float64,deposition='cubic')
    arrays={'science':science,'error':error,'valid':valid,'projection_coefficients':op.coefficients.numpy(),
            'kernels':kernels,'inclination_grid':ig,'inclination_logpdf':ip}
    for key,a in arrays.items():
        if hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()!=report['array_sha256'][key]:
            raise ValueError(f'Input changed: {key}')
    likelihood=StripContinuumLikelihood(science,error,valid,[e.pupil for e in target.exposures],degree=2)
    model=RealVelocityProfile(JaxNativeProjector(op),JaxLineMap(grid),JaxStripLikelihood(likelihood),ig,ip,
        correlation=s['correlation'],map_sigma=s['map_sigma'])
    objective=jax.jit(jax.value_and_grad(model.objective));components=jax.jit(model.components)
    best={f['vrot_0p5arcsec_kms']:f for f in report['best_by_velocity']}
    winner=min(best.values(),key=lambda f:f['objective'])
    with np.load(args.grid/winner['artifact']) as a:global_start=a['vector']
    failed=[v for v,f in best.items() if not f['stationarity_passed']]
    digest=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    old_code=args.grid/'polish_code.py'
    if old_code.exists():
        old_digest=hashlib.sha256(old_code.read_bytes()).hexdigest()
        (args.grid/f'polish_code_{old_digest[:12]}.py').write_bytes(old_code.read_bytes())
    report['polish_code_sha256']=digest
    (args.grid/f'polish_code_{digest[:12]}.py').write_bytes(Path(__file__).read_bytes())
    if args.propagate:
        anchor=int(winner['vrot_0p5arcsec_kms'])
        failed=list(range(anchor+1,s['vmax']+1))+list(range(anchor-1,-1,-1))
        continuation=global_start
    zero=float(np.sum(np.asarray(model.likelihood.residual_data)**2))
    for speed in failed:
        with np.load(args.grid/best[speed]['artifact']) as a:local_start=a['vector']
        if args.propagate and speed==anchor-1:continuation=global_start
        starts=(('outward',continuation),) if args.propagate else (('previous',local_start),('best_branch',global_start))
        for label,start in starts:
            t=time.perf_counter()
            def fun(x):
                value,gradient=objective(jnp.asarray(x),jnp.asarray(speed));return float(value),np.asarray(gradient)
            opt=minimize(fun,start,jac=True,method='L-BFGS-B',bounds=model.bounds,
                         options={'maxiter':160 if args.propagate else 1600,'gtol':args.gtol,'ftol':1e-12,'maxls':40,'maxcor':40})
            if projected_gradient(opt.x,opt.jac,model.bounds)>=.1:
                opt=minimize(fun,opt.x,jac=True,method='L-BFGS-B',bounds=model.bounds,
                    options={'maxiter':1600,'gtol':args.gtol,'ftol':1e-12,'maxls':40,'maxcor':40})
            if not np.isfinite(opt.fun) or not np.isfinite(opt.jac).all():raise RuntimeError('Nonfinite refit')
            _,parts=components(jnp.asarray(opt.x),jnp.asarray(speed))
            relative,incpen,otherpen,amplitudes,line,flux,velocity=map(np.asarray,parts)
            pg=projected_gradient(opt.x,opt.jac,model.bounds)
            tag=f"fit_{len(report['fits']):05d}.npz"
            f=model.describe(opt.x,speed)|{'objective':float(opt.fun),'chi_square':float(zero-2*relative),
                'inclination_penalty':float(incpen),'other_penalty':float(otherpen),'start':f'polish_{speed:g}_{label}',
                'stage':'polish','seconds':time.perf_counter()-t,'iterations':int(opt.nit),'evaluations':int(opt.nfev),
                'auxiliary_code_sha256':digest,'requested_optimizer_gtol':args.gtol,
                'optimizer_success':bool(opt.success),'message':str(opt.message),'projected_gradient_inf':pg,
                'stationarity_passed':pg<.1,'artifact':tag}
            np.savez_compressed(args.grid/tag,vector=opt.x,amplitudes=amplitudes,flux=flux,velocity=velocity)
            report['fits'].append(f)
            if f['objective']<best[speed]['objective']:best[speed]=f
            if args.propagate:continuation=opt.x
            report['best_by_velocity']=[best[k] for k in sorted(best)]
            path.write_text(json.dumps(report,indent=2));print(json.dumps(f),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--grid',type=Path,required=True)
    p.add_argument('--propagate',action='store_true',help='Follow the best branch outward over every integer velocity')
    p.add_argument('--gtol',type=float,default=.002,help='Optimizer tolerance; the separate stationarity check is unchanged')
    run(p.parse_args())
