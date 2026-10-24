"""Conditional multistart sensitivity checks after the real velocity grid.

These reoptimize speed continuously as well as nuisance parameters. They are
not posterior intervals or independently certified global optima.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from diagnostics.native_real_grid import (REPO,CODE,imaging_prior,response_kernels,projected_gradient)
import jax
import jax.numpy as jnp
import numpy as np
import torch
from scipy.optimize import minimize
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from dingo.exposures import load_native_target
from dingo.native_fitting import BilinearLineMap,StripContinuumLikelihood
from dingo.native_forward import NativeLineProjector
from dingo.native_jax import JaxNativeProjector,JaxLineMap,JaxStripLikelihood
from dingo.native_real import RealVelocityProfile
from dingo.native_posterior import radial_profiles


def run(args):
    if jax.default_backend() != 'gpu': raise RuntimeError('GPU required for this diagnostic')
    torch.set_num_threads(4)
    source=json.loads((args.grid/'grid.json').read_text())
    if not source.get('completed'): raise ValueError('Finish both integer-grid sweeps first')
    for p in CODE:
        if hashlib.sha256((REPO/p).read_bytes()).hexdigest()!=source['code_sha256'][p]:
            raise ValueError(f'Baseline code changed: {p}')
    settings=source['settings'];args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'sensitivity.json'
    own_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if path.exists():
        report=json.loads(path.read_text())
        if report['code_sha256'] != own_hash or report['source'] != str(args.grid):
            raise ValueError('Use fresh sensitivity directory')
    else:
        report={'source':str(args.grid),'code_sha256':own_hash,'completed':False,'fits':[],
                'interpretation':'Conditional optimization checks, not posterior uncertainty or model evidence'}
        (args.output/'code.py').write_bytes(Path(__file__).read_bytes())
    target=load_native_target(settings['bundle'],REPO/'data/ID3700/config_kinematics.yaml',settings['size'])
    imaging=Path(settings['imaging'])
    with np.load(imaging/'cutout.npz') as a:
        wcs=WCS(fits.Header.fromstring(str(a['wcs_header'])))
    catalog=SkyCoord(float(target.summary['ra'])*u.deg,float(target.summary['dec'])*u.deg)
    stellar_centers=[]
    for dirname in ('full_pixel','core_pixel_clean'):
        with np.load(imaging/dirname/'samples.npz') as a:
            sky=wcs.pixel_to_world(float(np.median(a['xc'])),float(np.median(a['yc'])))
        east,north=catalog.spherical_offsets_to(sky)
        stellar_centers.append([east.to_value(u.arcsec),north.to_value(u.arcsec)])
    center_mean=np.mean(stellar_centers,axis=0)
    report['stellar_center']={'full_and_masked_east_north_arcsec':stellar_centers,
        'mean_east_north_arcsec':center_mean.tolist(),'adopted_soft_prior_sigma_arcsec':.05,
        'note':'The 0.05 arcsec scale allows morphology/astrometry differences; not the formal centroid error.'}
    science,error,valid=[np.stack([getattr(e,k) for e in target.exposures]) for k in ('science','error','valid')]
    for key,array in zip(('science','error','valid'),(science,error,valid)):
        if hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()!=source['array_sha256'][key]:
            raise ValueError(f'Real input changed: {key}')
    pupils=[e.pupil for e in target.exposures]
    grid=BilinearLineMap(settings['controls'],settings['quadrature'],half_extent=.9,dtype=torch.float64)
    likelihood=StripContinuumLikelihood(science,error,valid,pupils,degree=2)
    best={f['vrot_0p5arcsec_kms']:f for f in source['best_by_velocity']}
    speeds=np.array(sorted(best))
    variants=[('baseline',.15,.3,1.,10.),('stellar_center',.05,.3,1.,10.),
              ('free_center',None,.6,1.,10.),
              ('center_bounds_0p6',.15,.6,1.,10.),('response_0p9',.15,.3,.9,10.),
              ('response_1p1',.15,.3,1.1,10.),('inclination_scatter_5',.15,.3,1.,5.),
              ('inclination_scatter_20',.15,.3,1.,20.),
              ('map_correlation_0p05',.15,.3,1.,10.),('map_correlation_0p20',.15,.3,1.,10.)]
    if args.variants:
        if set(args.variants)-{v[0] for v in variants}:raise ValueError('Unknown sensitivity variant')
        variants=[v for v in variants if v[0] in args.variants]
    done={(f['variant'],f['start_speed']) for f in report['fits']}
    for name,center_sigma,center_bound,response_scale,incscatter in variants:
        ig,ip,inc_report=imaging_prior(Path(settings['imaging']),incscatter)
        kernels,kernel_report=response_kernels(target,Path(settings['bundle']),response_scale)
        op=NativeLineProjector.from_target(target,grid.east.numpy(),grid.north.numpy(),kernels,
            np.ones(len(pupils)),spectral_samples=9,dtype=torch.float64,deposition='cubic')
        correlation={'map_correlation_0p05':.05,'map_correlation_0p20':.2}.get(name,settings['correlation'])
        model=RealVelocityProfile(JaxNativeProjector(op),JaxLineMap(grid),JaxStripLikelihood(likelihood),
            ig,ip,correlation=correlation,map_sigma=settings['map_sigma'])
        bounds=[(0.,settings['vmax']/100.)]+model.bounds.copy()
        bounds[6]=bounds[7]=(-center_bound/.1,center_bound/.1)
        center_mu=jnp.asarray(center_mean if name=='stellar_center' else [0.,0.])
        def objective(p):
            # Change just the explicit center penalty; preserve all other terms.
            center=.1*p[6:8]
            new=0. if center_sigma is None else .5*jnp.sum(((center-center_mu)/center_sigma)**2)
            extra=new-.5*jnp.sum((center/.15)**2)
            return model.objective(p[1:],100*p[0])+extra
        compiled=jax.jit(jax.value_and_grad(objective))
        for start_speed in (250,350,500,1000):
            if (name,start_speed) in done: continue
            f=best[float(speeds[np.argmin(abs(speeds-start_speed))])]
            with np.load(args.grid/f['artifact']) as a:
                start=np.r_[f['vrot_0p5arcsec_kms']/100.,a['vector']]
            def fun(p):
                value,grad=compiled(jnp.asarray(p));return float(value),np.asarray(grad)
            t=time.perf_counter()
            opt=minimize(fun,start,jac=True,method='L-BFGS-B',bounds=bounds,
                         options={'maxiter':args.maxiter,'gtol':.02,'ftol':1e-12,'maxls':30,'maxcor':40})
            if not np.isfinite(opt.fun): raise RuntimeError(f'Nonfinite {name} fit')
            pg=projected_gradient(opt.x,opt.jac,bounds)
            entry=model.describe(opt.x[1:],opt.x[0]*100)|{
                'variant':name,'start_speed':start_speed,'center_sigma_arcsec':center_sigma,
                'center_bound_arcsec':center_bound,'response_scale':response_scale,'inclination_scatter_deg':incscatter,
                'map_correlation_arcsec':correlation,
                'objective':float(opt.fun),'projected_gradient_inf':pg,'stationarity_passed':pg<.1,
                'optimizer_success':bool(opt.success),'message':str(opt.message),'iterations':int(opt.nit),
                'seconds':time.perf_counter()-t,'artifact':f'{name}_from_{start_speed}.npz'}
            np.savez_compressed(args.output/entry['artifact'],vector=opt.x)
            report['fits'].append(entry);path.write_text(json.dumps(report,indent=2))
            print(json.dumps(entry),flush=True)
        winner=min([f for f in report['fits'] if f['variant']==name],key=lambda f:f['objective'])
        with np.load(args.output/winner['artifact']) as a: p=a['vector']
        _,parts=jax.jit(model.components)(jnp.asarray(p[1:]),jnp.asarray(100*p[0]))
        _,_,_,amplitudes,line,flux,velocity=map(np.asarray,parts)
        continuum=likelihood.continuum(torch.from_numpy(line.copy())).numpy()
        residual=np.where(valid,(science-line-continuum)/error,np.nan)
        winner['chi_square']=float(np.nansum(residual**2))
        winner['residual_rms_R_C']=[float(np.sqrt(np.nanmean(residual[np.array(pupils)==pupil]**2)))
                                  for pupil in ('R','C')]
        winner['center_boundary_reached']=bool(np.max(abs(p[6:8]*.1))>=center_bound-.001)
        x=p[1:];east=np.asarray(model.line_map.east)-.1*x[5];north=np.asarray(model.line_map.north)-.1*x[6]
        a=np.cos(x[1])*east+np.sin(x[1])*north;b=-np.sin(x[1])*east+np.cos(x[1])*north
        radius=np.sqrt(a*a+(b/np.cos(x[2]))**2);order=np.argsort(radius)
        r10,r90=np.interp([.1,.9],np.cumsum(flux[order]),radius[order])
        winner['line_flux_radii_10_90_arcsec']=[float(r10),float(r90)]
        winner['reference_radius_in_flux_range']=bool(r10<=.5<=r90)
        radii=np.linspace(max(.1,r10),r90,60)
        from astropy.cosmology import Planck18
        scale=float(Planck18.kpc_proper_per_arcmin(1.1231).value/60)
        profiles=radial_profiles(100*p[0]*np.sin(x[2]),np.exp(x[0]),x[2],radii,kpc_per_arcsec=scale)
        np.savez_compressed(args.output/f'{name}_best.npz',vector=p,line=line,continuum=continuum,
            residual=residual,source_flux=flux,source_velocity=velocity,radii_arcsec=radii,
            kpc_per_arcsec=scale,**profiles)
        path.write_text(json.dumps(report,indent=2))
    report['completed']=True
    report['best_by_variant']=[min([f for f in report['fits'] if f['variant']==name],key=lambda f:f['objective'])
                               for name,*_ in variants]
    path.write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grid',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--maxiter',type=int,default=1600)
    p.add_argument('--variants',nargs='+')
    run(p.parse_args())
