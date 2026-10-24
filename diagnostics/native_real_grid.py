"""Real ID3700: F150W-informed, nuisance-refitted joint R+C velocity scan.

Exploratory penalized profile, not a posterior or a calibrated mass measurement.
Run coarse multistarts first, then --stage dense in the same output directory.
The dense grid includes every integer speed from zero through --vmax.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

from diagnostics import native_posterior_check  # x64 and GPU compiler settings
import jax
import jax.numpy as jnp
import numpy as np
import torch
from astropy.io import fits
from scipy.optimize import least_squares, minimize

from dingo.exposures import load_native_target
from dingo.native_fitting import BilinearLineMap, StripContinuumLikelihood
from dingo.native_forward import NativeLineProjector, projection_coefficients
from dingo.native_jax import JaxLineMap, JaxNativeProjector, JaxStripLikelihood
from dingo.native_real import (RealVelocityProfile, oblate_inclination,
    broadened_inclination_density, separable_response_kernel)

REPO = Path(__file__).resolve().parents[1]
CODE = ["diagnostics/native_real_grid.py", "dingo/native_real.py", "dingo/native_jax.py",
        "dingo/native_forward.py", "dingo/native_fitting.py", "dingo/exposures.py",
        "dingo/native_posterior.py"]


def imaging_prior(imaging, scatter_deg):
    rng = np.random.default_rng(3700)
    stellar, sources = [], []
    for name in ("full_pixel", "core_pixel_clean"):
        path = imaging/name
        report = json.loads((path/"imaging.json").read_text())
        if not report["completed"] or not report["diagnostics"]["convergence_gate_passed"] or not report["orientation_parameterization_ok"]:
            raise ValueError(f"Imaging diagnostics failed: {path}")
        with np.load(path/"samples.npz") as saved:
            q = saved["q"].ravel()[::8]
        thickness = rng.uniform(.05,.30,len(q))
        stellar.extend(oblate_inclination(q,thickness))
        sources.append({"path":str(path),"q_quantiles":report["quantiles_5_50_95"]["q"],
                        "samples_sha256":hashlib.sha256((path/"samples.npz").read_bytes()).hexdigest()})
    grid,logp = broadened_inclination_density(stellar,scatter_deg)
    cdf = np.r_[0.,np.cumsum((np.exp(logp[1:])+np.exp(logp[:-1]))/2*np.diff(grid))]
    return grid,logp,{"sources":sources,"q0_uniform":[.05,.30],"stellar_gas_scatter_deg":scatter_deg,
        "stellar_inclination_quantiles_deg":np.rad2deg(np.quantile(stellar,[.05,.5,.95])).tolist(),
        "gas_inclination_quantiles_deg":np.rad2deg(np.interp([.05,.5,.95],cdf,grid)).tolist(),
        "interpretation":"Equal mixture of full/core-masked morphology fits; these are not independent data. "
            "Provisional transferred orientation distribution with uniform thickness and phenomenological scatter; "
            "not a hierarchical imaging likelihood or calibrated misalignment population."}


def response_kernels(target,bundle,scale_factor):
    psfpath = bundle/"calibration/psf/fsun_grisms_4750/4750_f444w_ePSF.fits"
    with fits.open(psfpath) as h:
        psf = np.maximum(h[0].data,0.); psfscale = h[0].header["PIXSCALE"]
    yy,xx = np.indices(psf.shape)
    cy,cx = np.unravel_index(psf.argmax(),psf.shape)
    selected = np.hypot(xx-cx,yy-cy)*psfscale < .16
    x,y,z = xx[selected],yy[selected],psf[selected]
    def residual(p):
        a,mx,my,sx,sy=p
        return a*np.exp(-.5*((x-mx)**2/sx**2+(y-my)**2/sy**2))-z
    fit = least_squares(residual,[psf.max(),cx,cy,4.,4.],
                        bounds=([0,cx-3,cy-3,1,1],[1,cx+3,cy+3,10,10]))
    if not fit.success:
        raise RuntimeError("F444W core fit failed")
    # A provisional diffraction scaling of the broad-band core to the line.
    cross_sigma_arcsec = np.mean(fit.x[-2:])*psfscale*target.wavelength_um/4.44*scale_factor
    lsfpath = bundle/"calibration/grism_cal/LSF/LSF_param.dat"
    lsf = np.genfromtxt(lsfpath,names=True,dtype=None,encoding="utf-8")
    kernels, records = [], []
    for e in target.exposures:
        row = lsf[lsf["modpup"] == e.module+e.pupil][0]
        logwave = np.log10(target.wavelength_um/4.)
        frac = row["frac1_k"]*logwave+row["frac1_b"]
        fwhm = np.array([row[f"fwhm{j}_nm_k"]*logwave+row[f"fwhm{j}_nm_b"] for j in (1,2)])
        coeff = projection_coefficients(e.calibration,np.array([e.direct_xy]),target.wavelength_um,e.origin_xy)
        alongaxis = 0 if e.pupil == "R" else 1
        dispersion = abs(coeff[0,alongaxis,1])
        detector = e.source_detector_xy([0.,.01,0.],[0.,0.,.01])
        jac = (detector[1:]-detector[0]).T/.01
        pixel_per_arcsec = np.linalg.norm(jac[1-alongaxis])
        sigma = fwhm*.001*dispersion/np.sqrt(8*np.log(2))*scale_factor
        cross_sigma = cross_sigma_arcsec*pixel_per_arcsec
        kernels.append(separable_response_kernel(e.pupil,frac,sigma,cross_sigma))
        records.append({"path":str(e.path),"pupil":e.pupil,"module":e.module,
            "lsf_fraction":float(frac),"lsf_fwhm_nm":fwhm.tolist(),
            "dispersion_pixels_per_um":float(dispersion),"cross_sigma_pixels":float(cross_sigma),
            "along_sigma_pixels":sigma.tolist()})
    return np.stack(kernels),{"files_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (psfpath,lsfpath)},
        "F444W_core_sigma_pixels":fit.x[-2:].tolist(),"F444W_psf_pixel_scale":float(psfscale),
        "cross_sigma_arcsec_at_line":float(cross_sigma_arcsec),"response_scale":scale_factor,
        "exposures":records,"limitations":["Separable spectral double Gaussian and circularized spatial Gaussian core",
        "F444W core rescaled by wavelength/4.44; PSF wings/asymmetry not modeled",
        "Cubic deposition corrected in second moment, not exact response shape",
        "Empirical LSF already includes optical spectral response; no second optical convolution along dispersion"]}


def projected_gradient(x,grad,bounds):
    result=np.array(grad,copy=True)
    for j,(low,high) in enumerate(bounds):
        if low is not None and x[j] <= low+1e-8 and result[j]>0: result[j]=0.
        if high is not None and x[j] >= high-1e-8 and result[j]<0: result[j]=0.
    return float(np.max(abs(result)))


def run(args):
    torch.set_num_threads(4)
    if args.device == "cuda" and jax.default_backend() != "gpu":
        raise RuntimeError("GPU requested but unavailable")
    args.output.mkdir(parents=True,exist_ok=True)
    jax.config.update("jax_compilation_cache_dir",str(args.output/"jax_cache"))
    hashes={p:hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in CODE}
    settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()
              if k not in ("stage","maxiter","device")}
    reportpath=args.output/"grid.json"
    if reportpath.exists():
        report=json.loads(reportpath.read_text())
        if report["code_sha256"] != hashes or report["settings"] != settings:
            raise ValueError("Use a fresh directory after changing the model/settings")
    else:
        if args.stage != "coarse": raise ValueError("Run the coarse multistarts first")
        report={"source_id":3700,"real_observations":True,"completed":False,"settings":settings,
                "code_sha256":hashes,"fits":[],"interpretation":"Penalized joint R+C profile, not a marginalized posterior"}
        for p in CODE:
            dest=args.output/"code"/p;dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes((REPO/p).read_bytes())
    target=load_native_target(args.bundle,REPO/"data/ID3700/config_kinematics.yaml",args.size)
    science,error,valid=[np.stack([getattr(e,k) for e in target.exposures]) for k in ("science","error","valid")]
    pupils=[e.pupil for e in target.exposures]
    incgrid,incpdf,increport=imaging_prior(args.imaging,args.inclination_scatter)
    kernels,kernelreport=response_kernels(target,args.bundle,args.response_scale)
    grid=BilinearLineMap(args.controls,args.quadrature,half_extent=.9,dtype=torch.float64)
    operator=NativeLineProjector.from_target(target,grid.east.numpy(),grid.north.numpy(),kernels,
        np.ones(len(pupils)),spectral_samples=9,dtype=torch.float64,deposition="cubic")
    array_hashes={k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in
        {"science":science,"error":error,"valid":valid,"projection_coefficients":operator.coefficients.numpy(),
         "kernels":kernels,"inclination_grid":incgrid,"inclination_logpdf":incpdf}.items()}
    if "array_sha256" in report and report["array_sha256"] != array_hashes:
        raise ValueError("Real pixels, calibration or imaging prior changed since this scan began")
    report["array_sha256"]=array_hashes
    likelihood=StripContinuumLikelihood(science,error,valid,pupils,degree=2)
    model=RealVelocityProfile(JaxNativeProjector(operator),JaxLineMap(grid),JaxStripLikelihood(likelihood),
        incgrid,incpdf,correlation=args.correlation,map_sigma=args.map_sigma)
    objective=jax.jit(jax.value_and_grad(model.objective))
    components=jax.jit(model.components)
    report.update(inclination_prior=increport,instrument_response=kernelreport,
        exposures=[e.metadata for e in target.exposures],nominal_wavelength_um=target.wavelength_um,
        n_valid_pixels=int(valid.sum()),continuum_rank=likelihood.continuum_rank,
        constraints={"map_sigma":args.map_sigma,"map_correlation_arcsec":args.correlation,
                     "registration_sigma_pixels":.25,"kinematic_center_sigma_arcsec":.15,
                     "systemic_sigma_kms":300.,"intrinsic_width_halfnormal_sigma_kms":100.,
                     "line_amplitudes":"independent positive per exposure, analytically profiled",
                     "emission_PA_and_axis_ratio":"free, independent of stellar/kinematic orientation"})
    target.save(args.output/"native_data.npz")
    np.savez_compressed(args.output/"inclination_prior.npz",angle_rad=incgrid,log_density=incpdf)
    reportpath.write_text(json.dumps(report,indent=2))
    zero_chi=float(np.sum(np.asarray(model.likelihood.residual_data)**2))
    best={}
    for f in report["fits"]:
        v=f["vrot_0p5arcsec_kms"]
        if v not in best or f["objective"] < best[v]["objective"]: best[v]=f
    def vector(f):
        with np.load(args.output/f["artifact"]) as a: return a["vector"]
    def fit(speed,x,startlabel):
        tag=f"fit_{len(report['fits']):05d}"
        t=time.perf_counter()
        def fun(values):
            val,grad=objective(jnp.asarray(values),jnp.asarray(float(speed)))
            return float(val),np.asarray(grad)
        opt=minimize(fun,x,jac=True,method="L-BFGS-B",bounds=model.bounds,
                     options={"maxiter":args.maxiter,"gtol":.02,"ftol":1e-12,"maxls":30,"maxcor":30})
        if not np.isfinite(opt.fun) or not np.isfinite(opt.x).all():
            raise RuntimeError(f"Nonfinite fit at {speed}")
        _,parts=components(jnp.asarray(opt.x),jnp.asarray(float(speed)))
        relative,incpen,otherpen,amplitudes,line,flux,velocity=map(np.asarray,parts)
        pg=projected_gradient(opt.x,opt.jac,model.bounds)
        entry=model.describe(opt.x,speed)|{"objective":float(opt.fun),"chi_square":float(zero_chi-2*relative),
            "inclination_penalty":float(incpen),"other_penalty":float(otherpen),"start":startlabel,
            "stage":args.stage,"seconds":time.perf_counter()-t,"iterations":int(opt.nit),
            "evaluations":int(opt.nfev),"optimizer_success":bool(opt.success),"message":str(opt.message),
            "projected_gradient_inf":pg,"stationarity_passed":pg<.1,"artifact":tag+".npz"}
        # Detailed native-pixel products are regenerated for selected velocities.
        np.savez_compressed(args.output/entry["artifact"],vector=opt.x,amplitudes=amplitudes,
                            flux=flux,velocity=velocity)
        report["fits"].append(entry)
        if speed not in best or entry["objective"]<best[speed]["objective"]: best[speed]=entry
        reportpath.write_text(json.dumps(report,indent=2))
        print(json.dumps(entry),flush=True)
        return opt.x
    inc=np.deg2rad(increport["gas_inclination_quantiles_deg"][1])
    if args.stage == "coarse":
        done={f["start"] for f in report["fits"] if f["stage"]=="coarse"}
        for speed in (0,100,250,500,750,args.vmax):
            for j,theta in enumerate(np.deg2rad([0,90,180,270])):
                label=f"coarse_v{speed}_theta{j}"
                if label not in done: fit(speed,model.initial(theta,inc,rturn=.25 if j%2==0 else .7),label)
        report["coarse_completed"]=True
    elif args.stage == "dense":
        if not report.get("coarse_completed"): raise ValueError("Coarse search incomplete")
        # Traverse both ways. Compare previous continuation to a nearby coarse
        # branch before optimization, retaining the best stored solution.
        done={f["start"] for f in report["fits"] if f["stage"]=="dense"}
        for direction,values in (("up",range(args.vmax+1)),("down",range(args.vmax,-1,-1))):
            previous=None
            for speed in values:
                label=f"{direction}_{speed}"
                if label in done:
                    previous=vector(best[speed]);continue
                nearest=sorted(best,key=lambda v:abs(v-speed))[:3]
                starts=[vector(best[v]) for v in nearest]
                if previous is not None: starts.append(previous)
                scores=[float(objective(jnp.asarray(x),jnp.asarray(float(speed)))[0]) for x in starts]
                fit(speed,starts[int(np.argmin(scores))],label)
                previous=vector(best[speed])
        report["completed"]=True
    elif args.stage == "inspect":
        from diagnostics.native_real_grid_plot import make_plots
        make_plots(args.output,report,best,model,likelihood,components,science,error,valid,pupils)
    report["best_by_velocity"]=[best[k] for k in sorted(best)]
    reportpath.write_text(json.dumps(report,indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle",type=Path,default=Path("/home/aakavan/EDR_data/magnif_copy/sapphires_kinematics_inputs_2026-10-13"))
    p.add_argument("--imaging",type=Path,default=Path("/home/aakavan/kinematics/results/2026-10-15_real_ID3700/imaging"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--stage",choices=["coarse","dense","inspect"],default="coarse")
    p.add_argument("--maxiter",type=int,default=600)
    p.add_argument("--size",type=int,default=81)
    p.add_argument("--controls",type=int,default=21)
    p.add_argument("--quadrature",type=int,default=61)
    p.add_argument("--vmax",type=int,default=1000)
    p.add_argument("--inclination-scatter",type=float,default=10.)
    p.add_argument("--correlation",type=float,default=.10)
    p.add_argument("--map-sigma",type=float,default=2.)
    p.add_argument("--response-scale",type=float,default=1.)
    p.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    run(p.parse_args())
