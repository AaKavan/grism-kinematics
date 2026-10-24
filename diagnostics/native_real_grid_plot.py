"""Review artifacts for the conditional real joint-R+C velocity profile."""

import hashlib
import json
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import torch


def make_plots(output,report,best,model,likelihood,components,science,error,valid,pupils):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.cosmology import Planck18
    from dingo.native_posterior import radial_profiles
    output=Path(output)
    speeds=np.array(sorted(best));fits=[best[v] for v in speeds]
    scores=np.array([f["objective"] for f in fits]);index=int(np.argmin(scores))
    minfit=fits[index]; delta=2*(scores-scores.min())
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    axes[0,0].plot(speeds,delta)
    failed=np.array([not f["stationarity_passed"] for f in fits])
    axes[0,0].scatter(speeds[failed],delta[failed],s=5,c="red",label="stationarity not passed")
    axes[0,0].set(xlabel='Deprojected Vrot at 0.5 arcsec [km/s]',ylabel='2 Δ penalized objective',ylim=(0,min(200,float(delta.max())+1)))
    axes[0,0].legend(fontsize=8)
    axes[0,1].plot(speeds,[f["inclination_deg"] for f in fits])
    axes[0,1].set(xlabel='Vrot at 0.5 arcsec [km/s]',ylabel='Refitted inclination [deg]')
    axes[1,0].plot(speeds,[f["turnover_arcsec"] for f in fits])
    axes[1,0].set(xlabel='Vrot at 0.5 arcsec [km/s]',ylabel='Refitted turnover radius [arcsec]')
    axes[1,1].plot(speeds,[f["intrinsic_sigma_kms"] for f in fits])
    axes[1,1].set(xlabel='Vrot at 0.5 arcsec [km/s]',ylabel='Refitted intrinsic width [km/s]',
                  ylim=(0,max(10.,1.05*max(f['intrinsic_sigma_kms'] for f in fits))))
    fig.suptitle('Real ID3700, all 18 R+C exposures — conditional profile, not posterior intervals')
    fig.tight_layout();fig.savefig(output/'profile.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5))
    for prefix,label in [('up_','Ascending sweep'),('down_','Descending sweep'),
                          ('polish_','Additional branch / stationarity checks')]:
        branch={}
        for f in report['fits']:
            if f['start'].startswith(prefix):
                v=f['vrot_0p5arcsec_kms']
                branch[v]=min(branch.get(v,np.inf),f['objective'])
        values=sorted(branch)
        ax.plot(values,[2*(branch[v]-scores.min()) for v in values],lw=1,alpha=.8,label=label)
    ax.plot(speeds,delta,color='black',lw=1.5,ls='--',label='Best stored solution at each speed')
    ax.set(xlabel='Deprojected Vrot at 0.5 arcsec [km/s]',ylabel='2 Δ penalized objective',
           xlim=(150,650),ylim=(0,110),title='Nuisance branches depend on search direction; scores are not confidence levels')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(output/'search_paths.png',dpi=160);plt.close(fig)
    incl=np.load(output/'inclination_prior.npz')
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot(np.rad2deg(incl['angle_rad']),np.exp(incl['log_density'])*np.pi/180)
    ax.set(xlabel='Gas inclination [degrees]',ylabel='Provisional prior density per degree')
    fig.tight_layout();fig.savefig(output/'inclination_prior.png',dpi=150);plt.close(fig)

    # Display alternatives across the requested range, plus near-best examples.
    targets=[0,50,100,150,200,250,300,400,500,750,1000,int(speeds[index])]
    for threshold in (2,10,50):
        group=speeds[delta<=threshold]
        if len(group): targets.extend([int(group[0]),int(group[-1])])
    selected=sorted(set(float(speeds[np.argmin(abs(speeds-v))]) for v in targets))
    panels=output/'panels';panels.mkdir(exist_ok=True)
    candidate_records=[]
    radial_candidates=[]
    kpc_arcsec=float(Planck18.kpc_proper_per_arcmin(1.1231).value/60)
    for v in selected:
        f=best[v]
        with np.load(output/f['artifact']) as saved: x=saved['vector']
        _,parts=components(jnp.asarray(x),jnp.asarray(v))
        _,_,_,amplitudes,line,flux,velocity=map(np.asarray,parts)
        continuum=likelihood.continuum(torch.from_numpy(line.copy())).numpy()
        total=line+continuum
        residual=np.where(valid,(science-total)/error,np.nan)
        np.savez_compressed(output/f'candidate_{v:g}.npz',line=line,continuum=continuum,
            residual=residual,source_flux=flux,source_velocity=velocity,vector=x)
        # The map-weighted, deprojected radii describe where this particular fit
        # places its flux. They are not an independent resolution or coverage test.
        east=np.asarray(model.line_map.east)-.1*x[5];north=np.asarray(model.line_map.north)-.1*x[6]
        a=np.cos(x[1])*east+np.sin(x[1])*north
        b=-np.sin(x[1])*east+np.cos(x[1])*north
        radius=np.sqrt(a*a+(b/np.cos(x[2]))**2)
        order=np.argsort(radius);cum=np.cumsum(flux[order]);cum/=cum[-1]
        r10,r90=np.interp([.1,.9],cum,radius[order])
        # The fitted map has faint off-source/edge flux: its deprojected r90
        # is not a validated observed extent. Use an explicit illustration range.
        radii=np.linspace(.2,.8,60)
        profiles=radial_profiles(v*np.sin(x[2]),np.exp(x[0]),x[2],radii,kpc_per_arcsec=kpc_arcsec)
        np.savez_compressed(output/f'candidate_{v:g}_radial.npz',radii_arcsec=radii,**profiles)
        radial_candidates.append((v,radii,profiles))
        candidate_records.append(f|{'relative_penalized_score':float(2*(f['objective']-scores.min())),
            'line_flux_radii_10_90_arcsec':[float(r10),float(r90)],
            'native_residual_rms':float(np.sqrt(np.nanmean(residual**2)))})
        side=int(np.sqrt(len(flux)))
        brightness=flux.reshape(side,side)
        relative_velocity=(velocity-100*x[3]).reshape(side,side)
        extent=[-model.half_extent,model.half_extent]*2
        fig,axes=plt.subplots(1,2,figsize=(9,4))
        axes[0].imshow(np.arcsinh(brightness/brightness.max()*20),origin='lower',extent=extent,cmap='magma')
        axes[0].set_title('Fitted line-emission map')
        im=axes[1].imshow(np.where(brightness>.01*brightness.max(),relative_velocity,np.nan),
                          origin='lower',extent=extent,cmap='RdBu_r',vmin=-500,vmax=500)
        axes[1].set_title('LOS rotation; colors clipped at ±500 km/s')
        fig.colorbar(im,ax=axes[1],shrink=.8)
        for ax in axes:
            ax.plot(0,0,'w+',ms=10,mew=2,label='Catalog origin')
            ax.plot(.1*x[5],.1*x[6],'cx',ms=9,mew=2,label='Kinematic center')
            ax.set(xlabel='East offset [arcsec]',ylabel='North offset [arcsec]')
        axes[0].legend(fontsize=7,loc='upper left')
        fig.suptitle(f'Conditional source model: Vrot(0.5″)={v:g} km/s; i={f["inclination_deg"]:.1f}°')
        fig.tight_layout();fig.savefig(panels/f'source_v{v:g}.png',dpi=130);plt.close(fig)
        n=science.shape[-1];mid=n//2;cut=np.s_[mid-18:mid+19,mid-18:mid+19]
        for e in range(len(pupils)):
            data=np.where(valid[e],science[e],np.nan)[cut]
            limits=np.nanquantile(data,[.02,.995])
            fig,axes=plt.subplots(1,3,figsize=(10,3.3))
            for ax,arr,title in zip(axes,(data,total[e][cut],residual[e][cut]),('Observed SCI','Line + refitted continuum','(SCI − model) / ERR')):
                ax.imshow(arr,origin='lower',cmap='RdBu_r' if title.startswith('(') else 'magma',
                          **({'vmin':-5,'vmax':5} if title.startswith('(') else {'vmin':limits[0],'vmax':limits[1]}))
                ax.set_title(title,fontsize=10);ax.set_xticks([]);ax.set_yticks([])
            fig.suptitle(f'ID3700 | exposure {e+1} ({pupils[e]}) | Vrot(0.5″)={v:g} km/s | i={f["inclination_deg"]:.1f}°',fontsize=11)
            fig.tight_layout();fig.savefig(panels/f'v{v:g}_e{e}.png',dpi=110);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for v,radii,p in radial_candidates:
        if best[v]['objective']-scores.min()>25 and v!=0:continue
        axes[0].plot(radii*kpc_arcsec,p['deprojected_velocity_kms'],label=f'{v:g} km/s')
        axes[1].plot(radii*kpc_arcsec,p['rotation_only_mass_msun'])
    axes[0].set(xlabel='Model radius [kpc]',ylabel='Vrot [km/s]');axes[0].legend(fontsize=7)
    axes[1].set(xlabel='Model radius [kpc]',ylabel='r Vrot²/G [solar masses]')
    fig.suptitle('Conditional curves at 0.2–0.8 arcsec; illustrative radii, not measured coverage or confidence bands',fontsize=10)
    fig.tight_layout();fig.savefig(output/'candidate_profiles.png',dpi=160);plt.close(fig)
    summary={'best_conditional_fit':minfit,'grid_size':len(speeds),'range_kms':[float(speeds.min()),float(speeds.max())],
        'plot_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'all_integer_velocities_completed':bool(len(speeds)==report['settings']['vmax']+1 and np.all(np.diff(speeds)==1)),
        'stationarity_failures_among_best':int(failed.sum()),'candidates':candidate_records,
        'kpc_per_arcsec_Planck18_at_z1p1231':kpc_arcsec,
        'illustrative_radial_range_arcsec':[.2,.8],
        'limitations':['Profile optimization is not marginalization and its score is not a probability',
        'Flux-weighted radius range is model dependent and does not establish resolved radial information',
        'r Vrot²/G assumes circular support and spherical equivalent geometry; no pressure-support or lensing correction',
        'Response, morphology prior, contaminants and residual noise require validation before scientific mass intervals']}
    (output/'inspection.json').write_text(json.dumps(summary,indent=2))
    options=''.join(f'<option value="{e}">Exposure {e+1}: grism {p}</option>' for e,p in enumerate(pupils))
    records=json.dumps([{'v':v,'score':float(2*(best[v]['objective']-scores.min())),
                        'inclination':best[v]['inclination_deg'],'stationary':best[v]['stationarity_passed']} for v in selected])
    html='''<!doctype html><html><meta charset="utf-8"><title>ID3700 velocity review</title>
<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 16px;color:#222}img{width:100%}input{width:65%}select{padding:8px}small{display:block;color:#555}#value{font-weight:bold;margin:12px 0}</style>
<h1>ID3700: joint R+C velocity review</h1>
<p>All 18 exposures constrain one shared line-emission map and rotation field. Each displayed velocity has refitted nuisance parameters. Use the exposure menu to inspect R and C individually within that joint fit.</p>
<p>This is a conditional, penalized profile. A visually acceptable alternative remains an alternative; these scores are not posterior probabilities or calibrated confidence intervals.</p>
<label>Trial velocity <input id="speed" type="range" min="0" max="MAX" value="0" step="1"></label>
<select id="exposure">OPTIONS</select><div id="value"></div><img id="panel"><img id="source">
<small>SCI is unchanged between trials. The model includes refitted continuum; residual colors always span −5 to +5 times quoted ERR. The slider shows saved comparison velocities; the numerical scan is stored in grid.json.</small>
<h2>Velocity profile</h2><img src="profile.png"><h2>Search direction and local solutions</h2><img src="search_paths.png"><h2>Imaging-informed orientation prior</h2><img src="inclination_prior.png">
<h2>Selected rotation and mass-proxy profiles</h2><img src="candidate_profiles.png">
<p>These curves are examples, not uncertainty bands. The 0.2–0.8 arcsec range is illustrative: resolved radial information has not been established. Faint off-source/edge flux makes the fitted map's outer radius unreliable. Dynamical masses still require instrument, morphology, noise and pressure-support validation.</p>
<script>const records=RECORDS;const slider=document.getElementById('speed');const exposure=document.getElementById('exposure');function update(){const r=records[+slider.value];document.getElementById('panel').src=`panels/v${r.v}_e${exposure.value}.png`;document.getElementById('source').src=`panels/source_v${r.v}.png`;document.getElementById('value').textContent=`Vrot(0.5 arcsec) = ${r.v} km/s; inclination = ${r.inclination.toFixed(1)}°; 2 Δ objective = ${r.score.toFixed(2)}; stationarity ${r.stationary?'passed':'not passed'}`;}slider.oninput=update;exposure.onchange=update;update();</script></html>'''
    html=html.replace('MAX',str(len(selected)-1)).replace('OPTIONS',options).replace('RECORDS',records)
    html=html.replace('value="0" step="1"',f'value="{selected.index(float(speeds[index]))}" step="1"')
    (output/'inspect.html').write_text(html)
    (output/'inspection_plot_code.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in summary.items() if k!='candidates'}),flush=True)
