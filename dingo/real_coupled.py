"""CPU-only inspection of the new shared-flux real-source fits."""
from pathlib import Path
import re

import numpy as np

from .real_workflow import read_json, sha256, run_command
from .real_review import load_pixels

LABELS = {'positive': 'Shared flux + positive continuum', 'coupled': 'Shared flux only',
          'legacy': 'Previous independent-flux model (same pixel mask)'}


def fit_coupled(source, grid, *, run_name='coupled_repair_v3', modes=('positive','coupled','legacy'),
                detector_mask=None, calibration_sigma=.1, background_scale=1.,
                continuum_scale=1., maxiter=1200, refine=4000, device='cuda'):
    """One worker per model to release compiler memory between comparisons."""
    if not re.fullmatch(r'[A-Za-z0-9_-]+',run_name) or not modes or set(modes)-set(LABELS):
        raise ValueError('Use a valid run name and known, nonempty model selection.')
    grid=Path(grid).expanduser().resolve()
    grid_report=read_json(grid/'grid.json')
    if grid_report['source_id'] != source.source_id:
        raise ValueError('Choose this source\'s own saved grid.')
    initial=[]
    if not grid_report.get('best_by_velocity'):
        if not grid_report.get('preparation_completed') or not grid_report.get('initial_fits'):
            raise ValueError('Prepare this source\'s own native data and starting points first.')
        for seed in grid_report['initial_fits']:
            path=grid/seed['artifact']
            if sha256(path) != seed['sha256']:
                raise ValueError('Saved initialization artifact changed.')
            initial.append(str(path))
    root=source.directory/run_name
    for mode in modes:
        out=root/mode
        args=['-m','diagnostics.native_coupled_fit','--source-id',str(source.source_id),
            '--grid',str(grid),'--output',str(out),'--modes',mode,'--device',device,
            '--calibration-sigma',str(calibration_sigma),'--background-scale',str(background_scale),
            '--continuum-scale',str(continuum_scale),'--maxiter',str(maxiter),'--refine',str(refine)]
        if detector_mask is not None:
            args += ['--detector-mask',str(Path(detector_mask).expanduser().resolve())]
        if initial:
            args += ['--no-default-starts','--initial-fit',*initial]
        print(f'ID{source.source_id}: {LABELS[mode]}',flush=True)
        run_command(args,out/'run.log',device=device)
    return root


class CoupledResults:
    def __init__(self, path, source_id):
        self.path = Path(path).expanduser().resolve()
        self.report = read_json(self.path/'coupled.json')
        if (self.report.get('format') != 'native_coupled_v1'
                or self.report.get('real_observations') is not True
                or self.report.get('source_id') != int(source_id)):
            raise ValueError('Select this source\'s own coupled-model run.')
        self.source_id = int(source_id)
        if sha256(self.path/'native_data.npz') != self.report['native_data_sha256']:
            raise ValueError('Saved native pixels changed.')
        self.pixels, self.metadata = load_pixels(self.path/'native_data.npz', self.source_id)
        self.fits = self.report['fits']

    def best(self, mode='positive'):
        matches = [f for f in self.fits if f['mode'] == mode]
        if not matches:
            raise ValueError('No fit has been saved for this mode.')
        return min(matches, key=lambda f:f['objective'])

    def candidate(self, mode='positive', start=None):
        fit = self.best(mode) if start is None else next(f for f in self.fits if f['mode']==mode and f['start']==start)
        path = self.path/fit['artifact']
        if sha256(path) != fit['artifact_sha256']:
            raise ValueError('Saved candidate changed.')
        with np.load(path) as a:
            saved = {k:a[k] for k in a.files}
        if int(saved['source_id']) != self.source_id:
            raise ValueError('Candidate belongs to another source.')
        return fit, saved

    def table(self):
        import pandas as pd
        table=pd.DataFrame([dict(model=LABELS[f['mode']], start=f['start'],
            V_0p5_kms=f['vrot_0p5arcsec_kms'], sigma_kms=f['intrinsic_sigma_kms'],
            inclination_deg=f['inclination_deg'], turnover_arcsec=f['turnover_arcsec'],
            x0_arcsec=f['center_east_north_arcsec'][0], y0_arcsec=f['center_east_north_arcsec'][1],
            flux_1e17=f['line_flux_1e17'], score_above_best=2*(f['objective']-self.best(f['mode'])['objective']),
            chi2_R=f['chi_square_R'],chi2_C=f['chi_square_C'],
            min_gain=f['gain_min_max'][0],max_gain=f['gain_min_max'][1],
            gradient=f['projected_gradient_inf'],stationary=f['stationarity_passed']) for f in self.fits])
        seeds={}
        for i,path in enumerate(self.report['settings'].get('initial_fit',[])):
            if sha256(path) != self.report['input_sha256'][path]:
                raise ValueError('Saved starting point changed.')
            with np.load(path) as saved:
                if int(saved['source_id']) != self.source_id:
                    raise ValueError('Starting point belongs to another source.')
                p=saved['parameters']
                seeds[f'saved_{i}']=(100*p[0],100*p[5],np.rad2deg(p[2])%360)
        if seeds:
            for j,key in enumerate(('initial_V_kms','initial_sigma_kms','initial_theta_deg')):
                table.insert(2+j,key,[seeds.get(s,(np.nan,)*3)[j] for s in table['start']])
        return table

    def pair(self, r=0, c=None):
        pupils = [e['pupil'] for e in self.metadata['exposures']]
        c = pupils.index('C') if c is None else c
        if not 0<=r<len(pupils) or not 0<=c<len(pupils) or pupils[r]!='R' or pupils[c]!='C':
            raise ValueError('Choose zero-based R and C exposure indices.')
        return r,c

    def fit_figure(self, mode='positive', start=None, *, r_exposure=0, c_exposure=None, full_cutout=False):
        import matplotlib.pyplot as plt
        from matplotlib.colors import AsinhNorm
        fit,s = self.candidate(mode,start)
        fig,axes=plt.subplots(2,4,figsize=(17,8),layout='constrained')
        for row,e in enumerate(self.pair(r_exposure,c_exposure)):
            data=np.where(self.pixels['valid'][e],self.pixels['science'][e],np.nan)
            total=s['line'][e]+s['continuum'][e]+s['background'][e]
            mid=data.shape[0]//2
            lower,upper=(0,len(data)) if full_cutout else (mid-18,mid+19)
            cut=np.s_[lower:upper,lower:upper]
            lo,hi=np.nanquantile(data[cut],[.02,.995])
            norm=AsinhNorm(linear_width=max((hi-lo)/10,1e-12),vmin=lo,vmax=max(hi,lo+1e-12))
            pupil=self.metadata['exposures'][e]['pupil']
            for col,(v,label) in enumerate(zip((data,total,s['residual'][e]),('Observed SCI','Predicted SCI','Residual / ERR'))):
                style=(dict(cmap=plt.get_cmap('RdBu_r').with_extremes(bad='0.65'),vmin=-5,vmax=5)
                    if col==2 else dict(cmap=plt.get_cmap('gist_heat').with_extremes(bad='0.65'),norm=norm))
                im=axes[row,col].imshow(np.where(self.pixels['valid'][e],v,np.nan)[cut],origin='lower',
                    extent=[lower-.5,upper-.5]*2,interpolation='none',**style)
                axes[row,col].set(title=f'{pupil}, exposure {e+1}: {label}',xlabel='Cutout x [pixel]')
                fig.colorbar(im,ax=axes[row,col],shrink=.65,label='ERR' if col==2 else 'DN/s')
        side=int(np.sqrt(s['source_flux'].size));weights=np.ones(side);weights[[0,-1]]=.5
        flux=s['source_flux'].reshape(side,side)/np.outer(weights,weights);flux/=flux.max()
        los=s['source_velocity'].reshape(side,side)-fit['systemic_offset_kms']
        for row,v in enumerate((flux,np.where(flux>.01,los,np.nan))):
            style=(dict(cmap='gist_heat',norm=AsinhNorm(linear_width=.05,vmin=0,vmax=1)) if row==0
                   else dict(cmap='RdBu_r',vmin=-500,vmax=500))
            im=axes[row,3].imshow(v,origin='lower',extent=[-.9,.9]*2,**style)
            axes[row,3].set(title='Shared emission map' if row==0 else 'Shared LOS rotation',
                            xlabel='East [arcsec]',ylabel='North [arcsec]')
            axes[row,3].plot(0,0,'w+',ms=10);axes[row,3].plot(*fit['center_east_north_arcsec'],'cx')
            fig.colorbar(im,ax=axes[row,3],shrink=.65,label='Relative brightness' if row==0 else 'km/s')
        fig.suptitle(f'ID{self.source_id}: {LABELS[mode]}\n'
            f'Vrot(0.5″)={fit["vrot_0p5arcsec_kms"]:.1f}, σ={fit["intrinsic_sigma_kms"]:.1f} km/s; '
            f'optimizer check {"passed" if fit["stationarity_passed"] else "FAILED"}; conditional fit, no intervals',fontsize=12)
        return fig

    def morphology_figure(self, source, mode='positive', start=None):
        """WCS overlay without moving the observed stars onto the fitted gas."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import AsinhNorm
        from matplotlib.lines import Line2D
        from astropy.coordinates import SkyCoord
        from astropy.io.fits import Header
        from astropy.wcs import WCS
        import astropy.units as u
        if source.source_id != self.source_id:
            raise ValueError('Imaging and kinematics belong to different sources.')
        inputs=source.imaging_directory()
        if read_json(inputs/'input_provenance.json')['source_id'] != self.source_id:
            raise ValueError('Saved imaging belongs to another source.')
        fit,s=self.candidate(mode,start)
        side=int(np.sqrt(s['source_flux'].size)); weights=np.ones(side); weights[[0,-1]]=.5
        brightness=s['source_flux'].reshape(side,side)/np.outer(weights,weights)
        brightness/=brightness.max()
        with np.load(inputs/'cutout.npz') as a:
            data=a['image']; wcs=WCS(Header.fromstring(str(a['wcs_header'])),fix=False).celestial
            pixel_scale=float(a['pixel_scale_arcsec'])
        summary=self.metadata['summary']
        origin=SkyCoord(summary['ra'],summary['dec'],unit='deg')
        east,north=np.meshgrid(np.linspace(-.9,.9,side),np.linspace(-.9,.9,side))
        xx,yy=wcs.world_to_pixel(origin.spherical_offsets_by(east*u.arcsec,north*u.arcsec))
        center=origin.spherical_offsets_by(*[v*u.arcsec for v in fit['center_east_north_arcsec']])
        cx,cy=wcs.world_to_pixel(origin);lo,hi=np.nanquantile(data,[.05,.995])
        fig,ax=plt.subplots(figsize=(6,5.5),layout='constrained')
        ax.imshow(data,origin='lower',cmap='gist_heat',
                  norm=AsinhNorm(linear_width=max((hi-lo)/10,1e-12),vmin=lo,vmax=hi))
        ax.contour(xx,yy,brightness,levels=[.1,.25,.5,.75,.9],colors='cyan',linewidths=.9)
        ax.plot(cx,cy,'w+',ms=12,mew=2,label='Catalog position')
        ax.plot(*wcs.world_to_pixel(center),'cx',ms=9,mew=2,label='Kinematic center')
        handles,_=ax.get_legend_handles_labels()
        handles.append(Line2D([0],[0],color='cyan',label='Inferred line emission'))
        ax.legend(handles=handles,fontsize=8,loc='upper left')
        ax.set(xlim=(cx-1.1/pixel_scale,cx+1.1/pixel_scale),ylim=(cy-1.1/pixel_scale,cy+1.1/pixel_scale),
               xlabel='F150W cutout x [pixel]',ylabel='F150W cutout y [pixel]',
               title=f'ID{self.source_id}: F150W + fitted gas contours\n'
                     f'{fit["start"]}: Vrot(0.5″) = {fit["vrot_0p5arcsec_kms"]:.1f} km/s')
        return fig

    def decomposition_figure(self, mode='positive', start=None, exposures=None):
        import matplotlib.pyplot as plt
        fit,s=self.candidate(mode,start)
        exposures=self.pair() if exposures is None else tuple(exposures)
        if not exposures or any(i<0 or i>=len(self.metadata['exposures']) for i in exposures):
            raise ValueError('Choose valid zero-based exposure indices.')
        fig,axes=plt.subplots(len(exposures),4,figsize=(15,3.5*len(exposures)),squeeze=False,layout='constrained')
        for row,e in enumerate(exposures):
            err=self.pixels['error'][e]
            values=(self.pixels['science'][e]/err,s['line'][e]/err,
                    (s['continuum'][e]+s['background'][e])/err,s['residual'][e])
            for col,(v,label) in enumerate(zip(values,('Observed / ERR','Line / ERR','Continuum + background / ERR','Residual / ERR'))):
                im=axes[row,col].imshow(np.where(self.pixels['valid'][e],v,np.nan),origin='lower',
                    cmap=plt.get_cmap('RdBu_r').with_extremes(bad='0.65'),vmin=-5,vmax=5,interpolation='none')
                axes[row,col].set(title=f'{self.metadata["exposures"][e]["pupil"]} exposure {e+1}: {label}',xlabel='Cutout x')
                fig.colorbar(im,ax=axes[row,col],shrink=.65,extend='both')
        fig.suptitle(f'ID{self.source_id}: {LABELS[mode]}; separate components, clipped at ±5 ERR')
        return fig

    def show(self, mode='positive', save=False):
        import matplotlib.pyplot as plt
        from IPython.display import display,Markdown
        best,s=self.candidate(mode)
        display(Markdown(f'**ID{self.source_id}: {LABELS[mode]}.** '
            f'{len(self.fits)} saved starts; run {"complete" if self.report["completed"] else "partial"}. '
            '**These are optimization results, not MCMC measurements.** '
            'Stationarity checks local numerical convergence; residuals and nuisance assumptions still need inspection.'))
        display(self.table().round(4))
        pupils=[e['pupil'] for e in self.metadata['exposures']]
        cs=[i for i,p in enumerate(pupils) if p=='C']
        examples=tuple(dict.fromkeys([pupils.index('R'),cs[0],cs[-1]]))
        figures=[('fit',self.fit_figure(mode)),('components',self.decomposition_figure(mode,exposures=examples))]
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        for m in self.report['settings']['modes']:
            fs=[f for f in self.fits if f['mode']==m]
            if not fs:continue
            for f in fs:
                axes[0].scatter(f['vrot_0p5arcsec_kms'],2*(f['objective']-self.best(m)['objective']),
                    color={'positive':'C0','coupled':'C1','legacy':'C2'}[m],marker='o' if f['stationarity_passed'] else 'x')
            axes[0].plot([],[],'.',color={'positive':'C0','coupled':'C1','legacy':'C2'}[m],label=LABELS[m])
        axes[0].set(xlabel='Refitted Vrot(0.5″) [km/s]',ylabel='Fit score above best within each model',
                    yscale='symlog',title='Different starts; × fails optimizer check')
        axes[0].legend(fontsize=8)
        axes[1].plot(np.arange(1,len(s['gains'])+1),s['gains'],'o')
        sigma=self.report['settings']['calibration_sigma']
        if mode != 'legacy':
            axes[1].axhspan(np.exp(-sigma),np.exp(sigma),alpha=.15)
        axes[1].axhline(1,color='gray',lw=1)
        pupils=[e['pupil'] for e in self.metadata['exposures']]
        axes[1].set_xticks(np.arange(1,len(pupils)+1),[f'{i+1}\n{p}' for i,p in enumerate(pupils)],fontsize=8)
        axes[1].set(xlabel='Exposure number and grism',
                    ylabel='Relative fitted flux' if mode=='legacy' else 'Relative calibration gain',
                    title='Independent fluxes / response, normalized by median' if mode=='legacy'
                    else f'Prior log-gain SD {sigma:g}; provisional assumption')
        figures.append(('starts_gains',fig))
        for label,fig in figures:
            if save:
                out=self.path/'figures';out.mkdir(exist_ok=True)
                fig.savefig(out/f'{mode}_{label}.png',dpi=140)
                fig.savefig(out/f'{mode}_{label}.pdf')
            display(fig);plt.close(fig)
