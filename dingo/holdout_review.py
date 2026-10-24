"""CPU-only notebook views of the saved joint R+C exposure prediction test."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm
import numpy as np
import pandas as pd


class HoldoutReview:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        with np.load(self.path, allow_pickle=False) as saved:
            self.arrays = {k: saved[k].copy() for k in saved.files}
        a = self.arrays
        self.report = json.loads(str(a['report_json']))
        self.covariance = json.loads(str(a['covariance_assessment_json']))
        self.labels = list(map(str, a['labels']))
        self.fits = {f['start']: f for f in self.report['fits']}
        if self.labels != [f['start'] for f in self.report['fits']]:
            raise ValueError('Fitted parameter labels differ from the embedded report.')
        if not self.report.get('completed'):
            raise ValueError('This diagnostic has not completed its prediction assessment.')
        self.radius_pc = float(a['radius_reference_pc'])
        self.pc_per_arcsec = self.radius_pc/float(a['radius_reference_arcsec'])
        self.nominal = np.array([np.asarray(self.report['exposures'][int(i)]['line_xy'])-
            np.asarray(self.report['exposures'][int(i)]['origin_xy']) for i in a['held_exposure_indices']])
        self.line_data = a['science']-a['sideband_prediction']
        selected = a['valid'] & a['core']
        self.norm = AsinhNorm(linear_width=float(np.median(a['error'][selected])),
            vmin=0., vmax=max(float(np.max(self.line_data[selected])), 1e-8))
        self.heat = plt.get_cmap('gist_heat').copy(); self.heat.set_bad('white')
        self.diverging = plt.get_cmap('RdBu_r').copy(); self.diverging.set_bad('white')

    def exposure_table(self):
        held = set(self.report['held_indices'])
        return pd.DataFrame([dict(exposure=i+1, grism=e['pupil'],
            use='WITHHELD' if i in held else 'training',
            valid_pixels=e['valid_pixels'], filename=Path(e['path']).name)
            for i,e in enumerate(self.report['exposures'])])

    def candidate_table(self):
        best = min(f['objective'] for f in self.fits.values())
        return pd.DataFrame([{'start':name, f'Vrot at {self.radius_pc:.0f} pc [km/s]':f['vrot_0p5arcsec_kms'],
            'V sin(i) [km/s]':f['vproj_0p5arcsec_kms'], 'inclination [deg]':f['inclination_deg'],
            'turnover [pc]':f['turnover_arcsec']*self.pc_per_arcsec,
            'Δ training objective':f['objective']-best,
            'optimizer stationary':f['stationarity_passed'],
            'at a bound':', '.join(f['boundary_parameters']) or '—'}
            for name,f in self.fits.items()])

    def selected_exposures(self, r=1, c=1):
        selected = []
        for pupil, ordinal in (('R',r), ('C',c)):
            indices = np.flatnonzero(self.arrays['pupils']==pupil)
            if not isinstance(ordinal, (int,np.integer)) or not 1<=ordinal<=len(indices):
                raise ValueError(f'{pupil} exposure must be an integer from 1 to {len(indices)}.')
            selected.append(int(indices[ordinal-1]))
        return selected

    @staticmethod
    def _colorbar(fig, artist, ax, units, norm=None):
        bar = fig.colorbar(artist, ax=ax, fraction=.045, pad=.02, label=units)
        if norm is not None:
            ticks = np.linspace(norm.vmin,norm.vmax,4)
            bar.set_ticks(ticks, labels=[f'{v:.3g}' for v in ticks])

    def _markers(self, ax, exposure, label=None, half_width=20):
        ax.plot(*self.nominal[exposure], 'o', markerfacecolor='none',
            markeredgecolor='white', markersize=10, markeredgewidth=1.8)
        if label is not None:
            k=self.labels.index(label)
            ax.plot(*self.arrays['projected_kinematic_centers_xy'][k,exposure],
                'x', color='cyan', markersize=9, markeredgewidth=1.8)
        cx,cy=self.nominal[exposure]
        if half_width is not None:
            ax.set_xlim(cx-half_width,cx+half_width); ax.set_ylim(cy-half_width,cy+half_width)
        ax.set_xlabel('Cutout x [pixel]'); ax.set_ylabel('Cutout y [pixel]')

    def input_figure(self, r=1, c=1):
        """Raw withheld SCI, independently estimated continuum, and their difference."""
        a=self.arrays
        fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
        rows=self.selected_exposures(r,c)
        values=a['science'][a['valid'] & a['core']]
        norm=AsinhNorm(linear_width=float(np.median(a['error'][a['valid']])),
            vmin=0.,vmax=max(float(np.max(values)),self.norm.vmax))
        for row,e in enumerate(rows):
            for col,(array,title) in enumerate(((a['science'],'Raw SCI'),
                (a['sideband_prediction'],'Sideband continuum + background'),
                (self.line_data,'SCI − continuum/background'))):
                ax=axes[row,col]
                im=ax.imshow(np.ma.masked_where(~a['valid'][e],array[e]),origin='lower',
                    cmap=self.heat,norm=norm,interpolation='nearest')
                self._markers(ax,e,half_width=None)
                cx,cy=self.nominal[e]
                if a['pupils'][e]=='R':
                    ax.axvline(cx-28,color='cyan',ls=':',lw=1);ax.axvline(cx+28,color='cyan',ls=':',lw=1)
                else:
                    ax.axhline(cy-28,color='cyan',ls=':',lw=1);ax.axhline(cy+28,color='cyan',ls=':',lw=1)
                ax.set_title(f'{a["pupils"][e]} exposure {int(a["held_exposure_indices"][e])+1}\n{title}')
                self._colorbar(fig,im,ax,'DN/s',norm)
        fig.suptitle('Real withheld exposures · dotted lines bound the excluded line window\n'
            'Continuum uses only the outer sidebands; white pixels are excluded.',fontsize=12)
        return fig

    def fit_figure(self, label='saved_lower', r=1, c=1, *, all_exposures=False,
                   half_width=20, residual_limit=5.):
        if label not in self.fits:
            raise ValueError(f'Choose one of {self.labels}')
        if residual_limit<=0 or (half_width is not None and half_width<=0):
            raise ValueError('Positive image half-width and residual limit required.')
        a=self.arrays;k=self.labels.index(label);f=self.fits[label]
        rows=list(range(len(a['pupils']))) if all_exposures else self.selected_exposures(r,c)
        fig,axes=plt.subplots(len(rows),3,figsize=(13,3.6*len(rows)),squeeze=False,layout='constrained')
        for row,e in enumerate(rows):
            for col,(values,title) in enumerate(((self.line_data[e],'Observed emission'),
                (a['predicted_line'][k,e],'Frozen line prediction'),(a['residual_ERR'][k,e],'Residual / ERR'))):
                ax=axes[row,col];masked=np.ma.masked_where(~a['valid'][e],values)
                if col<2:
                    im=ax.imshow(masked,origin='lower',cmap=self.heat,norm=self.norm,interpolation='nearest')
                else:
                    im=ax.imshow(masked,origin='lower',cmap=self.diverging,
                        vmin=-residual_limit,vmax=residual_limit,interpolation='nearest')
                self._markers(ax,e,label,half_width)
                ax.set_title(f'{a["pupils"][e]} exposure {int(a["held_exposure_indices"][e])+1} · {title}',fontsize=10)
                self._colorbar(fig,im,ax,'ERR units' if col==2 else 'DN/s',self.norm if col<2 else None)
        boundary=', '.join(f['boundary_parameters']) or 'none'
        fig.suptitle(f'{label}: Vrot({self.radius_pc:.0f} pc) = {f["vrot_0p5arcsec_kms"]:.1f} km/s; '
            f'V sin(i) = {f["vproj_0p5arcsec_kms"]:.1f}; i = {f["inclination_deg"]:.1f}°\n'
            f'Parameters at a bound: {boundary}. Cyan × fitted center; white ○ nominal position.\n'
            'Residual / ERR omits continuum uncertainty; the score comparison below includes its correlations.',fontsize=11)
        return fig

    def source_figure(self, labels=('saved_lower','saved_higher')):
        a=self.arrays;n=int(round(np.sqrt(a['source_east'].size)))
        if n*n!=a['source_east'].size:
            raise ValueError('Expected the saved square quadrature grid.')
        east=a['source_east'].reshape(n,n);north=a['source_north'].reshape(n,n)
        dx=float(east[0,1]-east[0,0]);dy=float(north[1,0]-north[0,0])
        weights=np.ones(n);weights[[0,-1]]=.5
        area=np.outer(weights,weights)*dx*dy
        brightness=a['source_flux'].reshape(-1,n,n)/area
        selected=[self.labels.index(label) for label in labels]
        velocity=np.stack([a['source_velocity'][k].reshape(n,n)-
            self.fits[self.labels[k]]['systemic_offset_kms'] for k in selected])
        visible=np.stack([brightness[k]>.01*brightness[k].max() for k in selected])
        vmax=max(float(np.max(np.abs(velocity[visible]))),1.)
        bmax=float(brightness[selected].max())
        norm=AsinhNorm(linear_width=.03*bmax,vmin=0.,vmax=bmax)
        scale=self.pc_per_arcsec
        extent=[(east.min()-dx/2)*scale,(east.max()+dx/2)*scale,
                (north.min()-dy/2)*scale,(north.max()+dy/2)*scale]
        fig,axes=plt.subplots(len(labels),2,figsize=(11,4.5*len(labels)),squeeze=False,layout='constrained')
        for row,(k,label) in enumerate(zip(selected,labels)):
            left,right=axes[row]
            im=left.imshow(brightness[k],origin='lower',extent=extent,cmap=self.heat,norm=norm)
            self._colorbar(fig,im,left,'Normalized brightness [arcsec⁻²]',norm)
            im=right.imshow(np.ma.masked_where(~visible[row],velocity[row]),origin='lower',extent=extent,
                cmap=self.diverging,vmin=-vmax,vmax=vmax)
            self._colorbar(fig,im,right,'LOS rotation [km/s]')
            left.set_title(f'{label} · gas brightness model')
            right.set_title('LOS rotation relative to systemic')
            for ax in (left,right):
                center=np.asarray(self.fits[label]['center_east_north_arcsec'])*scale
                ax.plot(*center,'x',color='cyan',markersize=9,mew=1.8)
                ax.set_xlabel('East offset [pc]');ax.set_ylabel('North offset [pc]')
        fig.suptitle('One gas map and velocity field per solution, used for both R and C\n'
            'Velocity shown where model brightness exceeds 1% of its peak; this is not an observed-extent mask.',fontsize=11)
        return fig

    def rotation_figure(self, labels=('saved_lower','saved_higher')):
        radius=np.linspace(0,self.radius_pc,300)
        fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for label,style in zip(labels,('-', '--', ':', '-.')):
            f=self.fits[label];rt=f['turnover_arcsec']*self.pc_per_arcsec
            curve=f['vrot_0p5arcsec_kms']*np.arctan(radius/rt)/np.arctan(self.radius_pc/rt)
            axes[0].plot(radius,curve,color='black',ls=style,label=label)
            axes[1].plot(radius,curve*np.sin(np.deg2rad(f['inclination_deg'])),color='black',ls=style,label=label)
        for ax,title in zip(axes,('Deprojected model rotation','Projected model amplitude')):
            ax.set(xlabel='Disk radius [pc]',ylabel='Velocity [km/s]',title=title);ax.legend();ax.grid(alpha=.15)
        fig.suptitle('Conditional model curves · no posterior error bands\n'
            'These are not independently measured radial bins; the inner turnover remains unresolved.',fontsize=11)
        return fig

    def score_table(self):
        return pd.DataFrame([{'region':region.replace('_',' '),'grism':group,
            'raw Δχ² (high − low)':value['higher_minus_lower']['raw_chi_square'],
            'correlated ΔQ (high − low)':value['higher_minus_lower']['correlated_quadratic']}
            for region,groups in self.covariance['totals'].items() for group,value in groups.items()])

    def score_figure(self, region='line_window'):
        if region not in self.covariance['totals']:
            raise ValueError(f'Choose {list(self.covariance["totals"])}')
        records=self.covariance['exposures'];delta=[]
        for e in records:
            c=e['regions'][region]['candidates']
            delta.append(c['saved_higher']['correlated_quadratic']-c['saved_lower']['correlated_quadratic'])
        fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
        bars=ax.bar(np.arange(len(records)),delta,color='white',edgecolor='black',linewidth=1.4)
        for bar,e in zip(bars,records):
            if e['pupil']=='C':bar.set_hatch('///')
        ax.axhline(0,color='black',lw=1)
        ax.set_xticks(np.arange(len(records)),[f'{e["pupil"]} exposure {e["index"]+1}' for e in records])
        ax.set_ylabel('Q(high speed) − Q(lower speed)')
        ax.set_title(f'Withheld {region.replace("_"," ")} · positive favors the lower-speed solution\n'
            'Full continuum covariance, conditional on its active coefficients; not a significance level.')
        return fig

    def notes(self):
        lowest=min(self.fits.values(),key=lambda f:f['objective'])
        diff=self.covariance['totals']['line_window']['all']['higher_minus_lower']['correlated_quadratic']
        return (f'Lowest training objective: {lowest["start"]}, Vrot = {lowest["vrot_0p5arcsec_kms"]:.2f} km/s '
            f'at {self.radius_pc:.0f} pc.\nWithheld ΔQ (higher − lower) = {diff:.3f}; positive favors lower.\n'
            'This is an optimization/prediction diagnostic, not MCMC. There are no posterior error bars.\n'
            'Inspect boundary flags and residual structure before interpreting a rotation or mass profile.\n'
            f'Results loaded from: {self.path}')
