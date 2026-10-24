"""CPU-only tables and visual review of real multistart source comparisons."""
from pathlib import Path
import argparse
import numpy as np

from dingo.real_workflow import RealSource, read_json, write_json
from dingo.real_coupled import CoupledResults, LABELS


def run_root(source):
    if source.source_id in (3700,6170):
        return source.directory/'coupled_repair_v3'
    preferred=source.directory/'coupled_survey_v1'
    broad=source.directory/'coupled_survey_broad_v1'
    report=preferred/'positive/coupled.json'
    if report.exists() and read_json(report).get('completed'):
        return preferred
    return broad if (broad/'positive/coupled.json').exists() else preferred


def prior_kind(report):
    prior=report['inclination_prior']
    if 'kind' in prior:return prior['kind']
    if prior.get('sources') and 'stellar_inclination_quantiles_deg' in prior:return 'imaging'
    raise ValueError('Unknown inclination prior metadata.')


def summarize(result, mode):
    best,a=result.candidate(mode)
    pupils=np.array([e['pupil'] for e in result.metadata['exposures']])
    valid=result.pixels['valid']
    side=int(np.sqrt(a['source_flux'].size))
    xx,yy=np.meshgrid(np.linspace(-.9,.9,side),np.linspace(-.9,.9,side))
    flux=a['source_flux'].reshape(side,side)
    fs=[f for f in result.fits if f['mode']==mode]
    near=[f for f in fs if 2*(f['objective']-best['objective'])<10]
    boundaries=[]
    if min(abs(best['turnover_arcsec']-v) for v in (.02,2.))<1e-4:boundaries.append('turnover')
    if min(abs(best['inclination_deg']-v) for v in (5.,85.))<.01:boundaries.append('inclination')
    if any(abs(v)>=.2999 for v in best['center_east_north_arcsec']):boundaries.append('center')
    if min(abs(best['vrot_0p5arcsec_kms']-v) for v in (0.,1000.))<.01:boundaries.append('velocity')
    return dict(source_id=result.source_id,mode=mode,completed=result.report['completed'],
        prior=prior_kind(result.report),R=int(sum(pupils=='R')),C=int(sum(pupils=='C')),
        n_valid=int(valid.sum()),newly_masked=result.report.get('detector_mask',{}).get('newly_excluded',0),
        V_kms=best['vrot_0p5arcsec_kms'],V_sini_kms=best['vproj_0p5arcsec_kms'],
        sigma_kms=best['intrinsic_sigma_kms'],inclination_deg=best['inclination_deg'],
        rturn_arcsec=best['turnover_arcsec'],x0_arcsec=best['center_east_north_arcsec'][0],
        y0_arcsec=best['center_east_north_arcsec'][1],systemic_kms=best['systemic_offset_kms'],
        stationary=sum(f['stationarity_passed'] for f in fs),starts=len(fs),
        near_best_starts=len(near),near_best_V_min=min(f['vrot_0p5arcsec_kms'] for f in near),
        near_best_V_max=max(f['vrot_0p5arcsec_kms'] for f in near),
        boundaries=', '.join(boundaries),gain_min=best['gain_min_max'][0],gain_max=best['gain_min_max'][1],
        residual_rms_R=float(np.sqrt(best['chi_square_R']/valid[pupils=='R'].sum())),
        residual_rms_C=float(np.sqrt(best['chi_square_C']/valid[pupils=='C'].sum())),
        edge_flux_fraction=float(flux[(abs(xx)>=.8)|(abs(yy)>=.8)].sum()/flux.sum()),
        path=str(result.path),best_start=best['start'])


def write_report(source_ids,output,figures=True):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    rows=[];starts=[]
    for sid in source_ids:
        source=RealSource(sid)
        for mode in ('positive','coupled'):
            path=run_root(source)/mode
            if not (path/'coupled.json').exists():continue
            result=CoupledResults(path,sid)
            if not result.fits:continue
            rows.append(summarize(result,mode))
            table=result.table();table.insert(0,'source_id',sid);starts.append(table)
            if figures:
                out=path/'figures';out.mkdir(exist_ok=True)
                pupils=[e['pupil'] for e in result.metadata['exposures']]
                cs=[i for i,p in enumerate(pupils) if p=='C']
                exposures=tuple(dict.fromkeys([pupils.index('R'),cs[0],cs[-1]]))
                for name,fig in [('best_fit',result.fit_figure(mode)),
                    ('best_components',result.decomposition_figure(mode,exposures=exposures)),
                    ('best_morphology',result.morphology_figure(source,mode))]:
                    fig.savefig(out/f'{name}.png',dpi=125);fig.savefig(out/f'{name}.pdf');plt.close(fig)
            print(f'ID{sid} {mode}: V={rows[-1]["V_kms"]:.1f}, sigma={rows[-1]["sigma_kms"]:.1f}, boundaries={rows[-1]["boundaries"]}',flush=True)
    output=Path(output);output.mkdir(exist_ok=True,parents=True)
    write_json(output/'summary.json',dict(sources=rows,
        interpretation='Conditional optimized values, not MCMC measurements. Near-best ranges use 2 delta objective <10 among stored starts, not confidence intervals.'))
    pd.DataFrame(rows).to_csv(output/'summary.csv',index=False)
    if starts:pd.concat(starts,ignore_index=True).to_csv(output/'all_starts.csv',index=False)
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-ids',type=int,nargs='+',default=[8910,15665,16887,21619,21636,3700,6170])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--no-figures',action='store_true')
    a=p.parse_args();write_report(a.source_ids,a.output,not a.no_figures)
