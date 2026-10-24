"""Inspect a specified detector pixel in all associated exposures, without a fit."""
import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

from dingo.real_workflow import read_json,write_json,sha256


def finite(value):
    if isinstance(value,dict):return {k:finite(v) for k,v in value.items()}
    if isinstance(value,list):return [finite(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def run(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with np.load(args.native_data) as a:
        metadata=json.loads(str(a['metadata_json']))
    args.output.mkdir(parents=True,exist_ok=True)
    records=[]
    for i,e in enumerate(metadata['exposures']):
        if e['module'] != args.module:continue
        with fits.open(e['path'],memmap=False) as h:
            shape=h['SCI'].shape
            if not 2<=args.x<shape[1]-2 or not 2<=args.y<shape[0]-2:
                raise ValueError('Select a detector pixel with a complete 5x5 neighborhood.')
            arrays={k:np.asarray(h[k].section[args.y-2:args.y+3,args.x-2:args.x+3]) for k in ('SCI','ERR','DQ')}
        s,err,dq=(arrays[k] for k in ('SCI','ERR','DQ'))
        neighbors=np.isfinite(s)&np.isfinite(err)&(err>0)&((dq&3)==0);neighbors[2,2]=False
        records.append(dict(exposure=i+1,path=e['path'],pupil=e['pupil'],module=e['module'],
            science=float(s[2,2]),error=float(err[2,2]),dq=int(dq[2,2]),
            snr=float(s[2,2]/err[2,2]),neighbor_median=float(np.median(s[neighbors])),
            science_patch=s.tolist(),error_patch=err.tolist(),dq_patch=dq.tolist()))
    report=finite(dict(module=args.module,x_zero_based=args.x,y_zero_based=args.y,
        native_data_sha256=sha256(args.native_data),records=records,
        evidence='Same absolute detector coordinate checked in all associated native exposures, including those where the target is elsewhere. No model residual selection.'))
    write_json(args.output/'audit.json',report)
    fig,axes=plt.subplots(3,6,figsize=(14,7),layout='constrained')
    for ax,r in zip(axes.ravel(),records):
        s=np.array(r['science_patch']);err=np.array(r['error_patch'])
        im=ax.imshow(s/err,origin='lower',cmap='RdBu_r',vmin=-10,vmax=10,interpolation='none')
        ax.set(title=f'{r["pupil"]} exposure {r["exposure"]}\ncenter = {r["snr"]:.0f} ERR',xticks=[],yticks=[])
        ax.plot(2,2,'o',ms=17,mfc='none',mec='lime')
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.65,label='SCI / ERR (clipped at ±10)',extend='both')
    fig.suptitle(f'NRC{args.module}LONG detector (x,y)=({args.x},{args.y}), zero-based; 5×5 neighborhoods')
    fig.savefig(args.output/'detector_pixel.png',dpi=150);plt.close(fig)
    print('Independent pixel audit:',args.output/'audit.json',flush=True)
    for r in records:print(r['exposure'],r['pupil'],round(r['snr'],2),'DQ',r['dq'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--x',type=int,required=True);p.add_argument('--y',type=int,required=True)
    p.add_argument('--module',choices=['A','B'],required=True)
    run(p.parse_args())
