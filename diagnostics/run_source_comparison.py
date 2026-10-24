"""Run short, source-specific real R+C comparisons with durable progress logs."""
import argparse
from pathlib import Path
import time

from dingo.real_workflow import RealSource, REPO, read_json, write_json, run_command
from dingo.real_coupled import fit_coupled


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    status=dict(completed=False,source_ids=args.source_ids,sources=[],started_unix=time.time())
    for sid in args.source_ids:
        source=RealSource(sid)
        row=dict(source_id=sid,completed=False);status['sources'].append(row)
        write_json(args.output/'run_status.json',status)
        try:
            source.require_data()
            if not args.skip_imaging:
                try:
                    source.fit_imaging(device=args.device)
                except (ValueError,RuntimeError) as exc:
                    row['imaging_error']=str(exc)
                    if not args.allow_isotropic_preview:raise
            ready=all(r['ready'] for r in source.imaging_status())
            if not ready and not args.allow_isotropic_preview and args.prior!='isotropic':
                raise ValueError('No accepted imaging prior; explicitly allow an isotropic preview or inspect imaging first.')
            kind='isotropic' if args.prior=='isotropic' else 'imaging' if ready else 'isotropic'
            row.update(inclination_prior=kind,imaging_status=source.imaging_status())
            print(f'ID{sid}: inclination={kind}; 12 starting points per model',flush=True)
            inputs=source.directory/args.input_name
            run_command(['-m','diagnostics.native_source_prepare','--source-id',str(sid),
                '--output',str(inputs),'--inclination',kind],inputs/'prepare.log',device='cpu')
            out=fit_coupled(source,inputs,run_name=args.run_name,modes=('positive','coupled'),
                detector_mask=args.detector_mask,device=args.device)
            row.update(completed=True,path=str(out))
        except Exception as exc:
            row['error']=str(exc);print(f'ID{sid}: saved failure for inspection: {exc}',flush=True)
        write_json(args.output/'run_status.json',status)
    status.update(completed=all(r['completed'] for r in status['sources']),finished_unix=time.time())
    write_json(args.output/'run_status.json',status)
    if not status['completed']:
        raise RuntimeError('Some sources need review; completed results are preserved in run_status.json.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-ids',type=int,nargs='+',default=[8910,15665,16887,21619,21636])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--detector-mask',type=Path,default=REPO/'data/detector_masks/survey_checked_pixels_2026-10-18.json')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--skip-imaging',action='store_true')
    p.add_argument('--allow-isotropic-preview',action='store_true')
    p.add_argument('--prior',choices=['auto','isotropic'],default='auto')
    p.add_argument('--input-name',default='survey_inputs_v1')
    p.add_argument('--run-name',default='coupled_survey_v1')
    run(p.parse_args())
