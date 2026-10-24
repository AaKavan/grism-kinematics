"""Prepare real pixels, a checked inclination prior and unfitted multistart seeds.

Uses the existing engine checkpoint schema for reproducible reconstruction.
The preparation is explicitly not a fitted/completed velocity grid.
"""
import argparse
import fcntl
from pathlib import Path

from diagnostics.native_source_grid import build_engine, CODE as BASE_CODE
from dingo.real_workflow import RealSource, REPO, read_json, write_json, sha256
import numpy as np
import torch

CODE = list(dict.fromkeys(BASE_CODE + ['diagnostics/native_source_prepare.py']))


def prepare(args):
    source = RealSource(args.source_id)
    source.require_data()
    source.prepare_imaging()
    if args.inclination == 'imaging' and not all(x['ready'] for x in source.imaging_status()):
        raise ValueError('This source has no accepted imaging prior. Use an explicitly labeled isotropic preview.')
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    settings = dict(config=str(source.config), bundle=str(source.bundle),
        imaging=str(source.imaging_directory()), inclination=args.inclination,
        inclination_scatter=10., response_scale=1., correlation=.1, map_sigma=2.,
        size=81, controls=21, quadrature=61, center='catalog')
    hashes = {p: sha256(REPO/p) for p in CODE}
    with (out/'.workflow.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path=out/'grid.json'
        if path.exists():
            old=read_json(path)
            if old['settings'] != settings or old['code_sha256'] != hashes or old['source_id'] != source.source_id:
                raise ValueError('Preparation changed; use a fresh directory.')
            if old.get('preparation_completed'):
                build_engine(old)  # Verify current real pixels, response and prior.
                for seed in old['initial_fits']:
                    if sha256(out/seed['artifact']) != seed['sha256']:
                        raise ValueError('Initialization artifact changed.')
                print('Prepared inputs already verified:', out, flush=True)
                return
        report = dict(format='native_source_grid_v1', source_id=source.source_id,
            real_observations=True, completed=False, preparation_completed=False,
            fits=[], best_by_velocity=[], settings=settings, code_sha256=hashes,
            interpretation='Input preparation only. Initial vectors are unfitted; no velocity grid or inference.')
        target,engine,_,_,_,_=build_engine(report)
        target.save(out/'native_data.npz')
        np.savez_compressed(out/'inclination_prior.npz', angle_rad=np.asarray(engine.inclination_grid),
                            log_density=np.asarray(engine.inclination_logpdf))
        angles=np.asarray(engine.inclination_grid);pdf=np.exp(np.asarray(engine.inclination_logpdf))
        inclination=float(np.interp(.5,np.cumsum(pdf)/pdf.sum(),angles))
        starts=[]
        for speed,sigma,rturn in ((50.,20.,.1),(250.,60.,.3),(600.,120.,.7)):
            for angle in (0.,90.,180.,270.):
                vector=engine.initial(np.deg2rad(angle),inclination,rturn=rturn)
                vector[4]=sigma/100.
                parameters=np.r_[speed/100.,vector,3.,np.zeros(2*len(target.exposures)-1)]
                name=f'initial_{len(starts):02d}.npz'
                np.savez_compressed(out/name,source_id=source.source_id,parameters=parameters,vector=vector)
                starts.append(dict(artifact=name,sha256=sha256(out/name),
                    velocity_kms=speed,sigma_kms=sigma,theta_deg=angle,turnover_arcsec=rturn,
                    inclination_deg=float(np.rad2deg(inclination)),unfitted=True))
        for p in CODE:
            dest=out/'code'/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((REPO/p).read_bytes())
        report.update(initial_fits=starts,preparation_completed=True)
        write_json(path,report)
        print(f'ID{source.source_id}: prepared {len(target.exposures)} native exposures and {len(starts)} unfitted starts; inclination={args.inclination}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-id',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--inclination',choices=['imaging','isotropic'],required=True)
    torch.set_num_threads(4)
    prepare(p.parse_args())
