"""Explicit detector-coordinate exclusions, separate from immutable FITS/DQ."""
from pathlib import Path

from .real_workflow import read_json, sha256


def apply_detector_mask(target, manifest_path):
    """Apply a predeclared, exposure-scoped mask; never select on fit residuals."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = read_json(manifest_path)
    if manifest.get('format') != 'native_detector_mask_v1' or manifest.get('coordinate_origin') != 0:
        raise ValueError('Require a zero-based detector-mask manifest.')
    records = []
    for entry in manifest['pixels']:
        if (entry['module'] not in ('A','B') or any(type(entry[k]) is not int or entry[k]<0 for k in ('x','y'))
                or not entry.get('exposure_basenames') or not entry.get('reason')):
            raise ValueError('Each detector pixel needs coordinates, exposure scope and independent evidence.')
        for i,e in enumerate(target.exposures):
            if e.module != entry['module'] or e.path.name not in entry['exposure_basenames']:
                continue
            x,y=entry['x']-e.origin_xy[0],entry['y']-e.origin_xy[1]
            if 0<=x<e.valid.shape[1] and 0<=y<e.valid.shape[0]:
                was_valid=bool(e.valid[y,x]);e.valid[y,x]=False
                records.append(dict(exposure=i+1,pupil=e.pupil,path=str(e.path),
                    detector_x=entry['x'],detector_y=entry['y'],cutout_x=x,cutout_y=y,
                    newly_excluded=was_valid,reason=entry['reason']))
                e.metadata['valid_pixels']=int(e.valid.sum())
                e.metadata.setdefault('detector_exclusions',[]).append(records[-1])
    return dict(manifest=str(manifest_path),sha256=sha256(manifest_path),records=records,
                newly_excluded=sum(r['newly_excluded'] for r in records))
