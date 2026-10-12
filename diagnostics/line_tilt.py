#!/usr/bin/env python
'''Screening only: how much does the Pa-alpha centroid tilt across a source?

    python diagnostics/line_tilt.py 16887 --z 1.3457

For each spatial row of the EDR 2D line map, take the flux-weighted wavelength
centroid of Pa-alpha and express it as a velocity. Reports the gradient in both
dispersion directions, alongside what the galaxy's own morphology would produce
with no rotation at all.

THE TILT IS NOT A VELOCITY. In a slitless grism, position along the dispersion
axis is degenerate with wavelength: at a given cross-dispersion row the centroid
reflects the flux-weighted SPATIAL position of the galaxy at that row. An
inclined galaxy whose major axis is tilted against the dispersion direction
produces the full signature with zero rotation. On ID16887 the morphology term
accounted for 26% of the R tilt and 38% of the C tilt.

Separating the two is what the R+C forward model is for, and nothing here
replaces it. The morphology column below is a first-order, additive
approximation computed from the broadband direct image -- the grism maps the 2D
light distribution THROUGH the velocity field rather than adding two slopes, and
Pa-alpha traces clumpy star formation that need not follow the F444W continuum.
Treat a large residual as a reason to fit the source, never as a detection.

Useful because it costs milliseconds and needs no fit: it screens which of the
95 sources with Pa-alpha in both dispersion directions are worth GPU time.
'''
import argparse
import os
import sys

import numpy as np
from astropy.io import fits

C_KMS = 299792.458
PA_REST_UM = 1.875
EDR_2D = os.environ.get('EDR_2D_SPEC_DIR',
                        os.path.expanduser('~/EDR_data/sapphires_edr_2d_spec'))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _weighted_line(x, y, w):
    '''Weighted least-squares slope and its error.'''
    A = np.vstack([x, np.ones_like(x)]).T
    Aw, bw = A*np.sqrt(w)[:, None], y*np.sqrt(w)
    coef, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
    cov = np.linalg.inv(Aw.T @ Aw)
    return coef[0], coef[1], np.sqrt(cov[0, 0])


def centroid_gradient(ID, z, direction, field='M0416', filt='F444W', half=12,
                      flux_floor=0.15):
    '''(gradient km/s/arcsec, its error, km/s per pixel, pixel scale, n rows).'''
    path = os.path.join(EDR_2D, f'spec_2d_{field}_{filt}_ID{ID}_{direction}.fits')
    with fits.open(path) as h:
        line, wht, hd = h['LINE2D'].data, h['WHT2D'].data, h['LINE2D'].header
    w1, dw, pix = hd['WAVE_1'], hd['D_WAVE'], hd['PIXSCALE']

    lam0 = PA_REST_UM*(1 + z)
    xc = int(round((lam0 - w1)/dw))
    sub, wsub = line[:, xc-half:xc+half+1], wht[:, xc-half:xc+half+1]
    lam = w1 + dw*np.arange(xc-half, xc+half+1)
    sigma = np.sqrt(np.divide(1.0, wsub, out=np.full_like(wsub, np.inf), where=wsub > 0))

    rowflux = np.nansum(sub, axis=1)
    rows = np.where(rowflux > flux_floor*np.nanmax(rowflux))[0]
    y0 = rows.mean() if len(rows) else 0.0

    offs, vels, weights = [], [], []
    for r in rows:
        f, s = sub[r], sigma[r]
        good = np.isfinite(f) & np.isfinite(s) & (s < np.inf)
        if good.sum() < 5:
            continue
        wt = np.clip(f[good], 0, None)
        if wt.sum() <= 0:
            continue
        cen = np.sum(wt*lam[good])/wt.sum()
        err = np.sqrt(np.sum(((lam[good] - cen)/wt.sum())**2 * s[good]**2))
        offs.append((r - y0)*pix)
        vels.append(C_KMS*(cen - lam0)/lam0)
        weights.append(1.0/max(C_KMS*err/lam0, 1e-6)**2)

    if len(offs) < 3:
        return None
    slope, _, err = _weighted_line(np.array(offs), np.array(vels), np.array(weights))
    return slope, err, C_KMS*dw/lam0, pix, len(offs)


def morphology_gradient(ID, axis, kms_per_px, pix, flux_floor=0.15):
    '''What the direct image alone would produce -- no velocity in it at all.'''
    path = os.path.join(REPO, 'data', f'ID{ID}', 'image.fits')
    if not os.path.exists(path):
        return None
    im = fits.getdata(path)
    im = np.clip(np.where(np.isfinite(im), im, 0.0) - np.median(im), 0, None)
    if axis == 1:
        im = im.T
    pos = np.arange(im.shape[0])
    total = im.sum(axis=0)
    cols = np.where(total > flux_floor*total.max())[0]
    cen = np.array([(im[:, j]*pos).sum()/im[:, j].sum() for j in cols if im[:, j].sum() > 0])
    rows = cols[:len(cen)].astype(float)
    if len(cen) < 3:
        return None
    slope, _, _ = _weighted_line(rows - rows.mean(), cen, np.ones_like(cen))
    return slope*kms_per_px/pix


def report(ID, z, **kw):
    print(f'ID{ID}   z = {z}')
    print(f'{"":4} {"rows":>5} {"measured":>22} {"morphology":>12} {"residual":>10}')
    for axis, direction in enumerate('RC'):
        got = centroid_gradient(ID, z, direction, **kw)
        if got is None:
            print(f'  {direction}  (too few rows with signal)')
            continue
        slope, err, kms_per_px, pix, n = got
        morph = morphology_gradient(ID, axis, kms_per_px, pix)
        m_txt = f'{morph:+12.1f}' if morph is not None else f'{"n/a":>12}'
        r_txt = f'{slope - morph:+10.1f}' if morph is not None else f'{"":>10}'
        print(f'  {direction}  {n:5d} {slope:+10.1f} +- {err:5.1f} ({abs(slope)/err:4.1f}s)'
              f'{m_txt}{r_txt}   km/s/arcsec   [1 px = {kms_per_px:.1f} km/s]')
    print('\nA residual is a reason to FIT this source, not a detection -- see the '
          'module docstring.')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('source', help='EDR ID, e.g. 16887')
    ap.add_argument('--z', type=float, required=True, help='spectroscopic redshift')
    ap.add_argument('--field', default='M0416')
    ap.add_argument('--filter', default='F444W', dest='filt')
    a = ap.parse_args(argv)
    report(str(a.source).lstrip('ID'), a.z, field=a.field, filt=a.filt)
    return 0


if __name__ == '__main__':
    sys.exit(main())
