#!/usr/bin/env python
'''Is a systemic velocity separable from a pointing error in R+C data?

v_sys is only worth fitting if the data can tell it apart from a registration
error. The test is geometric and exact, so it is a measurement rather than an
argument.

Work in the 4-vector of model displacements (R_x, R_y, C_x, C_y). A systemic
shift dv moves each grism along ITS OWN dispersion axis -- R along x, C along y
-- so it points along roughly (-a, 0, 0, -a). A registration offset moves one
grism in one direction, i.e. along a coordinate axis. The question is whether
v_sys has any component OUTSIDE the subspace spanned by the offsets the config
actually leaves free; if not, the fit gains a flat direction instead of a
parameter.

Reported as the angle between v_sys and its projection onto that subspace:
0 deg means exactly degenerate, 90 deg means fully independent.
'''
import os
import sys

os.environ.setdefault('GRISM_CAL_DIR', '/home/aakavan/kinematics/grism_cal')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from dingo import grism, utils

C_KMS = 299792.458


def displacement_vectors(lam0, module='A', filt='F444W', dv=100.0):
    '''(v_sys direction, offset basis) as 4-vectors over (R_x, R_y, C_x, C_y).'''
    z = torch.zeros(1)
    d_vsys = np.zeros(4)
    for k, pupil in enumerate(('R', 'C')):
        _, sp, dp = grism.load_nircam_wfss_model(pupil, module, filt)
        fwd = utils.get_grism_model_torch(sp, dp, pupil, 1024, 1024,
                                          direction='forward', device=None)
        x0, y0 = fwd(z, z, torch.tensor([lam0]))
        x1, y1 = fwd(z, z, torch.tensor([lam0*(1 + dv/C_KMS)]))
        d_vsys[2*k] = float(x1 - x0)
        d_vsys[2*k + 1] = float(y1 - y0)
    # A registration offset translates that grism's model directly, so each free
    # offset is a unit vector in its own coordinate.
    basis = {'image.R.dx': np.array([1., 0, 0, 0]), 'image.R.dy': np.array([0, 1., 0, 0]),
             'image.C.dx': np.array([0, 0, 1., 0]), 'image.C.dy': np.array([0, 0, 0, 1.])}
    return d_vsys, basis


def angle_to_subspace(v, cols):
    '''Angle in degrees between v and its projection onto span(cols).'''
    if not cols:
        return 90.0
    A = np.column_stack(cols)
    proj = A @ np.linalg.lstsq(A, v, rcond=None)[0]
    nv, npj = np.linalg.norm(v), np.linalg.norm(proj)
    if nv == 0:
        return float('nan')
    return float(np.degrees(np.arccos(np.clip(npj/nv, 0.0, 1.0))))


def main():
    lam0 = 1.875*(1 + 1.1906)          # ID21636 Pa-alpha, observed
    dv = 100.0
    d, basis = displacement_vectors(lam0, dv=dv)
    print(f'lambda_obs = {lam0:.4f} um,  dv = {dv:g} km/s\n')
    print('model displacement for a systemic shift, in pixels:')
    for n, val in zip(('R_x', 'R_y', 'C_x', 'C_y'), d):
        print(f'    {n}  {val:+.4f}')
    scale = dv/max(abs(d[0]), abs(d[3]))
    print(f'\n  velocity scale: 1 px = {scale:.1f} km/s')
    print(f'  cross-term: R leaks {abs(d[1]/d[0]):.4%} into y, C leaks {abs(d[2]/d[3]):.4%} into x\n')

    cases = [
        ('C frozen (every current config)', ['image.R.dx', 'image.R.dy']),
        ('all four offsets free (ID6170, ID21619)',
         ['image.R.dx', 'image.R.dy', 'image.C.dx', 'image.C.dy']),
        ('R only, C frozen, R_y frozen too', ['image.R.dx']),
    ]
    print(f'{"free offsets":45s} {"angle to v_sys":>15s}   verdict')
    for label, names in cases:
        ang = angle_to_subspace(d, [basis[n] for n in names])
        verdict = ('DEGENERATE -- flat direction' if ang < 5 else
                   'separable' if ang > 20 else 'weak')
        print(f'{label:45s} {ang:12.1f} deg   {verdict}')

    print('\nSo v_sys is a measurement only because image.C.dx/dy are frozen. The')
    print('identifiability rests entirely on the C registration being right: an')
    print(f'error of d pixels there maps straight into {scale:.0f}*d km/s of v_sys.')


if __name__ == '__main__':
    sys.exit(main())
