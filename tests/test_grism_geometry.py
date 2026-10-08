'''Grism geometry: needs the calibration tables, but no staged data.'''
import numpy as np
import pytest
import torch

C_KMS = 299792.458


@pytest.fixture(scope='module')
def models(grism_cal):
    from dingo import grism, utils
    out = {}
    for pupil in ('R', 'C'):
        _, sp, dp = grism.load_nircam_wfss_model(pupil, 'A', 'F444W')
        out[pupil] = utils.get_grism_model_torch(sp, dp, pupil, 1024, 1024,
                                                 direction='forward', device=None)
    return out


LAM0 = 1.875*(1 + 1.1906)      # ID21636 Pa-alpha, observed


def _shift(model, dv):
    z = torch.zeros(1)
    x0, y0 = model(z, z, torch.tensor([LAM0]))
    x1, y1 = model(z, z, torch.tensor([LAM0*(1 + dv/C_KMS)]))
    return float(x1 - x0), float(y1 - y0)


def test_R_disperses_along_x_and_C_along_y(models):
    '''The identifiability argument for the whole method, as an assertion.

    R+C breaks the morphology-kinematics degeneracy only because the two grisms
    disperse along perpendicular axes. If a pupil/module mapping were ever
    mis-wired this is what would catch it, and nothing else would.
    '''
    dxR, dyR = _shift(models['R'], 100.0)
    dxC, dyC = _shift(models['C'], 100.0)
    assert abs(dxR) > 100*abs(dyR), (dxR, dyR)
    assert abs(dyC) > 100*abs(dxC), (dxC, dyC)
    # and the two directions carry the same velocity scale
    assert abs(dxR) == pytest.approx(abs(dyC), rel=0.02)


def test_velocity_to_pixel_scale_is_about_70_km_s(models):
    '''Fixes the conversion that sets what a registration error costs: an error of
    one pixel in the C-grism registration maps straight into ~70 km/s of systemic
    velocity, which is what bounds any future v_sys measurement.'''
    dxR, _ = _shift(models['R'], 100.0)
    km_s_per_px = 100.0/abs(dxR)
    assert 60.0 < km_s_per_px < 85.0, km_s_per_px


def test_doppler_shift_is_linear_and_signed(models):
    lo, hi = _shift(models['R'], -100.0)[0], _shift(models['R'], +100.0)[0]
    assert lo*hi < 0, 'opposite velocities must shift opposite ways'
    assert abs(lo) == pytest.approx(abs(hi), rel=1e-3)
