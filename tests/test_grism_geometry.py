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


@pytest.mark.parametrize('filt', ['F277W', 'F322W2', 'F335M', 'F356W',
                                  'F360M', 'F410M', 'F444W', 'F480M'])
@pytest.mark.parametrize('pupil', ['R', 'C'])
def test_every_advertised_filter_loads(grism_cal, filt, pupil):
    '''WRANGE used to be set for three filters while the tracing map covered eight,
    so five of them raised UnboundLocalError at the return rather than a message.
    That blocked the H-alpha sources, which sit in the short-wavelength filters.'''
    from dingo import grism
    wr, sp, dp = grism.load_nircam_wfss_model(pupil, 'A', filt)
    assert wr[0] < wr[1]
    assert len(sp) == 30 and len(dp) == 16


def test_unsupported_filter_raises_a_named_error(grism_cal):
    from dingo import grism
    with pytest.raises(NotImplementedError, match='F200W'):
        grism.load_nircam_wfss_model('R', 'A', 'F200W')


def test_kms_per_pixel_matches_the_forward_model_directly(grism_cal, models):
    '''The resolution floor must come from the trace, not a nominal dispersion.

    ID16887 fits V*sin(i) = 8.4 km/s and the null test calls it 5.97 sigma; one
    pixel is ~67 km/s there, so the gate that separates a measurement from a
    sub-pixel residual depends on this number being right.
    '''
    from dingo import utils
    dv, zero = 100.0, torch.zeros(1)
    moved = []
    for pupil, axis in (('R', 0), ('C', 1)):
        p0 = models[pupil](zero, zero, torch.tensor([LAM0]))
        p1 = models[pupil](zero, zero, torch.tensor([LAM0*(1 + dv/C_KMS)]))
        moved.append(abs(float(p1[axis] - p0[axis])))
    assert utils.kms_per_pixel(LAM0, 'F444W', 'A') == pytest.approx(dv/min(moved), rel=1e-9)


def test_kms_per_pixel_is_the_coarser_of_the_two_directions(grism_cal):
    '''R and C disperse along perpendicular axes and need not be equally
    sensitive. The floor is the larger km/s per pixel, so a source is required
    to be resolved in BOTH directions rather than only the finer one.'''
    from dingo import utils
    dv, zero = 100.0, torch.zeros(1)
    from dingo import grism
    per_dir = []
    for pupil, axis in (('R', 0), ('C', 1)):
        _, sp, dp = grism.load_nircam_wfss_model(pupil, 'A', 'F444W')
        fwd = utils.get_grism_model_torch(sp, dp, pupil, 1024, 1024,
                                          direction='forward', device=None)
        p0 = fwd(zero, zero, torch.tensor([LAM0]))
        p1 = fwd(zero, zero, torch.tensor([LAM0*(1 + dv/C_KMS)]))
        per_dir.append(dv/abs(float(p1[axis] - p0[axis])))
    assert utils.kms_per_pixel(LAM0, 'F444W', 'A') == pytest.approx(max(per_dir), rel=1e-9)


def test_kms_per_pixel_is_physically_sane_across_the_sample(grism_cal):
    '''Pa-alpha at z = 1.1-1.4 in F444W: one NIRCam pixel is tens of km/s. A
    value far outside that means the wrong calibration table was loaded.'''
    from dingo import utils
    for z in (1.1231, 1.1613, 1.1906, 1.3456):
        scale = utils.kms_per_pixel(1.875*(1 + z), 'F444W', 'A')
        assert 40.0 < scale < 120.0, f'z={z} gave {scale} km/s per pixel'
