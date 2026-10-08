'''The velocity model. Pure torch, no data, no calibration.'''
import numpy as np
import pytest
import torch

from dingo import kinematics as K

T = torch.tensor


def _grid(n=81):
    y, x = torch.meshgrid(torch.arange(float(n)), torch.arange(float(n)), indexing='ij')
    return x, y


BASE = dict(R_v=T(4.686), x0_v=T(42.3), y0_v=T(40.0), theta_v=T(2.36))


@pytest.mark.parametrize('V,inc', [(618.4, 0.456), (223.3, 1.0906), (1000.0, 0.12), (300.0, 1.4)])
def test_vsini_reparameterisation_is_exact(V, inc):
    '''V_sini must give a bit-identical field to V_rot*sin(inc).

    The whole point of the reparameterisation is that it changes which coordinate
    the sampler walks in and never the model. If this drifts, chains fitted in one
    parameterisation stop being comparable with the other.
    '''
    x, y = _grid()
    kw = dict(BASE, inc_v=T(inc))
    a = K.arctangent_disk_velocity_model(x, y, V_rot=T(V), **kw)
    b = K.arctangent_disk_velocity_model(x, y, V_sini=T(V*np.sin(inc)), **kw)
    assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max().item()


def test_amplitude_sign_flip_equals_pi_rotation():
    '''(V, theta) and (-V, theta+pi) are the same velocity field.

    This is the mirror mode the one-signed prior exists to kill, and the reason
    prepare_mcmc reflects a negative amplitude rather than refusing to sample.
    '''
    x, y = _grid()
    kw = dict(x0_v=T(42.3), y0_v=T(40.0), R_v=T(4.686), inc_v=T(0.9))
    a = K.arctangent_disk_velocity_model(x, y, V_rot=T(250.0), theta_v=T(1.1), **kw)
    b = K.arctangent_disk_velocity_model(x, y, V_rot=T(-250.0), theta_v=T(1.1 + np.pi), **kw)
    assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max().item()


def test_field_is_antisymmetric_about_the_centre():
    '''A rotating disk approaches on one side and recedes on the other by the same
    amount. A sign error anywhere in the projection breaks this and almost nothing
    else would notice.'''
    n = 81
    x, y = _grid(n)
    c = (n - 1)/2
    vz = K.arctangent_disk_velocity_model(
        x, y, V_rot=T(300.0), R_v=T(4.0), x0_v=T(c), y0_v=T(c),
        theta_v=T(0.7), inc_v=T(1.0))
    assert torch.allclose(vz, -torch.flip(vz, dims=(0, 1)), atol=1e-3)


def test_zero_amplitude_gives_no_rotation():
    x, y = _grid()
    vz = K.arctangent_disk_velocity_model(x, y, V_rot=T(0.0), inc_v=T(0.9), **BASE)
    assert float(vz.abs().max()) == 0.0


def test_face_on_disk_has_no_line_of_sight_velocity():
    '''inc = 0 means the rotation is entirely in the plane of the sky.'''
    x, y = _grid()
    vz = K.arctangent_disk_velocity_model(x, y, V_rot=T(500.0), inc_v=T(0.0), **BASE)
    assert float(vz.abs().max()) == pytest.approx(0.0, abs=1e-5)


def test_v_sys_default_is_bit_identical():
    '''The entire backwards-compatibility guarantee for v_sys.

    torch.equal, not allclose: `vz + 0.0` must be the same float, so that
    registering the parameter in every config cannot move a single existing
    result.
    '''
    x, y = _grid()
    kw = dict(BASE, inc_v=T(0.9), V_rot=T(250.0))
    assert torch.equal(K.arctangent_disk_velocity_model(x, y, **kw),
                       K.arctangent_disk_velocity_model(x, y, v_sys=0.0, **kw))
    assert torch.equal(K.arctangent_disk_velocity_model(x, y, **kw),
                       K.arctangent_disk_velocity_model(x, y, v_sys=T(0.0), **kw))


@pytest.mark.parametrize('dv', [-250.0, -1.0, 37.5, 500.0])
def test_v_sys_offsets_the_whole_field_by_exactly_itself(dv):
    x, y = _grid()
    kw = dict(BASE, inc_v=T(0.9), V_rot=T(250.0))
    base = K.arctangent_disk_velocity_model(x, y, **kw)
    shifted = K.arctangent_disk_velocity_model(x, y, v_sys=T(dv), **kw)
    assert torch.allclose(shifted - base, torch.full_like(base, dv), atol=1e-3)


def test_v_sys_is_separable_from_a_pointing_error_only_when_C_is_frozen():
    '''The measurement behind the guard in _setup_data.

    A systemic shift moves R along x and C along y; a registration offset moves
    one grism along one axis. With image.C.dx/dy frozen the free offsets span
    {R_x, R_y} and v_sys sits 45 deg out of that plane. With all four free it lies
    exactly inside it -- an exactly flat direction, not a parameter.
    '''
    import numpy as np
    d = np.array([-1.3974, 0.0032, -0.0023, -1.3983])    # measured, 100 km/s at 4.11 um
    e = np.eye(4)

    def angle(cols):
        A = np.column_stack(cols)
        proj = A @ np.linalg.lstsq(A, d, rcond=None)[0]
        return np.degrees(np.arccos(np.clip(np.linalg.norm(proj)/np.linalg.norm(d), 0, 1)))

    assert angle([e[0], e[1]]) == pytest.approx(45.0, abs=1.0)        # C frozen
    assert angle([e[0], e[1], e[2], e[3]]) == pytest.approx(0.0, abs=0.1)   # all free
