import numpy as np
import pytest
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from dingo.native_real import (oblate_inclination, broadened_inclination_density,
    separable_response_kernel, profile_positive_amplitude)


def test_oblate_inclination_recovers_geometry_and_rejects_impossible_shapes():
    inclination=np.deg2rad([0.,30.,70.,90.])
    q0=.2
    q=np.sqrt(np.cos(inclination)**2+q0*q0*np.sin(inclination)**2)
    np.testing.assert_allclose(oblate_inclination(q,q0),inclination,atol=1e-14)
    np.testing.assert_allclose(oblate_inclination([.3,.8],0),np.arccos([.3,.8]))
    with pytest.raises(ValueError): oblate_inclination(.1,.2)


def test_external_orientation_density_is_normalized_and_broadens():
    angles=np.deg2rad([67.,70.,73.])
    widths=[]
    for scatter in (5.,10.,20.):
        x,logp=broadened_inclination_density(angles,scatter)
        p=np.exp(logp)
        assert np.trapezoid(p,x)==pytest.approx(1.,abs=1e-12)
        mean=np.trapezoid(x*p,x)
        widths.append(np.trapezoid((x-mean)**2*p,x))
    assert widths[0] < widths[1] < widths[2]


@pytest.mark.parametrize("pupil",["R","C"])
def test_response_second_moments_include_cubic_deposition(pupil):
    fraction=.6
    sigma=np.array([1.05,2.5]);cross=1.1
    k=separable_response_kernel(pupil,fraction,sigma,cross)
    y,x=np.mgrid[-10:11,-10:11]
    assert np.min(k)>=0
    assert k.sum()==pytest.approx(1.,abs=1e-14)
    v=np.array([(k*x*x).sum(),(k*y*y).sum()])+1/3
    expected=[fraction*sigma[0]**2+(1-fraction)*sigma[1]**2,cross**2]
    np.testing.assert_allclose(v,expected if pupil=="R" else expected[::-1],atol=1e-10)


def test_positive_amplitude_profiles_instead_of_fitting_negative_lines():
    rng=np.random.default_rng(17)
    t=rng.normal(size=(2,7,7));data=np.array([3.,-2.])[:,None,None]*t
    amplitude,relative=profile_positive_amplitude(jnp.asarray(t),jnp.asarray(data))
    np.testing.assert_allclose(amplitude,[3.,0.],atol=1e-13)
    zero=(data**2).sum()
    fitted=((data-np.asarray(amplitude)[:,None,None]*t)**2).sum()
    assert float(relative)==pytest.approx((zero-fitted)/2,rel=1e-13)
    # Template normalization must not alter the profiled score.
    _,scaled=profile_positive_amplitude(jnp.asarray(4*t),jnp.asarray(data))
    assert float(scaled)==pytest.approx(float(relative),rel=1e-13)


def test_effective_psf_centering_preserves_width_when_resampled():
    from diagnostics.native_imaging_prior import resample_effective_psf
    y,x=np.mgrid[:103,:103]
    p=np.exp(-.5*((x-49.4)**2+(y-60.3)**2)/3.**2)
    result,report=resample_effective_psf(p,.015,.03)
    yy,xx=np.indices(result.shape);half=(len(result)-1)/2
    assert result.sum()==pytest.approx(1.,abs=1e-14)
    assert abs((result*(xx-half)).sum())<.06
    assert abs((result*(yy-half)).sum())<.06
    assert np.sqrt((result*(xx-half)**2).sum())==pytest.approx(1.5,abs=.03)
    with pytest.raises(ValueError):resample_effective_psf(np.zeros((9,9)),.015,.03)


def test_joint_profile_gradients_and_independent_emission_orientation():
    # Keep this independent compiled model from accumulating the mock-suite's
    # executable cache in the same CPU JAX process (a native compiler crash was
    # observed with the accumulated cache; the isolated test passes).
    jax.clear_caches()
    import torch
    from dingo.native_fitting import BilinearLineMap,StripContinuumLikelihood
    from dingo.native_forward import NativeLineProjector
    from dingo.native_jax import JaxLineMap,JaxNativeProjector,JaxStripLikelihood
    from dingo.native_real import RealVelocityProfile
    grid=BilinearLineMap(3,7,half_extent=.3,dtype=torch.float64)
    coeff=np.zeros((2,49,2,7))
    coeff[:,:,0,0]=10.3+grid.east.numpy()/.06
    coeff[:,:,1,0]=10.7+grid.north.numpy()/.06
    coeff[0,:,0,1]=1000.;coeff[1,:,1,1]=-900.
    projector=NativeLineProjector(coeff,4.,(23,23),np.ones((3,3)),[1.,1.],deposition="cubic")
    y,x=np.mgrid[:23,:23]
    data=np.stack([10*np.exp(-((x-11)**2+(y-10)**2)/8)]*2)
    likelihood=StripContinuumLikelihood(data,np.ones_like(data),np.ones_like(data,dtype=bool),["R","C"])
    ig,ip=broadened_inclination_density([1.1],10.)
    model=RealVelocityProfile(JaxNativeProjector(projector),JaxLineMap(grid),JaxStripLikelihood(likelihood),ig,ip,half_extent=.3)
    p=model.initial(.7,1.11123)
    p[3]=.2;p[5]=.12;p[10]=.3
    objective=jax.jit(model.objective)
    grad=np.asarray(jax.jit(jax.grad(model.objective))(jnp.asarray(p),200.))
    assert np.isfinite(grad).all()
    for i in (0,1,2,3,4,5,7,8,9,10,len(p)-1):
        step=np.zeros_like(p);step[i]=1e-5
        numeric=float(objective(p+step,200.)-objective(p-step,200.))/(2e-5)
        assert grad[i]==pytest.approx(numeric,abs=2e-5,rel=2e-4)
    _,a=model.components(jnp.asarray(p),200.)
    p2=p.copy();p2[1]+=.6
    _,b=model.components(jnp.asarray(p2),200.)
    np.testing.assert_allclose(a[-2],b[-2],atol=1e-15)
    assert np.min(a[-2])>0 and float(a[-2].sum())==pytest.approx(1.)
    _,zero=model.components(jnp.asarray(p),0.)
    np.testing.assert_allclose(zero[-1],20.,atol=1e-13)


def test_direct_imaging_renderer_supports_low_sersic_index():
    from pysersic.rendering import PixelRenderer
    y,x=np.mgrid[-4:5,-4:5]
    psf=np.exp(-.5*(x*x+y*y));psf/=psf.sum()
    renderer=PixelRenderer((41,41),jnp.asarray(psf))
    params={'xc':20.13,'yc':20.27,'flux':100.,'r_eff':5.,'n':.5,'ellip':.6,'theta':1.6}
    def moment(n):
        image=renderer.render_source(params|{'n':n},'sersic')
        return jnp.sum(image*jnp.asarray(np.indices((41,41))[0]**2))
    value,grad=jax.value_and_grad(moment)(.5)
    assert np.isfinite(value) and np.isfinite(grad)
