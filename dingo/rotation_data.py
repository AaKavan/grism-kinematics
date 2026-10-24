"""Prepare real native R+C observations for the rotation-only posterior.

The inclination transfer and nominal instrumental kernels retain the validated
October 18 definitions, now available without importing experiment runners.
This loader never fits data, initializes chains, or substitutes another source.
"""
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
import torch
from astropy.io import fits
from scipy.optimize import least_squares

from .exposures import load_native_target
from .native_coupled import calibration_response
from .native_fitting import BilinearLineMap, StripContinuumLikelihood
from .native_forward import NativeLineProjector, projection_coefficients
from .native_jax import JaxNativeProjector, JaxLineMap, JaxStripLikelihood
from .native_quality import apply_detector_mask
from .native_real import (RealVelocityProfile, oblate_inclination,
                          broadened_inclination_density, separable_response_kernel)
from .real_workflow import DEFAULT_RESULTS, REPO, RealSource, read_json, sha256
from .rotation_model import RotationOnlyModel


CODE = [
    'dingo/rotation_data.py', 'dingo/rotation_model.py',
    'dingo/exposures.py', 'dingo/grism.py', 'dingo/native_coupled.py',
    'dingo/native_fitting.py', 'dingo/native_forward.py', 'dingo/native_jax.py',
    'dingo/native_quality.py', 'dingo/native_real.py', 'dingo/native_posterior.py',
    'dingo/native_real_posterior.py', 'dingo/real_workflow.py',
]
DETECTOR_MASK = REPO / 'data/detector_masks/survey_checked_pixels_2026-10-18.json'


def inclination_prior(source_id, imaging, kind, scatter=10.):
    if kind not in ('imaging', 'isotropic'):
        raise ValueError('Inclination must be imaging or isotropic.')
    if kind == 'isotropic':
        grid = np.deg2rad(np.linspace(5, 85, 1601))
        pdf = np.sin(grid); pdf /= np.trapezoid(pdf, grid)
        return grid, np.log(pdf), {'kind': kind, 'source_id': source_id,
                                  'description': 'Broad isotropic orientation, truncated to 5–85 degrees; no imaging constraint.'}
    if imaging is None:
        raise ValueError('An accepted source-specific imaging directory is required.')
    rng = np.random.default_rng(source_id)
    angles, sources, rejected = [], [], []
    for name in ('full_pixel', 'core_pixel_clean'):
        path = Path(imaging) / name
        report = read_json(path / 'imaging.json')
        if report['source_id'] != source_id:
            raise ValueError(f'Imaging fit for a different source: {path}')
        if not (report['completed'] and report['diagnostics']['convergence_gate_passed'] and report['orientation_parameterization_ok']):
            raise ValueError(f'Imaging fit failed its diagnostics: {path}')
        for input_path, digest in report['input_sha256'].items():
            if sha256(input_path) != digest:
                raise ValueError(f'Imaging input changed: {input_path}')
        with np.load(path / 'samples.npz') as saved:
            q = saved['q'].ravel()[::8]
        thickness = rng.uniform(.05, .30, len(q))
        keep = (q >= thickness) & (q <= 1.) & np.isfinite(q)
        if not keep.any():
            raise ValueError('No valid axis-ratio/thickness pairs; inspect imaging model.')
        # Equal weights for the two alternative fits of the same image, even
        # if their physically valid sample counts differ.
        sample = oblate_inclination(q[keep], thickness[keep])
        angles.append(sample)
        rejected.append(float(1 - keep.mean()))
        sources.append({'path': str(path), 'samples_sha256': sha256(path / 'samples.npz')})
    n = min(map(len, angles))
    stellar = np.concatenate([x[np.linspace(0, len(x)-1, n, dtype=int)] for x in angles])
    grid, logp = broadened_inclination_density(stellar, scatter)
    return grid, logp, {'kind': 'imaging', 'source_id': source_id, 'sources': sources,
        'q0_uniform_before_rejection': [.05, .30], 'invalid_q_q0_pair_fraction': rejected,
        'additional_scatter_deg': scatter, 'alternative_fits_not_independent_data': True,
        'stellar_inclination_5_50_95_deg': np.rad2deg(np.quantile(stellar, [.05, .5, .95])).tolist()}


def response_kernels(target, bundle, scale_factor=1.):
    """Nominal empirical spectral response and wavelength-scaled spatial core."""
    bundle = Path(bundle)
    psfpath = bundle/"calibration/psf/fsun_grisms_4750/4750_f444w_ePSF.fits"
    with fits.open(psfpath) as h:
        psf = np.maximum(h[0].data,0.); psfscale = h[0].header["PIXSCALE"]
    yy,xx = np.indices(psf.shape)
    cy,cx = np.unravel_index(psf.argmax(),psf.shape)
    selected = np.hypot(xx-cx,yy-cy)*psfscale < .16
    x,y,z = xx[selected],yy[selected],psf[selected]
    def residual(p):
        a,mx,my,sx,sy=p
        return a*np.exp(-.5*((x-mx)**2/sx**2+(y-my)**2/sy**2))-z
    fit = least_squares(residual,[psf.max(),cx,cy,4.,4.],
                        bounds=([0,cx-3,cy-3,1,1],[1,cx+3,cy+3,10,10]))
    if not fit.success:
        raise RuntimeError("F444W core fit failed")
    # A provisional diffraction scaling of the broad-band core to the line.
    cross_sigma_arcsec = np.mean(fit.x[-2:])*psfscale*target.wavelength_um/4.44*scale_factor
    lsfpath = bundle/"calibration/grism_cal/LSF/LSF_param.dat"
    lsf = np.genfromtxt(lsfpath,names=True,dtype=None,encoding="utf-8")
    kernels, records = [], []
    for e in target.exposures:
        row = lsf[lsf["modpup"] == e.module+e.pupil][0]
        logwave = np.log10(target.wavelength_um/4.)
        frac = row["frac1_k"]*logwave+row["frac1_b"]
        fwhm = np.array([row[f"fwhm{j}_nm_k"]*logwave+row[f"fwhm{j}_nm_b"] for j in (1,2)])
        coeff = projection_coefficients(e.calibration,np.array([e.direct_xy]),target.wavelength_um,e.origin_xy)
        alongaxis = 0 if e.pupil == "R" else 1
        dispersion = abs(coeff[0,alongaxis,1])
        detector = e.source_detector_xy([0.,.01,0.],[0.,0.,.01])
        jac = (detector[1:]-detector[0]).T/.01
        pixel_per_arcsec = np.linalg.norm(jac[1-alongaxis])
        sigma = fwhm*.001*dispersion/np.sqrt(8*np.log(2))*scale_factor
        cross_sigma = cross_sigma_arcsec*pixel_per_arcsec
        kernels.append(separable_response_kernel(e.pupil,frac,sigma,cross_sigma))
        records.append({"path":str(e.path),"pupil":e.pupil,"module":e.module,
            "lsf_fraction":float(frac),"lsf_fwhm_nm":fwhm.tolist(),
            "dispersion_pixels_per_um":float(dispersion),"cross_sigma_pixels":float(cross_sigma),
            "along_sigma_pixels":sigma.tolist()})
    return np.stack(kernels),{"files_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (psfpath,lsfpath)},
        "F444W_core_sigma_pixels":fit.x[-2:].tolist(),"F444W_psf_pixel_scale":float(psfscale),
        "cross_sigma_arcsec_at_line":float(cross_sigma_arcsec),"response_scale":scale_factor,
        "exposures":records,"limitations":["Separable spectral double Gaussian and circularized spatial Gaussian core",
        "F444W core rescaled by wavelength/4.44; PSF wings/asymmetry not modeled",
        "Cubic deposition corrected in second moment, not exact response shape",
        "Empirical LSF already includes optical spectral response; no second optical convolution along dispersion"]}


def _imaging_directory(source, requested):
    if requested is not None:
        return Path(requested).expanduser().resolve()
    current = source.directory / 'imaging'
    # Resolve the historical ID3700 prior within the selected results root,
    # never silently borrow the default machine's prior for a different root.
    legacy = source.results / '2026-10-15_real_ID3700/imaging'
    if (source.source_id == 3700 and not (current / 'full_pixel/imaging.json').exists()
            and legacy.exists()):
        return legacy
    return current


def _array_sha256(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def prepare(source_id, bundle, config=None, imaging=None, inclination='imaging', results=None):
    """Return ``(model, target, metadata)`` for the source's own real pixels.

    The new model has 21x21 shared brightness controls, 121x121 spatial
    integration nodes, a single monochromatic spectral node, and sigma fixed
    exactly to zero. No fit or chain is started here. Input-file checksums cover
    configuration, associations, priors, masks and calibrations; array checksums
    cover the consumed native pixels and calibrated projection. Entire survey
    exposure files are neither copied nor hashed.
    """
    if inclination not in ('imaging', 'isotropic'):
        raise ValueError('Inclination must be imaging or isotropic.')
    source = RealSource(source_id, bundle=bundle,
                        results=DEFAULT_RESULTS if results is None else results, config=config)
    source.require_data()
    imaging_path = _imaging_directory(source, imaging)
    ig, ip, prior = inclination_prior(source.source_id, imaging_path, inclination, scatter=10.)
    # The native projection and source-map interpolation require float64.
    jax.config.update('jax_enable_x64', True)
    target = load_native_target(source.bundle, source.config, size=81)
    mask = apply_detector_mask(target, DETECTOR_MASK)
    pupils = [e.pupil for e in target.exposures]
    if set(pupils) != {'R', 'C'}:
        raise ValueError('A joint fit requires usable native exposures from both R and C.')
    science, error, valid, dq = [np.stack([getattr(e, key) for e in target.exposures])
                                for key in ('science', 'error', 'valid', 'dq')]
    if any(not e.valid.any() for e in target.exposures):
        raise ValueError('A detector exclusion removed every valid pixel in an exposure.')
    kernels, instrument = response_kernels(target, source.bundle)
    line_map = BilinearLineMap(21, 121, half_extent=.9, dtype=torch.float64)
    operator = NativeLineProjector.from_target(
        target, line_map.east.numpy(), line_map.north.numpy(), kernels,
        np.ones(len(target.exposures)), spectral_samples=1,
        dtype=torch.float64, deposition='cubic')
    likelihood = StripContinuumLikelihood(science, error, valid, pupils, degree=2)
    engine = RealVelocityProfile(
        JaxNativeProjector(operator), JaxLineMap(line_map), JaxStripLikelihood(likelihood),
        ig, ip, half_extent=.9, correlation=.1, map_sigma=2.)
    response, calibration = calibration_response(target, engine.projector, source.bundle)
    model = RotationOnlyModel(engine, target, response, mode='positive',
        calibration_sigma=.1, background_scale=1., continuum_scale=1.,
        center_kind='catalog', center_mean=(0., 0.))

    inputs = {str(source.config): sha256(source.config),
              str(DETECTOR_MASK): sha256(DETECTOR_MASK)}
    for exposure in target.exposures:
        stats_path = exposure.metadata['stats_path']
        if stats_path not in inputs:
            inputs[stats_path] = sha256(stats_path)
        inputs.update(exposure.calibration.provenance)
    inputs.update(instrument['files_sha256'])
    for record in calibration['records']:
        inputs[record['sensitivity_path']] = record['sensitivity_sha256']
    if inclination == 'imaging':
        for name in ('full_pixel', 'core_pixel_clean'):
            directory = imaging_path / name
            report_path = directory / 'imaging.json'
            inputs[str(report_path)] = sha256(report_path)
            inputs[str(directory / 'samples.npz')] = sha256(directory / 'samples.npz')
            inputs.update(read_json(report_path)['input_sha256'])

    arrays = dict(science=science, error=error, valid=valid, dq=dq,
        projection_coefficients=operator.coefficients.numpy(), kernels=operator.kernels.numpy(),
        inclination_grid=ig, inclination_logpdf=ip, calibrated_response=response,
        source_east=line_map.east.numpy(), source_north=line_map.north.numpy(),
        source_area=line_map.area.numpy(), map_basis=np.asarray(engine.basis))
    array_hashes = {key: _array_sha256(value) for key, value in arrays.items()}
    exposure_metadata = [e.metadata for e in target.exposures]
    array_hashes['exposure_metadata_json'] = hashlib.sha256(json.dumps(
        dict(source_summary=target.summary, exposures=exposure_metadata,
             excluded_exposures=target.excluded_exposures),
        sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    settings = dict(source_id=source.source_id, config=str(source.config),
        bundle=str(source.bundle), results=str(source.results),
        imaging=str(imaging_path) if inclination == 'imaging' else None,
        inclination=inclination, inclination_scatter=10., size=81,
        controls=21, quadrature=121, half_extent_arcsec=.9, spectral_samples=1,
        deposition='cubic', response_scale=1., correlation=.1, map_sigma=2.,
        registration_sigma_pixels=.25, center='catalog',
        center_mean_east_north_arcsec=[0., 0.], intrinsic_sigma_kms=0.,
        continuum='positive', continuum_scale=1., background_scale=1.,
        calibration_sigma=.1, detector_mask=str(DETECTOR_MASK))
    metadata = dict(source_id=source.source_id, real_observations=True,
        source_summary=target.summary, nominal_wavelength_um=target.wavelength_um,
        exposures=exposure_metadata, excluded_exposures=target.excluded_exposures,
        inclination_prior=prior, instrument_response=instrument, calibration=calibration,
        detector_mask=mask, array_sha256=array_hashes, input_sha256=inputs,
        code_sha256={name: sha256(REPO / name) for name in CODE}, settings=settings,
        n_valid_pixels=int(valid.sum()), n_exposures=len(target.exposures),
        source_grid=dict(shape=[121, 121], half_extent_arcsec=.9, spectral_quadrature_nodes=1),
        continuum_coefficient_count=int(np.prod(model.continuum_model.coefficient_shape)),
        background_sigma_DN_s=np.asarray(model.background_sigma).tolist(),
        prior=model.prior_description(),
        input_hashing_scope='Small input files plus consumed SCI/ERR/DQ/mask arrays, source grid, '
            'projection coefficients, kernels, calibration response and exposure/WCS metadata. '
            'Full rate FITS files are referenced by path but are not hashed or copied.')
    return model, target, metadata

