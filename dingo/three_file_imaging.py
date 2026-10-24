"""One-cell PySersic preparation and review before the three-file velocity search.

The stellar image is calibrated F150W data, not the display-stretched legacy
``image.fits``. These fits do not set the gas center or emission morphology.
"""
from pathlib import Path

import numpy as np

from .rotation_imaging import VARIANTS, prepare_imaging
from .real_workflow import sha256


def _show(value):
    from IPython.display import display
    display(value)


def _show_figure(figure):
    import matplotlib.pyplot as plt
    _show(figure)
    plt.close(figure)


def review_imaging(source_id, *, run=True, override=False, run_name='pysersic_v1',
                   results=None, device='cuda', warmup=1000, samples=2000,
                   fit_radius=2., corner_variant='full_pixel', show_core_corner=False,
                   bundle=None, config=None, imaging=None, science=None,
                   error=None, psf=None, cutout_size=181, chunk=50, seed=735):
    """Show the actual fitting image, fit/reuse PySersic, then show its posterior.

    ``run=False`` is read-only, including when ``override=True``. An accepted
    source-specific fit is reused unless override requests a new run. The input
    image is displayed before a fitting worker starts; retained samples are
    reviewed with PySersic's own corner plotting method after it returns.

    Returns an ``ImagingResults`` object. This function does not start a grism
    search or automatically apply an inclination prior.
    """
    if corner_variant not in dict(VARIANTS):
        raise ValueError('corner_variant must be full_pixel or core_pixel_clean.')

    def preview(result):
        _show_figure(result.input_figure(variant=corner_variant, fit_radius=fit_radius))

    result = prepare_imaging(source_id, run=run, override=override,
        run_name=run_name, results=results, device=device, warmup=warmup,
        samples=samples, fit_radius=fit_radius, bundle=bundle, config=config,
        imaging=imaging, science=science, error=error, psf=psf,
        cutout_size=cutout_size, chunk=chunk, seed=seed, preview_callback=preview)
    _show(result.summary())
    if any((result.path/name/'fit.npz').exists() for name, _ in VARIANTS):
        _show_figure(result.figure())
    variants = [corner_variant]
    if show_core_corner:
        variants += [name for name, _ in VARIANTS if name != corner_variant]
    for name in variants:
        if not (result.path/name/'samples.npz').exists():
            print(f'{name}: no retained PySersic samples yet. Reports: {result.path}', flush=True)
            continue
        print(f'{name}: native PySersic parameters.', flush=True)
        _show(result.parameter_summary(variant=name))
        print('Derived stellar inclination: median and 16th–84th percentile interval; '
              'fixed disk thickness, before stellar–gas scatter.', flush=True)
        for q0, row in result.inclination_summary(variant=name).iterrows():
            if row['status'] == 'compatible':
                print(f"  q0 = {q0:g}: i = {row['median_deg']:.3f} "
                      f"(+{row['plus_1sigma_deg']:.3f}/−{row['minus_1sigma_deg']:.3f}) deg", flush=True)
            else:
                print(f"  q0 = {q0:g}: incompatible with "
                      f"{100*row['incompatible_fraction']:.2f}% of axis-ratio draws (b/a < q0); "
                      'no inclination interval reported for this assumption.', flush=True)
        _show_figure(result.corner(variant=name, save=bool(run)))
        _show_figure(result.inclination_figure(variant=name, save=bool(run)))
    print('The imaging fit does not fix the gas center or gas-emission shape. '
          'Using its inclination in the velocity fit is a separate choice.', flush=True)
    return result


def inclination_prior(result, *, scatter_deg=10., q0_bounds=(.05, .30),
                      total_sigma_deg=None):
    """Approximate an accepted imaging mixture by a soft Gaussian gas prior.

    The full and core-masked fits are alternative fits to the same image and
    receive equal mixture weight. For each axis-ratio draw, assume an oblate
    disk with intrinsic thickness uniform over ``q0_bounds``. Add independent
    stellar/gas inclination scatter in quadrature by default. If supplied,
    ``total_sigma_deg`` instead sets the exact total Gaussian width selected by
    the user, retaining the imaging-derived mean; it is not added in quadrature
    and need not encompass the imaging mixture's spread. The returned mean and sigma
    describe the untruncated Gaussian; the grism fitter truncates to 5--85 deg.
    This is an explicit approximation, not a PySersic-sampled gas inclination.
    """
    if not np.isfinite(scatter_deg) or scatter_deg < 0:
        raise ValueError('scatter_deg must be finite and nonnegative.')
    if total_sigma_deg is not None and (
            isinstance(total_sigma_deg, (bool, np.bool_))
            or not isinstance(total_sigma_deg, (int, float, np.integer, np.floating))
            or not np.isfinite(total_sigma_deg) or total_sigma_deg <= 0):
        raise ValueError('total_sigma_deg must be a finite positive number, not a boolean.')
    bounds = np.asarray(q0_bounds, dtype=float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or not 0 <= bounds[0] < bounds[1] < 1:
        raise ValueError('q0_bounds must satisfy 0 <= lower < upper < 1.')
    result.require_ready()  # Checks source, convergence, and saved-file integrity.
    rng = np.random.default_rng(result.source_id)
    stellar, sources, rejected = [], [], []
    for name, _ in VARIANTS:
        path = Path(result.path)/name/'samples.npz'
        with np.load(path, allow_pickle=False) as saved:
            q = np.asarray(saved['q'], dtype=float).ravel()
        if not q.size or not np.isfinite(q).all() or np.any((q <= 0) | (q > 1)):
            raise ValueError(f'Invalid saved PySersic axis ratios: {path}')
        q0 = rng.uniform(bounds[0], bounds[1], q.size)
        keep = q >= q0
        if not keep.any():
            raise ValueError(f'No axis-ratio draws compatible with the assumed disk thickness: {name}')
        cos_i = np.sqrt((q[keep]**2-q0[keep]**2)/(1-q0[keep]**2))
        stellar.append(np.rad2deg(np.arccos(cos_i)))
        rejected.append(float(1-keep.mean()))
        sources.append(dict(variant=name, path=str(path), samples_sha256=sha256(path)))
    # Preserve equal variant weights even if physical sample rejection differs.
    means = np.array([x.mean() for x in stellar])
    mean = float(means.mean())
    variance = float(np.mean([x.var()+(m-mean)**2 for x, m in zip(stellar, means)]))
    sigma = (float(np.sqrt(variance+scatter_deg**2)) if total_sigma_deg is None
             else float(total_sigma_deg))
    if not 5 < mean < 85 or not sigma > 0:
        raise ValueError('Imaging inclination lies outside the grism prior support or has zero width; '
                         'inspect the image and disk assumption before selecting a prior.')
    prior = dict(inclination_deg=mean, inclination_sigma_deg=sigma,
        source_id=result.source_id, kind='imaging_gaussian_approximation',
        q0_uniform_before_rejection=bounds.tolist(), additional_scatter_deg=float(scatter_deg),
        stellar_inclination_mean_deg=mean, stellar_inclination_std_deg=float(np.sqrt(variance)),
        invalid_q_q0_pair_fraction=rejected, sources=sources,
        alternative_fits_not_independent_data=True, truncation_deg=[5., 85.])
    if total_sigma_deg is not None:
        prior.update(kind='imaging_mean_user_width_gaussian',
            additional_scatter_deg=None, total_sigma_deg=sigma,
            width_policy='User-selected total Gaussian width; no quadrature addition. '
                         'The center remains the imaging-derived mixture mean.')
    return prior
