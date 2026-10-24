"""Inspect saved line/continuum cancellation without refitting or GPU use."""
from pathlib import Path

import numpy as np


def line_continuum_diagnostic(review, velocity=None, *, exposures=None, output=None):
    """Return a decomposition figure and per-exposure diagnostic table.

    ``template_power_remaining`` is ||P (line / ERR)||² / ||line / ERR||²,
    on valid pixels, where P removes the fitted continuum subspace. It is a
    squared, noise-weighted norm fraction, not a fraction of emission flux.
    Exposure indices are zero based. Only previously cached fits are read.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import torch
    from .native_fitting import StripContinuumLikelihood
    from .real_workflow import write_json, sha256

    velocity = review.best['vrot_0p5arcsec_kms'] if velocity is None else float(velocity)
    fit = review.fits[velocity]
    saved = review.candidate(velocity, render=False)
    pixels = review.pixels
    metadata = review.metadata['exposures']
    pupils = [e['pupil'] for e in metadata]
    like = StripContinuumLikelihood(pixels['science'], pixels['error'], pixels['valid'],
                                   pupils, degree=2)
    line = torch.from_numpy(saved['line'].copy())
    white = torch.where(like.valid, line, 0.) * like.inverse_error
    projected = like.residualize(white).numpy()
    power = np.sum(white.numpy()**2, axis=(-2, -1))
    ratio = np.divide(np.sum(projected**2, axis=(-2, -1)), power,
                      out=np.full_like(power, np.nan), where=power > 0)
    with np.load(review.path / fit['artifact']) as a:
        amplitudes = a['amplitudes'].copy()
    chi2 = np.nansum(saved['residual']**2, axis=(-2, -1))
    # Verify that the decomposed prediction is the actual stored fit.
    np.testing.assert_allclose(chi2.sum(), fit['chi_square'], rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(like.chi_square(line).item(), chi2.sum(), rtol=1e-9, atol=1e-6)
    table = pd.DataFrame(dict(
        exposure=np.arange(1, len(metadata)+1), pupil=pupils,
        line_amplitude=amplitudes,
        line_sum_in_cutout=saved['line'].sum(axis=(-2, -1)),
        template_power_remaining=ratio, chi2=chi2,
        valid_pixels=pixels['valid'].sum(axis=(-2, -1))))
    if exposures is None:
        exposures = (pupils.index('R'), pupils.index('C'))
    exposures = tuple(exposures)
    if not exposures or any(i < 0 or i >= len(metadata) for i in exposures):
        raise ValueError('Select valid, zero-based exposure indices.')
    fig, axes = plt.subplots(len(exposures), 4, squeeze=False,
                             figsize=(15, 3.5*len(exposures)), layout='constrained')
    cmap = plt.get_cmap('RdBu_r').with_extremes(bad='0.65')
    for row, ei in enumerate(exposures):
        values = (pixels['science'][ei]/pixels['error'][ei],
                  saved['line'][ei]/pixels['error'][ei],
                  saved['continuum'][ei]/pixels['error'][ei], saved['residual'][ei])
        for col, (value, title) in enumerate(zip(values,
                ('Observed SCI / ERR', 'Line model / ERR', 'Continuum / ERR', 'Residual / ERR'))):
            ax = axes[row, col]
            im = ax.imshow(np.where(pixels['valid'][ei], value, np.nan),
                           origin='lower', cmap=cmap, vmin=-5, vmax=5, interpolation='none')
            ax.set(title=f'{pupils[ei]} exposure {ei+1}: {title}', xlabel='Cutout x [pixel]')
            if col == 0:
                ax.set_ylabel('Cutout y [pixel]')
            fig.colorbar(im, ax=ax, shrink=.65, extend='both')
    fig.suptitle(f'ID{review.source_id}: saved Vrot(0.5″)={velocity:g} km/s fit\n'
                 'Line and continuum shown separately; all panels clipped at ±5 ERR', fontsize=12)
    if output is not None:
        output = Path(output); output.mkdir(parents=True, exist_ok=True)
        fig.savefig(output/'line_continuum.png', dpi=140)
        fig.savefig(output/'line_continuum.pdf')
        table.to_csv(output/'exposures.csv', index=False)
        selected = {p: amplitudes[np.array(pupils) == p] for p in ('R', 'C')}
        denominator = float(np.median(selected['R']))
        write_json(output/'decomposition.json', dict(
            source_id=review.source_id, grid=str(review.path), grid_completed=review.report['completed'],
            grid_sha256=sha256(review.path/'grid.json'), fit=fit,
            fit_sha256=sha256(review.path/fit['artifact']),
            candidate_sha256=sha256(review.path/f'candidate_{velocity:g}.npz'),
            diagnostic_code_sha256=sha256(Path(__file__)),
            median_C_to_R_amplitude_ratio=(float(np.median(selected['C']))/denominator
                                           if denominator > 0 else None),
            chi2_check_passed=True, records=table.replace({np.nan: None}).to_dict('records'),
            interpretation='Template power is a noise-weighted squared-norm fraction, not emission flux. '
                           'Amplitude ratios have not been corrected for relative throughput. '
                           'This diagnostic does not certify a global optimum or a physical fit.'))
    return fig, table
