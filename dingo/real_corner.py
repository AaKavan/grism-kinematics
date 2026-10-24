"""CPU-only contour corner plots of saved, diagnostically accepted real chains."""
from pathlib import Path
import json

import numpy as np

from .real_workflow import read_json, sha256


DEFAULT_PARAMETERS = ('vrot', 'inclination', 'rturn', 'x0', 'y0', 'sigma_v')


def contour_corner(directory, source_id, *, parameters=DEFAULT_PARAMETERS, bins=40,
                   smooth=1.0, color='#235789', save=True):
    """Pool retained draws only; report marginal quantiles and explicit 2-D masses.

    Contours approximate 68% and 95% pairwise probability regions using a
    smoothed histogram. These are not Gaussian one/two-sigma contour radii.
    Saved warmup is never loaded, and no draw thinning or tail cuts are applied.
    """
    import corner
    import matplotlib.pyplot as plt
    import pandas as pd

    directory = Path(directory).expanduser().resolve()
    report = read_json(directory / 'posterior.json')
    if report.get('source_id') != source_id or report.get('real_observations') is not True:
        raise ValueError('Choose this source\'s own real-data posterior.')
    if not report.get('diagnostics', {}).get('convergence_gate_passed', False):
        raise ValueError('This run has not passed sampling checks. Use section 8\'s '
                         'plots by chain; pooled credible contours are withheld.')
    if not isinstance(bins, int) or bins < 10:
        raise ValueError('Use an integer bin count of at least 10.')
    if smooth is not None and (not np.isfinite(smooth) or smooth <= 0):
        raise ValueError('Smoothing must be positive (in histogram bins), or None.')
    with np.load(directory / 'samples.npz') as a:
        columns = {
            'vrot': (a['vrot'], r'$V_{\rm rot}(0.5^{\prime\prime})$ [km s$^{-1}$]', 'km/s', 1),
            'inclination': (np.rad2deg(a['inclination']), r'$i$ [deg]', 'deg', 1),
            'rturn': (np.exp(a['log_rturn']), r'$r_{\rm turn}$ [arcsec]', 'arcsec', 3),
            'x0': (a['center'][..., 0], r'$x_0$ [arcsec]', 'arcsec', 3),
            'y0': (a['center'][..., 1], r'$y_0$ [arcsec]', 'arcsec', 3),
            'sigma_v': (a['sigma_v'], r'$\sigma_{\rm int}$ [km s$^{-1}$]', 'km/s', 1),
            'systemic': (a['systemic'], r'$v_{\rm sys}$ [km s$^{-1}$]', 'km/s', 1),
        }
    # Existing notebook selections remain valid; saved sampler coordinates do not change.
    aliases = {'center_east': 'x0', 'center_north': 'y0'}
    parameters = tuple(aliases.get(k, k) for k in parameters)
    if len(parameters) < 2 or len(set(parameters)) != len(parameters) or any(k not in columns for k in parameters):
        raise ValueError(f'Select at least two distinct parameters from {tuple(columns)}.')
    shape = columns['vrot'][0].shape
    if len(shape) != 2 or shape[0] != report['settings']['chains'] or shape[1] != report['retained_draws_per_chain']:
        raise ValueError('Saved draws and report disagree. Wait for the current checkpoint to finish.')
    data = np.column_stack([columns[k][0].reshape(-1) for k in parameters])
    if not np.isfinite(data).all() or np.any(np.ptp(data, axis=0) == 0):
        raise ValueError('Selected parameters must have finite, nonconstant retained draws.')
    quantiles = np.quantile(data, [.16, .5, .84], axis=0)
    labels = [columns[k][1] for k in parameters]
    levels = [.68, .95]
    with plt.rc_context({'font.size': 10, 'axes.linewidth': .8, 'pdf.fonttype': 42}):
        fig = corner.corner(data, labels=labels, bins=bins, smooth=smooth, smooth1d=None,
            levels=levels, color=color, plot_datapoints=False, plot_density=False,
            plot_contours=True, fill_contours=True, quantiles=[.16, .5, .84],
            show_titles=False, max_n_ticks=3, use_math_text=True,
            label_kwargs={'fontsize': 12}, hist_kwargs={'linewidth': 1.5},
            contour_kwargs={'linewidths': 1.1})
        fig.set_size_inches(2.25*len(parameters), 2.25*len(parameters))
        axes = np.asarray(fig.axes).reshape(len(parameters), len(parameters))
        for i, key in enumerate(parameters):
            lo, med, hi = quantiles[:, i]
            precision = columns[key][3]
            title = (f'${med:.{precision}f}'
                     f'_{{-{med-lo:.{precision}f}}}^{{+{hi-med:.{precision}f}}}$')
            axes[i, i].set_title(title, fontsize=12, pad=9)
        fig.subplots_adjust(top=.89, bottom=.09, left=.09, right=.97, hspace=.07, wspace=.07)
        status = 'completed run' if report.get('completed') else 'partial run'
        fig.suptitle(f'ID{source_id}: joint R+C kinematic posterior', y=.985, fontsize=18)
        fig.text(.5, .958, 'Filled contours: 68% and 95% pairwise probability; '
                 'diagonal lines: 16th, 50th, 84th percentiles', ha='center', fontsize=11)
        fig.text(.5, .938, f'{shape[0]} chains × {shape[1]:,} retained draws; {status}; '
                 'sampling checks passed; conditional on the adopted model', ha='center', fontsize=10, color='0.35')
        fig.text(.5, .916, r'$x_0$: east; $y_0$: north — kinematic-center offsets from the catalog position',
                 ha='center', fontsize=10, color='0.35')
    summary = pd.DataFrame([dict(parameter=k, unit=columns[k][2],
        p16=float(quantiles[0, i]), median=float(quantiles[1, i]), p84=float(quantiles[2, i]))
        for i, k in enumerate(parameters)])
    paths = {}
    if save:
        output = directory / 'figures'
        output.mkdir(exist_ok=True)
        for extension in ('png', 'pdf'):
            paths[extension] = output / f'corner_contours.{extension}'
            fig.savefig(paths[extension], dpi=200, bbox_inches='tight', facecolor='white')
        metadata = dict(source_id=source_id, posterior_directory=str(directory),
            samples_sha256=sha256(directory/'samples.npz'), report_sha256=sha256(directory/'posterior.json'),
            parameters=parameters, chains=shape[0], retained_draws_per_chain=shape[1],
            contour_probabilities=levels, marginal_quantiles=[.16, .5, .84],
            bins=bins, gaussian_smoothing_bins=smooth, warmup_included=False, thinning=1,
            plot_ranges='Full retained sample ranges; no tail clipping',
            center_coordinates='x0 east, y0 north; arcsecond offsets from the catalog sky position',
            summary=summary.to_dict(orient='records'), corner_version=corner.__version__)
        paths['metadata'] = output/'corner_contours.json'
        paths['metadata'].write_text(json.dumps(metadata, indent=2)+'\n')
    return fig, summary, paths
