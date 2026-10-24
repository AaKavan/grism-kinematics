"""CPU-only inspection of flexible-morphology sensitivity and a linked control.

The viewer reads saved predictions and verifies their hashes and residuals
through SimpleDispersionResults. It never imports the fitting or JAX modules.
"""
from pathlib import Path
import re

import numpy as np

from .real_simple_dispersion import SimpleDispersionResults
from .real_workflow import read_json, run_command


SCENARIO_LABELS = {
    'baseline': 'Baseline: A=2, ℓ=0.10″',
    'amplitude1': 'Smaller fluctuations: A=1',
    'amplitude4': 'Larger fluctuations: A=4',
    'correlation0p05': 'Shorter correlation: ℓ=0.05″',
    'correlation0p20': 'Longer correlation: ℓ=0.20″',
}
MODEL_LABELS = {'flexible': 'Flexible gas map — primary model',
                'linked_exponential': 'Linked exponential disk — restrictive control'}


def fit_morphology_controls(source, run, seeds, *, run_name='morphology_controls_v1',
                            sigmas=(0, 20, 40, 80), device='cuda'):
    """Launch the checkpointed worker separately from the notebook process."""
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_name):
        raise ValueError('Use a simple run name with letters, digits, underscores or dashes.')
    run, seeds = Path(run).expanduser().resolve(), Path(seeds).expanduser().resolve()
    for directory in (run, seeds):
        report = read_json(directory/'dispersion.json')
        if report.get('source_id') != source.source_id or not report.get('completed'):
            raise ValueError('Select completed original and refined scans of this source.')
    output = source.directory/run_name
    arguments = ['-m', 'diagnostics.native_morphology_controls', '--source-id', str(source.source_id),
                 '--run', str(run), '--seeds', str(seeds), '--output', str(output),
                 '--sigmas', *map(str, sigmas), '--device', device]
    run_command(arguments, output/'run.log', device=device)
    return output


class MorphologyControlResults(SimpleDispersionResults):
    """Saved real R+C fits; field-prior changes and disk control remain distinct."""

    def __init__(self, path, source_id):
        super().__init__(path, source_id)
        if self.report.get('morphology_controls') is not True:
            raise ValueError('Select a morphology-controls run, not the earlier LSF comparison.')

    @staticmethod
    def scenario_label(model, variant):
        return ('Shared gas geometry; exponential light' if model == 'linked_exponential'
                else SCENARIO_LABELS.get(variant, variant))

    def table(self, all_starts=False):
        table = super().table(all_starts).rename(columns={'response': 'scenario'})
        by_artifact = {fit['artifact']: fit for fit in self.fits}
        selected = [by_artifact[name] for name in table.artifact]
        table['V_projected_0p5_kms'] = [f['vproj_0p5arcsec_kms'] for f in selected]
        table['map_penalty'] = [f.get('map_penalty', np.nan) for f in selected]
        table['edge_flux_percent'] = [100*f.get('source_edge_flux_fraction', np.nan) for f in selected]
        table['brightness_moment_q'] = [f.get('brightness_moment_axis_ratio', np.nan) for f in selected]
        table['brightness_moment_PA_deg'] = [f.get('brightness_moment_angle_deg', np.nan) for f in selected]
        for index, axis in enumerate(('x', 'y')):
            table[f'brightness_centroid_{axis}_arcsec'] = [
                f.get('brightness_centroid_arcsec', [np.nan, np.nan])[index] for f in selected]
        scenario_settings = self.report.get('scenario_settings', {})
        for column, setting in (('map_amplitude_A', 'map_sigma'), ('correlation_arcsec', 'correlation_arcsec')):
            table[column] = [scenario_settings.get(f['variant'], {}).get(setting, np.nan)
                             if f['model'] == 'flexible' else np.nan for f in selected]
        return table

    def comparison_table(self):
        """One best tested width per scenario; no posterior intervals or evidence."""
        import pandas as pd
        table = self.table()
        rows = [table.loc[group.two_delta_penalized.idxmin()]
                for _, group in table.groupby(['model', 'scenario'], sort=False)]
        result = pd.DataFrame(rows).reset_index(drop=True)
        order = {name: i for i, name in enumerate(SCENARIO_LABELS)}
        result = result.assign(_order=[order.get(row.scenario, 99) if row.model == 'flexible'
                                       else 100 for row in result.itertuples()])
        result = result.sort_values('_order').drop(columns='_order').reset_index(drop=True)
        return result[['model', 'scenario', 'sigma_kms', 'V_0p5_kms', 'V_projected_0p5_kms',
                       'inclination_deg', 'map_amplitude_A', 'correlation_arcsec', 'map_penalty',
                       'chi2_pixels', 'rms_R_ERR', 'rms_C_ERR', 'brightness_moment_q',
                       'brightness_moment_PA_deg', 'edge_flux_percent', 'stationary', 'boundaries']]

    def profile_figure(self):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(self.models), squeeze=False,
                                 figsize=(7*len(self.models), 4.8), layout='constrained')
        table = self.table()
        for ax, model in zip(axes.ravel(), self.models):
            for variant, group in table[table.model == model].groupby('scenario', sort=False):
                group = group.sort_values('sigma_kms')
                ax.plot(group.sigma_kms, group.two_delta_penalized, 'o-', ms=4,
                        label=self.scenario_label(model, variant))
                failed = group[~group.stationary]
                ax.scatter(failed.sigma_kms, failed.two_delta_penalized, marker='x', s=90, c='red', zorder=3)
            ax.axhline(0, color='.5', lw=.7)
            ax.set(xlabel='Fixed intrinsic dispersion σ₀ [km/s]',
                   ylabel='2 × change in penalized objective', title=MODEL_LABELS.get(model, model))
            ax.legend(fontsize=8)
        fig.suptitle(f'ID{self.source_id}: every curve has its own zero; all other parameters refitted\n'
                     'Nominal instrumental response throughout; no confidence thresholds'
                     + ('\nPARTIAL SCAN' if not self.report.get('completed') else ''))
        return fig

    def velocity_figure(self):
        import matplotlib.pyplot as plt
        table = self.comparison_table()
        labels = [self.scenario_label(row.model, row.scenario).replace(': ', '\n')
                  if row.model == 'flexible' else 'Linked disk\nrestrictive control'
                  for row in table.itertuples()]
        x = np.arange(len(table))
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout='constrained')
        for ax, column, label in zip(axes, ('V_0p5_kms', 'V_projected_0p5_kms'),
                                    ('Deprojected Vrot(0.5″) [km/s]', 'Projected Vrot(0.5″) sin(i) [km/s]')):
            for pos, row in enumerate(table.itertuples()):
                linked = row.model == 'linked_exponential'
                ax.plot(pos, getattr(row, column), 's' if linked else 'o',
                        color='C3' if linked else 'C0', ms=7)
                annotation = f'σ={row.sigma_kms:g}'
                if 'rotation speed' in row.boundaries:
                    annotation += '\nV at allowed limit'
                ax.annotate(annotation, (pos, getattr(row, column)),
                            xytext=(0, 7), textcoords='offset points', ha='center', fontsize=8)
                if not row.stationary:
                    ax.plot(pos, getattr(row, column), 'rx', ms=12)
                if linked:
                    ax.axvspan(pos-.4, pos+.4, color='C3', alpha=.06)
            baseline = table[(table.model == 'flexible') & (table.scenario == 'baseline')]
            if len(baseline):
                ax.axhline(float(baseline.iloc[0][column]), color='.5', ls='--', lw=.8)
            ax.set(ylabel=label)
            ax.margins(y=.2)
            ax.grid(axis='y', alpha=.2)
        axes[-1].set_xticks(x, labels, fontsize=8)
        fig.suptitle(f'ID{self.source_id}: fitted velocities at each scenario’s best tested dispersion\n'
                     'Points are optimization results; changes across priors are not error bars')
        return fig

    def fit_figure(self, model='flexible', variant='baseline', sigma=None, start=None, **kwargs):
        fig = super().fit_figure(model, variant, sigma, start, **kwargs)
        fit = self.best(model, variant, sigma) if start is None else self.candidate(model, variant, sigma, start)[0]
        fig.suptitle(f'ID{self.source_id}: {MODEL_LABELS.get(model, model)}\n'
                     f'{self.scenario_label(model, variant)}; fixed σ₀={fit["fixed_sigma_kms"]:g}; '
                     f'Vrot(0.5″)={fit["vrot_0p5arcsec_kms"]:.1f} km/s; '
                     f'stationarity {"passed" if fit["stationarity_passed"] else "FAILED"}\n'
                     'Observed/predicted share a stretch in each row; gray detector pixels are excluded')
        return fig

    def source_figure(self, model='flexible', variant='baseline', sigma=None, start=None):
        fig = super().source_figure(model, variant, sigma, start)
        fit = self.candidate(model, variant, sigma, start)[0]
        fig.suptitle(f'ID{self.source_id}: {MODEL_LABELS.get(model, model)}\n'
                     f'{self.scenario_label(model, variant)}; fixed σ₀={fit["fixed_sigma_kms"]:g} km/s\n'
                     'One shared intrinsic gas map and velocity field for every R+C exposure')
        return fig

    def source_comparison_figure(self):
        import matplotlib.pyplot as plt
        from matplotlib.colors import AsinhNorm
        table = self.comparison_table()
        columns = min(3, len(table)); rows = (len(table)+columns-1)//columns
        fig, axes = plt.subplots(rows, columns, squeeze=False, figsize=(5*columns, 4.6*rows),
                                 layout='constrained')
        half = self.report['source_grid']['half_extent_arcsec']
        for ax, row in zip(axes.ravel(), table.itertuples()):
            fit, saved = self.candidate(row.model, row.scenario, row.sigma_kms)
            side = int(np.sqrt(saved['source_flux'].size))
            weights = np.ones(side); weights[[0, -1]] = .5
            brightness = saved['source_flux'].reshape(side, side)/np.outer(weights, weights)
            brightness /= brightness.max()
            east = saved['source_east'].reshape(side, side)
            north = saved['source_north'].reshape(side, side)
            im = ax.pcolormesh(east, north, brightness, shading='nearest', cmap='gist_heat',
                              norm=AsinhNorm(linear_width=.05, vmin=0, vmax=1))
            ax.plot(*fit['center_east_north_arcsec'], 'cx', ms=8, mew=2)
            ax.set(xlabel='East from catalog [arcsec]', ylabel='North from catalog [arcsec]',
                   xlim=(-half, half), ylim=(-half, half), aspect='equal',
                   title=f'{self.scenario_label(row.model, row.scenario)}\n'
                         f'V={row.V_0p5_kms:.1f}, σ={row.sigma_kms:g} km/s; edge flux={row.edge_flux_percent:.1f}%')
        for ax in axes.ravel()[len(table):]:
            ax.set_visible(False)
        fig.colorbar(im, ax=list(axes.ravel()[:len(table)]), shrink=.8, label='Brightness / each map’s peak')
        fig.suptitle(f'ID{self.source_id}: gas brightness fitted jointly from R+C\n'
                     'Common stretch; cyan × is rotation center; linked disk is a restrictive control')
        return fig

    def save(self, output=None, model='flexible', variant='baseline', sigma=None, start=None):
        import matplotlib.pyplot as plt
        output = super().save(output, model, variant, sigma, start)
        for name, fig in (('velocity_morphology_comparison', self.velocity_figure()),
                          ('source_morphology_comparison', self.source_comparison_figure())):
            fig.savefig(output/f'{name}.png', dpi=150)
            fig.savefig(output/f'{name}.pdf')
            plt.close(fig)
        return output
