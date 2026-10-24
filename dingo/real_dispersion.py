"""CPU-only inspection and notebook launch for fixed-dispersion real R+C fits."""
from pathlib import Path
import re

import numpy as np

from .real_workflow import read_json, run_command, sha256
from .real_review import load_pixels

LABELS = {'baseline': 'Nominal response', 'lsf_0p9': 'Spectral width −10%',
          'lsf_1p1': 'Spectral width +10%', 'psf_0p9': 'Spatial width −10%',
          'psf_1p1': 'Spatial width +10%'}


def fit_dispersion(source, grid, run_name='dispersion_v1', *, sigmas=(0, 10, 20, 40, 80),
                   variants=tuple(LABELS), maxiter=1200, device='cuda'):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_name):
        raise ValueError('Use letters, numbers, underscores or dashes in the run name.')
    if set(variants)-set(LABELS) or not variants:
        raise ValueError('Unknown or empty response variants.')
    grid = Path(grid).expanduser().resolve()
    report = read_json(grid/'grid.json')
    if report.get('source_id') != source.source_id or report.get('real_observations') is not True:
        raise ValueError('Select a real grid belonging to this source.')
    output = source.directory/run_name
    args = ['-m', 'diagnostics.native_real_dispersion', '--source-id', str(source.source_id),
            '--grid', str(grid), '--output', str(output), '--device', device,
            '--maxiter', str(maxiter), '--sigmas', *map(str, sigmas), '--variants', *variants]
    run_command(args, output/'run.log', device)
    return output


class DispersionResults:
    def __init__(self, path, source_id):
        self.path = Path(path).expanduser().resolve()
        self.report = read_json(self.path/'dispersion.json')
        if (self.report.get('format') != 'native_real_dispersion_v1'
                or self.report.get('real_observations') is not True
                or self.report.get('source_id') != int(source_id)):
            raise ValueError('Dispersion results belong to a different source or format.')
        self.source_id = int(source_id)
        if 'native_data_sha256' not in self.report:
            raise ValueError('Setup has not saved a completed fit yet. Check run.log and resume the run.')
        if sha256(self.path/'native_data.npz') != self.report['native_data_sha256']:
            raise ValueError('Saved native pixels changed.')
        self.pixels, self.metadata = load_pixels(self.path/'native_data.npz', self.source_id)
        self.fits = self.report['best_by_sigma']
        self.variants = [v for v in self.report['settings']['variants'] if any(f['variant'] == v for f in self.fits)]

    def select(self, sigma, variant='baseline'):
        matches = [f for f in self.fits if f['variant'] == variant and f['fixed_sigma_kms'] == float(sigma)]
        if len(matches) != 1:
            raise ValueError('That dispersion/response combination has not been fitted.')
        return matches[0]

    def candidate(self, sigma, variant='baseline'):
        f = self.select(sigma, variant)
        path = self.path/f['artifact']
        if sha256(path) != f['artifact_sha256']:
            raise ValueError('Saved fit artifact changed.')
        with np.load(path) as a:
            result = {k: a[k] for k in a.files}
        if int(result['source_id']) != self.source_id:
            raise ValueError('Fit artifact belongs to a different source.')
        return result

    def table(self):
        import pandas as pd
        rows = []
        for variant in self.variants:
            fs = sorted((f for f in self.fits if f['variant'] == variant), key=lambda f: f['fixed_sigma_kms'])
            minimum = min(f['objective'] for f in fs)
            for f in fs:
                rows.append(dict(response=LABELS[variant], sigma_kms=f['fixed_sigma_kms'],
                    V_0p5_kms=f['vrot_0p5arcsec_kms'], score_above_best=2*(f['objective']-minimum),
                    chi2=f['chi_square'], chi2_R=f['chi_square_R'], chi2_C=f['chi_square_C'],
                    penalty=f['penalty'], inclination_deg=f['inclination_deg'],
                    gradient=f['projected_gradient_inf'], stationary=f['stationarity_passed']))
        return pd.DataFrame(rows)

    def profile_figure(self):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
        for i, variant in enumerate(self.variants):
            fs = sorted((f for f in self.fits if f['variant'] == variant), key=lambda f: f['fixed_sigma_kms'])
            sigmas = np.array([f['fixed_sigma_kms'] for f in fs])
            scores = np.array([f['objective'] for f in fs])
            color = f'C{i}'
            # Show both starts so a smooth winner curve cannot hide optimization dependence.
            for sweep in ('up', 'down'):
                trials = sorted((f for f in self.report['fits'] if f['variant'] == variant and f['sweep'] == sweep),
                                key=lambda f: f['fixed_sigma_kms'])
                axes[0].plot([f['fixed_sigma_kms'] for f in trials],
                             [2*(f['objective']-scores.min()) for f in trials],
                             ':', color=color, alpha=.25, lw=1)
            axes[0].plot(sigmas, 2*(scores-scores.min()), 'o-', color=color, label=LABELS[variant])
            axes[1].plot(sigmas, [f['vrot_0p5arcsec_kms'] for f in fs], 'o-', color=color)
            for j, f in enumerate(fs):
                if not f['stationarity_passed']:
                    axes[0].plot(sigmas[j], 2*(scores[j]-scores.min()), 'rx', ms=12, mew=2)
        axes[0].set(xlabel=r'Fixed intrinsic dispersion $\sigma_0$ [km/s]',
                    ylabel='Fit score above best for this response',
                    title='Lower is better; each response has its own zero')
        axes[0].set_yscale('symlog', linthresh=2)
        axes[0].legend(fontsize=8)
        axes[1].set(xlabel=r'Fixed intrinsic dispersion $\sigma_0$ [km/s]',
                    ylabel=r'Refitted $V_{\rm rot}(0.5^{\prime\prime})$ [km/s]',
                    title='Rotation is free at every dispersion')
        for ax in axes:
            ax.grid(alpha=.2)
        fig.suptitle(f'ID{self.source_id}: all R+C exposures jointly refitted\n'
                     'Penalized optimization diagnostic; no confidence thresholds', fontsize=12)
        return fig

    def exposure_pair(self, r_exposure=None, c_exposure=None):
        pair = []
        for pupil, selected in [('R', r_exposure), ('C', c_exposure)]:
            available = [i for i, e in enumerate(self.metadata['exposures']) if e['pupil'] == pupil]
            chosen = available[0] if selected is None else int(selected)-1
            if chosen not in available:
                raise ValueError(f'Select a {pupil} exposure number from {[i+1 for i in available]}.')
            pair.append(chosen)
        return pair

    def fit_figure(self, sigma=0., variant='baseline', *, r_exposure=None, c_exposure=None, full_cutout=False):
        import matplotlib.pyplot as plt
        from matplotlib.colors import AsinhNorm
        from matplotlib.ticker import FormatStrFormatter
        saved = self.candidate(sigma, variant)
        f = self.select(sigma, variant)
        pair = self.exposure_pair(r_exposure, c_exposure)
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout='constrained')
        for row, e in enumerate(pair):
            valid = self.pixels['valid'][e]
            data = np.where(valid, self.pixels['science'][e], np.nan)
            model = np.where(valid, saved['line'][e]+saved['continuum'][e], np.nan)
            n = data.shape[0]; mid = n//2
            lo, hi = (0, n) if full_cutout else (max(0, mid-18), min(n, mid+19))
            cut = np.s_[lo:hi, lo:hi]
            low, high = np.nanquantile(data[cut], [.02, .995])
            norm = AsinhNorm(linear_width=max((high-low)/10, 1e-12), vmin=low, vmax=max(high, low+1e-12))
            pupil = self.metadata['exposures'][e]['pupil']
            for col, (values, title) in enumerate([(data, 'Observed SCI'), (model, 'Predicted line + continuum'),
                                                  (saved['residual'][e], '(data − model) / ERR')]):
                style = (dict(cmap=plt.get_cmap('RdBu_r').with_extremes(bad='.65'), vmin=-5, vmax=5)
                         if col == 2 else dict(cmap=plt.get_cmap('gist_heat').with_extremes(bad='.65'), norm=norm))
                im = axes[row, col].imshow(np.where(valid, values, np.nan)[cut], origin='lower',
                                           extent=[lo-.5, hi-.5]*2, interpolation='none', **style)
                axes[row, col].set(title=title, xlabel='Cutout x [pixel]')
                if col == 0:
                    axes[row, col].set_ylabel(f'{pupil}, exposure {e+1}\nCutout y [pixel]')
                bar = fig.colorbar(im, ax=axes[row, col], shrink=.8)
                if col < 2:
                    bar.set_ticks([low, 0. if low < 0 < high else (low+high)/2, high])
                    bar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
                    bar.set_label(self.metadata['exposures'][e].get('unit', 'SCI units'), fontsize=8)
                bar.ax.tick_params(labelsize=8)
        fig.suptitle(f'ID{self.source_id} | {LABELS[variant]} | fixed σ = {sigma:g} km/s\n'
                     f'Refitted V(0.5″) = {f["vrot_0p5arcsec_kms"]:.1f} km/s | '
                     f'optimization {"PASS" if f["stationarity_passed"] else "FAIL"}', fontsize=13)
        return fig

    def residual_figure(self, variant='baseline', *, r_exposure=None, c_exposure=None):
        import matplotlib.pyplot as plt
        fs = sorted((f for f in self.fits if f['variant'] == variant), key=lambda f: f['fixed_sigma_kms'])
        if not fs:
            raise ValueError('No fitted points for that response.')
        pair = self.exposure_pair(r_exposure, c_exposure)
        fig, axes = plt.subplots(2, len(fs), figsize=(3*len(fs), 6.5), squeeze=False, layout='constrained')
        for col, f in enumerate(fs):
            saved = self.candidate(f['fixed_sigma_kms'], variant)
            for row, e in enumerate(pair):
                a = saved['residual'][e]; n = len(a); mid = n//2
                lo, hi = max(0, mid-18), min(n, mid+19)
                im = axes[row, col].imshow(a[lo:hi, lo:hi], origin='lower', interpolation='none',
                    cmap=plt.get_cmap('RdBu_r').with_extremes(bad='.65'), vmin=-5, vmax=5,
                    extent=[lo-.5, hi-.5]*2)
                axes[row, col].set_title(f'σ = {f["fixed_sigma_kms"]:g}; V = {f["vrot_0p5arcsec_kms"]:.0f} km/s')
                if col == 0:
                    axes[row, col].set_ylabel(f'{self.metadata["exposures"][e]["pupil"]}, exposure {e+1}\ny [pixel]')
                axes[row, col].set_xlabel('x [pixel]')
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.8, label='(data − model) / ERR')
        fig.suptitle(f'ID{self.source_id} | {LABELS[variant]} | same R/C exposures and color scale\n'
                     'All other parameters refitted; gray pixels are masked', fontsize=12)
        return fig

    def show(self, save=True):
        from IPython.display import display, Markdown
        import matplotlib.pyplot as plt
        display(Markdown(f'**ID{self.source_id}: {len(self.report["fits"])} saved refits; '
            f'{"complete" if self.report["completed"] else "partial"}.** '
            'One shared galaxy fits all R+C exposures. These scores are not posterior probabilities. '
            'Faint dotted curves show separate optimization sweeps; red crosses flag failed checks. '
            'Each response curve has its own zero; the ±10% scenarios are not calibration error bars.'))
        display(self.table().round(3))
        figures = [('dispersion_profile', self.profile_figure())]
        if 'baseline' in self.variants:
            figures.append(('baseline_residuals', self.residual_figure()))
        for name, fig in figures:
            if save:
                out = self.path/'figures'; out.mkdir(exist_ok=True)
                fig.savefig(out/f'{name}.png', dpi=160)
                fig.savefig(out/f'{name}.pdf')
            display(fig); plt.close(fig)
