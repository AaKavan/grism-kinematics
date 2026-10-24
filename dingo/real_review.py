"""CPU-only views of saved real-source fits, with optional isolated rendering."""
from pathlib import Path
import json
import numpy as np

from .real_workflow import read_json, run_command, record_review


def _show(fig):
    import matplotlib.pyplot as plt
    from IPython.display import display
    display(fig)
    plt.close(fig)


def load_pixels(path, source_id):
    with np.load(path) as a:
        meta = json.loads(str(a['metadata_json']))
        if meta['source_id'] != source_id:
            raise ValueError('Saved native pixels belong to a different source.')
        return {k: a[k] for k in ('science', 'error', 'valid', 'emline')}, meta


def show_imaging_fits(source):
    """Display even failed morphology models, but label their diagnostic gate."""
    from IPython.display import display, Markdown, Image
    found = False
    for row in source.imaging_status():
        path = Path(row['path'])
        if not path.exists():
            continue
        r = read_json(path)
        if r['source_id'] != source.source_id:
            raise ValueError('Imaging result belongs to another source.')
        found = True
        d = r.get('diagnostics', {})
        status = 'passed sampling checks' if row['ready'] else 'NOT accepted as an inclination prior'
        display(Markdown(f"**ID{source.source_id}, {row['fit']}: {status}.** "
            f"Rhat={d.get('max_rank_rhat', float('nan')):.4f}; "
            f"divergences={d.get('divergences', 'unknown')}. "
            'Check the residuals as well: sampling convergence does not certify the morphology model.'))
        if (path.parent / 'fit.png').exists():
            display(Image(filename=str(path.parent / 'fit.png'), width=1000))
    if not found:
        print('No F150W morphology fits saved for this source yet.')


def show_posterior(path, source_id, show_predictions=False):
    """Inspect real kinematic chains; failed diagnostics stay prominent."""
    import pandas as pd
    from IPython.display import display, Markdown, Image
    path = Path(path)
    r = read_json(path / 'posterior.json')
    if r.get('source_id') != source_id or r.get('real_observations') is not True:
        raise ValueError('Posterior belongs to another source or is not a real-data run.')
    d = r.get('diagnostics')
    display(Markdown(f'**Real ID{source_id} kinematic MCMC** — {r["n_exposures"]} exposures, '
                     f'{r.get("retained_draws_per_chain", 0)} retained draws per chain.\n\n'
                     f'Run status: **{"completed" if r.get("completed") else "partial / not finished"}**. '
                     f'Saved run: `{path}`. The preceding velocity-grid figures remain grid results; '
                     'the figures below come from these separate posterior chains.'))
    if r.get('adapted_from'):
        display(Markdown(f'Adaptation reused from `{r["adapted_from"]["directory"]}`. '
            'These are fresh retained draws; the earlier pilot samples are excluded. '
            'Legend velocities identify the original grid starting branches.'))
    if r.get('initialized_from'):
        display(Markdown(f'Initial positions taken from `{r["initialized_from"]["directory"]}`. '
            f'**Fresh warmup: {r["settings"]["warmup"]} steps per chain.** Earlier tuning and draws '
            'are excluded. Legend velocities show the actual starting positions for this run.'))
    if not d:
        print('No retained-chain diagnostics yet. The run may still be warming up; check its log.')
        return
    passed = d['convergence_gate_passed']
    if passed:
        display(Markdown('**Sampling checks passed.** Intervals remain conditional on the stated '
                         'instrument, noise, emission-map and inclination assumptions.'))
    else:
        display(Markdown('**NOT CONVERGED — diagnostic plots only.** Do not interpret the chain mixture '
                         'as a velocity or mass measurement. Colors identify individual chains; '
                         'pooled credible intervals are withheld.'))
    display(pd.DataFrame([dict(max_Rhat=d['max_rank_rhat'], min_bulk_ESS=d['min_bulk_ess'],
        min_tail_ESS=d['min_tail_ess'], divergences=d['divergences'],
        fraction_at_tree_limit=d['max_depth_fraction'])]))
    table = []
    with np.load(path / 'samples.npz') as a:
        for c in range(a['vrot'].shape[0]):
            table.append(dict(chain=c+1, starting_velocity=r['starts'][c].get('initial_velocity', r['starts'][c]['grid_velocity']),
                retained_median_velocity=float(np.median(a['vrot'][c])),
                retained_median_inclination=float(np.rad2deg(np.median(a['inclination'][c])))))
    display(pd.DataFrame(table))
    display(Markdown('**Reading these plots:** the traces show how each chain moves through successive draws. '
        'In the corner plot, diagonal panels show each parameter’s sampled distribution; off-diagonal '
        'panels show which parameter combinations occur together. Colors denote chains, not different '
        'grisms. Separated colors expose disagreement between starting solutions. Rhat should approach '
        '1 as chains agree; effective sample size measures how much independent information the draws contain.'))
    for name in ('trace.png', 'corner.png', 'profiles.png'):
        if (path / name).exists():
            display(Image(filename=str(path / name), width=1050))
    if show_predictions:
        for image in sorted(path.glob('chain_*_last_prediction.png')):
            display(Image(filename=str(image), width=1050))
    display(Markdown('The corner plot shows parameter correlations by chain. Gray in the **trace plot** '
        'marks excluded warmup; gray detector pixels in the image panels are data masks. '
        'The radius range is illustrative. Mass is the rotation-only spherical-equivalent proxy '
        '`r Vrot²/G`, without a pressure-support correction.'))


def show_real_data(source, run=None):
    """Overview uses native SCI. EMLINE never supplies a fitting likelihood."""
    import matplotlib.pyplot as plt
    from IPython.display import display, Markdown
    if run is None:
        from .exposures import load_native_target
        source.require_data()
        target = load_native_target(source.bundle, source.config, size=81)
        path = source.directory / 'preview/native_data.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        target.save(path)
    else:
        path = Path(run) / 'native_data.npz'
    data, meta = load_pixels(path, source.source_id)
    exposures = meta['exposures']
    display(Markdown(f"**ID{source.source_id} — z = {meta['summary']['z']}, "
                     f"line at {meta['wavelength_um']:.5f} μm.** "
                     f"{len(exposures)} exposures enter one joint R+C model."))
    fig, axes = plt.subplots(int(np.ceil(len(exposures)/6)), 6,
                              figsize=(14, 2.5*int(np.ceil(len(exposures)/6))), squeeze=False)
    for ax, i in zip(axes.ravel(), range(len(exposures))):
        image = np.where(data['valid'][i], data['science'][i], np.nan)
        mid = image.shape[0]//2; image = image[mid-18:mid+19, mid-18:mid+19]
        lo, hi = np.nanquantile(image, [.02, .995])
        ax.imshow(image, origin='lower', cmap='magma', vmin=lo, vmax=hi)
        ax.set_title(f"{i+1}: {exposures[i]['module']}{exposures[i]['pupil']}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[len(exposures):]:
        ax.axis('off')
    fig.suptitle(f'ID{source.source_id}: real native SCI near the line; separate display stretch per exposure')
    fig.tight_layout(); _show(fig)
    imaging = source.imaging_directory()
    if not (imaging / 'cutout.npz').exists():
        imaging = source.prepare_imaging()
    with np.load(imaging / 'cutout.npz') as a:
        image, xy = a['image'], a['center_xy']
    fig, ax = plt.subplots(figsize=(5, 5))
    lo, hi = np.nanquantile(image, [.05, .995])
    ax.imshow(np.arcsinh((image-lo)/max(hi-lo, 1e-12)*10), origin='lower', cmap='magma')
    ax.plot(*xy, 'c+', ms=12); ax.set_title(f'ID{source.source_id}: F150W stellar imaging')
    ax.set(xlabel='Cutout x [pixel]', ylabel='Cutout y [pixel]'); fig.tight_layout(); _show(fig)


class RealResults:
    def __init__(self, run, source_id=None):
        self.path = Path(run).resolve()
        self.report = read_json(self.path / 'grid.json')
        self.source_id = self.report['source_id']
        if source_id is not None and source_id != self.source_id:
            raise ValueError('Selected run belongs to a different source.')
        if self.report.get('real_observations') is not True:
            raise ValueError('This viewer requires a real-data run.')
        self.fits = {float(f['vrot_0p5arcsec_kms']): f for f in self.report['best_by_velocity']}
        if not self.fits:
            raise ValueError('No fitted velocities have been saved yet.')
        self.best = min(self.fits.values(), key=lambda f: f['objective'])
        self.pixels, self.metadata = load_pixels(self.path / 'native_data.npz', self.source_id)
        self.speeds = sorted(self.fits)

    def table(self, velocities=None):
        import pandas as pd
        selected = self.speeds if velocities is None else sorted(set(self.nearest(v) for v in velocities))
        rows = []
        for v in selected:
            f = self.fits[v]
            rows.append({'Vrot(0.5″) km/s': v, 'Fit score above best': 2*(f['objective']-self.best['objective']),
                'inclination deg': f['inclination_deg'], 'turnover arcsec': f['turnover_arcsec'],
                'center east ″': f['center_east_north_arcsec'][0], 'center north ″': f['center_east_north_arcsec'][1],
                'stationary': f['stationarity_passed'], 'search start': f['start']})
        return pd.DataFrame(rows)

    def nearest(self, velocity):
        return min(self.speeds, key=lambda v: abs(v-float(velocity)))

    def show_profile(self, search_details=False, score_limit=100):
        import matplotlib.pyplot as plt
        from IPython.display import display, Markdown
        scores = np.array([self.fits[v]['objective'] for v in self.speeds])
        delta = 2*(scores-scores.min())
        failed = [v for v in self.speeds if not self.fits[v]['stationarity_passed']]
        status = 'completed' if self.report['completed'] else 'PARTIAL — still needs a full scan'
        display(Markdown(f"**ID{self.source_id}: {status}.** Best speed found: "
            f"**{self.best['vrot_0p5arcsec_kms']:g} km/s at 0.5″**. "
            f"{len(self.speeds)} fitted speeds; {len(failed)} fail the stationarity check. "
            "**Lower is better; zero marks the best stored fit.** "
            "The score includes pixel mismatch and prior/regularization penalties. "
            "It is not a probability or a calibrated confidence interval."))
        fig, ax = plt.subplots(figsize=(9, 4.5), layout='constrained')
        ax.plot(self.speeds, delta, color='black', lw=2)
        vbest = self.best['vrot_0p5arcsec_kms']
        ax.axhline(0, color='0.7', lw=1)
        ax.axvline(vbest, color='tab:blue', ls=':', lw=1)
        ax.scatter([vbest], [0], marker='*', s=180, color='tab:blue', zorder=5,
                   label=f'Best found: {vbest:g} km/s (score = 0)')
        if failed:
            ax.scatter(failed, [2*(self.fits[v]['objective']-self.best['objective']) for v in failed],
                       color='red', s=12, label='Optimization check failed')
        top = max(1., float(delta.max())*1.05) if score_limit is None else float(score_limit)
        ax.set(xlabel='Trial rotation speed at 0.5 arcsec [km/s, deprojected]',
               ylabel='Fit score above best — lower is better', ylim=(-.04*top, top),
               title=f'ID{self.source_id}: which trial speed fits best?')
        ax.legend(); ax.grid(alpha=.2); _show(fig)
        if score_limit is not None and np.any(delta > score_limit):
            print(f'Scores above {score_limit:g} are outside this vertical zoom. '
                  'Use show_profile(score_limit=None) to see the full range.')
        if not search_details:
            return
        fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
        for label in ('up', 'down', 'outward'):
            rows = [f for f in self.report['fits'] if f['start'].startswith(label+'_')]
            if rows:
                rows = sorted(rows, key=lambda f: f['vrot_0p5arcsec_kms'])
                ax.plot([f['vrot_0p5arcsec_kms'] for f in rows],
                        [2*(f['objective']-self.best['objective']) for f in rows], label=label)
        if self.report.get('format') != 'native_source_grid_v1':
            rows = sorted([f for f in self.report['fits'] if f['start'].endswith('_outward')],
                          key=lambda f: f['vrot_0p5arcsec_kms'])
            ax.plot([f['vrot_0p5arcsec_kms'] for f in rows],
                    [2*(f['objective']-self.best['objective']) for f in rows], label='outward')
        ax.plot(self.speeds, delta, 'k--', label='Best stored at each speed')
        ax.set(xlabel='Trial rotation speed at 0.5 arcsec [km/s]',
               ylabel='Fit score above best', ylim=(-.04*top, top),
               title='Optional optimization check: different starting paths')
        ax.legend(); _show(fig)

    def candidate(self, velocity, render=False, device='cuda'):
        velocity = float(velocity)
        if velocity not in self.fits:
            raise ValueError('Choose a fitted velocity; unsampled velocities are not interpolated fits.')
        path = self.path / f'candidate_{velocity:g}.npz'
        with np.load(self.path / self.fits[velocity]['artifact']) as a:
            vector = a['vector']
        if path.exists():
            with np.load(path) as a:
                if np.array_equal(vector, a['vector']):
                    return {k: a[k] for k in a.files}
        if not render:
            raise FileNotFoundError(f'Pixel view for {velocity:g} km/s is not cached; click Prepare selected view.')
        run_command(['-m', 'diagnostics.native_source_grid', '--output', str(self.path), '--stage', 'render',
                     '--source-id', str(self.source_id), '--velocities', str(velocity), '--device', device],
                    self.path / 'render.log', device)
        return self.candidate(velocity)

    def _source_maps(self, saved):
        """Undo integration weights to display surface brightness at model nodes."""
        side = int(np.sqrt(saved['source_flux'].size))
        # The real-source models use trapezoidal integration on [-0.9, 0.9]².
        weights = np.ones(side); weights[[0, -1]] = .5
        brightness = saved['source_flux'].reshape(side, side) / np.outer(weights, weights)
        brightness /= brightness.max()
        los = saved['source_velocity'].reshape(side, side) - 100*saved['vector'][3]
        axis = np.linspace(-.9, .9, side)
        return axis, brightness, los

    def _exposure_pair(self, exposure=None, r_exposure=None, c_exposure=None):
        """Indices are zero based, as in the pre-existing single-exposure API."""
        exposures = self.metadata['exposures']
        selected = {'R': r_exposure, 'C': c_exposure}
        if exposure is not None:
            if not 0 <= int(exposure) < len(exposures):
                raise ValueError('Exposure index is outside the saved exposure list.')
            selected[exposures[int(exposure)]['pupil']] = int(exposure)
        for pupil in ('R', 'C'):
            indices = [i for i, e in enumerate(exposures) if e['pupil'] == pupil]
            if not indices:
                raise ValueError(f'No {pupil} exposure is available for the paired view.')
            if selected[pupil] is None:
                selected[pupil] = indices[0]
            if selected[pupil] not in indices:
                raise ValueError(f'Select a {pupil} exposure for the {pupil} row.')
        return selected['R'], selected['C']

    def show_fit(self, velocity=None, exposure=None, full_cutout=False, render=False,
                 device='cuda', *, r_exposure=None, c_exposure=None, source=None):
        """Old-notebook-style R/C rows, using the actual native-pixel fit."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import AsinhNorm
        from IPython.display import display, Markdown
        velocity = self.best['vrot_0p5arcsec_kms'] if velocity is None else float(velocity)
        pair = self._exposure_pair(exposure, r_exposure, c_exposure)
        saved = self.candidate(velocity, render, device)
        fit = self.fits[velocity]
        fig, axes = plt.subplots(2, 4, figsize=(17, 8), layout='constrained')
        for row, e in enumerate(pair):
            valid = self.pixels['valid'][e]
            data = np.where(valid, self.pixels['science'][e], np.nan)
            total = np.where(valid, saved['line'][e]+saved['continuum'][e], np.nan)
            residual = np.where(valid, saved['residual'][e], np.nan)
            mid = data.shape[0]//2
            lower, upper = (0, data.shape[0]) if full_cutout else (mid-18, mid+19)
            cut = np.s_[lower:upper, lower:upper]
            lo, hi = np.nanquantile(data[cut], [.02, .995])
            norm = AsinhNorm(linear_width=max((hi-lo)/10, 1e-12), vmin=lo, vmax=max(hi, lo+1e-12))
            meta = self.metadata['exposures'][e]; pupil = meta['pupil']
            for col, (values, title) in enumerate(zip((data, total, residual),
                    (f'{pupil} grism image', f'{pupil} predicted image', f'{pupil} residual'))):
                ax = axes[row, col]
                style = (dict(cmap=plt.get_cmap('RdBu_r').with_extremes(bad='0.65'), vmin=-5, vmax=5)
                         if col == 2 else dict(cmap=plt.get_cmap('gist_heat').with_extremes(bad='0.65'), norm=norm))
                im = ax.imshow(values[cut], origin='lower', interpolation='none',
                               extent=[lower-.5, upper-.5]*2, **style)
                ax.set(title=title, xlabel='Cutout x [pixel]')
                if col == 0:
                    ax.set_ylabel(f'{pupil}, exposure {e+1} (module {meta["module"]})\nCutout y [pixel]')
                bar = fig.colorbar(im, ax=ax, shrink=.75, pad=.02)
                bar.set_label('(data − model) / ERR' if col == 2 else meta.get('unit', 'SCI units'), fontsize=8)
                bar.ax.tick_params(labelsize=8)
        axis, brightness, los = self._source_maps(saved)
        half_pixel = (axis[1]-axis[0])/2
        extent = [axis[0]-half_pixel, axis[-1]+half_pixel]*2
        im = axes[0, 3].imshow(brightness, origin='lower', extent=extent, cmap='gist_heat',
                              norm=AsinhNorm(linear_width=.05, vmin=0, vmax=1))
        axes[0, 3].set_title('Shared emission map (inferred)')
        fig.colorbar(im, ax=axes[0, 3], shrink=.75, pad=.02, label='Relative surface brightness')
        im = axes[1, 3].imshow(np.where(brightness>.01, los, np.nan), origin='lower', extent=extent,
                               cmap='RdBu_r', vmin=-500, vmax=500)
        axes[1, 3].set_title('Shared line-of-sight velocity')
        fig.colorbar(im, ax=axes[1, 3], shrink=.75, pad=.02, extend='both', label='Rotation [km/s]')
        for ax in axes[:, 3]:
            ax.plot(0, 0, 'w+', ms=10, mew=2)
            ax.plot(*fit['center_east_north_arcsec'], 'cx', ms=8, mew=2)
            ax.set(xlabel='East offset [arcsec]', ylabel='North offset [arcsec]', xlim=(-.9, .9), ylim=(-.9, .9))
        score = 2*(fit['objective']-self.best['objective'])
        fig.suptitle(f'ID{self.source_id} | one joint R+C fit | Vrot(0.5″) = {velocity:g} km/s | '
                     f'inclination = {fit["inclination_deg"]:.1f}° | score above best = {score:.2f}', fontsize=14)
        _show(fig)
        display(Markdown(
            '**First three columns:** observed SCI, predicted line + continuum, and their residual in units of ERR. '
            'Gray pixels are masked. '
            f'These two example exposures belong to the joint fit of all **{len(self.metadata["exposures"])} exposures**. '
            '**Last column:** one inferred source map and its velocity field, shared by R and C. '
            'White +: catalog position; cyan ×: fitted kinematic center. '
            'Red velocity is receding; blue is approaching, relative to the fitted systemic velocity. '
            'Velocity colors are fixed at ±500 km/s and hidden below 1% of peak model brightness; this is a display mask, '
            'not a detection boundary. '
            f'Optimization check: **{"passed" if fit["stationarity_passed"] else "failed"}**.'))
        if source is not None:
            self.show_morphology(source, velocity, saved=saved)

    def show_morphology(self, source, velocity=None, *, saved=None):
        """Overlay the inferred gas on unshifted F150W using its saved sky WCS."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import AsinhNorm
        from matplotlib.lines import Line2D
        from astropy.coordinates import SkyCoord
        from astropy.io.fits import Header
        from astropy.wcs import WCS
        import astropy.units as u
        from IPython.display import display, Markdown
        if source.source_id != self.source_id:
            raise ValueError('Imaging and grism results belong to different sources.')
        inputs = source.imaging_directory()
        provenance = read_json(inputs / 'input_provenance.json')
        if provenance['source_id'] != self.source_id:
            raise ValueError('Saved imaging belongs to a different source.')
        velocity = self.best['vrot_0p5arcsec_kms'] if velocity is None else float(velocity)
        saved = self.candidate(velocity) if saved is None else saved
        axis, brightness, _ = self._source_maps(saved)
        with np.load(inputs / 'cutout.npz') as a:
            data = a['image']
            wcs = WCS(Header.fromstring(str(a['wcs_header'])), fix=False).celestial
            pixel_scale = float(a['pixel_scale_arcsec'])
        summary = self.metadata['summary']
        origin = SkyCoord(summary['ra'], summary['dec'], unit='deg')
        east, north = np.meshgrid(axis, axis)
        xx, yy = wcs.world_to_pixel(origin.spherical_offsets_by(east*u.arcsec, north*u.arcsec))
        center = origin.spherical_offsets_by(*[v*u.arcsec for v in self.fits[velocity]['center_east_north_arcsec']])
        cx, cy = wcs.world_to_pixel(origin)
        lo, hi = np.nanquantile(data, [.05, .995])
        fig, ax = plt.subplots(figsize=(6, 5.5), layout='constrained')
        ax.imshow(data, origin='lower', cmap='gist_heat',
                  norm=AsinhNorm(linear_width=max((hi-lo)/10, 1e-12), vmin=lo, vmax=hi))
        ax.contour(xx, yy, brightness, levels=[.1, .25, .5, .75, .9], colors='cyan', linewidths=.9)
        ax.plot(cx, cy, 'w+', ms=12, mew=2, label='Catalog position')
        ax.plot(*wcs.world_to_pixel(center), 'cx', ms=9, mew=2, label='Kinematic center')
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], color='cyan', label='Inferred line emission'))
        ax.legend(handles=handles, fontsize=8, loc='upper left')
        ax.set(xlim=(cx-1.1/pixel_scale, cx+1.1/pixel_scale),
               ylim=(cy-1.1/pixel_scale, cy+1.1/pixel_scale),
               xlabel='F150W cutout x [pixel]', ylabel='F150W cutout y [pixel]',
               title=f'ID{self.source_id}: F150W image + fitted gas contours\nVrot(0.5″) = {velocity:g} km/s')
        _show(fig)
        display(Markdown('Contours are 10, 25, 50, 75 and 90% of peak **inferred** line brightness. '
                         'The overlay uses the image WCS; the stellar and gas centers have not been shifted to coincide. '
                         'F150W is PSF blurred; the gas contours are intrinsic model structure, not an independent gas image.'))

    def inspector(self, device='cuda', source=None):
        import ipywidgets as widgets
        from IPython.display import display, clear_output
        velocity = widgets.SelectionSlider(options=[(f'{v:g} km/s', v) for v in self.speeds],
            value=self.best['vrot_0p5arcsec_kms'], description='Velocity', continuous_update=False,
            layout=widgets.Layout(width='95%'))
        selectors = [widgets.Dropdown(options=[(f"{i+1}: module {e['module']}", i)
            for i, e in enumerate(self.metadata['exposures']) if e['pupil'] == pupil],
            description=f'Grism {pupil}', layout=widgets.Layout(width='230px')) for pupil in ('R', 'C')]
        full = widgets.Checkbox(value=False, description='Full detector cutout')
        button = widgets.Button(description='Prepare selected view', layout=widgets.Layout(width='200px'))
        output = widgets.Output()
        def update(change=None, render=False):
            with output:
                clear_output(wait=True)
                try:
                    self.show_fit(velocity.value, full_cutout=full.value, render=render, device=device,
                                  r_exposure=selectors[0].value, c_exposure=selectors[1].value, source=source)
                except (FileNotFoundError, ValueError, RuntimeError) as exc:
                    print(str(exc))
        button.on_click(lambda _: update(render=True))
        for control in (velocity, *selectors, full):
            control.observe(update, names='value')
        verdict = widgets.Dropdown(options=['plausible', 'needs investigation', 'reject: documented model failure'],
                                  description='Assessment', value='needs investigation', layout=widgets.Layout(width='450px'))
        notes = widgets.Textarea(placeholder='Describe residual structure, contamination or the reason for keeping/rejecting this trial.',
                                 layout=widgets.Layout(width='95%', height='85px'))
        save = widgets.Button(description='Save inspection note', layout=widgets.Layout(width='200px'))
        status = widgets.Output()
        def save_note(_):
            with status:
                clear_output(wait=True)
                try:
                    print('Saved:', record_review(self.path, velocity.value, verdict.value, notes.value))
                except ValueError as exc:
                    print(exc)
        save.on_click(save_note)
        display(widgets.VBox([velocity, widgets.HBox(selectors), full, button, output,
                              verdict, notes, save, status]))
        update()

    def show_radial_comparison(self, velocities=None, radii_arcsec=(.2, .3, .5, .8)):
        import matplotlib.pyplot as plt
        import pandas as pd
        from astropy.cosmology import Planck18
        from IPython.display import display, Markdown
        radii = np.asarray(radii_arcsec, dtype=float)
        if radii.ndim != 1 or len(radii) < 2 or not np.isfinite(radii).all() or np.any(radii <= 0) or np.any(np.diff(radii) <= 0):
            raise ValueError('Provide at least two positive increasing finite radii.')
        velocities = velocities or [self.best['vrot_0p5arcsec_kms']]
        velocities = sorted(set(self.nearest(v) for v in velocities))
        z = self.metadata['summary']['z']
        scale = float(Planck18.kpc_proper_per_arcmin(z).value/60)
        display(Markdown(f"**Conditional curves at illustrative radii, not measured radial bins or uncertainty bands.** "
                         f"For this source, z={z}, 1″={scale:.3f} kpc (Planck18). "
                         "Mass is the rotation-only spherical-equivalent proxy r Vrot²/G; no pressure-support correction."))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained'); rows = []
        curve_radii = np.linspace(radii[0], radii[-1], 300)
        for index, speed in enumerate(velocities):
            f = self.fits[speed]; rt = f['turnover_arcsec']
            velocity = speed*np.arctan(radii/rt)/np.arctan(.5/rt)
            mass = radii*scale*velocity**2/4.30091727003628e-6
            curve_velocity = speed*np.arctan(curve_radii/rt)/np.arctan(.5/rt)
            curve_mass = curve_radii*scale*curve_velocity**2/4.30091727003628e-6
            is_best = speed == self.best['vrot_0p5arcsec_kms']
            label = f'{"Best found: " if is_best else "Trial: "}{speed:g} km/s at 0.5″'
            style = dict(color='black' if is_best else f'C{index%10}', lw=2 if is_best else 1.3)
            axes[0].plot(curve_radii*scale, curve_velocity, label=label, **style)
            if speed > 0:
                axes[1].plot(curve_radii*scale, curve_mass, label=label, **style)
            if radii[0] <= .5 <= radii[-1]:
                axes[0].plot(.5*scale, speed, 'o', color=style['color'])
                if speed > 0:
                    axes[1].plot(.5*scale, .5*scale*speed**2/4.30091727003628e-6, 'o', color=style['color'])
            rows.extend({'trial km/s': speed, 'radius arcsec': r, 'radius kpc': r*scale,
                         'Vrot km/s': v, 'Mrot solar masses': m} for r, v, m in zip(radii, velocity, mass))
        axes[0].set(xlabel='Disk radius [kpc]', ylabel='Deprojected rotation speed [km/s]',
                    title='Rotation curve — speed versus radius', ylim=(0, None))
        axes[1].set(xlabel='Disk radius [kpc]', ylabel='Rotation mass proxy [solar masses]',
                    title='Enclosed mass proxy: M(<r) = r Vrot(r)²/G', yscale='log')
        if all(v == 0 for v in velocities):
            axes[1].set_ylim(1e6, 1e12)
        for ax in axes:
            ax.grid(True, which='both', alpha=.2)
            if radii[0] <= .5 <= radii[-1]:
                ax.axvline(.5*scale, color='0.5', ls=':', lw=1)
        axes[0].legend(fontsize=8)
        fig.suptitle(f'ID{self.source_id}: conditional model curves; plotted radius range is illustrative', fontsize=11)
        _show(fig)
        if 0 in velocities:
            print('The zero-rotation trial has Mrot = 0 and cannot appear on the logarithmic mass axis.')
        print('Dots and dotted line mark the 0.5 arcsec reference radius; they are model evaluations, not radial measurements.')
        frame = pd.DataFrame(rows); display(frame)
        return frame
