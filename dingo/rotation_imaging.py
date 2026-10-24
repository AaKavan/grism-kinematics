"""CPU orchestration and inspection of source-specific PySersic inclination fits.

Accepted older fits are reusable. Retries live in their own imaging_runs tree;
they never overwrite the inputs or reports used by a previous grism posterior.
"""
import os
from pathlib import Path
import re
import tarfile

import numpy as np

from .real_workflow import (DEFAULT_BUNDLE, DEFAULT_RESULTS, RealSource,
                            read_json, run_command, sha256, write_json)


VARIANTS = (('full_pixel', 0.), ('core_pixel_clean', .18))


def imaging_ready(path, source_id):
    return all(_variant_ready(Path(path)/name, source_id) for name, _ in VARIANTS)


def _variant_ready(path, source_id):
    report_path = path/'imaging.json'
    if not report_path.exists():
        return False
    report = read_json(report_path)
    if not (report.get('source_id') == source_id and report.get('completed')
            and (report.get('diagnostics') or {}).get('convergence_gate_passed')
            and report.get('orientation_parameterization_ok') and (path/'samples.npz').exists()):
        return False
    for filename, digest in report.get('input_sha256', {}).items():
        if sha256(filename) != digest:
            raise ValueError(f'Imaging input checksum changed: {filename}')
    if report.get('samples_sha256') and sha256(path/'samples.npz') != report['samples_sha256']:
        raise ValueError(f'Imaging samples checksum changed: {path}')
    return True


def imaging_directory(source):
    """Discover an accepted source prior, including named PySersic runs."""
    current = source.directory/'imaging'
    legacy = source.results/'2026-10-15_real_ID3700/imaging'
    candidates = [current]
    if source.source_id == 3700:
        candidates.append(legacy)
    new_runs = source.directory/'imaging_runs'
    if new_runs.exists():
        candidates.extend(sorted((p for p in new_runs.iterdir() if p.is_dir()),
                                  key=lambda p: p.stat().st_mtime, reverse=True))
    for path in candidates:
        if imaging_ready(path, source.source_id):
            return path
    return current


def prepare_inputs(source, output, *, cutout_size=181, fit_radius=2., science=None, error=None, psf=None):
    """Extract this source's real F150W/ERR data and matching empirical PSF."""
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.nddata import Cutout2D
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_scales
    if not isinstance(cutout_size, (int, np.integer)) or cutout_size < 9 or cutout_size % 2 != 1:
        raise ValueError('cutout_size must be an odd integer of at least 9 pixels.')
    if not np.isfinite(fit_radius) or fit_radius <= .18:
        raise ValueError('fit_radius must exceed the 0.18 arcsec central-mask radius.')
    output = Path(output).expanduser().resolve()
    provenance = output/'input_provenance.json'
    default_root = Path.home()/'EDR_data'
    science_path = Path(science or os.environ.get('DINGO_F150W_SCI',
        default_root/'sapphires_edr_nircam_sci/4750_F150W_v05_sci.fits')).expanduser().resolve()
    error_path = Path(error or os.environ.get('DINGO_F150W_ERR',
        default_root/'sapphires_edr_nircam_err/4750_F150W_v05_err.fits')).expanduser().resolve()
    psf_path = psf or os.environ.get('DINGO_F150W_PSF')
    psf_path = Path(psf_path).expanduser().resolve() if psf_path else None
    archive = source.bundle/'calibration/psf/fsun_grisms_4750/SAP-EDR-PSF.tar.gz'
    settings = dict(source_id=source.source_id, ra_deg=source.summary['ra'], dec_deg=source.summary['dec'],
                    cutout_size=int(cutout_size), science_path=str(science_path), error_path=str(error_path),
                    psf_path=str(psf_path) if psf_path else None, psf_archive=str(archive) if psf_path is None else None)
    if provenance.exists():
        meta = read_json(provenance)
        if meta.get('preparation_settings') != settings:
            raise ValueError('Imaging inputs or cutout settings changed. Choose a new IMAGING_RUN_NAME.')
        for path, digest in meta['input_sha256'].items():
            if sha256(path) != digest:
                raise ValueError(f'Imaging input checksum changed: {path}')
        with np.load(output/'cutout.npz') as a:
            _check_aperture(a['image'].shape, a['center_xy'], float(a['pixel_scale_arcsec']), fit_radius)
        return output
    for path in (science_path, error_path, psf_path or archive):
        if not path.exists():
            raise FileNotFoundError(f'Required F150W input missing: {path}. '
                'Supply science/error/psf paths or set DINGO_F150W_SCI, DINGO_F150W_ERR, DINGO_F150W_PSF.')
    coord = SkyCoord(source.summary['ra'], source.summary['dec'], unit='deg')
    with fits.open(science_path, memmap=True) as hs, fits.open(error_path, memmap=True) as he:
        sci_hdu = hs['SCI'] if 'SCI' in hs else hs[0]
        err_hdu = he['ERR'] if 'ERR' in he else he[0]
        if sci_hdu.data.shape != err_hdu.data.shape:
            raise ValueError('The F150W science and error mosaics must have matching shapes.')
        wcs = WCS(sci_hdu.header, fix=False).celestial
        position = wcs.world_to_pixel(coord)
        if 'CTYPE1' in err_hdu.header and 'CTYPE2' in err_hdu.header:
            ewcs = WCS(err_hdu.header, fix=False).celestial
            probes = np.array([position, np.asarray(position)+[cutout_size/2, 0],
                               np.asarray(position)+[0, cutout_size/2]])
            mapped = np.stack(ewcs.world_to_pixel(wcs.pixel_to_world(probes[:, 0], probes[:, 1])), axis=-1)
            if not np.allclose(mapped, probes, atol=1e-4, rtol=0):
                raise ValueError('The F150W error mosaic is not aligned with the science mosaic.')
        cut = Cutout2D(sci_hdu.data, position, (cutout_size, cutout_size), wcs=wcs, mode='strict', copy=True)
        errors = np.array(err_hdu.data[cut.slices_original], dtype=float, copy=True)
        scale = float(np.mean(proj_plane_pixel_scales(wcs))*3600)
        unit = sci_hdu.header.get('BUNIT', '')
    center = np.asarray(cut.wcs.world_to_pixel(coord))
    _check_aperture(cut.data.shape, center, scale, fit_radius)
    yy, xx = np.indices(cut.data.shape)
    central = np.hypot(xx-center[0], yy-center[1])*scale < .5
    valid = np.isfinite(cut.data) & np.isfinite(errors) & (errors > 0)
    if valid[central].mean() < .8:
        raise ValueError('Insufficient valid F150W coverage at this source.')
    output.mkdir(parents=True, exist_ok=True)
    with (output/'cutout.tmp').open('wb') as stream:
        np.savez_compressed(stream, image=cut.data, error=errors, pixel_scale_arcsec=scale,
                            center_xy=center, wcs_header=cut.wcs.to_header().tostring())
    (output/'cutout.tmp').replace(output/'cutout.npz')
    member = 'SAP-EDR-PSF/4750_f150w_ePSF.fits'
    if psf_path:
        psf_bytes = psf_path.read_bytes()
    else:
        with tarfile.open(archive) as tar:
            with tar.extractfile(member) as stream:
                psf_bytes = stream.read()
    (output/'F150W_ePSF_original.fits').write_bytes(psf_bytes)
    meta = settings | dict(band='F150W', preparation_settings=settings, pixel_scale_arcsec=scale,
        unit=unit, source_pixel_in_cutout=center.tolist(), psf_archive_member=member if not psf_path else None,
        valid_pixel_fraction=float(valid.mean()), source_config=str(source.config),
        source_config_sha256=sha256(source.config),
        input_sha256={str(p): sha256(p) for p in (output/'cutout.npz', output/'F150W_ePSF_original.fits')})
    write_json(provenance, meta)
    return output


def _check_aperture(shape, center, scale, radius):
    extent = min(center[0]+.5, center[1]+.5, shape[1]-.5-center[0], shape[0]-.5-center[1])*scale
    if not np.isfinite([scale, extent]).all() or scale <= 0 or radius >= extent:
        raise ValueError('The fitting aperture must fit inside the cutout. Increase cutout_size or reduce fit_radius.')


def _recoverable_native_crash(exc, variant, source_id):
    if not any(f'code {code}.' in str(exc) for code in (-11, 139)):
        return False
    try:
        report = read_json(variant/'imaging.json')
        name = report['checkpoint_file']
        return (report['source_id'] == source_id and isinstance(name, str)
                and Path(name).name == name and name.startswith('checkpoint_')
                and sha256(variant/name) == report['checkpoint_sha256'])
    except (OSError, ValueError, KeyError):
        return False


def prepare_imaging(source_id, *, run=True, run_name='pysersic_v1', bundle=None, results=None,
                    config=None, imaging=None, device='cuda', warmup=1000, samples=2000,
                    cutout_size=181, fit_radius=2., chunk=50, seed=735,
                    science=None, error=None, psf=None, override=False, preview_callback=None):
    """Reuse an accepted prior or run full/core PySersic fits in a fresh directory.

    ``run=False`` only reads existing files. Explicit ``imaging`` selects an
    existing prepared directory or the location for a new fit. Otherwise named
    retries are under ID*/imaging_runs/run_name, leaving historical files intact.
    Both variants are attempted even if one fails, then returned for inspection.
    With ``run=True, override=True``, clear the named imaging_runs output and
    refit both variants, ignoring accepted historical fits or explicit imaging
    paths. ``run=False`` remains read-only regardless of override.
    ``preview_callback(result)`` displays the prepared input before a worker
    starts, and is also called when opening cached imaging results.
    """
    source = RealSource(source_id,
        bundle=bundle or os.environ.get('DINGO_BUNDLE', DEFAULT_BUNDLE),
        results=results or os.environ.get('DINGO_RESULTS', DEFAULT_RESULTS), config=config)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', str(run_name)):
        raise ValueError('Use letters, digits, underscores or dashes for IMAGING_RUN_NAME.')
    if not isinstance(override, bool):
        raise ValueError('override must be True or False.')
    if device not in ('cuda', 'cpu'):
        raise ValueError('device must be cuda or cpu.')
    for name, value in dict(warmup=warmup, samples=samples, chunk=chunk).items():
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'{name} must be a positive integer.')
    fresh = run and override
    existing = (source.directory/'imaging_runs'/run_name if fresh else
                Path(imaging).expanduser().resolve() if imaging else imaging_directory(source))
    if not fresh and imaging_ready(existing, source.source_id):
        print(f'Using accepted F150W PySersic fits for ID{source.source_id}: {existing}', flush=True)
        result = ImagingResults(existing, source.source_id)
        if preview_callback is not None:
            preview_callback(result)
        return result
    output = (existing if imaging and not fresh else source.directory/'imaging_runs'/run_name)
    if not run:
        if not output.exists() and existing.exists():
            output = existing
        result = ImagingResults(output, source.source_id)
        if preview_callback is not None:
            preview_callback(result)
        return result
    if not fresh and imaging and (output/'input_provenance.json').exists():
        provenance = read_json(output/'input_provenance.json')
        if 'preparation_settings' not in provenance:
            raise ValueError('This is a historical imaging directory. Leave IMAGING=None and choose '
                             'a new IMAGING_RUN_NAME to retry without overwriting its inputs.')
    from .run_output import output_run
    with output_run(output, source, kind='imaging', override=bool(fresh)):
        prepare_inputs(source, output, cutout_size=cutout_size, fit_radius=fit_radius,
                       science=science, error=error, psf=psf)
        if preview_callback is not None:
            preview_callback(ImagingResults(output, source.source_id))
        return _fit_imaging(source, output, cutout_size=cutout_size, fit_radius=fit_radius,
                            warmup=warmup, samples=samples, chunk=chunk, seed=seed, device=device)


def _fit_imaging(source, output, *, cutout_size, fit_radius, warmup, samples, chunk, seed, device):
    print(f'ID{source.source_id}: F150W PySersic, {cutout_size}×{cutout_size} pixels, '
          f'{fit_radius:g} arcsec fitting radius. Four chains per full/core variant.\n'
          f'Imaging sampling: {warmup} warmup + {samples} retained draws per chain.\n{output}', flush=True)
    errors = {}
    for name, core in VARIANTS:
        variant = output/name
        args = ['-m', 'dingo.imaging_fit', '--inputs', str(output), '--output', str(variant),
                '--mask-core', str(core), '--fit-radius', str(fit_radius), '--warmup', str(warmup),
                '--samples', str(samples), '--chunk', str(chunk), '--seed', str(seed), '--device', device, '--resume']
        for attempt in range(2):
            try:
                run_command(args, output/f'{name}.log', device=device)
                break
            except RuntimeError as exc:
                if attempt == 0 and _recoverable_native_crash(exc, variant, source.source_id):
                    print(f'{name}: native JAX worker crashed. Retrying once from its verified checkpoint.', flush=True)
                    continue
                errors[name] = str(exc)
                print(f'{name} did not finish: {exc}\nKeeping its files and continuing the imaging review.', flush=True)
                break
    return ImagingResults(output, source.source_id, errors=errors)


class ImagingResults:
    """Source-checked CPU display, including failed or interrupted fits."""
    def __init__(self, path, source_id, *, errors=None):
        self.path, self.source_id = Path(path), int(source_id)
        self.errors = errors or {}
        self.reports = {}
        provenance = self.path/'input_provenance.json'
        if provenance.exists() and read_json(provenance).get('source_id') != self.source_id:
            raise ValueError('Imaging input directory belongs to a different source.')
        for name, _ in VARIANTS:
            path = self.path/name/'imaging.json'
            if path.exists():
                report = read_json(path)
                if report.get('source_id') != self.source_id:
                    raise ValueError('Imaging report belongs to a different source.')
                self.reports[name] = report

    @property
    def ready(self):
        return imaging_ready(self.path, self.source_id)

    def require_ready(self):
        if not self.ready:
            raise ValueError(f'ID{self.source_id}: PySersic fits are not yet accepted. '
                'Inspect the imaging table and residuals above. Increase imaging draws for poor mixing; '
                'set OVERRIDE=True to restart after changing warmup, aperture or model inputs. '
                'A grism search requesting this imaging prior must wait for an accepted imaging fit. '
                f'Reports and logs: {self.path}')
        return self.path

    def summary(self):
        import pandas as pd
        rows = []
        for name, _ in VARIANTS:
            r = self.reports.get(name, {})
            d = r.get('diagnostics') or {}
            q = (r.get('quantiles_5_50_95') or {}).get('q', [None, None, None])
            ready = _variant_ready(self.path/name, self.source_id)
            rows.append(dict(fit=name, completed=r.get('completed', False), ready=ready,
                max_Rhat=d.get('max_rank_rhat'), min_bulk_ESS=d.get('min_bulk_ess'),
                min_tail_ESS=d.get('min_tail_ess'), divergences=d.get('divergences'),
                q_median=q[1], residual_RMS=r.get('residual_rms_inside_mask'),
                parameter_checks_passed=r.get('orientation_parameterization_ok'),
                note=self.errors.get(name, 'Not run' if not r else
                    'Inspect residuals; passing sampler checks does not validate a single Sérsic galaxy model.')))
        return pd.DataFrame(rows)

    def _report(self, variant):
        if variant not in dict(VARIANTS):
            raise ValueError(f'Unknown imaging variant: {variant}')
        return self.reports.get(variant, {})

    def _check_inputs(self, report=None):
        provenance = self.path/'input_provenance.json'
        metadata = read_json(provenance) if provenance.exists() else {}
        if metadata and metadata.get('source_id') != self.source_id:
            raise ValueError('Imaging input directory belongs to a different source.')
        checks = dict(metadata.get('input_sha256', {}))
        checks.update((report or {}).get('input_sha256', {}))
        for filename, expected in checks.items():
            if sha256(filename) != expected:
                raise ValueError(f'Imaging input checksum changed: {filename}')
        return metadata

    def _physical_samples(self, variant='full_pixel'):
        """Read constrained PySersic coordinates, retaining chain/draw axes."""
        report = self._report(variant)
        self._check_inputs(report)
        path = self.path/variant/'samples.npz'
        if not path.exists():
            raise FileNotFoundError(f'No saved PySersic samples yet: {path}')
        if report.get('samples_sha256') and sha256(path) != report['samples_sha256']:
            raise ValueError(f'Imaging samples checksum changed: {path}')
        names = ('xc', 'yc', 'flux', 'r_eff', 'n', 'ellip', 'theta', 'sky_back')
        with np.load(path, allow_pickle=False) as arrays:
            missing = [name for name in names if name not in arrays]
            if missing:
                raise ValueError(f'Saved PySersic samples are missing physical parameters: {missing}')
            samples = {name: np.asarray(arrays[name], dtype=float) for name in names}
            if 'q' in arrays and not np.allclose(arrays['q'], 1-samples['ellip'], atol=1e-10):
                raise ValueError('Saved axis ratio does not agree with PySersic ellipticity.')
        shape = samples['xc'].shape
        if (len(shape) != 2 or min(shape) < 1
                or any(a.shape != shape or not np.isfinite(a).all() for a in samples.values())):
            raise ValueError('PySersic samples must be finite arrays with matching chain/draw axes.')
        if (np.any((samples['ellip'] < 0) | (samples['ellip'] >= 1))
                or np.any(samples['r_eff'] <= 0) or np.any(samples['n'] <= 0)):
            raise ValueError('Saved PySersic shape parameters are outside their physical domain.')
        return samples

    def inclination_samples(self, variant='full_pixel', *, q0=0.):
        """Derive stellar inclination in degrees for a specified intrinsic q0.

        Inclination is not a fitted PySersic parameter. q0=0 assumes a thin
        circular disk; finite q0 assumes an oblate disk with known thickness.
        No stellar-to-gas alignment uncertainty is added to these draws.
        """
        if not np.isfinite(q0) or not 0 <= q0 < 1:
            raise ValueError('Intrinsic disk axis ratio q0 must be between 0 and 1.')
        q = 1-self._physical_samples(variant)['ellip']
        if np.any(q < q0 - 4*np.finfo(float).eps):
            raise ValueError('Some measured axis ratios are below q0; this fixed disk thickness '
                             'is incompatible with those draws. Choose a smaller q0.')
        cos_squared = (q*q-q0*q0)/(1-q0*q0)
        return np.rad2deg(np.arccos(np.sqrt(np.clip(cos_squared, 0., 1.))))

    def inclination_summary(self, variant='full_pixel', *, q0_values=(0., .2)):
        """Compare fixed disk thicknesses using the same imaging posterior draws.

        Intervals are 16/50/84 percentiles before any stellar/gas scatter. If
        draws have b/a below a chosen thickness, report their fraction and no
        interval for that assumption; do not silently discard or clip them.
        """
        import pandas as pd
        thicknesses = np.asarray(q0_values, dtype=float)
        if (thicknesses.ndim != 1 or not thicknesses.size
                or not np.isfinite(thicknesses).all()
                or np.any((thicknesses < 0) | (thicknesses >= 1))):
            raise ValueError('q0_values must contain fixed disk thicknesses between 0 and 1.')
        q = 1-self._physical_samples(variant)['ellip']
        rows = []
        for q0 in thicknesses:
            incompatible = q < q0 - 4*np.finfo(float).eps
            low, median, high = np.nan, np.nan, np.nan
            if not incompatible.any():
                cos_squared = (q*q-q0*q0)/(1-q0*q0)
                values = np.rad2deg(np.arccos(np.sqrt(np.clip(cos_squared, 0., 1.))))
                low, median, high = np.quantile(values, [.16, .5, .84])
            rows.append(dict(q0=float(q0), median_deg=median,
                minus_1sigma_deg=median-low, plus_1sigma_deg=high-median,
                incompatible_fraction=float(incompatible.mean()),
                status='incompatible' if incompatible.any() else 'compatible'))
        return pd.DataFrame(rows).set_index('q0')

    def parameter_summary(self, variant='full_pixel', *, q0=0.):
        """Native parameters and derived quantities, with pooled 16/50/84 percentiles."""
        import pandas as pd
        samples = self._physical_samples(variant)
        labels = dict(xc='Center x', yc='Center y', flux='Total Sérsic flux',
                      r_eff='Effective radius', n='Sérsic index', ellip='Ellipticity = 1 − b/a',
                      theta='PySersic angle', sky_back='Constant sky level')
        units = dict(xc='pixel', yc='pixel', flux='normalized image sum', r_eff='pixel',
                     n='', ellip='', theta='rad', sky_back='normalized image value')
        samples['q'] = 1-samples['ellip']
        samples['inclination_deg'] = self.inclination_samples(variant, q0=q0)
        labels.update(q='Axis ratio b/a (derived)',
                      inclination_deg=f'Stellar inclination (derived, q0={q0:g})')
        units.update(q='', inclination_deg='deg')
        rows = []
        for name, values in samples.items():
            low, median, high = np.quantile(values, [.16, .5, .84])
            rows.append(dict(parameter=name, meaning=labels[name], median=median,
                             minus_1sigma=median-low, plus_1sigma=high-median,
                             p16=low, p84=high, unit=units[name]))
        table = pd.DataFrame(rows).set_index('parameter')
        table.attrs['note'] = ('Flux and sky use the normalized image fitted by PySersic: '
            'divide the calibrated image by the median ERR of fitted pixels. '
            'Inclination assumes a circular oblate disk; these are conditional statistical errors.')
        return table

    def input_figure(self, variant='full_pixel', *, fit_radius=None, save=False):
        """Display the actual calibrated science cutout, including the fitting mask.

        This also works before sampling starts. For a new run, fit_radius must
        match the aperture passed to prepare_imaging (default 2 arcsec).
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import AsinhNorm
        report = self._report(variant)
        metadata = self._check_inputs(report)
        fit_path = self.path/variant/'fit.npz'
        if fit_path.exists():
            if report.get('fit_sha256') and sha256(fit_path) != report['fit_sha256']:
                raise ValueError(f'Imaging prediction checksum changed: {fit_path}')
            with np.load(fit_path, allow_pickle=False) as a:
                data, error, mask = (a[k] for k in ('image', 'error', 'mask'))
            valid = ~mask.astype(bool)
        else:
            with np.load(self.path/'cutout.npz', allow_pickle=False) as a:
                data, error, center = a['image'], a['error'], a['center_xy']
                scale = float(a['pixel_scale_arcsec'])
            args = report.get('arguments', {})
            radius = float(args.get('fit_radius', 2.) if fit_radius is None else fit_radius)
            if not np.isfinite(radius) or radius <= 0:
                raise ValueError('fit_radius must be a positive finite number.')
            core = float(args.get('mask_core', dict(VARIANTS)[variant]))
            yy, xx = np.indices(data.shape)
            distance = np.hypot(xx-center[0], yy-center[1])*scale
            valid = (np.isfinite(data) & np.isfinite(error) & (error > 0)
                     & (distance <= radius) & (distance >= core))
        if data.ndim != 2 or error.shape != data.shape or valid.shape != data.shape:
            raise ValueError('Imaging input arrays have inconsistent shapes.')
        if not valid.any():
            raise ValueError('No valid imaging pixels in the fitting aperture.')
        low, high = np.quantile(data[valid], [.02, .995])
        norm = AsinhNorm(linear_width=max((high-low)/20, 1e-12),
                         vmin=low, vmax=max(high, low+1e-12))
        fig, ax = plt.subplots(figsize=(6.5, 6), layout='constrained')
        image = ax.imshow(np.where(valid, data, np.nan), origin='lower', norm=norm,
                          cmap=plt.get_cmap('gray').with_extremes(bad='.7'))
        ax.set(title=f'ID{self.source_id}: F150W image fitted by PySersic\n'
                     f'{variant}; gray pixels are excluded',
               xlabel='Image x [pixel]', ylabel='Image y [pixel]')
        bar = fig.colorbar(image, ax=ax, shrink=.8, ticks=norm.inverse([.15, .5, .85]), format='%.3g')
        bar.set_label(metadata.get('unit') or 'Calibrated image units')
        bar.minorticks_off()
        self._save_review_figure(fig, f'{variant}_input', save)
        return fig

    def corner(self, variant='full_pixel', *, save=False):
        """Use PySersic's default corner plot on saved constrained samples.

        Reconstructs only its display container, not a fitter or sampler. Base
        variables are omitted. The stored angle chart is retained to avoid a
        modulo wrap splitting a single posterior across opposite plot edges.
        No contour, label, quantile, or title styling is overridden here.
        """
        import arviz as az
        import warnings
        with warnings.catch_warnings():
            # PySersic's old type annotation probes this removed ArviZ alias;
            # the display container below already supports the new DataTree.
            warnings.filterwarnings('ignore', message='arviz.InferenceData is no longer available.*')
            from pysersic.results import PySersicResults
        samples = self._physical_samples(variant)
        native = PySersicResults(data=None, rms=None, psf=None, loss_func=None,
                                renderer=None, mask=None)
        try:
            native.idata = az.from_dict(posterior=samples)
        except TypeError:
            # ArviZ >= 1 uses a nested group dictionary and an xarray DataTree.
            native.idata = az.from_dict({'posterior': samples})
        native.runtype = 'sampling (saved constrained samples; display only)'
        fig = native.corner()
        self._save_review_figure(fig, f'{variant}_corner', save)
        return fig

    def inclination_figure(self, variant='full_pixel', *, q0=0., save=False):
        """Show the measured axis ratio and its conditional inclination transform."""
        import matplotlib.pyplot as plt
        q = 1-self._physical_samples(variant)['ellip']
        inclination = self.inclination_samples(variant, q0=q0)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout='constrained')
        for ax, values, label in zip(axes, (q, inclination),
                                    ('Axis ratio b/a', 'Stellar inclination [deg]')):
            low, median, high = np.quantile(values, [.16, .5, .84])
            minus, plus = median-low, high-median
            positive = [error for error in (minus, plus) if error > 0]
            precision = min(8, max(0, 1-int(np.floor(np.log10(min(positive)))))) if positive else 3
            ax.hist(values.ravel(), bins=40, density=True, histtype='step', color='black')
            for quantile in (low, median, high):
                ax.axvline(quantile, color='black', linestyle='-' if quantile == median else ':', linewidth=.8)
            ax.set(xlabel=label, ylabel='Posterior density',
                   title=rf'${median:.{precision}f}_{{-{minus:.{precision}f}}}^{{+{plus:.{precision}f}}}$')
        geometry = 'thin circular disk' if q0 == 0 else f'oblate circular disk, intrinsic q0={q0:g}'
        fig.suptitle(f'ID{self.source_id}: {variant}; inclination derived assuming a {geometry}\n'
                     'Formal imaging uncertainty; stellar–gas misalignment is not included.')
        self._save_review_figure(fig, f'{variant}_inclination', save)
        return fig

    def _save_review_figure(self, fig, name, save):
        if save:
            directory = self.path/'figures'
            directory.mkdir(parents=True, exist_ok=True)
            for suffix in ('png', 'pdf'):
                fig.savefig(directory/f'{name}.{suffix}', dpi=150, bbox_inches='tight')

    def figure(self):
        import matplotlib.pyplot as plt
        from matplotlib.colors import AsinhNorm
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout='constrained')
        for row, (name, _) in enumerate(VARIANTS):
            fit_path = self.path/name/'fit.npz'
            if not fit_path.exists():
                for ax in axes[row]:
                    ax.text(.5, .5, f'{name}: no saved model yet', transform=ax.transAxes, ha='center')
                    ax.set_axis_off()
                continue
            expected = self.reports.get(name, {}).get('fit_sha256')
            if expected and sha256(fit_path) != expected:
                raise ValueError(f'Imaging prediction checksum changed: {fit_path}')
            with np.load(fit_path) as a:
                data, model, residual, mask = (a[k] for k in ('image', 'model', 'residual', 'mask'))
                err = a['error']
            if any(v.shape != data.shape for v in (model, residual, mask, err)):
                raise ValueError('Imaging prediction arrays have inconsistent shapes.')
            valid = ~mask.astype(bool)
            np.testing.assert_allclose(residual[valid], (data[valid]-model[valid])/err[valid], rtol=1e-6, atol=1e-6)
            low, high = np.nanquantile(data, [.02, .995])
            norm = AsinhNorm(linear_width=max((high-low)/20, 1e-12), vmin=low, vmax=max(high, low+1e-12))
            for col, (values, label) in enumerate(zip((data, model, np.where(valid, residual, np.nan)),
                    ('F150W observed', 'PySersic model', '(Observed − model) / ERR'))):
                ax = axes[row, col]
                style = (dict(cmap=plt.get_cmap('RdBu_r').with_extremes(bad='.7'), vmin=-5, vmax=5)
                         if col == 2 else dict(cmap='magma', norm=norm))
                im = ax.imshow(values, origin='lower', **style)
                if valid.any() and not valid.all():
                    ax.contour(valid, levels=[.5], colors='lime', linewidths=.7)
                ax.set(title=f'{name}\n{label}', xlabel='Image x [pixel]', ylabel='Image y [pixel]')
                colorbar = fig.colorbar(im, ax=ax, shrink=.75,
                    **(dict(ticks=norm.inverse([.15, .5, .85]), format='%.3g') if col < 2 else {}))
                colorbar.minorticks_off()
        fig.suptitle(f'ID{self.source_id}: stellar imaging only; gas morphology remains independent\n'
                     'Green boundary: fitted pixels. Inspect structure in residuals before using the inclination prior.')
        return fig
