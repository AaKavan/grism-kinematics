"""Reusable, source-level single-Sérsic imaging fits for notebook workflows.

The stellar fit is independent of a particular kinematic experiment. Each
explicit refit receives a new directory, so old experiments keep referencing
the exact posterior they used. Importing this module does not initialize JAX.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import os
from pathlib import Path
import re

import numpy as np

from .real_workflow import DEFAULT_BUNDLE, RealSource, read_json, sha256, write_json
from .rotation_imaging import ImagingResults, prepare_imaging


MODEL = 'One PSF-convolved Sérsic component plus a constant sky level'
VARIANTS = ('full_pixel',)
CHAINS = 4


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _path(value):
    return Path(value).expanduser().resolve()


def _file_identity(path):
    """Identify a large raw input without rereading a multi-GB mosaic."""
    path = _path(path)
    stat = path.stat()
    if not path.is_file():
        raise ValueError(f'Expected an imaging input file: {path}')
    return dict(path=str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def _signature(source, *, science, error, psf, settings):
    default = Path.home()/'EDR_data'
    science = _path(science or os.environ.get(
        'DINGO_F150W_SCI', default/'sapphires_edr_nircam_sci/4750_F150W_v05_sci.fits'))
    error = _path(error or os.environ.get(
        'DINGO_F150W_ERR', default/'sapphires_edr_nircam_err/4750_F150W_v05_err.fits'))
    psf = psf or os.environ.get('DINGO_F150W_PSF')
    psf = _path(psf) if psf else None
    archive = source.bundle/'calibration/psf/fsun_grisms_4750/SAP-EDR-PSF.tar.gz'
    raw = dict(science=_file_identity(science), error=_file_identity(error),
               psf=_file_identity(psf or archive))
    versions = {}
    for package in ('pysersic', 'numpyro', 'jax', 'jaxlib', 'numpy', 'scipy'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'not installed'
    # These are the same scientific files that protect the worker checkpoint.
    code = {name: sha256(Path(__file__).parent/name) for name in
            ('imaging_fit.py', 'rotation_fit.py', 'real_workflow.py')}
    identity = dict(source_id=source.source_id, band='F150W', model=MODEL,
        config_path=str(source.config), config_sha256=sha256(source.config),
        ra_deg=float(source.summary['ra']), dec_deg=float(source.summary['dec']),
        raw_inputs=raw, psf_archive_member=None if psf else 'SAP-EDR-PSF/4750_f150w_ePSF.fits',
        settings=settings, code_sha256=code, versions=versions)
    return identity, dict(science=science, error=error, psf=psf)


@contextmanager
def _cache_lock(directory):
    """Serialize shared-cache operations and detect workers from dead kernels."""
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory/'.morphology.lock'
    if lock_path.is_symlink():
        raise ValueError(f'The morphology lock must not be a symlink: {lock_path}')
    with lock_path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another morphology job is using {directory}. Wait for it to finish.') from exc
        # A subprocess can survive its notebook kernel. Its own lock is the
        # reliable indication that it is still writing, even after our lock
        # was released when that kernel died.
        for path in directory.glob('fit_*/full_pixel/.fit.lock'):
            with path.open('a') as worker:
                try:
                    fcntl.flock(worker, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeError(f'An imaging worker is still running in {path.parent}. '
                                       'Wait for it to finish before reusing or refitting morphology.') from exc
                fcntl.flock(worker, fcntl.LOCK_UN)
        yield


def _selected_fit(directory):
    pointer = directory/'current.json'
    if not pointer.exists():
        return None
    name = read_json(pointer).get('fit_directory', '')
    if not isinstance(name, str) or not re.fullmatch(r'fit_[0-9]+', name):
        raise ValueError(f'Invalid morphology cache pointer: {pointer}')
    fit = directory/name
    if fit.is_symlink() or not (fit/'morphology.json').is_file():
        raise ValueError(f'The selected morphology fit is missing or unsafe: {fit}')
    return fit


def _new_fit(directory):
    numbers = [int(path.name[4:]) for path in directory.iterdir()
               if re.fullmatch(r'fit_[0-9]+', path.name)]
    path = directory/f'fit_{max(numbers, default=0)+1:03d}'
    path.mkdir()
    return path


def _atomic_text(path, contents):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(contents)
    temporary.replace(path)


def _clean(value):
    """Keep unavailable diagnostic values readable and JSON standards compliant."""
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _collect_result(metadata, result):
    """Record only completed work evidenced by a worker report or saved samples."""
    report = result.reports.get('full_pixel', {})
    diagnostics = report.get('diagnostics') or {}
    counts = dict(chains=diagnostics.get('chains'),
                  warmup_per_chain=int(report.get('warmup_completed', 0)),
                  retained_draws_per_chain=int(report.get('retained_draws_per_chain', 0)))
    metadata.update(worker_completed=bool(report.get('completed')), completed=counts,
                    diagnostics=diagnostics,
                    orientation_parameterization_ok=report.get('orientation_parameterization_ok'),
                    worker_phase=report.get('phase', 'not started'))
    if result.errors:
        metadata['worker_errors'] = dict(result.errors)
    provenance = result.path/'input_provenance.json'
    if provenance.exists():
        metadata['prepared_input_provenance'] = read_json(provenance)
    if (result.path/'full_pixel/samples.npz').exists():
        samples = result._physical_samples('full_pixel')
        counts['chains'], counts['retained_draws_per_chain'] = samples['xc'].shape
        parameters = result.parameter_summary('full_pixel', q0=0.)
        inclinations = result.inclination_summary('full_pixel', q0_values=(0., .2))
        metadata['parameter_summary'] = parameters.reset_index().to_dict(orient='records')
        metadata['inclination_summary'] = inclinations.reset_index().to_dict(orient='records')
        metadata['parameter_note'] = parameters.attrs.get('note', '')
        text = ('PySersic posterior: pooled 16th/50th/84th percentiles of retained draws\n'
                'Intervals are conditional statistical uncertainties; inspect convergence first.\n\n'
                +parameters.to_string(float_format=lambda value: f'{value:.7g}')+'\n\n'
                +metadata['parameter_note']+'\n\n'
                'Inclination is derived from axis ratio, not sampled by PySersic.\n'
                'cos(i)^2 = ((b/a)^2 - q0^2) / (1 - q0^2)\n'
                'Any draws with b/a < q0 make that fixed thickness incompatible; no clipping.\n'
                +inclinations.to_string(float_format=lambda value: f'{value:.7g}')+'\n')
        _atomic_text(result.path/'parameters.txt', text)


def _write_report(path, metadata):
    """Save a readable report even when initialization or sampling failed."""
    metadata = _clean(metadata)
    write_json(path/'morphology.json', metadata)
    identity, settings = metadata['identity'], metadata['identity']['settings']
    done = metadata.get('completed', {})
    diagnostics = metadata.get('diagnostics', {})
    raw = identity['raw_inputs']
    gate = diagnostics.get('convergence_gate_passed')
    lines = [f'ID{identity["source_id"]} — F150W morphology fit',
        f'Model: {MODEL}', f'Fit directory: {path}', f'Status: {metadata["status"]}',
        f'First started (UTC): {metadata["started_at"]}',
        f'Finished (UTC): {metadata.get("finished_at") or ("not recorded; completed while controller was unavailable" if metadata.get("worker_completed") else "not finished")}',
        f'Last attempt started (UTC): {metadata["last_attempt_started_at"]}',
        f'Last attempt ended (UTC): {metadata.get("last_attempt_ended_at") or ("running" if metadata["status"] == "running" else "not recorded")}',
        f'Attempts: {len(metadata.get("attempts", []))}', '',
        'Sampling',
        'Backend: PySersic model / NumPyro NUTS / JAX; four vectorized chains',
        f'Requested device: {settings["device"]} (the worker verifies a GPU when cuda is requested)',
        f'Seed: {settings["seed"]} (worker transition seed = seed + 1)',
        f'Requested chains: {settings["chains"]}; saved chains: {done.get("chains") or "not available"}',
        f'Warmup per chain: requested {settings["warmup"]}; completed {done.get("warmup_per_chain", 0)}',
        f'Retained draws per chain: requested {settings["draws"]}; saved {done.get("retained_draws_per_chain", 0)}',
        f'Checkpoint chunk: {settings["chunk"]} iterations per chain',
        f'Fitting radius: {settings["fit_radius_arcsec"]} arcsec; central mask: none',
        f'Cutout: {settings["cutout_size"]} × {settings["cutout_size"]} pixels', '',
        'Inputs', f'Source config: {identity["config_path"]}',
        f'Source config SHA256: {identity["config_sha256"]}',
        f'Sky position: RA {identity["ra_deg"]} deg; Dec {identity["dec_deg"]} deg',
        f'Science image: {raw["science"]["path"]}', f'Error image: {raw["error"]["path"]}',
        f'PSF or PSF archive: {raw["psf"]["path"]}',
        f'Archive member: {identity.get("psf_archive_member") or "not applicable"}',
        'Raw input identities use path, byte size and modification time; they are not content hashes.',
        'Prepared cutout/PSF checksums, versions and scientific code checksums are in morphology.json.', '',
        'Convergence and interpretation',
        f'Sampler convergence gate: {"passed" if gate is True else "failed" if gate is False else "not available"}',
        f'Orientation/boundary checks: {metadata.get("orientation_parameterization_ok")}',
        f'Maximum rank Rhat: {diagnostics.get("max_rank_rhat")}',
        f'Minimum bulk/tail ESS: {diagnostics.get("min_bulk_ess")} / {diagnostics.get("min_tail_ess")}',
        f'Divergences: {diagnostics.get("divergences")}',
        'A completed sampler is not necessarily converged; inspect the image and residuals as well.',
        'Inclination is DERIVED from b/a assuming a circular disk, not directly fitted by PySersic.',
        'Both q0=0 and q0=0.2 are reported. Stellar/gas misalignment is not in these formal errors.', '']
    if metadata.get('error'):
        lines += ['Error: '+metadata['error'], '']
    if metadata.get('completion_observed_at'):
        lines += [f'Completion first observed (UTC): {metadata["completion_observed_at"]}',
                  'This is a later cache inspection time, not the actual run finish time.',
                  f'Completed worker report modification time (UTC): {metadata["completed_report_modified_at"]}', '']
    if metadata.get('worker_errors'):
        lines += [f'Worker error ({key}): {value}' for key, value in metadata['worker_errors'].items()]
    if metadata.get('review_errors'):
        lines += [f'Review product error ({key}): {value}' for key, value in metadata['review_errors'].items()]
    parameter_path = path/'parameters.txt'
    if parameter_path.exists():
        lines += ['', parameter_path.read_text().rstrip(), '']
    else:
        lines += ['No retained parameter samples are available yet.', '']
    lines += ['Files', 'morphology.json: machine-readable settings, status, provenance and summaries',
              'parameters.txt: native parameters and derived inclinations with 16/50/84 percentiles (when available)',
              'full_pixel/samples.npz: posterior arrays, preserving chain/draw axes (when available)',
              'full_pixel/imaging.json: worker convergence report and checkpoint identity (when available)',
              'full_pixel.log: worker progress and error messages',
              'corner.png and corner.pdf: native PySersic corner plot in this fit directory (when available)',
              'figures/: input image, fit/residual view and native PySersic corner plot (when available)', '']
    info_path = path / f'pysersic_{identity["source_id"]}_info.cat'
    _atomic_text(info_path, '\n'.join(lines))


def _input_preview(result, *, fit_radius, display):
    """Save the actual fitting image before starting expensive sampling."""
    import matplotlib.pyplot as plt
    figure = result.input_figure('full_pixel', fit_radius=fit_radius)
    for axis in figure.axes:
        for item in axis.images:
            item.set_cmap(plt.get_cmap('magma').with_extremes(bad='white'))
        if axis.images:
            axis.set_title(axis.get_title().replace('gray pixels', 'White pixels'))
    result._save_review_figure(figure, 'full_pixel_input', True)
    if display:
        from IPython.display import display as show
        show(figure)
    plt.close(figure)


def _review(result, *, display, center_reference=False, show_center_values=True):
    """Save native review plots and return plotting errors without losing a fit."""
    import matplotlib.pyplot as plt
    errors = {}
    if display:
        from IPython.display import display as show
        show(result.summary())
    fit_view = (('full_pixel_center', lambda: result.center_figure(
                    annotate_center=show_center_values)) if center_reference else
                ('full_pixel_fit', result.figure))
    for label, make in [fit_view,
                        ('full_pixel_corner', lambda: result.corner('full_pixel'))]:
        if label.endswith('corner') and not (result.path/'full_pixel/samples.npz').exists():
            continue
        try:
            figure = make()
            result._save_review_figure(figure, label, True)
            if label.endswith('corner'):
                for suffix in ('png', 'pdf'):
                    figure.savefig(result.path/f'corner.{suffix}', dpi=150, bbox_inches='tight')
            if display:
                show(figure)
            plt.close(figure)
        except Exception as exc:
            errors[label] = f'{type(exc).__name__}: {exc}'
            print(f'Could not make {label}: {exc}. Saved sampling results are retained.', flush=True)
    if display and (result.path/'parameters.txt').exists():
        parameters = result.parameter_summary('full_pixel', q0=0.)
        if not show_center_values:
            parameters = parameters.drop(index=['xc', 'yc'], errors='ignore')
        show(parameters)
        show(result.inclination_summary('full_pixel', q0_values=(0., .2)))
    return errors


def fit_morphology(source_id, *, directory, config_path, refit=False, device='cuda',
                   warmup=1000, draws=2000, seed=735, fit_radius=2., cutout_size=181,
                   chunk=50, science=None, error=None, psf=None, bundle=None, display=True,
                   center_reference=False, show_center_values=True):
    """Fit or reuse a source's F150W morphology, independently of kinematic runs.

    Parameters
    ----------
    source_id : int or str
        Catalog ID, checked against ``config_path``.
    directory : path-like
        Shared root, normally ``WORKING_DIR/<ID>/morph_fits/F150W``. Each fit
        lives in ``fit_001``, ``fit_002``, etc.; ``current.json`` selects one.
    config_path : path-like
        Source configuration containing coordinates, redshift and line.
    refit : bool
        Create a new fit directory, preserving all older results. Otherwise
        reuse the matching completed fit or resume its incomplete checkpoint.
        Changed settings/inputs require this explicit choice.
    device : {'cuda', 'cpu'}
        Device used by the PySersic/NumPyro/JAX subprocess. No GPU library is
        initialized merely by importing this module.
    warmup, draws : int
        Warmup iterations and retained draws per chain; four chains are used.
    seed : int
        Reproducible initialization seed; the transition worker uses seed+1.
    fit_radius : float
        Circular fitting aperture in arcsec; there is no core mask.
    cutout_size : int
        Odd cutout width in pixels, large enough to contain the aperture.
    chunk : int
        Iterations per checkpoint/progress update.
    science, error, psf : path-like, optional
        Calibrated F150W mosaic, matching ERR mosaic and empirical PSF. If
        omitted, use DINGO_F150W_SCI/ERR/PSF and the existing local defaults.
    bundle : path-like, optional
        Calibration bundle containing the default PSF archive when ``psf``
        is omitted. No grism exposures are read by this function.
    display : bool
        Show the input before sampling and the fit, native corner and tables
        afterward. Review products are saved even when display is false.
    center_reference : bool
        Show image/model panels annotated with posterior center coordinates
        instead of the image/model/residual view. Native corner and parameter
        tables are unchanged. This is display-only and does not change cache
        identity, sampling, or automatic acceptance of an imaging prior.
    show_center_values : bool
        Print the imaging-pixel center in the center-reference figure and
        displayed parameter table. False hides those values while retaining
        the center marker, complete native corner plot and saved parameters.
        This display-only choice does not change the cached posterior.

    Returns
    -------
    ImagingResults
        A single-full-image fit, including nonconverged completed runs for
        inspection. Call ``require_ready()`` before using an inclination prior.

    Notes
    -----
    Completed cache reads preserve original start/finish timestamps. Native
    samples retain chain and draw dimensions. Inclination is derived from axis
    ratio for q0=0 and 0.2, and is not a parameter sampled by PySersic.
    """
    if not isinstance(refit, bool) or not isinstance(display, bool):
        raise ValueError('refit and display must be True or False.')
    if not isinstance(center_reference, bool):
        raise ValueError('center_reference must be True or False.')
    if not isinstance(show_center_values, bool):
        raise ValueError('show_center_values must be True or False.')
    if device not in ('cpu', 'cuda'):
        raise ValueError('device must be cpu or cuda.')
    for name, value in dict(warmup=warmup, draws=draws, chunk=chunk).items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'{name} must be a positive integer.')
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32-1:
        raise ValueError('seed must be an integer between 0 and 2**32-2.')
    if (isinstance(cutout_size, bool) or not isinstance(cutout_size, (int, np.integer))
            or cutout_size < 9 or cutout_size % 2 != 1):
        raise ValueError('cutout_size must be an odd integer of at least 9.')
    if not np.isfinite(fit_radius) or fit_radius <= .18:
        raise ValueError('fit_radius must exceed 0.18 arcsec.')
    directory = _path(directory)
    source = RealSource(source_id, config=_path(config_path),
        bundle=bundle or os.environ.get('DINGO_BUNDLE', DEFAULT_BUNDLE),
        results=directory/'_coordination')
    settings = dict(chains=CHAINS, warmup=int(warmup), draws=int(draws), chunk=int(chunk),
                    seed=int(seed), device=device, fit_radius_arcsec=float(fit_radius),
                    cutout_size=int(cutout_size), variants=list(VARIANTS))
    identity, paths = _signature(source, science=science, error=error, psf=psf, settings=settings)
    with _cache_lock(directory):
        previous = _selected_fit(directory)
        metadata = read_json(previous/'morphology.json') if previous else None
        if metadata and not refit and metadata.get('identity') != identity:
            raise ValueError('Morphology settings, input identities or scientific code differ from the '
                             f'saved fit at {previous}. Use refit=True to create a new fit and preserve '
                             'the old posterior; do not overwrite it.')
        if previous and not refit:
            output = previous
            result = ImagingResults(output, source.source_id, variants=VARIANTS)
            result._check_inputs(result.reports.get('full_pixel'))
            if result.reports.get('full_pixel', {}).get('completed'):
                # Completed but nonconverged posteriors remain inspectable and
                # are never silently rerun or promoted to accepted priors.
                before = _clean(metadata)
                _collect_result(metadata, result)
                status = 'complete' if result.ready else 'complete_not_accepted'
                if metadata['status'] != status:
                    metadata['status'] = status
                    metadata['completion_observed_at'] = _utc_now()
                    modified = (output/'full_pixel/imaging.json').stat().st_mtime
                    metadata['completed_report_modified_at'] = datetime.fromtimestamp(
                        modified, timezone.utc).isoformat(timespec='seconds')
                    # The worker does not record a wall-clock finish time, so
                    # do not fabricate one when its controller disappeared.
                    # Keep prior attempt errors as evidence of that exit.
                print(f'Reusing F150W morphology for ID{source.source_id}: {output}', flush=True)
                _input_preview(result, fit_radius=fit_radius, display=display)
                metadata['review_errors'] = _review(result, display=display,
                    center_reference=center_reference, show_center_values=show_center_values)
                info_path = output / f'pysersic_{source.source_id}_info.cat'
                if _clean(metadata) != before or not info_path.exists():
                    _write_report(output, metadata)
                return result
        else:
            output = _new_fit(directory)
            now = _utc_now()
            metadata = dict(format='dingo_shared_morphology_v1', identity=identity,
                            started_at=now, finished_at=None, attempts=[])
        now = _utc_now()
        metadata.update(status='running', last_attempt_started_at=now,
                        last_attempt_ended_at=None, error=None, worker_errors={})
        metadata['attempts'].append(dict(started_at=now, ended_at=None, status='running'))
        _write_report(output, metadata)
        write_json(directory/'current.json', dict(source_id=source.source_id, fit_directory=output.name))
        try:
            result = prepare_imaging(source.source_id, config=source.config,
                bundle=source.bundle, results=source.results, imaging=output,
                run_name=output.name, device=device, warmup=warmup, samples=draws,
                seed=seed, cutout_size=cutout_size, fit_radius=fit_radius, chunk=chunk,
                variants=VARIANTS, **paths,
                preview_callback=lambda current: _input_preview(current, fit_radius=fit_radius, display=display))
            _collect_result(metadata, result)
            metadata['status'] = ('complete' if result.ready else 'complete_not_accepted'
                                  if metadata['worker_completed'] else 'failed')
            if metadata['worker_completed']:
                metadata['finished_at'] = _utc_now()
            metadata['review_errors'] = _review(result, display=display,
                center_reference=center_reference, show_center_values=show_center_values)
        except BaseException as exc:
            metadata.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                            error=f'{type(exc).__name__}: {exc}')
            try:
                result = ImagingResults(output, source.source_id, variants=VARIANTS)
                _collect_result(metadata, result)
            except Exception as report_error:
                metadata['report_error'] = f'{type(report_error).__name__}: {report_error}'
            raise
        finally:
            metadata['last_attempt_ended_at'] = _utc_now()
            metadata['attempts'][-1].update(ended_at=metadata['last_attempt_ended_at'],
                status=metadata['status'], error=metadata.get('error'),
                worker_errors=dict(metadata.get('worker_errors', {})))
            _write_report(output, metadata)
        info_path = output / f'pysersic_{source.source_id}_info.cat'
        print(f'PySersic report: {info_path}', flush=True)
        return result
