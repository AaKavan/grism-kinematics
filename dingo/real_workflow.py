"""Source discovery and notebook orchestration for real joint R+C fits.

Keep JAX out of the notebook process: fits/rendering run in interruptible
subprocesses, while saved pixels and plots can be inspected on a CPU.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = Path.home() / 'EDR_data/magnif_copy/sapphires_kinematics_inputs_2026-10-13'
DEFAULT_RESULTS = REPO.parent / 'results'
LEGACY_3700 = DEFAULT_RESULTS / '2026-10-15_real_ID3700'


def source_number(value):
    match = re.fullmatch(r'(?:ID)?([1-9][0-9]*)', str(value).strip(), re.I)
    if not match:
        raise ValueError('Enter a positive source ID, for example 3700 or ID3700.')
    return int(match[1])


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    """Atomic checkpoint replacement; interruption keeps the previous JSON."""
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class RealSource:
    source_id: int
    bundle: Path = DEFAULT_BUNDLE
    results: Path = DEFAULT_RESULTS
    config: Path | None = None

    def __post_init__(self):
        self.source_id = source_number(self.source_id)
        self.bundle, self.results = Path(self.bundle).expanduser().resolve(), Path(self.results).expanduser().resolve()
        self.config = Path(self.config or REPO / f'data/ID{self.source_id}/config_kinematics.yaml').resolve()
        if not self.config.exists():
            raise FileNotFoundError(f'No configuration for ID{self.source_id}: {self.config}. '
                                    'A source configuration must specify its own coordinates, redshift and line.')
        self.summary = yaml.safe_load(self.config.read_text())['summary']
        if source_number(self.summary['name']) != self.source_id:
            raise ValueError('Configuration belongs to a different source.')
        values = [self.summary[k] for k in ('ra', 'dec', 'z', 'lambda_rest')]
        if not np.isfinite(values).all() or self.summary['z'] <= 0:
            raise ValueError('Invalid coordinates, redshift or line wavelength.')

    @property
    def directory(self):
        return self.results / 'real_sources' / f'ID{self.source_id}'

    @property
    def wavelength(self):
        return self.summary['lambda_rest'] * (1 + self.summary['z'])

    def availability(self):
        """Check exact STATS-associated filenames, not just directory names."""
        from astropy.io import fits
        missing, counts = [], {}
        for pupil in ('R', 'C'):
            path = self.bundle / f'grism/target_spectra/GRISM{pupil}/spec_2d_M0416_F444W_ID{self.source_id}_{pupil}.fits'
            if not path.exists():
                missing.append(str(path)); counts[pupil] = 0
                continue
            with fits.open(path) as h:
                rows = h['STATS'].data.copy()
            counts[pupil] = len(rows)
            for row in rows:
                stem = str(row['name']).strip()
                if Path(stem).name != stem:
                    raise ValueError('Invalid exposure filename in STATS.')
                rate = self.bundle / f'grism/GRISM{pupil}/rate_lv1.5/{stem}_rate_lv1.5.fits'
                if not rate.exists():
                    missing.append(str(rate))
        supported = self.summary['filter'] == 'F444W' and 3.8 <= self.wavelength <= 5.1
        return {'source_id': self.source_id, 'z': self.summary['z'], 'line_um': self.wavelength,
                'R_exposures': counts['R'], 'C_exposures': counts['C'],
                'ready': bool(not missing and all(counts.values()) and supported),
                'missing': missing, 'supported_line': supported}

    def require_data(self):
        status = self.availability()
        if not status['ready']:
            raise FileNotFoundError('Native joint R+C inputs unavailable: ' + json.dumps(status, indent=2))

    def imaging_directory(self):
        current = self.directory / 'imaging'
        if (current / 'full_pixel/imaging.json').exists():
            return current
        if self.source_id == 3700 and (LEGACY_3700 / 'imaging').exists():
            return LEGACY_3700 / 'imaging'
        return current

    def saved_runs(self):
        paths = sorted(self.directory.glob('*/grid.json'), key=lambda p: p.stat().st_mtime, reverse=True)
        if self.source_id == 3700 and (LEGACY_3700 / 'grid/grid.json').exists():
            paths.append(LEGACY_3700 / 'grid/grid.json')
        rows = []
        for path in paths:
            report = read_json(path)
            if report['source_id'] != self.source_id:
                raise ValueError(f'Wrong source in saved results: {path}')
            fits = report.get('best_by_velocity', [])
            winner = min(fits, key=lambda f: f['objective']) if fits else {}
            rows.append({'run': path.parent.name, 'path': str(path.parent),
                'completed': report.get('completed', False), 'trial_velocities': len(fits),
                'best_velocity_kms': winner.get('vrot_0p5arcsec_kms'),
                'center': report.get('settings', {}).get('center', 'catalog')})
        return rows

    def latest_run(self):
        runs = self.saved_runs()
        return Path(next((r['path'] for r in runs if r['completed']), runs[0]['path'] if runs else '')) if runs else None

    def imaging_status(self):
        root = self.imaging_directory()
        rows = []
        for name in ('full_pixel', 'core_pixel_clean'):
            path = root / name / 'imaging.json'
            if not path.exists():
                rows.append({'fit': name, 'ready': False, 'path': str(path)})
                continue
            r = read_json(path)
            same = r['source_id'] == self.source_id
            gate = r.get('diagnostics', {}).get('convergence_gate_passed', False)
            rows.append({'fit': name, 'ready': bool(same and r['completed'] and gate and r['orientation_parameterization_ok']),
                         'q_5_50_95': r.get('quantiles_5_50_95', {}).get('q'), 'path': str(path)})
        return rows

    def prepare_imaging(self):
        """Read only the actual source's F150W mosaic cutout and supplied ePSF."""
        from astropy.coordinates import SkyCoord
        from astropy.io import fits
        from astropy.nddata import Cutout2D
        from astropy.wcs import WCS
        from astropy.wcs.utils import proj_plane_pixel_scales
        import astropy.units as u
        out = self.directory / 'imaging'
        provenance = out / 'input_provenance.json'
        if provenance.exists():
            meta = read_json(provenance)
            if meta['source_id'] != self.source_id or meta['ra_deg'] != self.summary['ra'] or meta['dec_deg'] != self.summary['dec']:
                raise ValueError('Imaging inputs changed. Use a fresh results directory.')
            if not all(sha256(p) == digest for p, digest in meta['input_sha256'].items()):
                raise ValueError('Saved imaging input checksum mismatch.')
            return out
        science_path = Path.home() / 'EDR_data/sapphires_edr_nircam_sci/4750_F150W_v05_sci.fits'
        error_path = Path.home() / 'EDR_data/sapphires_edr_nircam_err/4750_F150W_v05_err.fits'
        archive = self.bundle / 'calibration/psf/fsun_grisms_4750/SAP-EDR-PSF.tar.gz'
        for path in (science_path, error_path, archive):
            if not path.exists():
                raise FileNotFoundError(f'Required F150W input missing: {path}')
        coord = SkyCoord(self.summary['ra'], self.summary['dec'], unit='deg')
        with fits.open(science_path, memmap=True) as hs, fits.open(error_path, memmap=True) as he:
            wcs = WCS(hs['SCI'].header, fix=False).celestial
            position = wcs.world_to_pixel(coord)
            # Cutout2D strict mode rejects sources outside this mosaic; never
            # substitute another object's cutout or fill a non-overlap with zeros.
            cut = Cutout2D(hs['SCI'].data, position, (121, 121), wcs=wcs, mode='strict', copy=True)
            error = np.array(he[0].data[cut.slices_original], dtype=float, copy=True)
            scale = float(np.mean(proj_plane_pixel_scales(wcs)) * 3600)
            unit = hs['SCI'].header.get('BUNIT', '')
        xy = cut.wcs.world_to_pixel(coord)
        yy, xx = np.indices(cut.data.shape)
        central = np.hypot(xx-xy[0], yy-xy[1])*scale < .5
        if np.mean(np.isfinite(cut.data[central]) & np.isfinite(error[central]) & (error[central] > 0)) < .8:
            raise ValueError('Insufficient valid F150W coverage at this source.')
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / 'cutout.npz', image=cut.data, error=error,
                            pixel_scale_arcsec=scale, center_xy=np.array(xy),
                            wcs_header=cut.wcs.to_header().tostring())
        member = 'SAP-EDR-PSF/4750_f150w_ePSF.fits'
        with tarfile.open(archive) as tar:
            (out / 'F150W_ePSF_original.fits').write_bytes(tar.extractfile(member).read())
        meta = {'source_id': self.source_id, 'ra_deg': self.summary['ra'], 'dec_deg': self.summary['dec'],
                'band': 'F150W', 'science_path': str(science_path), 'error_path': str(error_path),
                'source_pixel_in_cutout': [float(x) for x in xy], 'pixel_scale_arcsec': scale, 'unit': unit,
                'psf_archive': str(archive), 'psf_archive_member': member,
                'input_sha256': {str(p): sha256(p) for p in (out / 'cutout.npz', out / 'F150W_ePSF_original.fits')}}
        write_json(provenance, meta)
        return out

    def fit_imaging(self, device='cuda', warmup=500, samples=1000):
        if all(r['ready'] for r in self.imaging_status()):
            print(f'Using completed F150W fits for ID{self.source_id}.')
            return self.imaging_directory()
        inputs = self.prepare_imaging()
        for name, core in [('full_pixel', 0.), ('core_pixel_clean', .18)]:
            out = inputs / name
            if (out / 'imaging.json').exists():
                r = read_json(out / 'imaging.json')
                if r['source_id'] != self.source_id:
                    raise ValueError(f'Imaging fit belongs to another source: {out}')
                if r['completed'] and r['diagnostics']['convergence_gate_passed'] and r['orientation_parameterization_ok']:
                    continue
                raise ValueError(f'Imaging diagnostics need attention: {out}. Inspect its model/residuals. '
                    'For an exploratory grism fit without this prior, explicitly choose '
                    'INCLINATION="isotropic" and CENTER="catalog" or "free".')
            run_command(['-m', 'diagnostics.native_imaging_prior', '--inputs', str(inputs), '--output', str(out),
                         '--mask-core', str(core), '--device', device, '--warmup', str(warmup), '--samples', str(samples)],
                        inputs / f'{name}.log', device)
            r = read_json(out / 'imaging.json')
            if not (r['completed'] and r['diagnostics']['convergence_gate_passed'] and r['orientation_parameterization_ok']):
                raise ValueError(f'F150W fit did not pass diagnostics: {out}. '
                    'Inspect its model/residuals before using an imaging inclination prior. '
                    'INCLINATION="isotropic" with CENTER="catalog" or "free" enables an explicit exploratory alternative.')
        return inputs

    def fit(self, run_name='visual_grid', step=25, vmax=1000, center='catalog',
            inclination='imaging', device='cuda', maxiter=160, controls=21, quadrature=61):
        """Run/resume real data only. No injections or synthetic science data."""
        if not re.fullmatch(r'[A-Za-z0-9_-]+', run_name):
            raise ValueError('Run name must contain only letters, numbers, underscore or dash.')
        self.require_data()
        if inclination == 'imaging' or center == 'stellar':
            if not all(r['ready'] for r in self.imaging_status()):
                raise ValueError('Run the F150W fitting cell first; the required source-specific prior is not ready.')
        output = self.directory / run_name
        args = ['-m', 'diagnostics.native_source_grid', '--source-id', str(self.source_id),
                '--config', str(self.config), '--bundle', str(self.bundle), '--output', str(output),
                '--imaging', str(self.imaging_directory()), '--step', str(step), '--vmax', str(vmax),
                '--center', center, '--inclination', inclination, '--device', device,
                '--maxiter', str(maxiter), '--controls', str(controls), '--quadrature', str(quadrature)]
        run_command(args, output / 'run.log', device)
        return output

    def posterior_runs(self):
        """Real kinematic chains only; never substitute imaging or mock chains."""
        rows = []
        for path in self.directory.glob('posterior*/posterior.json'):
            r = read_json(path)
            if r.get('source_id') != self.source_id or r.get('real_observations') is not True:
                continue
            rows.append(dict(path=str(path.parent), completed=r.get('completed', False),
                draws_per_chain=r.get('retained_draws_per_chain', 0),
                sampling_checks_passed=r.get('diagnostics', {}).get('convergence_gate_passed', False),
                modified=path.stat().st_mtime))
        return sorted(rows, key=lambda r: r['modified'], reverse=True)

    def sample_posterior(self, grid_run, run_name='posterior_v1', *, warmup=800,
                         samples=1000, chunk=100, chains=4, max_tree_depth=9, device='cuda',
                         from_posterior=None, initial_posterior=None):
        """Sample the real grid's model; stream progress and stop on interruption."""
        if not re.fullmatch(r'posterior[A-Za-z0-9_-]*', run_name):
            raise ValueError('Posterior run name must begin with posterior and use letters, digits, underscores or dashes.')
        grid_run = Path(grid_run).expanduser().resolve()
        grid = read_json(grid_run / 'grid.json')
        if grid.get('source_id') != self.source_id or grid.get('real_observations') is not True:
            raise ValueError('Posterior initialization needs this source\'s own real-data grid.')
        output = self.directory / run_name
        existing = read_json(output / 'posterior.json') if (output / 'posterior.json').exists() else {}
        from_posterior = from_posterior or existing.get('adapted_from', {}).get('directory')
        if initial_posterior == 'latest' and existing.get('initialized_from'):
            initial_posterior = existing['initialized_from']['directory']
        elif initial_posterior == 'latest':
            initial_posterior = None
            for row in self.posterior_runs():
                candidate = Path(row['path']).resolve()
                r = read_json(candidate / 'posterior.json')
                if (candidate != output.resolve() and r.get('completed') and
                        r.get('warmup_completed') and r.get('retained_draws_per_chain', 0) >= 4 and
                        Path(r['settings']['grid']).resolve() == grid_run):
                    initial_posterior = candidate
                    break
            if initial_posterior is None:
                print('No completed posterior for this grid; starting fresh from competing grid solutions.')
        initial_posterior = initial_posterior or existing.get('initialized_from', {}).get('directory')
        if initial_posterior is not None and from_posterior is not None:
            raise ValueError('Choose fresh warmup from positions OR reuse adaptation, not both.')
        common = ['--source-id', str(self.source_id), '--grid', str(grid_run),
                  '--output', str(output), '--device', device, '--samples', str(samples),
                  '--chunk', str(chunk), '--max-tree-depth', str(max_tree_depth)]
        if initial_posterior is not None:
            initial_posterior = Path(initial_posterior).expanduser().resolve()
            parent = read_json(initial_posterior / 'posterior.json')
            if parent.get('source_id') != self.source_id or parent.get('real_observations') is not True:
                raise ValueError('Initialization must use this source\'s own real-data posterior.')
            if Path(parent['settings']['grid']).resolve() != grid_run:
                raise ValueError('Initialization must use the selected grid.')
            if parent['settings']['chains'] != chains:
                raise ValueError('Fresh adaptation must keep the initialization source\'s chain count.')
            if initial_posterior == output.resolve():
                raise ValueError('Use a fresh run tag for fresh adaptation.')
            print(f'Initial positions: {initial_posterior}\n'
                  f'Fresh warmup: {warmup} steps per chain; previous tuning and draws are excluded.')
            args = ['-m', 'diagnostics.native_real_posterior_rewarm',
                    '--initial-posterior', str(initial_posterior), *common,
                    '--warmup', str(warmup), '--chains', str(chains)]
        elif from_posterior is not None:
            print('Reusing the selected posterior\'s adapted chains, priors and step sizes; '
                  'warmup and chain count come from that checkpoint.')
            args = ['-m', 'diagnostics.native_real_posterior_extend',
                    '--from-posterior', str(Path(from_posterior).expanduser().resolve()), *common]
        else:
            args = ['-m', 'diagnostics.native_real_posterior', *common,
                    '--warmup', str(warmup), '--chains', str(chains)]
        if (output / 'posterior.json').exists():
            args.append('--resume')
        print(f'Posterior output: {output}\nLog: {output / "run.log"}', flush=True)
        run_command(args, output / 'run.log', device)
        return output


def run_command(arguments, log_path, device='cuda'):
    """Stream progress, preserve logs, terminate child on notebook interruption."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['JAX_PLATFORMS'] = 'cpu' if device == 'cpu' else 'cuda'
    env.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    env.setdefault('XLA_FLAGS', '--xla_gpu_enable_triton_gemm=false')
    with log_path.open('a') as log:
        proc = subprocess.Popen([sys.executable, *arguments], cwd=REPO, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in proc.stdout:
                log.write(line); log.flush()
                # Full diagnostics remain in the log; concise progress in cells.
                print(line.rstrip(), flush=True)
            code = proc.wait()
            if code:
                raise RuntimeError(f'Job exited with code {code}. See {log_path}')
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()


def available_sources(bundle=DEFAULT_BUNDLE, results=DEFAULT_RESULTS):
    return [RealSource(source_number(p.parent.name), bundle, results, p).availability()
            for p in sorted((REPO / 'data').glob('ID*/config_kinematics.yaml'))]


def record_review(run, velocity, verdict, notes):
    """Append a human assessment without changing a fitted score or winner."""
    allowed = {'plausible', 'needs investigation', 'reject: documented model failure'}
    if verdict not in allowed or not notes.strip():
        raise ValueError('Choose a listed verdict and write the reason for the assessment.')
    run = Path(run)
    report = read_json(run / 'grid.json')
    fits = [f for f in report['best_by_velocity'] if f['vrot_0p5arcsec_kms'] == float(velocity)]
    if not fits:
        raise ValueError('Review velocity must be a saved fitted trial.')
    row = {'utc': datetime.now(timezone.utc).isoformat(), 'source_id': report['source_id'],
           'velocity_kms': float(velocity), 'verdict': verdict, 'notes': notes.strip(),
           'artifact': fits[0]['artifact'], 'objective': fits[0]['objective'],
           'fit_sha256': sha256(run / fits[0]['artifact']), 'run': str(run.resolve())}
    path = run / 'visual_reviews.jsonl'
    with path.open('a') as stream:
        stream.write(json.dumps(row) + '\n')
    return path
