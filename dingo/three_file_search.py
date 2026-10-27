"""Broad candidate search and sampling with one shared, fixed comparison model.

This is a separate worker so saved legacy searches keep their original worker
identity and can still be opened/resumed. Uses only the three legacy FITS inputs.
"""
from __future__ import annotations
import copy
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import time
import numpy as np
import torch
import yaml
from astropy.io import fits
from .masked_comparison import InputMaskedFitter
from .candidate_search import PosteriorSearch, grid_seeds, diverse_indices, profile, polish_profile
from .three_file_fit import (REPO, build_config, validate_inclination_metadata, velocity_definition,
    _json, _npz, _sha, _guard_identity, _choose_candidate, _search_config, parser as base_parser)
from .three_file_plot import physical_scale, velocity_summary

LOG = logging.getLogger(__name__)
NOISE_NOTE = ('Approximate stacked-image likelihood. One common mask, variance map and '
    'effective correlation area are calculated at a fixed zero-velocity, zero-registration '
    'reference, saved with the search, and reused unchanged by every candidate and MCMC. '
    'A fitted log noise scale is profiled during search and sampled during MCMC. '
    'Stack/subtraction covariance is not fully propagated; intervals remain conditional '
    'on this noise prescription. Scores from legacy candidate-specific noise runs are not comparable.')
TENSOR_FIELDS = ('mcmc_mask', 'mcmc_var_map', 'mcmc_foot_R', 'mcmc_foot_C')
SCALAR_FIELDS = ('sigma_R', 'sigma_C', 'mcmc_corr_area', 'mcmc_var_base', 'mcmc_n_eff',
                 'n_mcmc_pix', 'mcmc_temper', '_mcmc_lnvar_const', 'mcmc_flux_masked_out')


class SharedComparisonFitter(InputMaskedFitter):
    def _mcmc_prepare(self):
        if self.mcmc_mask is not None:
            return
        path = Path(self.config['shared_comparison'])
        with np.load(path, allow_pickle=False) as saved:
            for name in TENSOR_FIELDS:
                setattr(self, name, torch.as_tensor(saved[name].copy(), device=self.device))
            for name in SCALAR_FIELDS:
                setattr(self, name, saved[name].item())
        self._mcmc_inv_var = None
        self.validate_input_mask_support()

    def mcmc_fingerprint(self):
        return dict(super().mcmc_fingerprint(), shared_comparison_sha256=_sha(Path(self.config['shared_comparison'])))


def save_comparison(config_path, path, device):
    """Reference fixed independently of the candidates; never rebuilt during sampling."""
    f = InputMaskedFitter(str(config_path), device=device)
    from .startpoint import amplitude_name
    amplitude = amplitude_name(f.config)
    f.set_params_dict({f'velocity.{amplitude}': 0., 'image.R.dx': 0., 'image.R.dy': 0.})
    f.prepare_mcmc()
    values = {name: getattr(f, name).detach().cpu().numpy() for name in TENSOR_FIELDS}
    values.update({name: np.asarray(getattr(f, name)) for name in SCALAR_FIELDS})
    if f.input_excluded:
        values['input_comparison_excluded'] = f.input_comparison_excluded.cpu().numpy()
        values.update({f'input_excluded_{pupil}': mask.cpu().numpy()
                       for pupil, mask in f.input_excluded.items()})
    _npz(path, **values)


def records_and_pixels(f, searcher, points, scores, metadata, source_dir, path, kinds):
    rows, rr, cc = [], [], []
    for cid, (theta, score, kind) in enumerate(zip(points, scores, kinds)):
        actual = f.log_probability(theta)
        if not np.isfinite(actual) or abs(actual-score) > .2:
            raise RuntimeError(f'Candidate {cid}: scalar/batched score mismatch ({actual} vs {score}).')
        r, c, _, _ = f.get_fitting_results()
        full = dict(zip(f.mcmc_param_names, map(float, theta)))
        params = {k.split('.', 1)[1]: v for k, v in full.items() if k.startswith('velocity.')}
        params['v_sys'] = 0.
        full.update({'velocity.v_sys': 0., 'image.C.dx': 0., 'image.C.dy': 0.})
        v = velocity_summary(params, metadata)
        # A masked run's displayed residual sum must use its actual comparison
        # pixels; otherwise the discarded contaminant still dominates the table.
        residual = (r-c)[f.mcmc_mask.cpu().numpy()] if f.input_excluded else r-c
        rows.append(dict(candidate_id=cid, kind=kind, params=params, full_params=full,
            loss=float(np.sum(residual**2)), log_posterior=float(actual),
            log_prior=float(f.log_prior(theta)), log_likelihood=float(actual-f.log_prior(theta)),
            velocity_summary=dict(V_sini_kms=v['v_sini'], V_inf_kms=v['v_inf'],
                inclination_deg=v['inclination_deg'], R_v_pixels=params['R_v']),
            flux_retained=f._flux_retained()))
        rr.append(r.copy()); cc.append(c.copy())
    r, c = np.stack(rr), np.stack(cc)
    observed_r, observed_c = f.get_true_images()
    _npz(path, candidate_ids=np.arange(len(points)), image=fits.getdata(source_dir/'image.fits'),
         observed_R=observed_r, observed_C=observed_c, rectified_R=r, rectified_C=c,
         difference=r-c, mask=f.mcmc_mask.cpu().numpy())
    return rows


def run_search(args, config, output, config_path, metadata, source_dir):
    f = SharedComparisonFitter(str(config_path), device=torch.device(args.device))
    s = PosteriorSearch(f, radius_pixels=args.profile_radius_pc/physical_scale(metadata)['pc_per_pixel'])
    stage = output/'search_progress.npz'
    def checkpoint(points, scores, step):
        _npz(stage, points=points, log_posterior=scores, iteration=np.array(step))
    seed_path = output/'grid_seeds.npz'
    if seed_path.exists():
        with np.load(seed_path) as saved:
            seeds, scores = saved['points'], saved['log_posterior']
        LOG.info('Reusing completed geometry grid (%d seeds)', len(seeds))
    else:
        seeds, scores = grid_seeds(s, args.amp_step, args.n_theta, checkpoint)
        _npz(seed_path, points=seeds, log_posterior=scores)
    refined_path = output/'refined_starts.npz'
    if refined_path.exists():
        with np.load(refined_path) as saved:
            points, scores = saved['points'], saved['log_posterior']
    else:
        points, scores = s.optimize(seeds, iterations=args.refine_iterations, callback=checkpoint)
        # Explicit neighbouring alignment alternatives, plus independent center perturbations.
        ids = diverse_indices(points, scores, s, 4)
        extra = []
        for idx in ids:
            for dx, dy in ((x,y) for x in (-1.,0.,1.) for y in (-1.,0.,1.) if x or y):
                p = points[idx].copy()
                p[s.index['image.R.dx']] += dx; p[s.index['image.R.dy']] += dy
                extra.append(p)
            for dx, dy in [(-2.,0.), (2.,0.), (0.,-2.), (0.,2.)]:
                p = points[idx].copy()
                p[s.index['velocity.x0_v']] += dx; p[s.index['velocity.y0_v']] += dy
                extra.append(p)
        if extra:
            ep, es = s.optimize(extra, iterations=args.refine_iterations, label='Alignment/center starts', callback=checkpoint)
            points, scores = np.concatenate([points, ep]), np.concatenate([scores, es])
        _npz(refined_path, points=points, log_posterior=scores)
    profile_path = output/'initial_velocity_profile.npz'
    if profile_path.exists():
        with np.load(profile_path) as saved:
            velocities, pp, ps = saved['velocity_kms'], saved['points'], saved['log_posterior']
    else:
        ids = diverse_indices(points, scores, s, 3)
        starts = points[ids].copy()
        # Include different inclinations, even when the best free fits cluster.
        spec = dict(f._mcmc_priors)['velocity.inc_v']
        for offset in (-1., 1.):
            p = starts[0].copy()
            p[s.inc] = np.clip(spec.get('mu', np.pi/3)+offset*spec.get('sigma', .2),
                              spec['min']+.001, spec['max']-.001)
            starts = np.vstack([starts, p])
        velocities = np.arange(0., args.profile_max+args.profile_step*.5, args.profile_step)
        pp, ps = profile(s, starts, velocities, iterations=args.profile_iterations, checkpoint=checkpoint)
        if args.profile_fine_step > 0 and np.isfinite(ps).any():
            best = velocities[np.argmax(ps)]
            fine = np.arange(max(0., best-args.profile_step), min(args.profile_max, best+args.profile_step)
                             +args.profile_fine_step*.5, args.profile_fine_step)
            fine = fine[~np.isclose(fine[:, None], velocities[None, :]).any(axis=1)]
            fp, fs = profile(s, pp[np.argsort(-ps)[:2]], fine,
                             iterations=args.profile_iterations, checkpoint=checkpoint)
            velocities, pp, ps = np.r_[velocities, fine], np.vstack([pp, fp]), np.r_[ps, fs]
            order = np.argsort(velocities); velocities, pp, ps = velocities[order], pp[order], ps[order]
        _npz(profile_path, velocity_kms=velocities, points=pp, log_posterior=ps,
             radius_pc=np.array(args.profile_radius_pc), param_names=np.array(s.names))
    # Let the best fixed-velocity solutions move freely; these are MCMC candidates.
    releases = output/'profile_released.npz'
    if releases.exists():
        with np.load(releases) as saved:
            rp, rs = saved['points'], saved['log_posterior']
    else:
        ids = diverse_indices(pp, ps, s, args.n_keep)
        rp, rs = s.optimize(pp[ids], iterations=args.refine_iterations,
                            label='Free refinement from profile', callback=checkpoint)
        _npz(releases, points=rp, log_posterior=rs)
    points, scores = np.vstack([points, rp]), np.r_[scores, rs]
    final_profile = output/'velocity_profile.npz'
    if final_profile.exists():
        with np.load(final_profile) as saved:
            pp, ps = saved['points'], saved['log_posterior']
    else:
        anchors = points[diverse_indices(points, scores, s, 4)]
        pp, ps, gain = polish_profile(s, pp, ps, velocities, anchors,
            iterations=args.profile_iterations, checkpoint=checkpoint)
        _npz(final_profile, velocity_kms=velocities, points=pp, log_posterior=ps,
             radius_pc=np.array(args.profile_radius_pc), param_names=np.array(s.names),
             last_pass_max_logp_gain=np.array(gain))
    # Incorporate newly found profile branches into the free-candidate pool too.
    ids = diverse_indices(pp, ps, s, args.n_keep)
    rp, rs = s.optimize(pp[ids], iterations=args.refine_iterations,
                        label='Final free refinement', callback=checkpoint)
    points, scores = np.vstack([points, rp]), np.r_[scores, rs]
    ids = diverse_indices(points, scores, s, args.n_keep)
    chosen, chosen_scores = points[ids], scores[ids]
    kinds = ['free candidate']*len(chosen)
    # Retain representative constrained alternatives explicitly for visual inspection.
    # They are not advertised as local maxima; MCMC may move away from them.
    for lo, hi in [(0., 250.), (250., 500.), (500., 750.), (750., args.profile_max+1)]:
        ix = np.flatnonzero((velocities >= lo) & (velocities < hi) & np.isfinite(ps))
        if not len(ix):
            continue
        j = ix[np.argmax(ps[ix])]
        if np.min(np.abs(s.velocity(chosen)-velocities[j])) < args.profile_step:
            continue
        chosen = np.vstack([chosen, pp[j]]); chosen_scores = np.r_[chosen_scores, ps[j]]
        kinds.append('fixed-velocity comparison; free MCMC can move')
    rows = records_and_pixels(f, s, chosen, chosen_scores, metadata, source_dir, output/'candidates.npz', kinds)
    return rows


def run(args):
    amplitude = getattr(args, 'velocity_parameter', 'V_sini')
    definition = velocity_definition(amplitude)
    LOG.info('Sampling definition: %s; uniform amplitude support 0..%g km/s.',
             definition, args.max_velocity)
    started = time.monotonic()
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    if 'GRISM_CAL_DIR' not in os.environ:
        os.environ['GRISM_CAL_DIR'] = str(REPO.parent/'grism_cal')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use the GPU-enabled environment or device="cpu".')
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    source_dir = Path(args.data_dir).resolve()/f'ID{args.source}'
    original = Path(args.config or source_dir/'config_kinematics.yaml')
    config = build_config(args.source, Path(args.data_dir), original, output,
        warmup=args.warmup, draws=args.draws, walkers=args.walkers, seed=args.seed,
        inclination_deg=args.inclination_deg, inclination_sigma_deg=args.inclination_sigma_deg,
        optimizer_steps=tuple(args.optimizer_steps), device=args.device, max_velocity=args.max_velocity,
        velocity_parameter=amplitude)
    # Regions remain in original input-pixel coordinates. InputMaskedFitter
    # builds a fixed comparison exclusion covering their remapped footprint.
    masks = {}
    for pupil in ('R', 'C'):
        value = getattr(args, f'mask_{pupil.lower()}', None)
        if value is not None:
            path = Path(value).expanduser().resolve()
            masks[pupil] = dict(path=str(path), sha256=_sha(path))
    if masks:
        config['input_masks'] = masks
    inclination_metadata = validate_inclination_metadata(
        __import__('json').loads(args.inclination_metadata_json) if args.inclination_metadata_json else None,
        args.source, args.inclination_deg, args.inclination_sigma_deg)
    metadata = dict(source_id=args.source, redshift=float(config['summary']['z']),
                    pixel_scale_arcsec=args.pixel_scale, r_fit_pixels=config['summary']['r_fit'],
                    velocity_parameter=amplitude, velocity_definition=definition)
    args.profile_radius_pc = args.profile_radius_pc or physical_scale(metadata)['reference_radius_pc']
    config['shared_comparison'] = str(output/'comparison.npz')
    settings = dict(method='broad', amp_step=args.amp_step, n_theta=args.n_theta, n_keep=args.n_keep,
        max_velocity=args.max_velocity, optimizer_steps=args.optimizer_steps,
        refine_iterations=args.refine_iterations, profile_iterations=args.profile_iterations,
        profile_step=args.profile_step, profile_fine_step=args.profile_fine_step,
        profile_max=args.profile_max, profile_radius_pc=args.profile_radius_pc)
    inputs = {n: dict(path=str(source_dir/f'{n}.fits'), sha256=_sha(source_dir/f'{n}.fits')) for n in ('image','R','C')}
    inputs.update({f'mask_{pupil}': info for pupil, info in masks.items()})
    from .grism import get_grism_cal_dir
    cal = Path(get_grism_cal_dir())
    identity = dict(source_id=args.source, config=_search_config(config), inputs=inputs,
        settings=settings, inclination_metadata=inclination_metadata, pixel_scale=args.pixel_scale,
        posterior={k: config['mcmc'][k] for k in ('priors', 'noise', 'likelihood', 'xy_iters', 'xy_tol')},
        code={n: _sha(REPO/'dingo'/n) for n in ('candidate_search.py','three_file_search.py','three_file_fit.py',
                                             'fitting.py','kinematics.py','startpoint.py','grism.py','utils.py')},
        calibration={str(p.relative_to(cal)): _sha(p) for p in sorted(cal.rglob('*.dat'))})
    if masks:
        identity['code'].update({name: _sha(REPO/'dingo'/name)
                                 for name in ('input_masks.py', 'masked_comparison.py')})
    _guard_identity(output/'search_identity.json', identity, description='broad search')
    config_path = output/'search_config.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    if not (output/'comparison.npz').exists():
        if (output/'start_search.json').exists():
            raise ValueError('Saved comparison model is missing; preserve the run and use a new name.')
        save_comparison(config_path, output/'comparison.npz', torch.device(args.device))
    comparison_hash = _sha(output/'comparison.npz')
    _guard_identity(output/'comparison_identity.json', {'sha256': comparison_hash}, description='comparison weights')
    start_path = output/'start_search.json'
    if start_path.exists():
        search = __import__('json').loads(start_path.read_text())
        for name in ('candidates', 'velocity_profile', 'comparison'):
            if _sha(output/f'{name}.npz') != search[f'{name}_sha256']:
                raise ValueError(f'Saved {name} changed; do not reuse this search.')
        LOG.info('Reusing completed broad search.')
    else:
        rows = run_search(args, config, output, config_path, metadata, source_dir)
        search = dict(metadata, schema='dingo-three-file-candidates-v1', trials=rows,
            selected=int(np.argmax([t['log_posterior'] for t in rows])), search_settings=settings,
            inclination_prior=config['mcmc']['priors']['velocity.inc_v'],
            inclination_settings=dict(inclination_deg=args.inclination_deg,
                inclination_sigma_deg=args.inclination_sigma_deg if args.inclination_deg is not None else 10.),
            inclination_metadata=inclination_metadata, noise_note=NOISE_NOTE,
            search_identity_sha256=_sha(output/'search_identity.json'),
            selection_note='Finite multistart search; no guarantee all solutions were found. '
                'Fixed-velocity comparison rows are constrained alternatives, not free local maxima.',
            display_mask_note='Same saved fixed comparison mask for all candidates and sampling.',
            elapsed_seconds=time.monotonic()-started, finished_utc=datetime.now(timezone.utc).isoformat(),
            **{f'{n}_sha256': _sha(output/f'{n}.npz') for n in ('candidates','velocity_profile','comparison')})
        _json(start_path, search)
    if args.search_only:
        LOG.info('Saved %d candidates and velocity profile. No MCMC was run.', len(search['trials']))
        return search
    return sample(args, config, output, source_dir, inputs, inclination_metadata, search, identity, started)


def sample(args, config, output, source_dir, inputs, inclination_metadata, search, identity, started):
    from .startpoint import amplitude_name
    amplitude = amplitude_name(config)
    chosen_id, trial = _choose_candidate(search, args.candidate)
    config = copy.deepcopy(config)
    config["mcmc"]["noise"]["ln_f"] = trial["full_params"]["noise.ln_f"]
    identity_config = copy.deepcopy(config)
    # A search is reusable when the user chooses MCMC lengths/walkers afterward.
    # Once sampling starts, only increasing the production length is safe.
    identity_config["mcmc"].pop("nsteps")
    identity = {"search_identity": identity, "comparison_sha256": _sha(output/"comparison.npz"), "config": identity_config,
                "chosen_candidate_id": chosen_id, "chosen_params": trial["full_params"]}
    _guard_identity(output/"identity.json", identity, description="sampling run (including its chosen candidate)")
    report_path = output / "report.json"
    if report_path.exists():
        previous = __import__("json").loads(report_path.read_text())
        if previous.get("status") == "complete" and previous.get("draws") == args.draws:
            for name in ("samples", "predictions"):
                if _sha(output / f"{name}.npz") != previous.get(f"{name}_sha256"):
                    raise ValueError(f"Saved {name} changed; inspect this run before resuming.")
            LOG.info("Requested %d retained steps already saved in %s; loading without further sampling.",
                     args.draws, output)
            return previous
    run_config = output / "run_config.yaml"
    run_config.write_text(yaml.safe_dump(config, sort_keys=False))
    fitter = SharedComparisonFitter(str(run_config), device=torch.device(args.device))
    fitter.set_params_dict(trial["full_params"])
    LOG.info("Chosen candidate %d: sampled %s=%.1f km/s, V_inf=%.1f km/s, R_v=%.3f px, inverse R-C loss=%.6g",
             chosen_id, amplitude, trial["params"][amplitude], trial["velocity_summary"]["V_inf_kms"],
             trial["params"]["R_v"], trial["loss"])
    fitter.fit_MCMC()
    raw_chain, raw_logp = fitter._mcmc_raw_chain, fitter._mcmc_raw_log_prob
    # The legacy sampler catches Ctrl-C. Never relabel an interrupted warmup
    # as posterior draws when exporting the simple workflow's results.
    if raw_chain.shape[0] <= args.warmup:
        raise InterruptedError("Stopped during warmup; checkpoint saved, no retained posterior yet.")
    retained, logp = raw_chain[args.warmup:], raw_logp[args.warmup:]
    complete = len(retained) >= args.draws
    names = list(fitter.mcmc_param_names)
    if fitter.mcmc_sampler.iteration:
        acceptance = fitter.mcmc_sampler.acceptance_fraction
    else:
        with np.load(output / "checkpoint.npz", allow_pickle=False) as saved:
            acceptance = saved["acceptance_fraction"]
    fitter._mcmc_acceptance = acceptance
    _npz(output / "samples.npz", chain=retained, log_prob=logp,
         param_names=np.asarray(names, dtype=str), warmup=np.asarray(args.warmup),
         acceptance_fraction=acceptance)
    diag = fitter.get_mcmc_diagnostics()
    ratios = np.asarray(list(diag["n_steps_over_tau"].values()), dtype=float)
    shifts = np.asarray(list(diag["split_half_shift"].values()), dtype=float)
    sampling_pass = bool(complete and np.isfinite(ratios).all() and np.all(ratios >= 50)
                         and np.isfinite(shifts).all() and np.all(shifts <= 0.2)
                         and diag["frac_neg_inf"] == 0)
    theta = fitter.set_params_to_mcmc("map")
    fitter.log_probability(theta)  # Render the exact deterministic sampling model.
    rectified_r, rectified_c, vz_r, vz_c = fitter.get_fitting_results()
    observed_r, observed_c = fitter.get_true_images()
    from astropy.io import fits
    _npz(output / "predictions.npz", image=fits.getdata(source_dir / "image.fits"),
         observed_R=observed_r, observed_C=observed_c,
         rectified_R=rectified_r, rectified_C=rectified_c,
         difference=rectified_r-rectified_c, velocity_R=vz_r, velocity_C=vz_c,
         mask=fitter.mcmc_mask.detach().cpu().numpy(),
         variance=fitter.mcmc_var_map.detach().cpu().numpy(),
         theta=theta, param_names=np.asarray(names, dtype=str))
    fixed = {key: float(cfg.tensor.detach().cpu()) for key, cfg in fitter._mcmc_stage_cfg().items()
             if isinstance(cfg.tensor, torch.Tensor) and cfg.tensor.numel() == 1 and key not in names}
    report = {"schema": "dingo-three-file-v1", "source_id": args.source,
              "status": "complete" if complete else "interrupted", "sampling_pass": sampling_pass,
              "model": "Original DINGO inverse rectification of line-only R+C stacks",
              "intrinsic_dispersion_kms": 0.0, "velocity_parameter": amplitude,
              "velocity_definition": velocity_definition(amplitude),
              "radius_definition": "R_v is arctan turnover radius in input stack pixels",
              "pixel_scale_arcsec": args.pixel_scale, "pixel_scale_source": "explicit notebook/worker setting",
              "redshift": float(config["summary"]["z"]), "r_fit_pixels": config["summary"]["r_fit"],
              "inputs": inputs, "sampler": "emcee ensemble MCMC with PyTorch likelihood",
              "device": args.device, "walkers": retained.shape[1], "warmup": args.warmup,
              "draws": retained.shape[0], "param_names": names, "fixed_params": fixed,
              "inclination_prior": config["mcmc"]["priors"]["velocity.inc_v"],
              "inclination_metadata": inclination_metadata,
              "settings": {"pixel_scale_arcsec": args.pixel_scale, "warmup": args.warmup,
                           "draws": args.draws, "walkers": args.walkers, "device": args.device,
                           "velocity_parameter": amplitude},
              "initialization": search, "best_fit": dict(zip(names, theta)),
              "diagnostics": diag, "percentiles_16_50_84": fitter.get_mcmc_results(),
              "noise_note": NOISE_NOTE, "noise": {"sigma_R": fitter.sigma_R, "sigma_C": fitter.sigma_C,
                                                   "correlation_area": fitter.mcmc_corr_area,
                                                   "effective_pixels": fitter.mcmc_n_eff},
              "chosen_candidate_id": chosen_id,
              "selection_method": "explicit candidate" if args.candidate is not None else "automatic maximum posterior",
              "selection_notes": args.selection_notes,
              "selection_note": "MCMC starts around the selected candidate; intervals are conditional on that "
                                "choice and the local fixed noise/mask prescription. Alternative candidates "
                                "are retained separately and are not pooled. No hard boundary isolates this "
                                "mode: inspect traces for excursions before reporting local intervals.",
              "elapsed_seconds": time.monotonic()-started,
              "completed_utc": datetime.now(timezone.utc).isoformat() if complete else None,
              "comparison_sha256": _sha(output/"comparison.npz"),
              "samples_sha256": _sha(output / "samples.npz"),
              "predictions_sha256": _sha(output / "predictions.npz")}
    _json(output / "report.json", report)
    LOG.info("Saved %d retained steps per walker to %s; sampling checks %s.",
             retained.shape[0], output, "passed" if sampling_pass else "not yet passed")
    return report


def parser():
    p = base_parser()
    p.set_defaults(n_keep=8)
    p.add_argument('--refine-iterations', type=int, default=100)
    p.add_argument('--profile-iterations', type=int, default=80)
    p.add_argument('--profile-step', type=float, default=25.)
    p.add_argument('--profile-fine-step', type=float, default=1.)
    p.add_argument('--profile-max', type=float, default=1000.)
    p.add_argument('--profile-radius-pc', type=float)
    p.add_argument('--mask-r', help='DS9 exclusion regions in original R input-pixel coordinates.')
    p.add_argument('--mask-c', help='DS9 exclusion regions in original C input-pixel coordinates.')
    return p


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    args = parser().parse_args(argv)
    positive = [args.amp_step, args.max_velocity, args.n_theta, args.n_keep,
                args.refine_iterations, args.profile_iterations, args.profile_step,
                args.profile_max, args.pixel_scale]
    if not np.isfinite(positive).all() or min(positive) <= 0 or args.n_theta < 2:
        raise ValueError('Search steps, counts and ranges must be finite and positive.')
    if args.profile_fine_step < 0 or not np.isfinite(args.profile_fine_step):
        raise ValueError('profile_fine_step must be finite and nonnegative (0 disables fine grid).')
    if args.profile_radius_pc is not None and (not np.isfinite(args.profile_radius_pc) or args.profile_radius_pc <= 0):
        raise ValueError('profile_radius_pc must be finite and positive.')
    run(args)


if __name__ == '__main__':
    main()
