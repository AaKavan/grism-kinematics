#!/usr/bin/env python
'''
Measure a galaxy's inclination from its direct image, to use as a prior on
`velocity.inc_v` in the DINGO kinematics fit.

Why this is needed: the grism kinematics constrain V_rot*sin(i) very well
(+-0.7% for ID15665) but V_rot and inc_v individually are degenerate at
corr = -0.986, so V_rot alone spans 482-790 km/s. An external constraint on the
inclination is what turns V_rot*sin(i) into a V_rot and hence a dynamical mass.

Method: fit a PSF-convolved Sersic profile to a linear SCI cutout, weighted by
the matching ERR map, sampling with emcee; then convert the axis-ratio posterior
into an inclination posterior.

The conversion is exact rather than approximate, because DINGO's two models share
a convention:

    galaxy.sersic_model_torch      R = sqrt(x_rot^2 + (y_rot/q)^2)
    kinematics.arctangent_...      r = sqrt(x_p^2   + (y_p/cos inc_v)^2)

so within DINGO, q IS cos(inc_v) for an infinitely thin disk -- which is the
geometry the velocity model itself assumes. A finite intrinsic thickness q0
instead gives cos^2 i = (q^2 - q0^2)/(1 - q0^2); both are reported, and the thin
disk is the self-consistent default.

Note the direct image and the grism source frame are rotated relative to one
another, so the POSITION ANGLE from this fit is not directly comparable to
theta_v. The axis ratio is rotation invariant, so the inclination is unaffected.

Usage:
    python measure_inclination.py --id 15665 --ra 64.007840 --dec -24.148910
'''

import argparse
import logging
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))   # pipeline/
ROOT = os.path.dirname(HERE)                        # DINGO_development/
GK   = os.path.dirname(ROOT)                        # Galaxy_Kinematics/
os.environ.setdefault('GRISM_CAL_DIR', os.path.join(GK, 'grism_cal'))

import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats
import astropy.units as u

from dingo import galaxy, fitting

warnings.filterwarnings('ignore')
LOG = logging.getLogger('inclination')

SCI_DIR = os.path.join(GK, 'EDR_data/sapphires_edr_nircam_sci')
ERR_DIR = os.path.join(GK, 'EDR_data/sapphires_edr_nircam_err')
DEFAULT_PSF = os.path.join(
    GK, 'github_backups/galaxy_kinematics/15665/geko_fitting/psfs/webbPSF_F444W.fits')

PARAM_NAMES = ['log_I_e', 'R_e', 'n', 'x0', 'y0', 'q', 'theta', 'sky']


def load_cutouts(ra, dec, band, size_arcsec):
    coord = SkyCoord(ra*u.deg, dec*u.deg)
    with fits.open(os.path.join(SCI_DIR, f'4750_{band}_v05_sci.fits')) as h:
        hdu = h['SCI']
        wcs = WCS(hdu.header)
        if not wcs.footprint_contains(coord):
            raise ValueError(f'{coord} is outside the {band} mosaic')
        sci = Cutout2D(hdu.data, coord, size_arcsec*u.arcsec, wcs=wcs)
        pixscale = float(np.mean(np.abs(np.diag(wcs.pixel_scale_matrix))))*3600.0
        sci_shape = hdu.data.shape
    # The ERR mosaics carry no WCS of their own, but they are pixel-aligned with
    # the SCI mosaic, so reuse the SCI cutout's pixel slices.
    with fits.open(os.path.join(ERR_DIR, f'4750_{band}_v05_err.fits')) as h:
        name = 'ERR' if 'ERR' in [x.name for x in h] else h[0].name
        if h[name].data.shape != sci_shape:
            raise ValueError(f'ERR shape {h[name].data.shape} != SCI shape {sci_shape}; '
                             'they are not pixel-aligned, so the slices cannot be reused')
        err = h[name].data[sci.slices_original]
    return (np.array(sci.data, dtype=np.float64),
            np.array(err, dtype=np.float64), pixscale)


def load_psf(path, target_pixscale, halfsize=20):
    '''Detector-sampled, distorted webbPSF, resampled to the mosaic pixel grid.'''
    from scipy.ndimage import zoom
    with fits.open(path) as h:
        ext = 'DET_DIST' if 'DET_DIST' in [x.name for x in h] else 'DET_SAMP'
        psf = np.array(h[ext].data, dtype=np.float64)
        psf_pixscale = float(h[0].header['PIXELSCL'])
    factor = psf_pixscale/target_pixscale
    if abs(factor - 1) > 1e-3:
        psf = zoom(psf, factor, order=3)
        LOG.info(f'resampled PSF {psf_pixscale:.5f} -> {target_pixscale:.5f} arcsec/pix '
                 f'(x{factor:.4f}), now {psf.shape}')
    c = psf.shape[0]//2
    psf = psf[c-halfsize:c+halfsize+1, c-halfsize:c+halfsize+1]
    psf = np.clip(psf, 0, None)
    return psf/psf.sum()


def run_pysersic(image, err, psf, fit_mask, num_warmup, num_samples, num_chains, seed):
    '''
    Axis-ratio posterior from pysersic (JAX + NumPyro NUTS).

    Preferred over the hand-rolled emcee fit here: it is a maintained, tested
    implementation with proper PSF rendering, and it is what geko uses for its
    morphology priors, so the two pipelines stay comparable.

    pysersic's `mask` marks pixels to EXCLUDE, so the core mask and the outer
    radius cut are passed through inverted. It parameterises the shape as an
    ellipticity, so q = 1 - ellip.
    '''
    import jax
    from pysersic import FitSingle
    from pysersic.priors import SourceProperties
    from pysersic.loss import gaussian_loss
    from pysersic.rendering import HybridRenderer

    exclude = np.asarray(~fit_mask, dtype=bool)
    props = SourceProperties(np.asarray(image, dtype=np.float32), mask=exclude)
    prior = props.generate_prior('sersic', sky_type='flat')
    LOG.info(f'pysersic prior:\n{prior}')

    fitter = FitSingle(
        data=np.asarray(image, dtype=np.float32),
        rms=np.asarray(err, dtype=np.float32),
        psf=np.asarray(psf, dtype=np.float32),
        mask=exclude,
        prior=prior,
        loss_func=gaussian_loss,
        renderer=HybridRenderer,
    )
    LOG.info(f'NUTS: {num_chains} chains x {num_samples} samples '
             f'(+{num_warmup} warmup)')

    # NOTE: we drive NUTS ourselves rather than calling fitter.sample(). That
    # method hands the sampler to PySersicResults._parse_injested_data, which
    # does `data.posterior.drop_vars(...)`; under arviz >= 1.0 az.from_numpyro
    # returns an xarray DataTree instead of an InferenceData, and a DataTree
    # node has no .drop_vars, so it raises AttributeError *after* the (slow)
    # sampling has already finished. Everything below is what sample() does
    # minus that post-processing, so we keep pysersic's model and lose only
    # its arviz bookkeeping.
    from numpyro import infer

    model = fitter.build_model(return_model=False)
    sampler = infer.MCMC(
        infer.NUTS(model, init_strategy=infer.init_to_sample),
        num_chains=num_chains, num_samples=num_samples, num_warmup=num_warmup,
    )
    sampler.run(jax.random.PRNGKey(seed))
    try:
        sampler.print_summary(exclude_deterministic=True)
    except Exception as exc:
        LOG.warning(f'numpyro summary unavailable: {exc}')

    post = sampler.get_samples()

    def flat(name):
        return np.asarray(post[name]).reshape(-1)

    missing = [k for k in ('ellip', 'theta', 'r_eff', 'n', 'xc', 'yc')
               if k not in post]
    if missing:
        raise KeyError(f'pysersic posterior is missing {missing}; '
                       f'available: {sorted(post)}')

    # pysersic wraps theta into [0, pi) in its own post-processing; do the same
    # here so the position angle is comparable with the emcee backend.
    ellip = flat('ellip')
    return {'q': 1.0 - ellip,
            'theta': np.remainder(flat('theta') + np.pi, np.pi),
            'r_eff': flat('r_eff'), 'n': flat('n'),
            'xc': flat('xc'), 'yc': flat('yc')}


def make_log_prob(image, err, psf, mask, priors):
    ny, nx = image.shape
    yy, xx = torch.meshgrid(torch.arange(ny), torch.arange(nx), indexing='ij')
    xx = xx.to(torch.float32)
    yy = yy.to(torch.float32)
    t_psf = torch.tensor(psf, dtype=torch.float32)
    t_img = torch.tensor(image, dtype=torch.float32)
    t_err = torch.tensor(err, dtype=torch.float32)
    t_mask = torch.tensor(mask)
    n_data = int(t_mask.sum())

    def model_image(theta):
        log_I_e, R_e, n, x0, y0, q, th, sky = [torch.tensor(float(v)) for v in theta]
        model = galaxy.full_sersic_model_torch(
            xx, yy, t_psf, I_e=10.0**log_I_e, R_e=R_e, n=n, x0=x0, y0=y0, q=q, theta=th)
        return model + sky

    def log_prob(theta):
        for value, (lo, hi) in zip(theta, priors):
            if not (lo <= value <= hi):
                return -np.inf
        with torch.no_grad():
            model = model_image(theta)
            if not torch.isfinite(model).all():
                return -np.inf
            chi2 = float(torch.sum((((t_img - model)/t_err)[t_mask])**2))
        return -0.5*chi2

    return log_prob, model_image, n_data


def estimate_error_inflation(image, err, psf, mask, priors, peak,
                             q_mom, th_mom, cx, cy):
    '''
    sqrt(chi2/dof) at the best fit, the factor the ERR map must be scaled by so
    that chi2/dof = 1.

    A single Sersic cannot describe a real galaxy at ~1000 sigma per pixel, so
    the formal ERR-based errors are meaninglessly small: the uncertainty on q is
    dominated by model inadequacy, not photon noise. Both backends must apply
    this, otherwise they are not sampling the same likelihood and cannot be
    compared. Returns 1.0 if the fit is already acceptable.
    '''
    from scipy.optimize import minimize

    log_prob, _, n_data = make_log_prob(image, err, psf, mask, priors)
    p_start = np.array([np.log10(max(peak, 1e-3)/3), 6.0, 1.0, cx, cy,
                        np.clip(q_mom, 0.15, 0.98), th_mom, 0.0])
    nll = lambda t: -log_prob(t)
    best = minimize(nll, p_start, method='Nelder-Mead',
                    options={'maxiter': 40000, 'maxfev': 40000,
                             'xatol': 1e-5, 'fatol': 1e-5})
    best = minimize(nll, best.x, method='Nelder-Mead',
                    options={'maxiter': 40000, 'maxfev': 40000,
                             'xatol': 1e-5, 'fatol': 1e-5})
    chi2_dof = 2*best.fun/(n_data - len(PARAM_NAMES))
    if chi2_dof <= 1:
        LOG.info(f'chi2/dof = {chi2_dof:.3f}, no error inflation needed')
        return 1.0
    inflate = float(np.sqrt(chi2_dof))
    LOG.info(f'chi2/dof = {chi2_dof:.3f} -> inflating ERR by '
             f'sqrt(chi2/dof) = {inflate:.2f}')
    return inflate


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--id', default='15665')
    ap.add_argument('--ra', type=float, default=None,
                    help='RA in degrees; looked up from the EDR catalogue by --id '
                         'when omitted')
    ap.add_argument('--dec', type=float, default=None,
                    help='Dec in degrees; looked up from the EDR catalogue by --id '
                         'when omitted')
    ap.add_argument('--band', default='F444W',
                    help='F444W matches the grism band and has a matching webbPSF')
    ap.add_argument('--psf', default=DEFAULT_PSF)
    ap.add_argument('--size', type=float, default=3.2, help='cutout size in arcsec')
    ap.add_argument('--fit-radius', type=float, default=1.1,
                    help='only fit pixels within this radius, in arcsec')
    ap.add_argument('--mask-core', type=float, default=0.20,
                    help='exclude pixels inside this radius, in arcsec. The bright '
                         'core is round and PSF-dominated and carries overwhelming '
                         'statistical weight at ~1000 sigma, so an unmasked single '
                         'Sersic fits the core and returns q ~ 1 regardless of the '
                         'disk. Set 0 to disable.')
    ap.add_argument('--q0', type=float, default=0.2,
                    help='intrinsic axis ratio for the thick-disk conversion')
    ap.add_argument('--backend', default='pysersic', choices=['pysersic', 'emcee'],
                    help='pysersic (JAX/NUTS, preferred) or the built-in emcee fit')
    ap.add_argument('--nwalkers', type=int, default=32)
    ap.add_argument('--nsteps', type=int, default=4000)
    ap.add_argument('--nburn', type=int, default=1000)
    ap.add_argument('--num-chains', type=int, default=2, help='pysersic backend')
    ap.add_argument('--num-samples', type=int, default=1000, help='pysersic backend')
    ap.add_argument('--num-warmup', type=int, default=1000, help='pysersic backend')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--outdir', default=os.path.join(ROOT, 'results'))
    ap.add_argument('--no-plots', action='store_true')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        datefmt='%H:%M:%S')
    torch.set_num_threads(1)
    args.outdir = os.path.join(args.outdir, f'ID{args.id}')
    os.makedirs(args.outdir, exist_ok=True)

    # RA/Dec default to the EDR catalogue entry for --id, so an ID is enough
    if args.ra is None or args.dec is None:
        from catalogs import lookup_source
        src = lookup_source(args.id, require_z=False)
        if args.ra is None:
            args.ra = src['ra']
        if args.dec is None:
            args.dec = src['dec']
        z = 'unknown' if src['zspec'] is None else f"{src['zspec']:.4f}"
        LOG.info(f"ID {args.id} from the EDR catalogue: RA {args.ra:.6f}, "
                 f"Dec {args.dec:+.6f}, zspec {z}")
        if src['q_phot'] is not None:
            inc_phot = np.degrees(np.arccos(np.clip(src['q_phot'], 0, 1)))
            LOG.info(f'  phot-cat second moments: q = {src["q_phot"]:.4f} '
                     f'-> i = {inc_phot:.1f} deg (rough check only)')
    else:
        LOG.info(f'ID {args.id}: RA {args.ra:.6f}, Dec {args.dec:+.6f} (given)')

    image, err, pixscale = load_cutouts(args.ra, args.dec, args.band, args.size)
    psf = load_psf(args.psf, pixscale)
    ny, nx = image.shape
    LOG.info(f'{args.band} cutout {image.shape} at {pixscale:.5f} arcsec/pix, '
             f'PSF {psf.shape}')

    bad = ~np.isfinite(image) | ~np.isfinite(err) | (err <= 0)
    image = np.nan_to_num(image)
    err = np.where(bad, 1e10, err)
    yy, xx = np.mgrid[:ny, :nx]
    cy, cx = (ny - 1)/2.0, (nx - 1)/2.0

    # centroid on the core, so the mask and the moments are centred on the source
    rr0 = np.hypot(xx - cx, yy - cy)
    core = rr0 < 0.3/pixscale
    tot = image[core].sum()
    cx = float((xx[core]*image[core]).sum()/tot)
    cy = float((yy[core]*image[core]).sum()/tot)
    rr = np.hypot(xx - cx, yy - cy)

    mask = (rr < args.fit_radius/pixscale) & ~bad
    if args.mask_core > 0:
        mask &= rr >= args.mask_core/pixscale
    LOG.info(f'centroid ({cx:.2f}, {cy:.2f}); fitting {int(mask.sum())} pixels with '
             f'{args.mask_core}" <= r < {args.fit_radius}"')

    # background from a blank annulus -- a sigma-clipped estimate over the whole
    # cutout is contaminated by the galaxy, which fills it
    ann = (rr > 2.0/pixscale) & ~bad
    bkg_sigma = (float(sigma_clipped_stats(image[ann])[2]) if ann.sum() > 100
                 else float(np.median(err[~bad])))
    peak = float(np.nanmax(image))
    LOG.info(f'peak/ERR = {peak/np.median(err[mask]):.0f}, blank-annulus sigma = '
             f'{bkg_sigma:.5g} (median ERR there = {np.median(err[ann]):.5g})')

    # moments of the masked (disk-dominated) region, as the starting guess
    wt = np.clip(image, 0, None)*mask
    t = wt.sum()
    mxx = ((xx - cx)**2*wt).sum()/t
    myy = ((yy - cy)**2*wt).sum()/t
    mxy = ((xx - cx)*(yy - cy)*wt).sum()/t
    dd = np.hypot((mxx - myy)/2, mxy)
    q_mom = float(np.sqrt(max((mxx + myy)/2 - dd, 1e-6)/((mxx + myy)/2 + dd)))
    th_mom = float(0.5*np.arctan2(2*mxy, mxx - myy))
    LOG.info(f'moments of the fitted region: q = {q_mom:.3f}, theta = {th_mom:+.3f} rad')

    # (lo, hi) bounds = uniform priors
    priors = [
        (-3.0, 4.0),                 # log_I_e
        (0.5, 60.0),                 # R_e   [pix]
        (0.3, 6.0),                  # n
        (cx - 10, cx + 10),          # x0
        (cy - 10, cy + 10),          # y0
        (0.1, 1.0),                  # q  (sersic_model_torch clamps at 0.1)
        (-np.pi, 2*np.pi),           # theta
        (-5*bkg_sigma, 5*bkg_sigma),  # sky
    ]
    if args.backend == 'pysersic':
        # The emcee backend rescales ERR so chi2/dof = 1 before sampling (see the
        # comment in run_emcee_backend). pysersic must get the SAME likelihood or
        # the two backends are not a like-for-like cross-check. It also matters
        # numerically: on the raw ERR map F444W sits at chi2/dof ~ 700, and NUTS
        # responds to that curvature by collapsing its step size to ~1e-8 and
        # hitting max tree depth (1023 leapfrog steps) every iteration, which
        # neither converges nor terminates in reasonable time.
        inflate = estimate_error_inflation(image, err, psf, mask, priors,
                                           peak, q_mom, th_mom, cx, cy)
        post = run_pysersic(image, err*inflate, psf, mask, args.num_warmup,
                            args.num_samples, args.num_chains, args.seed)
        chain = np.column_stack([post[k] for k in ('q', 'theta', 'r_eff', 'n')])
        chain_names = ['q', 'theta', 'r_eff', 'n']
        LOG.info('--- pysersic posterior ---')
        for j, name in enumerate(chain_names):
            lo, med, hi = np.percentile(chain[:, j], [16, 50, 84])
            unit = f'  ({med*pixscale:.4f}")' if name == 'r_eff' else ''
            LOG.info(f'  {name:8s} {med:10.5f}  -{med-lo:.5f} +{hi-med:.5f}{unit}')
        q, model_image, best = post['q'], None, None
    else:
        q, chain, chain_names, model_image, best = run_emcee_backend(
            image, err, psf, mask, priors, peak, bkg_sigma, q_mom, th_mom, cx, cy,
            pixscale, args)

    report_inclination(q, chain, chain_names, args, pixscale, image, err, mask,
                       model_image, best)
    return 0


def run_emcee_backend(image, err, psf, mask, priors, peak, bkg_sigma,
                      q_mom, th_mom, cx, cy, pixscale, args):
    '''Built-in Sersic fit, kept as an independent cross-check on pysersic.'''
    log_prob, model_image, n_data = make_log_prob(image, err, psf, mask, priors)
    ndim = len(PARAM_NAMES)

    p_start = np.array([np.log10(max(peak, 1e-3)/3), 6.0, 1.0, cx, cy,
                        np.clip(q_mom, 0.15, 0.98), th_mom, 0.0])
    if not np.isfinite(log_prob(p_start)):
        raise RuntimeError('starting point has zero probability')

    # Nelder-Mead polish first, restarted, so the walkers start near the mode
    from scipy.optimize import minimize
    nll = lambda t: -log_prob(t)
    best = minimize(nll, p_start, method='Nelder-Mead',
                    options={'maxiter': 40000, 'maxfev': 40000,
                             'xatol': 1e-5, 'fatol': 1e-5})
    best = minimize(nll, best.x, method='Nelder-Mead',
                    options={'maxiter': 40000, 'maxfev': 40000,
                             'xatol': 1e-5, 'fatol': 1e-5})
    dof = n_data - ndim
    chi2_dof = 2*best.fun/dof
    LOG.info(f'Nelder-Mead: chi2/dof = {chi2_dof:.3f}')
    for name, value in zip(PARAM_NAMES, best.x):
        LOG.info(f'    {name:8s} = {value:.5f}')

    # A single Sersic cannot describe a real galaxy at ~1000 sigma per pixel, so the
    # formal errors from the ERR map alone are meaninglessly small: the uncertainty
    # on q is dominated by model inadequacy, not by photon noise. Rescale the errors
    # so chi2/dof = 1, the standard remedy, which propagates a realistic width.
    if chi2_dof > 1:
        inflate = np.sqrt(chi2_dof)
        LOG.info(f'inflating ERR by sqrt(chi2/dof) = {inflate:.2f} so that the '
                 f'posterior width reflects model inadequacy')
        log_prob, model_image, n_data = make_log_prob(
            image, err*inflate, psf, mask, priors)
    else:
        inflate = 1.0

    emcee = fitting._import_emcee()
    rng = np.random.default_rng(args.seed)
    scale = np.array([0.01, 0.05, 0.02, 0.02, 0.02, 0.01, 0.01, 1e-4*max(bkg_sigma, 1e-6)])
    p0 = np.empty((args.nwalkers, ndim))
    for j in range(args.nwalkers):
        while True:
            cand = best.x + scale*rng.standard_normal(ndim)
            if np.isfinite(log_prob(cand)):
                p0[j] = cand
                break

    sampler = emcee.EnsembleSampler(args.nwalkers, ndim, log_prob)
    LOG.info(f'sampling {args.nwalkers} walkers x {args.nsteps} steps '
             f'(+{args.nburn} burn)')
    sampler.run_mcmc(p0, args.nburn + args.nsteps, progress=False)
    chain = sampler.get_chain(discard=args.nburn, flat=True)
    LOG.info(f'acceptance = {np.mean(sampler.acceptance_fraction):.3f}')
    try:
        tau = emcee.autocorr.integrated_time(sampler.get_chain(discard=args.nburn),
                                             quiet=True)
        LOG.info(f'n_steps/tau (min) = {args.nsteps/np.max(tau):.1f}')
    except Exception:
        pass

    LOG.info('--- Sersic posterior (emcee backend) ---')
    for j, name in enumerate(PARAM_NAMES):
        lo, med, hi = np.percentile(chain[:, j], [16, 50, 84])
        unit = f'  ({med*pixscale:.4f}")' if name == 'R_e' else ''
        LOG.info(f'  {name:8s} {med:10.5f}  -{med-lo:.5f} +{hi-med:.5f}{unit}')
    return chain[:, PARAM_NAMES.index('q')], chain, list(PARAM_NAMES), model_image, best


def report_inclination(q, chain, chain_names, args, pixscale, image, err, mask,
                       model_image=None, best=None):
    '''Axis-ratio posterior -> inclination posterior, prior snippet, plots.'''
    q_lo, q_med, q_hi = np.percentile(q, [16, 50, 84])
    LOG.info(f'\naxis ratio q = {q_med:.4f} -{q_med-q_lo:.4f} +{q_hi-q_med:.4f}')
    if q_med < args.q0 + 0.05 or q_hi > 0.97:
        LOG.warning('q is near a bound; the inclination conversion is unreliable there')

    # thin disk: q == cos(inc_v) exactly, in DINGO's own convention
    inc_thin = np.arccos(np.clip(q, 1e-6, 1.0))
    # thick disk: cos^2 i = (q^2 - q0^2)/(1 - q0^2)
    cos2 = (q**2 - args.q0**2)/(1.0 - args.q0**2)
    inc_thick = np.arccos(np.sqrt(np.clip(cos2, 0.0, 1.0)))

    for tag, inc in [('thin disk (q = cos i)', inc_thin),
                     (f'thick disk (q0 = {args.q0})', inc_thick)]:
        lo, med, hi = np.percentile(inc, [16, 50, 84])
        LOG.info(f'  inc_v {tag:28s} = {med:.4f} -{med-lo:.4f} +{hi-med:.4f} rad '
                 f'({np.degrees(med):.2f} deg)')

    mu = float(np.mean(inc_thin))
    sigma = float(np.std(inc_thin))
    LOG.info('\n--- paste into the mcmc.priors block of the kinematics config ---')
    LOG.info(f'    velocity.inc_v:   {{type: gaussian, mu: {mu:.4f}, sigma: {sigma:.4f}, '
             f'min: 0.0873, max: 1.4835}}')
    LOG.info('(thin-disk value: it is the geometry arctangent_disk_velocity_model itself '
             'assumes. Widen sigma if you want to allow for intrinsic thickness.)')

    out = os.path.join(args.outdir,
                       f'ID{args.id}_sersic_{args.band}_{args.backend}.npz')
    np.savez_compressed(
        out, chain=chain, param_names=np.array(chain_names, dtype=object),
        q=q, inc_thin=inc_thin, inc_thick=inc_thick, pixscale=pixscale,
        q0=args.q0, inc_mu=mu, inc_sigma=sigma, band=args.band,
        backend=args.backend, ra=args.ra, dec=args.dec,
        best=(best.x if best is not None else np.array([])),
    )
    LOG.info(f'saved {out}')

    if not args.no_plots and model_image is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        with torch.no_grad():
            model = model_image(np.median(chain, axis=0)).numpy()
        resid = np.where(mask, (image - model)/err, np.nan)
        _, axs = plt.subplots(1, 3, figsize=(13, 4))
        vmax = np.nanpercentile(image, 99.5)
        for ax, img, title, kw in [
                (axs[0], image, f'{args.band} data', dict(vmin=0, vmax=vmax)),
                (axs[1], model, 'Sersic (x) PSF + sky', dict(vmin=0, vmax=vmax)),
                (axs[2], resid, '(data - model)/err',
                 dict(vmin=-5, vmax=5, cmap='seismic'))]:
            im = ax.imshow(img, origin='lower', **kw)
            ax.set_title(title)
            ax.grid(False)
            plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        png = os.path.join(args.outdir,
                           f'ID{args.id}_sersic_{args.band}_{args.backend}.png')
        plt.savefig(png, dpi=140, bbox_inches='tight')
        LOG.info(f'saved {png}')
        n_fit = int(mask.sum())
        LOG.info(f'chi2/dof at the posterior median = '
                 f'{float(np.nansum(resid[mask]**2))/(n_fit - len(chain_names)):.4f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
