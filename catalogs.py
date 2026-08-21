'''
Look up a SAPPHIRES source by ID.

Give it an ID and it returns RA, Dec and the spectroscopic redshift from the EDR
catalogues, so nothing downstream has to carry hand-copied coordinates or a
hand-copied z. That matters here: `Archive_6/ID15665/config_kinematics.yaml`
carries `summary.z: 1.123`, which is ID3700's redshift, not ID15665's (1.1613),
and the error is invisible in the fit -- see MCMC_STATUS.md.

    from catalogs import lookup_source, print_source
    src = lookup_source(15665)
    print_source(src)
    ra, dec, z = src['ra'], src['dec'], src['zspec']

Catalogue location resolution order:
    1. `cat_dir` argument
    2. $SAPPHIRES_CAT_DIR
    3. the first of CANDIDATE_DIRS that exists (local Mac, then magnif)
'''

import os
import logging

import numpy as np
from astropy.io import fits

LOG = logging.getLogger(__name__)

SPEC_CAT = 'sapphires_edr_spec_cat.fits'
PHOT_CAT = 'sapphires_edr_phot_cat.fits'

CANDIDATE_DIRS = [
    '/Users/aakavan/Astro_Research/Galaxy_Kinematics/EDR_data',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'EDR_data'),
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), 'EDR_data'),
    '../EDR',            # magnif, relative to the notebook directory
    '../../EDR',
]

_CACHE = {}


def find_cat_dir(cat_dir=None):
    '''Directory holding the two EDR catalogues.'''
    if cat_dir is not None:
        candidates = [cat_dir]
    elif os.environ.get('SAPPHIRES_CAT_DIR'):
        candidates = [os.environ['SAPPHIRES_CAT_DIR']]
    else:
        candidates = CANDIDATE_DIRS
    for d in candidates:
        if os.path.exists(os.path.join(d, SPEC_CAT)):
            return d
    raise FileNotFoundError(
        f'could not find {SPEC_CAT} in any of {candidates}. Pass cat_dir= or set '
        f'$SAPPHIRES_CAT_DIR.')


def _load(cat_dir, name):
    key = (os.path.abspath(cat_dir), name)
    if key not in _CACHE:
        path = os.path.join(cat_dir, name)
        _CACHE[key] = fits.getdata(path)
        LOG.debug(f'loaded {path}')
    return _CACHE[key]


def lookup_source(source_id, cat_dir=None, require_z=True):
    '''
    RA/Dec/redshift for one source ID.

    Returns a dict with:
        id, ra, dec                  -- degrees
        zspec, zconf, nlines, name   -- None if the ID is not in the spec catalogue
        q_phot, pa_phot              -- axis ratio and orientation from the photometric
                                        catalogue's second moments, or None. A crude
                                        sanity check on a Sersic fit's `q`, not a
                                        substitute for one: it is unweighted, uses no
                                        PSF, and is measured on the detection image.
        in_spec_cat, in_phot_cat     -- bool

    require_z=True raises if the ID has no spectroscopic redshift.
    '''
    cat_dir = find_cat_dir(cat_dir)
    source_id = int(source_id)
    out = {'id': source_id, 'cat_dir': cat_dir,
           'ra': None, 'dec': None, 'zspec': None, 'zconf': None,
           'nlines': None, 'name': None, 'q_phot': None, 'pa_phot': None,
           'in_spec_cat': False, 'in_phot_cat': False}

    spec = _load(cat_dir, SPEC_CAT)
    hit = spec[spec['ID'] == source_id]
    if len(hit):
        r = hit[0]
        out.update(in_spec_cat=True,
                   ra=float(r['RA']), dec=float(r['DEC']),
                   zspec=float(r['zspec']), zconf=float(r['zconf']),
                   nlines=int(r['nlines']),
                   name=str(r['name']).strip())

    try:
        phot = _load(cat_dir, PHOT_CAT)
    except FileNotFoundError:
        phot = None
    if phot is not None:
        hit = phot[phot['ID'] == source_id]
        if len(hit):
            r = hit[0]
            out['in_phot_cat'] = True
            if out['ra'] is None:
                out['ra'], out['dec'] = float(r['RA']), float(r['DEC'])
            a, b = float(r['semimajor_sigma']), float(r['semiminor_sigma'])
            if a > 0:
                out['q_phot'] = b/a
            out['pa_phot'] = float(r['orientation'])

    if out['ra'] is None:
        raise KeyError(f'ID {source_id} is in neither {SPEC_CAT} nor {PHOT_CAT} '
                       f'under {cat_dir}')
    if require_z and out['zspec'] is None:
        raise KeyError(f'ID {source_id} has no spectroscopic redshift in {SPEC_CAT}')
    return out


SPEC1D_DIRNAME = 'sapphires_edr_1d_spec'

# Per-source starting values and rotation sense, from the grad student's working
# batch in 1.1-astr540-plot.ipynb. sign = -1 means the galaxy rotates the other
# way; his loop sets V_rot = 300*sign. Starting a sign=-1 galaxy at +V_rot with
# theta_v = 0 makes Adam escape through inc -> 0 (which forces vz = 0 everywhere)
# instead of finding the theta_v ~ pi solution -- that is exactly how ID3700 and
# ID21619 failed here.
SOURCE_HINTS = {
    3700:  {'sign': -1, 'R_v': 4, 'x0_v': 40, 'y0_v': 40},
    6170:  {'sign':  1, 'R_v': 2, 'x0_v': 40, 'y0_v': 40},
    8910:  {'sign':  1, 'R_v': 2, 'x0_v': 40, 'y0_v': 40},
    15665: {'sign': -1, 'R_v': 2, 'x0_v': 40, 'y0_v': 40},
    16887: {'sign':  1, 'R_v': 2, 'x0_v': 40, 'y0_v': 40},
    21619: {'sign': -1, 'R_v': 1, 'x0_v': 44, 'y0_v': 30},
    21636: {'sign':  1, 'R_v': 2, 'x0_v': 40, 'y0_v': 40},
}


def lookup_module(source_id, grism='R', filter='F444W', cat_dir=None):
    '''
    Which NIRCam module a source was observed on, from the EDR 1D spectra.

    The cutouts carry no MODULE keyword, but spec_1d_M0416_<filter>_ID<n>_<R|C>.fits
    records N_A and N_B, the number of coadded exposures per module. Whichever is
    non-zero is the module. This matters: `module` selects the tracing and
    dispersion calibration tables, so guessing it wrong silently gives the wrong
    trace. Three of the seven Pa-alpha sources are on B, not A.

    Returns 'A' or 'B', or None when no spectrum file exists.
    '''
    cat_dir = cat_dir or find_cat_dir()
    path = os.path.join(cat_dir, SPEC1D_DIRNAME,
                        f'spec_1d_M0416_{filter}_ID{int(source_id)}_{grism}.fits')
    if not os.path.exists(path):
        LOG.warning(f'no 1D spectrum at {path}; cannot determine module')
        return None
    h = fits.getheader(path, 1)
    n_a, n_b = h.get('N_A', 0) or 0, h.get('N_B', 0) or 0
    if n_a == n_b:
        LOG.warning(f'ID{source_id} {grism}: N_A == N_B == {n_a}, module ambiguous')
        return None
    return 'A' if n_a > n_b else 'B'


def print_source(src):
    '''Print a lookup_source() result.'''
    print(f"ID {src['id']}")
    print(f"  RA, Dec    = {src['ra']:.6f}, {src['dec']:+.6f} deg")
    if src['zspec'] is not None:
        extra = []
        if src['zconf'] is not None:
            extra.append(f"zconf {src['zconf']:.2f}")
        if src['nlines'] is not None:
            extra.append(f"{src['nlines']} lines")
        if src['name']:
            extra.append(src['name'])
        print(f"  zspec      = {src['zspec']:.4f}" +
              (f"   ({', '.join(extra)})" if extra else ''))
    else:
        print('  zspec      = not in the spec catalogue')
    if src['q_phot'] is not None:
        inc = np.degrees(np.arccos(np.clip(src['q_phot'], 0, 1)))
        print(f"  q (phot cat second moments) = {src['q_phot']:.4f}  "
              f"-> i = {inc:.1f} deg   [rough check only]")
    print(f"  catalogues: {src['cat_dir']}")


def check_config_redshift(config_path, source_id=None, cat_dir=None):
    '''
    Compare a kinematics config's `summary.z` against the catalogue redshift.

    Returns (z_config, z_catalog). Warns loudly when they disagree, because the
    fit stays internally self-consistent with a wrong z and so converges
    normally while biasing every velocity by (1+z_cat)/(1+z_cfg) - 1.
    '''
    import yaml
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    z_config = float(cfg['summary']['z'])

    if source_id is None:
        base = os.path.basename(os.path.dirname(os.path.abspath(config_path)))
        digits = ''.join(ch for ch in base if ch.isdigit())
        if not digits:
            raise ValueError(f'cannot infer an ID from {config_path}; pass source_id=')
        source_id = int(digits)

    z_catalog = lookup_source(source_id, cat_dir=cat_dir)['zspec']
    if abs(z_config - z_catalog) > 1e-4:
        bias = (1 + z_catalog)/(1 + z_config) - 1
        LOG.warning(
            f'{config_path}: summary.z = {z_config} but ID {source_id} has '
            f'zspec = {z_catalog:.4f} -> velocities biased by {100*bias:+.2f}%')
    return z_config, z_catalog


SAMPLE_IDS = [3700, 6170, 8910, 15665, 16887, 21619, 21636]


def _main():
    import sys, argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('ids', nargs='*', type=int, help='source IDs (default: all 7)')
    ap.add_argument('--make-config', action='store_true',
                    help='write data/ID<n>/config_kinematics.yaml for each ID')
    ap.add_argument('--module-R', default='A')
    ap.add_argument('--module-C', default='A')
    ap.add_argument('--inc-mu', type=float, default=None,
                    help='photometric inclination in rad; omitted -> sin_i prior')
    ap.add_argument('--inc-sigma', type=float, default=0.0436)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    for i in (args.ids or SAMPLE_IDS):
        try:
            print_source(lookup_source(i, require_z=False))
            if args.make_config:
                make_config(i, module_R=args.module_R, module_C=args.module_C,
                            inc_mu=args.inc_mu, inc_sigma=args.inc_sigma,
                            overwrite=args.overwrite)
        except (KeyError, FileExistsError) as exc:
            print(f'ID {i}: {exc}')
        print()




# ── config generation ─────────────────────────────────────────────────────────

CONFIG_TEMPLATE = '''\
# Generated by catalogs.make_config() -- regenerate rather than hand-edit the
# header, so RA/Dec/z always come from the EDR catalogue. ID15665 once carried
# ID3700's redshift because a config was copied by hand; that cannot happen here.

summary:
  name: ID{id}
  ra: {ra:.6f}
  dec: {dec:.6f}
  z: {z:.4f}          # SAPPHIRES EDR zspec for ID{id}
  lambda_rest: {lambda_rest}
  filter: {filter}
  mode: kinematics
  r_fit: {r_fit}

# NOTE: no explicit `cutout:` here on purpose. The cutout position depends on z
# through lambda_rest*(1+z), so a hard-coded one goes stale the moment the
# redshift is corrected (it moved ~73 px for ID15665). Letting the r_fit
# fallback derive it keeps the two consistent by construction.

image:
  R:
    module: {module_R}
    path: ['ID{id}/R.fits', 0]
    dx: 0
    dy: 0
  C:
    module: {module_C}
    path: ['ID{id}/C.fits', 0]
    dx: 0
    dy: 0

velocity:
  model: arctan
  V_rot: {V_rot}
  R_v: {R_v}
  x0_v: {x0_v}
  y0_v: {y0_v}
  theta_v: 0
  inc_v: {inc_init}

# ----------------------------------------------------------------------------
# Fitting strategy (also defines the free-parameter set that fit_MCMC samples)
# ----------------------------------------------------------------------------

fitting:
  - method: Adam
    maxiter: 2500
    scheduler: StepLR
    default:
      lr: 0.05
      min: -1e10
      max: 1e10
      fit: true
    override:
      image.R.dx, image.R.dy, image.C.dx, image.C.dy:
        lr: 0.5
      velocity.theta_v:
        fit: true
      # R_v is a turnover RADIUS: without this floor Adam happily returns a
      # negative one (ID21619 gave -0.011), which is unphysical and then blocks
      # MCMC initialisation. The grad student's working batch sets the same bound.
      velocity.R_v:
        min: 0
      velocity.inc_v:
        min: 0
        max: 1.57

  - method: Adam
    maxiter: 1500
    scheduler: ReduceLROnPlateau
    default:
      lr: 0.003
      min: -1e10
      max: 1e10
      fit: true
    override:
      velocity.theta_v:
        fit: false

# ----------------------------------------------------------------------------
# MCMC posterior sampling (read only by fit_MCMC)
# ----------------------------------------------------------------------------

mcmc:
  # Stage 1 (index 0): stage 2 freezes theta_v for the polish, so sampling there
  # would silently drop it from the free set.
  stage: 0
  nwalkers: {nwalkers}
  nsteps: {nsteps}
  nburn: {nburn}
  xy_iters: 40
  xy_tol: 5e-3
  log_cadence: 200
  seed: 42
  checkpoint: ../../results/ID{id}/ID{id}_chain_checkpoint.npz
  checkpoint_every: 500

  noise:
    fit_ln_f: true

  init_scatter:
    velocity.V_rot:  2.0
    velocity.R_v:    0.05
    velocity.x0_v:   0.05
    velocity.y0_v:   0.05
    velocity.theta_v: 0.01
    velocity.inc_v:  0.01
    image.R.dx:      0.02
    image.R.dy:      0.02
    image.C.dx:      0.02
    image.C.dy:      0.02
    noise.ln_f:      0.05

  priors:
    # one-signed: (V_rot, theta_v) and (-V_rot, theta_v+pi) give an identical vz field
    velocity.V_rot:   {{type: uniform,     min: 50.0,   max: 1200.0}}
    # lower bound keeps the damped fixed-point iteration contracting
    velocity.R_v:     {{type: log_uniform, min: 0.5,    max: 40.0}}
    # Centred on the cutout with a width that still admits an off-centre source:
    # ID6170's centre fits at y0 = 54.9, which a +-10 px window would exclude and
    # then fail at walker initialisation.
    velocity.x0_v:    {{type: gaussian,    mu: {x0_v},  sigma: 5.0, min: {c_lo}, max: {c_hi}}}
    velocity.y0_v:    {{type: gaussian,    mu: {y0_v},  sigma: 5.0, min: {c_lo}, max: {c_hi}}}
    velocity.theta_v: {{type: uniform,     min: 0.0,    max: 6.283185}}
{inc_prior_block}
    image.R.dx:       {{type: gaussian,    mu: 0.0,     sigma: 2.0, min: -10.0, max: 10.0}}
    image.R.dy:       {{type: gaussian,    mu: 0.0,     sigma: 2.0, min: -10.0, max: 10.0}}
    image.C.dx:       {{type: gaussian,    mu: 0.0,     sigma: 2.0, min: -10.0, max: 10.0}}
    image.C.dy:       {{type: gaussian,    mu: 0.0,     sigma: 2.0, min: -10.0, max: 10.0}}
    noise.ln_f:       {{type: uniform,     min: -2.0,   max: 2.0}}
'''

INC_UNINFORMATIVE = '''\
    # No photometric measurement supplied. sin_i is the isotropic-orientation
    # prior; it does NOT break the V_rot--inc_v degeneracy (corr ~ -0.99), so
    # V_rot alone will be poorly constrained. Pass inc_mu/inc_sigma, or let the
    # notebook overwrite this from its pysersic fit.
    velocity.inc_v:   {type: sin_i,   min: 0.0873, max: 1.4835}'''

INC_GAUSSIAN = '''\
    # Photometric inclination, via q = cos(inc_v) -- exact in DINGO's convention,
    # since sersic_model_torch and arctangent_disk_velocity_model share the same
    # deprojection. Do not tighten sigma to a single fit's formal error: the
    # spread BETWEEN methods and bands is what actually limits this.
    velocity.inc_v:   {{type: gaussian, mu: {mu:.4f}, sigma: {sigma:.4f}, min: 0.0873, max: 1.4835}}'''


def make_config(source_id, outdir=None, module_R=None, module_C=None,
                filter='F444W', lambda_rest=1.875, r_fit=40,
                inc_mu=None, inc_sigma=0.0436, sign=None,
                V_rot=300.0, R_v=None, x0_v=None, y0_v=None,
                nwalkers=32, nsteps=8000, nburn=2000,
                cat_dir=None, overwrite=False):
    '''
    Write data/ID<n>/config_kinematics.yaml with RA/Dec/z filled from the
    catalogue.

    Everything source-specific is derived: only the ID and the grism module
    letters are inputs. Deliberately omits `cutout:` (see the template comment).

    inc_mu : photometric inclination in radians. None -> a sin_i prior, which
             leaves the V_rot--inc_v degeneracy unbroken.

    Returns the path written.
    '''
    src = lookup_source(source_id, cat_dir=cat_dir)
    source_id = int(source_id)
    hints = SOURCE_HINTS.get(source_id, {})

    # module drives which calibration table is loaded -- never guess it
    if module_R is None:
        module_R = lookup_module(source_id, 'R', filter, cat_dir) or 'A'
    if module_C is None:
        module_C = lookup_module(source_id, 'C', filter, cat_dir) or 'A'
    if sign is None:
        sign = hints.get('sign', 1)
    if R_v is None:
        R_v = hints.get('R_v', 2)

    if outdir is None:
        outdir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', f'ID{source_id}')
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, 'config_kinematics.yaml')
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f'{path} exists; pass overwrite=True to replace it')

    centre = r_fit          # the cutout is (2*r_fit+1)^2, so its centre is r_fit
    x0_v = hints.get('x0_v', centre) if x0_v is None else x0_v
    y0_v = hints.get('y0_v', centre) if y0_v is None else y0_v
    inc_block = (INC_UNINFORMATIVE if inc_mu is None
                 else INC_GAUSSIAN.format(mu=float(inc_mu), sigma=float(inc_sigma)))

    text = CONFIG_TEMPLATE.format(
        id=source_id, ra=src['ra'], dec=src['dec'], z=src['zspec'],
        lambda_rest=lambda_rest, filter=filter, r_fit=r_fit,
        module_R=module_R, module_C=module_C,
        V_rot=float(V_rot)*sign, R_v=R_v, x0_v=float(x0_v), y0_v=float(y0_v),
        c_lo=float(centre - 0.75*r_fit), c_hi=float(centre + 0.75*r_fit),
        inc_init=1.2,
        nwalkers=nwalkers, nsteps=nsteps, nburn=nburn,
        inc_prior_block=inc_block)

    with open(path, 'w') as fh:
        fh.write(text)
    LOG.info(f'wrote {path}  (z={src["zspec"]:.4f}, '
             f'RA/Dec {src["ra"]:.6f} {src["dec"]:+.6f}, '
             f'inc prior: {"sin_i" if inc_mu is None else f"gaussian mu={inc_mu:.4f}"})')

    missing = [f for f in ('R.fits', 'C.fits')
               if not os.path.exists(os.path.join(outdir, f))]
    if missing:
        LOG.warning(f'{outdir} is missing {missing}; the config will not load until '
                    f'those cutouts are in place')
    return path


if __name__ == '__main__':
    _main()
