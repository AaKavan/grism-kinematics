'''
Non-parametric rotation curves for DINGO.

WHY THIS EXISTS
---------------
`dingo.kinematics.arctangent_disk_velocity_model` hard-wires

    V_circ(r) = (2/pi) * V_rot * arctan(r / R_v)

which is monotonically rising and asymptotically flat BY CONSTRUCTION. It cannot
represent a declining outer rotation curve, and it cannot tell you whether the
data prefer a different shape -- any departure is absorbed into R_v. That is a
problem if the science question is "are z ~ 1 disks already flat, rising, or
mildly declining", because the model answers it by assumption.

This module replaces the two-parameter shape with a free velocity per radial
ring -- a tilted-ring fit, the standard non-parametric approach (3DBarolo,
GalPaK3D) -- and fits the ring velocities against DINGO's existing R-C objective.
The geometry (centre, position angle, inclination) is held fixed at the values
the parametric fit already found, so only the SHAPE of V(r) is free.

What makes this possible here is the pair of perpendicular dispersion
directions: requiring R and C to agree constrains the velocity locally, without
needing a light model. A single-dispersion code cannot do this.

Everything below leaves `dingo/` untouched. The velocity function is swapped in
temporarily through a context manager and always restored.

    from rotation_curve import fit_rotation_curve, plot_rotation_curve
    rc = fit_rotation_curve(fitter, n_rings=7)
    plot_rotation_curve(rc, kpc_per_pixel=0.5198)
'''

import contextlib
import logging

import numpy as np
import torch

from dingo import kinematics

LOG = logging.getLogger(__name__)

GEOM_KEYS = ('x0_v', 'y0_v', 'theta_v', 'inc_v')


# ── the velocity model ────────────────────────────────────────────────────────

def ring_circular_speed(r, r_knots, v_knots):
    '''
    Piecewise-linear V_circ(r) through (r_knots, v_knots), held flat beyond the
    outermost knot. Differentiable in v_knots, which is what the fit needs.

    torch has no interp, and torch.searchsorted does not backpropagate through
    the index, so this is written as an explicit weighted sum over segments --
    slower than a lookup but differentiable and only ~8 terms.
    '''
    r = r.reshape(-1)
    v = torch.zeros_like(r)
    # inside the first knot: linear from 0 at r=0 (a disk has V(0) = 0)
    seg = r < r_knots[0]
    v = v + seg*(v_knots[0]*r/r_knots[0])
    for i in range(len(r_knots) - 1):
        lo, hi = r_knots[i], r_knots[i + 1]
        seg = (r >= lo) & (r < hi)
        t = (r - lo)/(hi - lo)
        v = v + seg*(v_knots[i] + t*(v_knots[i + 1] - v_knots[i]))
    v = v + (r >= r_knots[-1])*v_knots[-1]        # flat outside
    return v


def make_ring_velocity_fn(r_knots, v_knots):
    '''
    A drop-in replacement for `arctangent_disk_velocity_model` that uses the ring
    profile instead of the arctangent. The projection is copied EXACTLY from
    dingo/kinematics.py -- same deprojected radius, same sin(i)*cos(phi) term --
    so the only difference between this and the parametric model is the shape of
    V_circ(r). V_rot and R_v arrive in kwargs and are ignored.
    '''
    def vz_fn(x, y, V_rot=None, R_v=None, x0_v=0., y0_v=0.,
              theta_v=0., inc_v=0., **kwargs):
        dx, dy = x - x0_v, y - y0_v
        cos_t, sin_t = torch.cos(theta_v), torch.sin(theta_v)
        x_p = cos_t*dx + sin_t*dy
        y_p = -sin_t*dx + cos_t*dy
        r = torch.sqrt(x_p**2 + (y_p/torch.cos(inc_v))**2 + 1e-8)
        V_circ = ring_circular_speed(r, r_knots, v_knots).reshape(r.shape)
        return V_circ*torch.sin(inc_v)*(x_p/(r + 1e-8))
    return vz_fn


@contextlib.contextmanager
def ring_model(r_knots, v_knots):
    '''
    Temporarily swap the package-level velocity model.

    `iteratively_find_xy` calls `arctangent_disk_velocity_model` by name at
    module scope, so this is the least invasive way to change the velocity law
    without editing the grad student's file. Always restored, including on error.
    '''
    original = kinematics.arctangent_disk_velocity_model
    kinematics.arctangent_disk_velocity_model = make_ring_velocity_fn(r_knots, v_knots)
    try:
        yield
    finally:
        kinematics.arctangent_disk_velocity_model = original


# ── fitting ───────────────────────────────────────────────────────────────────

def _masked_sse(fitter, mask):
    '''R-C residual over the frozen mask, differentiable.'''
    fitter._reset_state()
    fitter.loss()                       # populates image_R / image_C
    d = fitter.image_R - fitter.image_C
    return torch.sum(d[mask]**2)


def fit_rotation_curve(fitter, n_rings=7, r_max=None, r_min=None,
                       maxiter=400, lr=8.0, seed_from_arctan=True,
                       profile_errors=True, verbose=True):
    '''
    Fit one circular velocity per radial ring, geometry held fixed.

    Parameters
    ----------
    fitter : an ALREADY-FITTED KinematicsFitter, with setup_noise_model() called
    n_rings : number of free ring velocities
    r_max : outer knot in pixels. Defaults to the 90th percentile of the
        deprojected radius over the noise mask -- i.e. where there is still
        signal, rather than the corner of the cutout.
    seed_from_arctan : start the rings on the fitted arctangent curve. Starting
        from zero works too but takes longer and can find a mirrored solution.
    profile_errors : per-ring 1-sigma from a delta-chi2 = 1 scan. Costs
        n_rings * ~12 likelihood evaluations.

    Returns a dict with r_knots (pixels), v_knots, v_err, the arctangent curve
    evaluated at the same radii, and the SSE of each model for comparison.
    '''
    if getattr(fitter, 'mcmc_mask', None) is None:
        raise RuntimeError('call fitter.setup_noise_model() first -- the ring fit '
                           'uses the same frozen mask as the likelihood')
    mask = fitter.mcmc_mask
    vel = fitter._get_model_params('velocity')
    geom = {k: float(vel[k]) for k in GEOM_KEYS}
    V_rot0, R_v0 = abs(float(vel['V_rot'])), float(vel['R_v'])

    # deprojected radius over the mask, to place the knots where there is signal
    ny, nx = fitter.true_grism_R.shape
    yy, xx = torch.meshgrid(torch.arange(ny, dtype=torch.float32),
                            torch.arange(nx, dtype=torch.float32), indexing='ij')
    dx, dy = xx - geom['x0_v'], yy - geom['y0_v']
    ct, st = np.cos(geom['theta_v']), np.sin(geom['theta_v'])
    x_p, y_p = ct*dx + st*dy, -st*dx + ct*dy
    r_map = torch.sqrt(x_p**2 + (y_p/np.cos(geom['inc_v']))**2 + 1e-8)
    r_in_mask = r_map[mask].detach().cpu().numpy()

    # Knot placement matters more than it looks. The noise mask is a COVERAGE
    # mask -- it keeps ~96% of the frame -- so radius percentiles over it just
    # describe the cutout geometry, not where the galaxy is. Placing the inner
    # knot there (10.7 px, with R_v = 3.3 px) put the entire turnover inside the
    # first segment, where the profile is a straight line from zero: nothing like
    # an arctangent, and the "more flexible" model fit WORSE than the parametric
    # one. Anchor the inner knot near the turnover instead, and space the knots
    # logarithmically so the rise is resolved and the flat part is not
    # oversampled.
    if r_max is None:
        r_max = float(np.percentile(r_in_mask[r_in_mask > 0], 75))
    if r_min is None:
        r_min = max(0.6, 0.4*R_v0)
    r_knots = torch.tensor(np.geomspace(r_min, r_max, n_rings), dtype=torch.float32)

    v0 = ((2/np.pi)*V_rot0*np.arctan(r_knots.numpy()/R_v0) if seed_from_arctan
          else np.full(n_rings, 0.5*V_rot0))
    v_knots = torch.tensor(v0, dtype=torch.float32, requires_grad=True)

    if verbose:
        LOG.info(f'[RC] {n_rings} rings from {r_min:.1f} to {r_max:.1f} px '
                 f'(geometry fixed: inc={np.rad2deg(geom["inc_v"]):.1f} deg, '
                 f'PA={geom["theta_v"]:.3f} rad)')

    # sanity: seeded on the arctangent, the ring model must reproduce the
    # parametric SSE. If it does not, the knots cannot represent the curve and
    # every number below is meaningless -- fail loudly instead of fitting junk.
    with torch.no_grad():
        sse_parametric = float(_masked_sse(fitter, mask))
    with ring_model(r_knots, v_knots), torch.no_grad():
        sse_seed = float(_masked_sse(fitter, mask))
    frac = abs(sse_seed - sse_parametric)/sse_parametric
    if verbose:
        LOG.info(f'[RC] seed check: parametric {sse_parametric:.6f}  '
                 f'rings-at-seed {sse_seed:.6f}  ({100*frac:+.2f}%)')
    if frac > 0.05 and seed_from_arctan:
        raise RuntimeError(
            f'the ring knots cannot reproduce the arctangent at the seed '
            f'({100*frac:.1f}% off). The knots do not resolve the turnover -- '
            f'R_v = {R_v0:.2f} px but the inner knot is at {float(r_knots[0]):.2f} px. '
            f'Lower r_min or raise n_rings.')

    opt = torch.optim.Adam([v_knots], lr=lr)
    with ring_model(r_knots, v_knots):
        for it in range(maxiter):
            opt.zero_grad()
            sse = _masked_sse(fitter, mask)
            sse.backward()
            opt.step()
            with torch.no_grad():                 # a rotation curve is positive
                v_knots.clamp_(min=0.0)
            if verbose and (it % max(1, maxiter//8) == 0 or it == maxiter - 1):
                LOG.info(f'[RC] step {it:4d}  SSE={float(sse):.6f}')
        v_fit = v_knots.detach().clone()
        with torch.no_grad():
            sse_ring = float(_masked_sse(fitter, mask))

    # the parametric fit, same mask, for a like-for-like number
    sse_arctan = sse_parametric

    v_err = np.full(n_rings, np.nan)
    if profile_errors:
        var = fitter.mcmc_var_base
        for i in range(n_rings):
            v_err[i] = _delta_chi2_error(fitter, mask, r_knots, v_fit, i,
                                         sse_ring, var)
        if verbose:
            LOG.info('[RC] per-ring 1-sigma from delta-chi2 = 1')

    r_np, v_np = r_knots.numpy(), v_fit.numpy()
    return dict(r_knots=r_np, v_knots=v_np, v_err=v_err,
                v_arctan=(2/np.pi)*V_rot0*np.arctan(r_np/R_v0),
                sse_ring=sse_ring, sse_arctan=sse_arctan,
                n_pix=int(mask.sum()), var_base=fitter.mcmc_var_base,
                geometry=geom, V_rot=V_rot0, R_v=R_v0)


def _delta_chi2_error(fitter, mask, r_knots, v_fit, i, sse0, var, span=0.35):
    '''1-sigma on ring i by scanning it until the masked chi2 rises by 1.'''
    best = float(v_fit[i])
    scale = max(abs(best), 20.0)
    for frac in np.linspace(0.02, span, 12):
        trial = v_fit.clone()
        trial[i] = best + frac*scale
        with ring_model(r_knots, trial), torch.no_grad():
            if (float(_masked_sse(fitter, mask)) - sse0)/var >= 1.0:
                return frac*scale
    return span*scale        # unconstrained within the scan window


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_rotation_curve(rc, kpc_per_pixel=None, ax=None, filename=None,
                        title=None, fs_label=14, fs_tick=12):
    '''Ring velocities against the fitted arctangent, with the SSE comparison.'''
    import matplotlib.pyplot as plt

    unit, scale = ('kpc', kpc_per_pixel) if kpc_per_pixel else ('pixels', 1.0)
    r = rc['r_knots']*scale
    rr = np.linspace(0, r.max()*1.05, 240)
    arctan = (2/np.pi)*rc['V_rot']*np.arctan((rr/scale)/rc['R_v'])

    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.plot(rr, arctan, color='crimson', lw=2.2,
            label=(f"arctangent fit\n$V_{{\\rm rot}}$={rc['V_rot']:.0f}, "
                   f"$R_v$={rc['R_v']*scale:.2f} {unit}"))
    ax.errorbar(r, rc['v_knots'], yerr=rc['v_err'], fmt='o', ms=8,
                color='k', capsize=4, lw=1.6, label='free rings (this fit)')
    ax.set_xlabel(f'deprojected radius [{unit}]', fontsize=fs_label)
    ax.set_ylabel(r'$V_{\rm circ}$ [km s$^{-1}$]', fontsize=fs_label)
    ax.tick_params(labelsize=fs_tick)
    ax.grid(False)

    d = (rc['sse_arctan'] - rc['sse_ring'])/rc['var_base']
    ax.set_title(title or (f"free rings improve $\\chi^2$ by {d:.1f} "
                           f"for {len(r) - 2} extra parameters"), fontsize=fs_label)
    ax.legend(frameon=True, fontsize=fs_tick)
    if filename:
        ax.figure.savefig(filename, dpi=200, bbox_inches='tight')
    return ax
