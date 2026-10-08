"""The physics: velocity field, Doppler mapping, and rectification.

arctangent_disk_velocity_model is the model. Note what it does and does not
constrain: V_rot and inc_v enter the line-of-sight field almost entirely through
the product V_rot*sin(inc_v), with inclination surviving on its own only through
the cos(inc_v) deprojection in the radius -- a strong handle for an edge-on disk
and almost none for a face-on one. Sampling V_sini instead of V_rot removes that
degenerate direction from the chain.

iteratively_find_xy is the forward model actually used by the fitter: a damped
fixed point that solves for where each grism pixel came from, given that the
wavelength it was dispersed by depends on the velocity at the position being
solved for. fit_MCMC runs it with tol=0 and a fixed iteration count so the
likelihood is a deterministic function of theta -- an early exit at a
theta-dependent step makes ln L discontinuous, which emcee cannot sample.

"dispersion" throughout this module means GRISM dispersion, never velocity
dispersion. The model has no line-width term.
"""

import torch
from . import galaxy, utils

import matplotlib.pyplot as plt

import logging
LOG = logging.getLogger(__name__)  # This will inherit the root logger's config if set
LOG.addHandler(logging.NullHandler())  # Prevents "No handler found" warning if root isn't configured


#%% kinematics models in torch
def arctangent_disk_velocity_model(x, y, V_rot=200., R_v=2., x0_v=0., y0_v=0., theta_v=0.,
                                   inc_v=0., V_sini=None, v_sys=0., **kwargs):
    """
    TODO: change the name to torch
    Compute vx, vy, vz on the sky for a rotating disk using arctangent rotation curve.

    Parameters
    ----------
    x, y : torch.Tensor
        Coordinates (float32) of shape (H, W) or (N,) in arcsec or pixels.
    V_rot : float or torch.Tensor
        Maximum (deprojected) rotational velocity. Ignored when V_sini is given.
    V_sini : float or torch.Tensor, optional
        The projected amplitude V_rot*sin(inc_v). Supplying this instead of V_rot
        reparameterises the model onto the quantity the data actually constrain --
        see the note at the amplitude below.
    R_v : float or torch.Tensor
        Turnover radius (scale radius).
    x0_v, y0_v : float or torch.Tensor
        Dynamical center (same units as x, y).
    theta : float (radians)
        Position angle of major axis (CCW from x-axis).
    inc : float (radians)
        Inclination angle (0=face-on, pi/2=edge-on)
    v_sys : float or torch.Tensor
        Systemic velocity offset in km/s, added to the whole field. See the note
        at the return.

    Returns
    -------
    vz : torch.Tensor
        OBSERVED line-of-sight velocity (same shape as x), i.e. the rotation field
        plus v_sys. With the default v_sys=0 this is the rotation field alone.
    """
    # Shift to center
    dx = x - x0_v
    dy = y - y0_v

    # Rotate coordinates by -theta (align major axis with x')
    cos_t, sin_t = torch.cos(theta_v), torch.sin(theta_v)
    x_p = cos_t * dx + sin_t * dy
    y_p = -sin_t * dx + cos_t * dy

    # Deproject radius (inclination)
    r = torch.sqrt(x_p**2 + (y_p / torch.cos(inc_v))**2 + 1e-8)

    # Amplitude. V_rot and inc_v enter the line-of-sight field only through the
    # product V_rot*sin(inc_v); inclination survives on its own ONLY through the
    # cos(inc_v) deprojection in r above. That handle is strong for an edge-on disk
    # and almost absent for a face-on one, so for a round galaxy the two parameters
    # are nearly perfectly degenerate: on ID15665 (inc ~ 26 deg) the chain reached
    # corr(V_rot, inc_v) = -0.97 with tau ~ 800, against corr = -0.15 for ID21636
    # (inc ~ 62 deg). Sampling V_sini instead of V_rot removes that direction from
    # the chain: the likelihood then depends on V_sini alone, and inc_v is set by
    # its prior plus the weak geometry. V_rot = V_sini/sin(inc_v) is recovered after
    # sampling, and inherits the inclination prior's width honestly.
    amp = V_rot * torch.sin(inc_v) if V_sini is None else V_sini
    V_circ = (2 / torch.pi) * amp * torch.atan(r / R_v)

    # Only vz (los velocity) is non-zero, vx = vy = 0
    vz = V_circ * (x_p / (r + 1e-8))

    # Systemic velocity. It lives HERE, inside the velocity model, rather than
    # being threaded through each Doppler site separately -- fitting.py computes
    # lambda_obs in four places, and two of them (_xy_converged and
    # _xy_converged_batch) re-implement the arithmetic as a CONVERGENCE TEST. If
    # the test used a different lambda from the iteration it is testing, the fixed
    # point would never look converged and log_likelihood would return -inf for
    # every theta, killing the chain at walker initialisation with a message that
    # says nothing about the cause. Returning it as part of vz makes that
    # impossible by construction.
    #
    # lambda_rest is fixed from summary.z, so without this a spectroscopic
    # redshift error has nowhere to go but into the rotation amplitude. A fitted
    # v_sys of X km/s means summary.z is off by X/c*(1+z).
    #
    # Default 0.0: `vz + 0.0` is bit-identical to `vz` for every IEEE-754 float
    # that occurs here, so the default-off path is provably a no-op.
    vz = vz + v_sys

    # # Optionally return full velocity field (vx, vy = 0 for circular disk)
    # vx = torch.zeros_like(vz)
    # vy = torch.zeros_like(vz)

    # return vx, vy, vz
    return vz

def dispersion_model(x, y, vz, wavelength_rest, grism_model, dx=0., dy=0., **kwargs):
    """
    Map rest-frame emission line to grism image using Doppler shift and spatial model.

    Parameters
    ----------
    x, y : (H, W) torch.Tensor
        Image coordinates.
    vz : (H, W) torch.Tensor
        Line-of-sight velocity in km/s.
    wavelength_rest : float
        Rest-frame wavelength in microns (e.g., 0.6563 for Hα).
    grism_model : function
        Function grism_model(x, y, wavelength) returning (x_G, y_G).
    wavs, dxs, dys : (M,) torch.Tensor
        Grism trace model used inside grism_model.

    Returns
    -------
    x_G, y_G : (H, W) torch.Tensor
        Projected grism coordinates accounting for Doppler shift.
    """
    x = x + dx
    y = y + dy
    c = 299792.458  # speed of light in km/s
    lambda_obs = wavelength_rest * (1.0 + vz / c)
    x_G, y_G = grism_model(x, y, lambda_obs)

    return x_G, y_G

def bilinear_interpolte_intensity_torch(xg, yg, intensities, cutout):
        """Scatter `intensities` through the (xg, yg) map onto the cutout canvas.

        Accepts an optional leading batch dimension: xg/yg of shape (..., Hin, Win) give an
        output of shape (..., h, w). With no batch dims this is the original scalar path.

        Implementation note: the original masked with `intensities[valid]`, which works for a
        single parameter set but collapses a batch into one ragged vector (each batch element
        has a different number of valid pixels), so the result is unrecoverable. Here the
        weight is multiplied by `valid` instead and the indices are clamped, which is
        numerically identical -- the invalid entries contribute exactly 0.0 -- and keeps the
        batch dimension intact. `scatter_add_` also replaces `index_put_(accumulate=True)`:
        it is the same accumulation, but it has a deterministic CUDA implementation, which
        `index_put_` does not.
        """
        device = intensities.device
        x0_c, y0_c, w_cut, h_cut = cutout
        H, W = h_cut, w_cut

        batch = tuple(xg.shape[:-2])
        n_in = xg.shape[-2] * xg.shape[-1]

        # Transform coordinates to cutout frame
        xg_cut = xg - x0_c
        yg_cut = yg - y0_c

        xi0 = torch.floor(xg_cut).long()
        yi0 = torch.floor(yg_cut).long()
        dx = xg_cut - xi0.to(xg_cut.dtype)
        dy = yg_cut - yi0.to(yg_cut.dtype)

        flat = torch.zeros(*batch, H * W, device=device, dtype=intensities.dtype)

        for i in range(2):
            for j in range(2):
                wt = ((1 - dx) if i == 0 else dx) * ((1 - dy) if j == 0 else dy)
                xi = xi0 + i
                yj = yi0 + j
                valid = (xi >= 0) & (xi < W) & (yj >= 0) & (yj < H)
                lin = (yj.clamp(0, H - 1) * W + xi.clamp(0, W - 1)).reshape(*batch, n_in)
                vals = (intensities * wt * valid.to(intensities.dtype)).reshape(*batch, n_in)
                flat.scatter_add_(-1, lin, vals)

        return flat.reshape(*batch, H, W)

def scatter_shifted_psf(xg, yg, intensities, psf, cutout):
    """
    Scatter intensity-weighted, subpixel-shifted PSFs into a full image.

    Parameters
    ----------
    xg, yg : (N,) float tensors - Subpixel locations in grism image.
    intensities : (N,) float tensor - Flux values.
    psf : (P, P) tensor - Centered PSF kernel.
    image_shape : (H, W) - Shape of full image canvas.
    x0, y0 : int - Starting coordinates of cutout region.
    w, h : int or None - Width and height of cutout region. If None, use full image.

    Returns
    -------
    image : (h, w) tensor - Model image of specified cutout region.
    """

    image = bilinear_interpolte_intensity_torch(xg, yg, intensities, cutout)

    convolved = utils.fftconvolve_torch(image, psf, mode='same')

    return convolved

    # return image

#%% full grism models

def full_grism_model_torch(x, y, psf, cutout, emline_mask=1., **kwargs):

    """
    Generate a full model image using centered FFT-based convolution.

    Parameters
    ----------
    x, y : 2D torch tensors
        Coordinate grid.
    psf : 2D torch tensor
        PSF kernel, assumed centered and normalized.
    kwargs : dict
        velocity model parameters: V_rot, R_v, x0_v, y0_v, theta_v, inc_v
        dispersion model parameters: wavelength_rest, grism_model
        Sérsic model parameters: I_e, R_e, n, x0, y0, q, theta
    emline_mask: 2D torch tensor or float
        Filter the galaxy flux to the emline flux
        If tensor then the shape should be the same as x, y

    Returns
    -------
    image_conv : 2D torch tensor
        The convolved model image.
    """


    vz = arctangent_disk_velocity_model(x, y, **kwargs)
    x_G, y_G = dispersion_model(x, y, vz, **kwargs)
    image_model = galaxy.sersic_model_torch(x, y, **kwargs)
    
    # plt.scatter(x_G.detach().numpy(), 
    #             y_G.detach().numpy(), 
    #             c=image_model.detach().numpy())
    # plt.show()

    # convolve with psf
    I_flat = image_model.flatten()
    xG_flat = x_G.flatten()
    yG_flat = y_G.flatten()
    # Build PSF model image
    model_img = scatter_shifted_psf(xG_flat, yG_flat, I_flat, psf, cutout)
    model_img = model_img*emline_mask

    return model_img

def full_grism_model_nonparametric_torch(image_model, psf, cutout, emline_mask=1., **kwargs):

    """
    Generate a full model image using centered FFT-based convolution.

    Parameters TODO: change it
    ----------
    x, y : 2D torch tensors
        Coordinate grid.
    psf : 2D torch tensor
        PSF kernel, assumed centered and normalized.
    kwargs : dict
        velocity model parameters: V_rot, R_v, x0_v, y0_v, theta_v, inc_v
        dispersion model parameters: wavelength_rest, grism_model
        Sérsic model parameters: I_e, R_e, n, x0, y0, q, theta

    Returns
    -------
    image_conv : 2D torch tensor
        The convolved model image.
    """
    device = image_model.device
    nx, ny = image_model.shape
    y, x = torch.meshgrid(
        torch.linspace(0, nx - 1, nx, device=device),
        torch.linspace(0, ny - 1, ny, device=device),
        indexing='ij'
    )

    vz = arctangent_disk_velocity_model(x, y, **kwargs)
    x_G, y_G = dispersion_model(x, y, vz, **kwargs)
    
    # plt.scatter(x_G.detach().numpy(), 
    #             y_G.detach().numpy(), 
    #             c=image_model.detach().numpy())
    # plt.show()

    # convolve with psf
    I_flat = image_model.flatten()
    xG_flat = x_G.flatten()
    yG_flat = y_G.flatten()
    # Build PSF model image
    model_img = scatter_shifted_psf(xG_flat, yG_flat, I_flat, psf, cutout)
    model_img = model_img*emline_mask

    return model_img

# model ok; idea does not work

def forward_dispersion_model(x_G, y_G, lambda_obs, forward_model, dx=0., dy=0., **kwargs):

    x_G = x_G + dx
    y_G = y_G + dy
    x, y = forward_model(x_G, y_G, lambda_obs)

    return x, y

def full_forward_grism_model_torch(grism_image, lambda_test, cutout, oversample=2, 
                                       **kwargs):
    # TODO: oversample
    nx_G, ny_G = grism_image.shape
    y_G, x_G = torch.meshgrid(torch.arange(nx_G), torch.arange(ny_G))
    x, y = forward_dispersion_model(x_G, y_G, lambda_test, **kwargs)

    # print('cutout', cutout)
    # plt.scatter(x.detach().numpy(), 
    #             y.detach().numpy(), 
    #             c=grism_image.detach().numpy())
    # plt.show()


    image = bilinear_interpolte_intensity_torch(x, y, grism_image, cutout)

    dx = cutout[0] #+ cutout[2]//2
    dy = cutout[1] #+ cutout[3]//2
    vz = arctangent_disk_velocity_model(x-dx, y-dy, **kwargs)


    return image, vz

# finally worked version, iteratively find xy

def iteratively_find_xy(x_init, y_init, cutout, lambda_rest, x_G, y_G, 
                        maxiter=50, tol=1e-3, alpha=0.5, **kwargs):
    c = 299792.458  # speed of light in km/s

    # find x and y
    dx = cutout[0] #+ cutout[2]//2
    dy = cutout[1] #+ cutout[3]//2
    x = x_init.detach()
    y = y_init.detach()
    converged = False
    if maxiter < 1:
        # vz_new and k would be undefined below, giving a NameError from a config
        # that merely asked for no iterations.
        raise ValueError(f'iteratively_find_xy needs maxiter >= 1, got {maxiter} '
                         f'(check mcmc.xy_iters)')
    for k in range(maxiter):
        vz_new = arctangent_disk_velocity_model(x-dx, y-dy, **kwargs)
        lambda_new = lambda_rest * (1.0 + vz_new / c)
        x_new, y_new = forward_dispersion_model(x_G, y_G, lambda_new, **kwargs)
        # print(torch.max(torch.abs(x_new - x)), torch.max(torch.abs(y_new - y)))
        # `tol <= 0` means the caller wants a fixed iteration count (fit_MCMC does), so the
        # test can never pass -- short-circuit it. Each evaluation is two full-grid
        # reductions plus a device sync, 80 per likelihood call at maxiter=40 x 2 grisms.
        if tol > 0 and torch.max(torch.abs(x_new - x)) < tol and \
           torch.max(torch.abs(y_new - y)) < tol:
            x = alpha*x + (1 - alpha)*x_new
            y = alpha*y + (1 - alpha)*y_new
            converged = True
            break
        x = alpha*x + (1 - alpha)*x_new
        y = alpha*y + (1 - alpha)*y_new
    # NOTE: testing k+1>=maxiter would also fire when the loop converged on its
    # very last iteration, so track convergence explicitly instead.
    #
    # tol == 0 means the caller deliberately wants a FIXED iteration count rather
    # than early exit -- fit_MCMC does this so the likelihood is a deterministic
    # function of theta (an early exit at a theta-dependent k makes ln L
    # discontinuous, which emcee cannot sample). The convergence test can never
    # pass in that mode, so warning about it is pure noise.
    if not converged and tol > 0:
        n_bad = int(torch.sum((torch.abs(x_new - x) >= tol) |
                              (torch.abs(y_new - y) >= tol)))
        worst = float(torch.max(torch.maximum(torch.abs(x_new - x),
                                              torch.abs(y_new - y))))
        LOG.warning(
            f'iteratively_find_xy: {n_bad}/{x.numel()} pixels still above '
            f'tol={tol} after {maxiter} iterations (worst {worst:.3g} px). '
            f'This is usually harmless: the max runs over the whole cutout, and '
            f'pixels far outside the trace never settle while the ones carrying '
            f'signal converge to ~1e-7.')

    return x, y, vz_new, k 