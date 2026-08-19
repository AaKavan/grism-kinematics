# plot.py

from astropy.stats import sigma_clipped_stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

#%% --------------------------------------------------------------------------
# helper normalization functions
# ----------------------------------------------------------------------------

class AsinhNorm(Normalize):
    def __init__(self, std, vmin=None, vmax=None, clip=False):
        super().__init__(vmin=vmin, vmax=vmax, clip=clip)
        self.std = std

    def __call__(self, value, clip=None):
        # forward: map raw data to [0,1] via arcsinh stretch
        vmin = self.vmin if self.vmin is not None else np.min(value)
        vmax = self.vmax if self.vmax is not None else np.max(value)
        y    = np.arcsinh(value/self.std)
        ymin = np.arcsinh(vmin/self.std)
        ymax = np.arcsinh(vmax/self.std)
        return (y - ymin) / (ymax - ymin)

    def inverse(self, value):
        # inverse: map [0,1] back to raw data
        vmin = self.vmin
        vmax = self.vmax
        ymin = np.arcsinh(vmin/self.std)
        ymax = np.arcsinh(vmax/self.std)
        y    = ymin + value*(ymax - ymin)
        return self.std * np.sinh(y)

def asinhstretch(im):
    _, _, std = sigma_clipped_stats(im)
    return np.arcsinh(im / std)

#%% --------------------------------------------------------------------------
# plotting utilities
# ----------------------------------------------------------------------------

def plot_kinematics_fitting_result(true_im, model_im, vz, diff_im, title_prefix, 
                                   axs_row, velocity_params, dispersion_params):
    
    dx = dispersion_params['dx']
    dy = dispersion_params['dy']
    x0_v = velocity_params['x0_v']
    y0_v = velocity_params['y0_v']
    residual = model_im - diff_im
    resid_max = np.max(np.abs(residual))
    vz_max = np.max(np.abs(vz))

    im0 = axs_row[0].imshow(true_im)
    axs_row[0].set_title(f'{title_prefix} Grism Image')
    axs_row[0].grid(False)
    # axs_row[0].scatter(x0_v-dx, y0_v-dy, marker='o', s=100, c='none', edgecolor='lime')
    axs_row[0].scatter(x0_v-dx, y0_v-dy, marker='+', s=100, c='lime')
    plt.colorbar(im0, ax=axs_row[0], fraction=0.046, pad=0.04)

    im1 = axs_row[1].imshow(model_im)
    axs_row[1].set_title(f'{title_prefix} velocity corrected')
    axs_row[1].grid(False)
    # axs_row[1].scatter(x0_v, y0_v, marker='o', s=100, c='none', edgecolor='lime')
    axs_row[1].scatter(x0_v, y0_v, marker='+', s=100, c='lime')
    plt.colorbar(im1, ax=axs_row[1], fraction=0.046, pad=0.04)

    im2 = axs_row[2].imshow(residual, cmap='seismic', vmin=-resid_max, vmax=resid_max)
    axs_row[2].set_title(f'{title_prefix} Residual')
    axs_row[2].grid(False)
    plt.colorbar(im2, ax=axs_row[2], fraction=0.046, pad=0.04)

    im2 = axs_row[3].imshow(vz, cmap='seismic', vmin=-vz_max, vmax=vz_max)
    axs_row[3].set_title(f'{title_prefix} velocity field')
    axs_row[3].grid(False)
    plt.colorbar(im2, ax=axs_row[3], fraction=0.046, pad=0.04)

def plot_loss_and_lr(losses, lrs, steps=None, filename=None):

    if steps is None:
        steps = range(len(losses))
    
    fig, ax1 = plt.subplots(figsize=(8,5))

    color_loss = 'tab:blue'
    ax1.plot(steps, losses, color=color_loss, label='Loss')
    ax1.set_yscale('log')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss', color=color_loss)
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()  # create second y-axis sharing the same x-axis

    color_lr = 'tab:red'
    ax2.plot(steps, lrs, color=color_lr, linestyle='--', label='Learning Rate')
    ax2.set_yscale('log')
    ax2.set_ylabel('Learning Rate', color=color_lr)
    ax2.tick_params(axis='y')

    fig.tight_layout()
    plt.title('Loss and Learning Rate Evolution')
    if filename:
        plt.savefig(filename)
    plt.show()

def plot_image_fitting_result(true_image, gal_model, psf_model):

    full_model  = gal_model + psf_model
    residual    = true_image - full_model
    psf_removed = true_image - psf_model

    panels = [
        ('True Image',            true_image),
        ('Full Model',            full_model),
        ('Galaxy Model',          gal_model),
        ('AGN Removed', psf_removed),
        ('Residual',              residual),
    ]

    resid_max = np.max(np.abs(residual))

    fig, axs = plt.subplots(1, 5, figsize=(22, 5))
    for ax, (title, img) in zip(axs, panels):
        if title == 'Residual':
            # keep residual linear but symmetric
            norm = Normalize(vmin=-resid_max, vmax=resid_max)
            im = ax.imshow(img, cmap='seismic', norm=norm)
        else:
            # compute per‐panel σ for asinhstretch
            _, _, std = sigma_clipped_stats(img)
            if std<=0: std = 1e-5
            norm = AsinhNorm(std, vmin=np.min(img), vmax=np.max(img))
            im = ax.imshow(img, norm=norm)

        ax.set_title(title)
        ax.grid(False)
        if np.max(img)>np.min(img):
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # cbar.set_label('Pixel value')   # now shows true image units

    plt.tight_layout()
    plt.show()

#%% --------------------------------------------------------------------------
# MCMC diagnostics
# ----------------------------------------------------------------------------

def plot_mcmc_chains(chain, names, nburn=None, filename=None, show=True):
    '''
    Trace plot, one panel per parameter, every walker overplotted.

    Parameters
    ----------
    chain : (nsteps, nwalkers, ndim) array, i.e. emcee's get_chain() with no discard
    names : list of ndim parameter names
    nburn : if given, shade the burn-in region
    '''
    chain = np.asarray(chain)
    nsteps, nwalkers, ndim = chain.shape
    fig, axs = plt.subplots(ndim, 1, figsize=(9, 1.5*ndim), sharex=True)
    axs = np.atleast_1d(axs)
    for j, ax in enumerate(axs):
        ax.plot(chain[:, :, j], color='k', alpha=0.25, lw=0.5)
        if nburn:
            ax.axvspan(0, nburn, color='0.85', zorder=0)
        ax.set_ylabel(names[j].split('.')[-1], fontsize=8)
        ax.grid(False)
    axs[-1].set_xlabel(f'step  ({nwalkers} walkers)')
    plt.tight_layout()
    if filename: plt.savefig(filename, dpi=150, bbox_inches='tight')
    if show: plt.show()
    else: plt.close(fig)
    return fig

def plot_mcmc_corner(flat_chain, names, truths=None, filename=None, show=True):
    '''
    Corner plot of a flattened chain.

    Uses the `corner` package when it is installed, and otherwise falls back to a
    plain matplotlib pair grid, so that this never becomes a hard dependency.
    '''
    flat_chain = np.asarray(flat_chain)
    labels = [n.split('.')[-1] for n in names]
    try:
        import corner
    except ImportError:
        return _corner_fallback(flat_chain, labels, truths, filename, show)
    fig = corner.corner(
        flat_chain, labels=labels, truths=truths,
        quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt='.3g',
        title_kwargs={'fontsize': 8}, label_kwargs={'fontsize': 9}
    )
    if filename: fig.savefig(filename, dpi=150, bbox_inches='tight')
    if show: plt.show()
    else: plt.close(fig)
    return fig

def _corner_fallback(flat_chain, labels, truths=None, filename=None, show=True):
    '''Minimal corner plot for when the `corner` package is unavailable.'''
    ndim = flat_chain.shape[1]
    fig, axs = plt.subplots(ndim, ndim, figsize=(1.35*ndim, 1.35*ndim))
    axs = np.atleast_2d(axs)
    for i in range(ndim):
        for j in range(ndim):
            ax = axs[i, j]
            ax.grid(False)
            if j > i:
                ax.axis('off')
                continue
            if i == j:
                ax.hist(flat_chain[:, i], bins=40, color='0.3', histtype='step')
                lo, med, hi = np.percentile(flat_chain[:, i], [16, 50, 84])
                ax.set_title(f'{med:.4g}\n$-${med-lo:.2g} $+${hi-med:.2g}', fontsize=6)
                for v in (lo, med, hi):
                    ax.axvline(v, color='0.5', ls='--', lw=0.7)
                ax.set_yticks([])
            else:
                ax.hist2d(flat_chain[:, j], flat_chain[:, i], bins=40, cmap='Greys')
            if truths is not None:
                if i == j:
                    ax.axvline(truths[i], color='tab:red', lw=1)
                else:
                    ax.axvline(truths[j], color='tab:red', lw=0.7)
                    ax.axhline(truths[i], color='tab:red', lw=0.7)
            if i == ndim-1: ax.set_xlabel(labels[j], fontsize=7)
            else: ax.set_xticklabels([])
            if j == 0 and i > 0: ax.set_ylabel(labels[i], fontsize=7)
            elif j > 0: ax.set_yticklabels([])
            ax.tick_params(labelsize=5)
    plt.tight_layout()
    if filename: fig.savefig(filename, dpi=150, bbox_inches='tight')
    if show: plt.show()
    else: plt.close(fig)
    return fig
