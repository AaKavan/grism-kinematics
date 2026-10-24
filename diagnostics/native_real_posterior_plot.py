"""CPU-only posterior diagnostics; failed chains never imply pooled intervals."""
import json
from pathlib import Path
import numpy as np


def physical_columns(samples):
    return [samples['vrot'], np.rad2deg(samples['inclination']), np.exp(samples['log_rturn']),
            samples['center'][..., 0], samples['center'][..., 1], samples['sigma_v']], [
            'Vrot(0.5″) [km/s]', 'Inclination [deg]', 'Turnover [arcsec]',
            r'$x_0$ [arcsec, east]', r'$y_0$ [arcsec, north]', 'Intrinsic width [km/s]']


def plot(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    path = Path(directory)
    report = json.loads((path/'posterior.json').read_text())
    with np.load(path/'samples.npz') as a:
        samples = dict(a)
    arrays, labels = physical_columns(samples)
    passed = report['diagnostics']['convergence_gate_passed']
    status = 'Sampling checks passed; conditional model' if passed else 'NOT CONVERGED — diagnostic only'
    nchain, ndraw = samples['vrot'].shape
    legend = [Line2D([], [], color=f'C{c}', label=f'Chain {c+1}: {report["starts"][c].get("initial_velocity", report["starts"][c]["grid_velocity"]):.1f} km/s start')
              for c in range(nchain)]
    warmup = None
    if (path/'warmup.npz').exists():
        with np.load(path/'warmup.npz') as a:
            warmup = physical_columns(dict(a))[0]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
    for j, (ax, values, label) in enumerate(zip(axes.ravel(), arrays, labels)):
        nwarm = warmup[j].shape[1] if warmup is not None else 0
        for c in range(nchain):
            if nwarm:
                ax.plot(np.arange(nwarm), warmup[j][c], color=f'C{c}', alpha=.45, lw=.7)
            ax.plot(nwarm+np.arange(ndraw), values[c], color=f'C{c}', lw=.8)
        if nwarm:
            ax.axvspan(0, nwarm, color='0.9', zorder=-2)
            ax.axvline(nwarm, color='0.5', ls=':', lw=1)
        ax.set(xlabel='Iteration (gray region = excluded warmup)' if nwarm else
               'Fresh retained draw (adaptation reused)', ylabel=label)
    fig.legend(handles=legend, loc='outside lower center', ncol=2, fontsize=9)
    fig.suptitle(f'Real ID{report["source_id"]}: chain movement\n{status}', color='black' if passed else 'darkred')
    fig.savefig(path/'trace.png', dpi=130); plt.close(fig)
    # A corner diagram can diagnose chains before convergence. Never draw a
    # pooled density/credible contour when their mixture weights are unknown.
    n = len(arrays)
    fig, axes = plt.subplots(n, n, figsize=(14, 14))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.axis('off'); continue
            for c in range(nchain):
                if i == j:
                    lo, hi = np.min(arrays[j]), np.max(arrays[j])
                    if hi <= lo:
                        hi = lo+1e-6
                    ax.hist(arrays[j][c], bins=np.linspace(lo, hi, 26), histtype='step',
                            density=True, color=f'C{c}', lw=1)
                else:
                    stride = max(1, ndraw//500)
                    ax.scatter(arrays[j][c, ::stride], arrays[i][c, ::stride],
                               s=3, alpha=.25, color=f'C{c}', rasterized=True)
            if i == n-1:
                ax.set_xlabel(labels[j], fontsize=9)
            else:
                ax.tick_params(labelbottom=False)
            if j == 0 and i:
                ax.set_ylabel(labels[i], fontsize=9)
            elif i != j:
                ax.tick_params(labelleft=False)
            ax.tick_params(labelsize=7)
    fig.suptitle(f'Real ID{report["source_id"]}: kinematic corner plot\n{status}; each color is one chain',
                 color='black' if passed else 'darkred')
    fig.legend(handles=legend, loc='upper right', bbox_to_anchor=(.94, .94), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(path/'corner.png', dpi=130); plt.close(fig)
    with np.load(path/'profiles.npz') as a:
        radii = a['radii_arcsec']*report['distance']['kpc_per_arcsec']
        curves = [a['deprojected_velocity_kms'], a['rotation_only_mass_msun']]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for ax, values, label in zip(axes, curves, ['Rotation [km/s]', 'Rotation-only mass proxy [solar masses]']):
        if passed:
            q = np.quantile(values.reshape(-1, len(radii)), [.05, .5, .95], axis=0)
            ax.fill_between(radii, q[0], q[2], color='0.7', alpha=.5, label='Conditional pointwise 90% interval')
            ax.plot(radii, q[1], color='black', label='Pooled median')
            ax.legend(fontsize=8)
        else:
            for c in range(nchain):
                # Show movement/branch disagreement, with no shaded interval.
                ax.plot(radii, np.median(values[c], axis=0), color=f'C{c}', label=f'Chain {c+1} median')
            ax.legend(fontsize=8)
        ax.set(xlabel='Illustrative disk radius [kpc]', ylabel=label)
        ax.grid(alpha=.2)
    axes[1].set_yscale('log')
    fig.suptitle(f'ID{report["source_id"]}: {status}\nRadius coverage is not established; mass assumes circular support')
    fig.savefig(path/'profiles.png', dpi=140); plt.close(fig)
    # One actual last draw per chain, with conditional-mean line amplitudes.
    native = Path(report['settings']['grid'])/'native_data.npz'
    with np.load(native) as a:
        science, valid = a['science'], a['valid']
        meta = json.loads(str(a['metadata_json']))
    pair = [next(i for i, e in enumerate(meta['exposures']) if e['pupil'] == p) for p in ('R', 'C')]
    for c in range(nchain):
        pred = path/f'chain_{c}_last_prediction.npz'
        if not pred.exists():
            continue
        with np.load(pred) as a:
            total = a['line']+a['continuum']; residual = a['residual']; speed = float(a['vrot'])
        fig, axes = plt.subplots(2, 3, figsize=(11, 7), layout='constrained')
        mid = science.shape[-1]//2; cut = np.s_[mid-18:mid+19, mid-18:mid+19]
        for row, e in enumerate(pair):
            data = np.where(valid[e], science[e], np.nan)[cut]
            lo, hi = np.nanquantile(data, [.02, .995])
            for col, (values, name) in enumerate(zip((science[e], total[e], residual[e]), ('Observed SCI', 'Predicted SCI', '(data−model)/ERR'))):
                values = np.where(valid[e], values, np.nan)[cut]
                image = axes[row, col].imshow(values, origin='lower',
                    cmap=plt.get_cmap('RdBu_r' if col == 2 else 'gist_heat').with_extremes(bad='0.65'),
                    vmin=-5 if col == 2 else lo, vmax=5 if col == 2 else hi)
                axes[row, col].set_title(f'{meta["exposures"][e]["pupil"]}: {name}')
                axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
                fig.colorbar(image, ax=axes[row, col], shrink=.7)
        fig.suptitle(f'ID{report["source_id"]}, chain {c+1}, last retained draw: {speed:.1f} km/s\n'
                     f'{status}; conditional-mean amplitude/continuum; not a predictive uncertainty band')
        fig.savefig(path/f'chain_{c}_last_prediction.png', dpi=130); plt.close(fig)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    plot(parser.parse_args().directory)
