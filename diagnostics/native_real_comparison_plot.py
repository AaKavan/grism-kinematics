"""Summarize completed real-ID3700 center fits without running a new fit.

The curves are conditional examples at illustrative radii, not credible bands
or a measurement of the radial extent. Run with --root <real-output-root>.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
import astropy.units as u

from dingo.native_posterior import radial_profiles


def run(root):
    out = root / 'comparison'
    out.mkdir(exist_ok=True)
    source = root / 'sensitivity/sensitivity.json'
    report = json.loads(source.read_text())
    if not report['completed']:
        raise ValueError('Finish center comparisons first')
    winners = {f['variant']: f for f in report['best_by_variant']}
    variants = ['stellar_center', 'free_center']
    labels = ['Soft stellar center', 'Free center']
    colors = ['#2166ac', '#d95f02']
    stellar = np.array(report['stellar_center']['mean_east_north_arcsec'])
    scale = float(Planck18.kpc_proper_per_arcmin(1.1231).value / 60)
    radii = np.linspace(.2, .8, 121)
    saved, rows = {}, []
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for j, (name, label, color) in enumerate(zip(variants, labels, colors)):
        winner = winners[name]
        with np.load(root / 'sensitivity' / f'{name}_best.npz') as a:
            saved[name] = {k: a[k] for k in a.files}
        trials = [f for f in report['fits'] if f['variant'] == name]
        for f in trials:
            delta = 2 * (f['objective'] - winner['objective'])
            axes[0, 0].scatter(f['vrot_0p5arcsec_kms'], delta, color=color,
                               marker='o' if f['start_speed'] < 500 else 'x', s=48)
        axes[0, 0].plot([], [], color=color, label=label)
        for branch, linestyle, alpha in [(winner, '-', 1.),
                (min((f for f in trials if f['start_speed'] >= 500), key=lambda f: f['objective']), ':', .7)]:
            center = np.array(branch['center_east_north_arcsec'])
            axes[0, 1].scatter(*center, color=color, marker='o' if linestyle == '-' else 'x', s=55)
            profiles = radial_profiles(branch['vproj_0p5arcsec_kms'], branch['turnover_arcsec'],
                np.deg2rad(branch['inclination_deg']), radii, kpc_per_arcsec=scale)
            axes[1, 0].plot(radii * scale, profiles['deprojected_velocity_kms'], linestyle,
                            color=color, alpha=alpha, label=label if linestyle == '-' else None)
            axes[1, 1].plot(radii * scale, profiles['rotation_only_mass_msun'] / 1e10,
                            linestyle, color=color, alpha=alpha)
        p = radial_profiles(winner['vproj_0p5arcsec_kms'], winner['turnover_arcsec'],
                            np.deg2rad(winner['inclination_deg']), [.5], kpc_per_arcsec=scale)
        offset = float(np.linalg.norm(np.array(winner['center_east_north_arcsec']) - stellar))
        rows.append(winner | {'offset_from_stellar_arcsec': offset,
                             'offset_from_stellar_kpc': offset * scale,
                             'rotation_only_mass_at_0p5_msun': float(p['rotation_only_mass_msun'][0])})
    axes[0, 0].set(xlabel='Optimized Vrot(0.5 arcsec) [km/s]',
                   ylabel='2 Δ objective within each center model')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].set_title('Four starts per model; dots: low starts, crosses: high starts', fontsize=9)
    axes[0, 1].scatter(*stellar, marker='*', color='black', s=80, label='F150W centroid mixture')
    axes[0, 1].add_patch(Circle(stellar, .05, fill=False, color=colors[0], alpha=.6))
    axes[0, 1].set(xlabel='East from catalog [arcsec]', ylabel='North from catalog [arcsec]',
                   xlim=(-.16, .16), ylim=(-.43, .12), aspect='equal')
    axes[0, 1].legend(fontsize=8, loc='lower left')
    axes[0, 1].set_title('Circle: adopted 0.05 arcsec soft-center scale', fontsize=9)
    axes[1, 0].set(xlabel='Illustrative disk radius [kpc]', ylabel='Vrot(r) [km/s]')
    axes[1, 0].legend(fontsize=9)
    axes[1, 1].set(xlabel='Illustrative disk radius [kpc]', ylabel='r Vrot²/G [10¹⁰ solar masses]')
    for ax in axes[1]:
        ax.axvline(.5 * scale, color='gray', lw=.8, alpha=.6)
    fig.suptitle('ID3700 joint R+C: solid = preferred fits; dotted = worse local solutions\n'
                 'Conditional curves, not uncertainty bands; radial information is not yet validated', fontsize=12)
    fig.tight_layout()
    fig.savefig(out / 'center_comparison.png', dpi=170)
    plt.close(fig)

    with np.load(root / 'grid/native_data.npz') as a:
        science, valid = a['science'], a['valid']
        metadata = json.loads(str(a['metadata_json']))
    pupils = np.array([e['pupil'] for e in metadata['exposures']])
    residual_changes = {}
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    mid = science.shape[-1] // 2
    crop = np.s_[mid-18:mid+19, mid-18:mid+19]
    soft, free = (saved[name] for name in variants)
    difference = soft['residual'] - free['residual']
    for row, pupil in enumerate(['R', 'C']):
        e = int(np.flatnonzero(pupils == pupil)[0])
        data = np.where(valid[e], science[e], np.nan)[crop]
        lo, hi = np.nanquantile(data, [.02, .995])
        axes[row, 0].imshow(data, origin='lower', cmap='magma', vmin=lo, vmax=hi)
        for col, array in enumerate([soft['residual'][e], free['residual'][e], difference[e]], 1):
            im = axes[row, col].imshow(array[crop], origin='lower', cmap='RdBu_r',
                                       vmin=-5 if col < 3 else -.1, vmax=5 if col < 3 else .1)
        fig.colorbar(im, ax=axes[row, 3], shrink=.7, label='Δ residual / ERR')
        axes[row, 0].set_ylabel(f'Exposure {e+1}: {pupil}')
        residual_changes[pupil] = float(np.sqrt(np.nanmean(difference[pupils == pupil]**2)))
    for ax, title in zip(axes[0], ['Observed SCI', 'Soft center: residual / ERR',
                                  'Free center: residual / ERR', 'Soft − free residual']):
        ax.set_title(title, fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle('Representative exposures within the joint fit; residual panels span ±5 ERR')
    fig.tight_layout()
    fig.savefig(out / 'center_residuals.png', dpi=170)
    plt.close(fig)

    # Independent catalog context: nearby imaging detections alone cannot
    # establish that any particular dispersed pixel is contaminated.
    origin = SkyCoord(metadata['summary']['ra'] * u.deg, metadata['summary']['dec'] * u.deg)
    catalog_paths = [Path('/home/aakavan/EDR_data/sapphires_edr_catalogs') /
                     f'sapphires_edr_{kind}_cat.fits' for kind in ['phot', 'spec']]
    neighbors = {}
    for kind, path in zip(['phot', 'spec'], catalog_paths):
        table = Table.read(path, hdu=1)
        sky = SkyCoord(table['RA'], table['DEC'], unit='deg')
        sep = origin.separation(sky).arcsec
        records = []
        for ix in np.argsort(sep)[:12]:
            east, north = origin.spherical_offsets_to(sky[ix])
            record = {'id': int(table['ID'][ix]), 'separation_arcsec': float(sep[ix]),
                      'east_arcsec': float(east.arcsec), 'north_arcsec': float(north.arcsec)}
            if kind == 'spec':
                record['zspec'] = float(table['zspec'][ix])
            records.append(record)
        neighbors[kind] = records
    with np.load(root / 'imaging/cutout.npz') as a:
        wcs = WCS(fits.Header.fromstring(str(a['wcs_header'])))
        image = a['image']
    yy, xx = np.indices(image.shape)
    east, north = origin.spherical_offsets_to(wcs.pixel_to_world(xx, yy))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    hi = np.nanquantile(image, .995)
    axes[0].pcolormesh(east.arcsec, north.arcsec, np.arcsinh(image / hi * 20),
                       shading='nearest', cmap='magma', vmin=0, vmax=np.arcsinh(20))
    for neighbor in neighbors['phot']:
        if neighbor['separation_arcsec'] < 2.:
            x, y = neighbor['east_arcsec'], neighbor['north_arcsec']
            axes[0].plot(x, y, '+', color='cyan', ms=8)
            axes[0].annotate(str(neighbor['id']), (x, y), color='cyan', fontsize=8, xytext=(4, 5),
                             textcoords='offset points')
    axes[0].add_patch(Rectangle((-.9, -.9), 1.8, 1.8, fill=False, color='white', ls='--'))
    axes[0].set_title('F150W + photometric catalog; box = line-map support')
    flux = soft['source_flux'].reshape(61, 61)
    axes[1].imshow(np.arcsinh(flux / flux.max() * 20), extent=[-.9, .9, -.9, .9],
                    origin='lower', cmap='magma')
    axes[1].plot(*stellar, 'c*', ms=10)
    axes[1].set_title('Fitted line map: faint edge/off-source emission')
    for ax in axes:
        ax.set(xlabel='East offset [arcsec]', ylabel='North offset [arcsec]', aspect='equal')
    fig.tight_layout()
    fig.savefig(out / 'source_context.png', dpi=170)
    plt.close(fig)
    edge = np.ones(flux.shape, dtype=bool)
    edge[3:-3, 3:-3] = False
    profile_rows = []
    for name in variants:
        f = winners[name]
        r = np.array([.2, .3, .5, .8])
        p = radial_profiles(f['vproj_0p5arcsec_kms'], f['turnover_arcsec'],
                            np.deg2rad(f['inclination_deg']), r, kpc_per_arcsec=scale)
        profile_rows.append({'variant': name, 'illustrative_radii_arcsec': r.tolist(),
                             'radii_kpc': (r * scale).tolist(),
                             **{k: v.tolist() for k, v in p.items()}})
    summary = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'catalog_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in catalog_paths},
        'center_comparison': rows, 'profiles': profile_rows,
        'rms_residual_difference_in_ERR_R_C': residual_changes,
        'stellar_center_map_outer_three_rows_columns_flux_fraction': float(flux[edge].sum() / flux.sum()),
        'nearest_catalog_objects': neighbors,
        'limitations': ['Curves are conditional examples, not posterior intervals.',
            'Illustrative 0.2–0.8 arcsec radii do not establish observed or resolved coverage.',
            'Faint map edge flux is not independently confirmed source emission.',
            'Nearby catalog objects do not prove which grism pixels are contaminated.',
            'r Vrot²/G is a spherical-equivalent rotation-only mass proxy.']}
    (out / 'comparison.json').write_text(json.dumps(summary, indent=2))
    (out / 'code.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'centers': rows, 'residual_difference': residual_changes,
                      'edge_flux_fraction': summary['stellar_center_map_outer_three_rows_columns_flux_fraction']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
