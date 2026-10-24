"""CPU assessment of frozen real-exposure predictions, with sideband calibration."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm
import numpy as np

from dingo.native_holdout_continuum import fit_sideband_continuum
from dingo.real_workflow import read_json, write_json, sha256


def summary(residual, selected):
    values = np.asarray(residual)[selected]
    return dict(n_pixels=int(values.size), sum_squared=float(values @ values),
                rms=float(np.sqrt(np.mean(values**2))), mean=float(np.mean(values)),
                n_abs_gt5=int(np.sum(np.abs(values) > 5.)))


def scores(residual, mask, pupils):
    return dict(all=summary(residual, mask),
                by_grism={p: summary(residual, mask & (pupils == p)[:, None, None])
                          for p in ('R', 'C')},
                by_exposure=[summary(r, m) for r, m in zip(residual, mask)])


def leakage(line, error, valid, sideband, core):
    records = []
    for prediction, err, good, side, center in zip(line, error, valid, sideband, core):
        standardized = np.divide(prediction, err, out=np.zeros_like(err), where=good)
        total = float(np.sum(prediction[good]))
        records.append(dict(line_flux_fraction_in_sidebands=float(np.sum(prediction[side])/total)
                            if total > 0 else None,
            sideband_line_ERR_norm=float(np.linalg.norm(standardized[side])),
            sideband_line_max_abs_ERR=float(np.max(np.abs(standardized[side]))),
            core_line_ERR_norm=float(np.linalg.norm(standardized[good & center]))))
    return records


def draw_rows(path, rows, data_line, line, raw_residual, valid, pupils, held,
              nominal, projected, record, norm):
    fig, axes = plt.subplots(len(rows), 3, figsize=(12, 3.6*len(rows)), squeeze=False,
                             constrained_layout=True)
    cmap = plt.get_cmap('gist_heat').copy()
    cmap.set_bad('white')
    residual_cmap = plt.get_cmap('RdBu_r').copy()
    residual_cmap.set_bad('white')
    for row, e in enumerate(rows):
        for column, (values, label) in enumerate(zip(
                (data_line[e], line[e], raw_residual[e]),
                ('Data − sideband continuum/background', 'Frozen line prediction', 'Residual / ERR'))):
            ax = axes[row, column]
            image = np.ma.masked_where(~valid[e], values)
            if column < 2:
                artist = ax.imshow(image, origin='lower', cmap=cmap, norm=norm,
                                   interpolation='nearest')
            else:
                artist = ax.imshow(image, origin='lower', cmap=residual_cmap, vmin=-5., vmax=5.,
                                   interpolation='nearest')
            ax.plot(*nominal[e], 'o', color='white', markeredgecolor='black',
                    markerfacecolor='none', markersize=9., markeredgewidth=1.3)
            if projected is not None:
                ax.plot(*projected[e], 'x', color='cyan', markersize=9., markeredgewidth=1.8)
            cx, cy = nominal[e]
            ax.set_xlim(cx-25., cx+25.)
            ax.set_ylim(cy-25., cy+25.)
            ax.set_title(f'{pupils[e]} · exposure {int(held[e])+1}\n{label}', fontsize=10)
            ax.set_xlabel('Detector cutout x [pixel]')
            ax.set_ylabel('Detector cutout y [pixel]')
            colorbar = fig.colorbar(artist, ax=ax, fraction=.046, pad=.025,
                                    label='ERR units' if column == 2 else 'DN/s')
            if column < 2:
                ticks = np.linspace(norm.vmin, norm.vmax, 4)
                colorbar.set_ticks(ticks, labels=[f'{value:.3f}' for value in ticks])
    desc = (f'ID21636 · {record["start"]} · frozen prediction\n'
            f'Vrot(0.5″)={record["vrot_0p5arcsec_kms"]:.1f} km/s; '
            f'V sin i={record["vproj_0p5arcsec_kms"]:.1f} km/s; '
            f'i={record["inclination_deg"]:.1f}°; σ=0\n'
            '○ nominal calibrated line position'
            + ('; cyan × projected fitted kinematic center' if projected is not None else '')
            + '; white pixels excluded')
    fig.suptitle(desc, fontsize=12)
    fig.savefig(path, dpi=155)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    report = read_json(output/'holdout.json')
    if not report.get('training_completed'):
        raise ValueError('Wait for all declared training starts to finish before assessment.')
    if report.get('completed'):
        raise ValueError('Assessment already complete; preserve the saved assessment.')
    if len(report['fits']) != len(report['settings']['starts']):
        raise ValueError('Expected all declared training starts.')
    if sha256(output/'observations.npz') != report['observation_sha256']:
        raise ValueError('Observation artifact changed after training.')
    with np.load(output/'observations.npz', allow_pickle=False) as a:
        held = a['held_indices']
        data, error, valid, pupils, nominal, core = [a[k][held].copy() for k in
            ('science', 'error', 'valid', 'pupils', 'line_centers_xy', 'core')]
    nuisance = report['nuisance_settings']
    calibration = fit_sideband_continuum(data, error, valid, pupils, nominal,
        guard=report['settings']['sideband_guard_pixels'],
        coefficient_scale=nuisance['continuum_scale'], background_scale=nuisance['background_scale'])
    sideband = calibration['sideband_mask']
    line_window = valid & ~sideband
    center_mask = valid & core & ~sideband
    independent = data-calibration['prediction']
    effective_error = np.sqrt(error**2+calibration['prediction_variance'])
    np.savez_compressed(output/'heldout_sideband_calibration.npz', held_indices=held,
        **{k: v for k, v in calibration.items() if isinstance(v, np.ndarray)},
        line_window=line_window, core=center_mask)
    report['heldout_sideband_calibration'] = dict(
        artifact='heldout_sideband_calibration.npz',
        sha256=sha256(output/'heldout_sideband_calibration.npz'),
        metadata=calibration['metadata'], background_z=calibration['background_z'].tolist(),
        background_DN_s=calibration['background'].tolist(),
        policy=calibration['calibration_policy'], variance_interpretation=calibration['variance_interpretation'])
    report['assessment_policy'] = ('Same frozen sideband prediction for every training candidate. '
        'Raw scores use reported ERR. The second residual scaling adds the diagonal continuum '
        'prediction variance; it omits cross-pixel covariance and is descriptive, not a calibrated '
        'chi-square confidence level or Bayes factor. No held-out score selects or refits candidates.')
    centers_path = output/'projected_centers.json'
    projected_centers = read_json(centers_path) if centers_path.exists() else None
    if projected_centers is not None:
        report['projected_centers_sha256'] = sha256(centers_path)
    # One data-determined stretch for all candidates and exposures. Asinh keeps
    # faint structure visible without saturating the observed core maxima.
    values = independent[center_mask]
    lower = float(np.percentile(values, 1.))
    upper = float(np.max(values))
    norm = AsinhNorm(linear_width=max(float(np.median(error[center_mask])), 1.e-8),
                     vmin=lower, vmax=max(upper, lower+1.e-8))
    representative = [int(np.flatnonzero(pupils == p)[0]) for p in ('R', 'C')]
    for record in report['fits']:
        artifact = output/record['artifact']
        if sha256(artifact) != record['artifact_sha256']:
            raise ValueError(f'Training artifact changed: {artifact}')
        with np.load(artifact, allow_pickle=False) as a:
            line = a['line'][held].copy()
            if not np.allclose(a['gains'][held], 1., rtol=0., atol=1.e-14):
                raise ValueError('Held-out gains must remain at the nominal response.')
        raw = np.divide(independent-line, error, out=np.full_like(data, np.nan), where=valid)
        adjusted = np.divide(independent-line, effective_error,
                             out=np.full_like(data, np.nan), where=valid)
        metrics = dict(line_window_raw_ERR=scores(raw, line_window, pupils),
                       core_raw_ERR=scores(raw, center_mask, pupils),
                       line_window_with_diagonal_continuum_variance=scores(adjusted, line_window, pupils),
                       core_with_diagonal_continuum_variance=scores(adjusted, center_mask, pupils),
                       predicted_line_in_sidebands=leakage(line, error, valid, sideband, core))
        record['heldout'] = metrics
        name = record['start']
        predicted_center = None
        if projected_centers is not None and name in projected_centers:
            predicted_center = np.asarray(projected_centers[name]['centers_xy'])[held]
        saved = output/f'heldout_{name}.npz'
        np.savez_compressed(saved, held_indices=held, science=data, error=error, valid=valid,
            line=line, sideband_prediction=calibration['prediction'],
            sideband_prediction_variance=calibration['prediction_variance'],
            total=line+calibration['prediction'], residual_raw_ERR=raw,
            residual_with_diagonal_continuum_variance=adjusted, line_window=line_window,
            core=center_mask, line_centers_xy=nominal,
            projected_kinematic_centers_xy=np.empty((0, 2)) if predicted_center is None else predicted_center)
        record['heldout_artifact'] = saved.name
        record['heldout_artifact_sha256'] = sha256(saved)
        record['heldout_plots'] = [f'heldout_{name}_RC.png', f'heldout_{name}_all.png']
        for filename, rows in zip(record['heldout_plots'], (representative, range(len(held)))):
            draw_rows(output/filename, rows, independent, line, raw, valid, pupils,
                      held, nominal, predicted_center, record, norm)
        print(f'{name}: V={record["vrot_0p5arcsec_kms"]:.2f}, '
              f'held core raw RMS={metrics["core_raw_ERR"]["all"]["rms"]:.4f}, '
              f'raw squared sum={metrics["core_raw_ERR"]["all"]["sum_squared"]:.3f}', flush=True)
    report.update(completed=True, assessment_finished_utc=datetime.now(timezone.utc).isoformat(),
                  assessment_code_sha256=sha256(__file__),
                  continuum_code_sha256=sha256(Path(__file__).parents[1]/'dingo/native_holdout_continuum.py'))
    write_json(output/'holdout.json', report)
    print('Completed CPU held-out assessment:', output, flush=True)


if __name__ == '__main__':
    main()
