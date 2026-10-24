"""CPU covariance assessment of frozen, sideband-conditioned native predictions.

The active positive-continuum coefficients and signed background are held to
their sideband-selected active set. Their Gaussian curvature is propagated with
all cross-pixel covariance. This is a conditional approximation, not a Bayesian
evidence calculation or a new galaxy/continuum fit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def design_matrix(shape, pupil, coefficients, background_sigma):
    """Flattened detector design, with active strip coefficients then background."""
    yy, xx = np.indices(shape)
    strip, disp = (yy, xx) if pupil == "R" else (xx, yy)
    t = disp.ravel() / (shape[1] - 1)
    basis = np.stack(((1-t)**2, 2*t*(1-t), t*t), axis=1)
    active = np.argwhere(coefficients > 0.)
    design = np.zeros((strip.size, len(active) + 1))
    for column, (s, k) in enumerate(active):
        selected = strip.ravel() == s
        design[selected, column] = basis[selected, k]
    design[:, -1] = background_sigma
    return design


def gaussian_quadratic(design, error, mask, precision, residuals):
    """Woodbury quadratic for columns of residuals; prior precision is sideband H."""
    selected = np.asarray(mask).ravel()
    sigma = np.asarray(error).ravel()[selected]
    a = design[selected] / sigma[:, None]
    r = np.asarray(residuals)[selected] / sigma[:, None]
    full_precision = precision + a.T @ a
    factor = cho_factor(full_precision, lower=True, check_finite=True)
    rhs = a.T @ r
    raw = np.sum(r*r, axis=0)
    correction = np.sum(rhs*cho_solve(factor, rhs), axis=0)
    quadratic = raw - correction
    if np.any(quadratic < -1.e-7):
        raise AssertionError("Negative covariance quadratic")
    return raw, correction, quadratic, factor


def toy_check():
    """Independent dense pixel-covariance inverse checks the Woodbury identity."""
    rng = np.random.default_rng(21636)
    a = rng.normal(size=(11, 4))
    q = rng.normal(size=(4, 4))
    h = q.T@q + np.eye(4)
    err = rng.uniform(.3, 1.2, size=11)
    residual = rng.normal(size=(11, 3))
    actual = gaussian_quadratic(a, err, np.ones(11, dtype=bool), h, residual)[2]
    covariance = np.diag(err**2) + a@np.linalg.solve(h, a.T)
    expected = np.sum(residual*np.linalg.solve(covariance, residual), axis=0)
    np.testing.assert_allclose(actual, expected, rtol=1.e-12, atol=1.e-12)
    return float(np.max(np.abs(actual-expected)))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assess(directory):
    directory = Path(directory).resolve()
    metadata = json.loads((directory/"holdout.json").read_text())
    observations = np.load(directory/"observations.npz", allow_pickle=False)
    side = np.load(directory/"heldout_sideband_calibration.npz", allow_pickle=False)
    held = observations["held_indices"]
    np.testing.assert_array_equal(side["held_indices"], held)
    names = [f["start"] for f in metadata["fits"]]
    candidate_files = [directory/f"heldout_{name}.npz" for name in names]
    candidates = [np.load(p, allow_pickle=False) for p in candidate_files]
    coefficient_scale = float(metadata["nuisance_settings"]["continuum_scale"])
    exposures = []
    variance_check_max = 0.
    for e, index in enumerate(held):
        pupil = str(observations["pupils"][index])
        error = observations["error"][index]
        design = design_matrix(error.shape, pupil, side["coefficients"][e],
                               side["background_sigma"][e])
        prior_diagonal = np.full(design.shape[1], 1./coefficient_scale**2)
        prior_diagonal[-1] = 1.  # Standardized signed background has N(0,1) prior.
        side_selected = side["sideband_mask"][e].ravel()
        weighted_side = design[side_selected]/error.ravel()[side_selected, None]
        precision = np.diag(prior_diagonal) + weighted_side.T@weighted_side
        factor_side = cho_factor(precision, lower=True)
        variance = np.sum(design*cho_solve(factor_side, design.T).T, axis=1)
        variance_difference = float(np.max(np.abs(
            variance-side["prediction_variance"][e].ravel())))
        variance_check_max = max(variance_check_max, variance_difference)
        np.testing.assert_allclose(variance, side["prediction_variance"][e].ravel(),
                                   rtol=1.e-7, atol=1.e-12)
        residual = np.stack([(c["science"][e]-c["total"][e]).ravel()
                             for c in candidates], axis=1)
        exposure = dict(index=int(index), pupil=pupil,
                        active_coefficients=int(design.shape[1]-1), regions={})
        for region, mask in (("line_window", side["line_window"][e]),
                             ("core_15_pixels", side["core"][e])):
            if np.any(mask & side["sideband_mask"][e]):
                raise AssertionError("Scored pixels overlap fitted sidebands")
            raw, correction, quadratic, factor = gaussian_quadratic(
                design, error, mask, precision, residual)
            logdet_extra = float(2.*np.sum(np.log(np.diag(factor[0])))
                              - 2.*np.sum(np.log(np.diag(factor_side[0]))))
            exposure["regions"][region] = dict(
                n_pixels=int(mask.sum()), logdet_covariance_over_noise=logdet_extra,
                candidates={name:dict(raw_chi_square=float(raw[k]),
                    covariance_correction=float(correction[k]),
                    correlated_quadratic=float(quadratic[k]))
                    for k, name in enumerate(names)})
        exposures.append(exposure)
    totals = {}
    for region in ("line_window", "core_15_pixels"):
        totals[region] = {}
        for group in ("all", "R", "C"):
            group_exposures = [e for e in exposures if group == "all" or e["pupil"] == group]
            values = {name: {stat: float(sum(e["regions"][region]["candidates"][name][stat]
                        for e in group_exposures))
                        for stat in ("raw_chi_square", "covariance_correction", "correlated_quadratic")}
                       for name in names}
            delta = {stat: values["saved_higher"][stat]-values["saved_lower"][stat]
                     for stat in ("raw_chi_square", "correlated_quadratic")}
            totals[region][group] = dict(n_pixels=sum(e["regions"][region]["n_pixels"]
                        for e in group_exposures), candidates=values,
                        higher_minus_lower=delta)
    sources = [directory/"observations.npz", directory/"heldout_sideband_calibration.npz",
               *candidate_files]
    result = dict(format="native_holdout_active_continuum_covariance_v1",
        completed_utc=datetime.now(timezone.utc).isoformat(), source_id=metadata["source_id"],
        interpretation="Frozen galaxy predictions; Gaussian continuum uncertainty conditional on the sideband-selected positive-coefficient active set. Includes full cross-pixel coefficient/background covariance. Common covariance determinant cancels candidate differences. Not a full posterior predictive likelihood or Bayes factor.",
        limitations=["Positivity-boundary and active-set uncertainty are omitted.",
            "Galaxy, calibration, and instrument-response uncertainty are not propagated.",
            "One retrospective exposure split; previous full-data analyses informed model choice."],
        formula="C = Dscore + A_score Hside^-1 A_score^T; Q = r^T Dscore^-1 r - b^T (Hside + A_score^T Dscore^-1 A_score)^-1 b",
        toy_dense_inverse_max_absolute_difference=toy_check(),
        saved_diagonal_covariance_max_absolute_difference=variance_check_max,
        input_sha256={str(p):sha256(p) for p in sources}, code_sha256=sha256(__file__),
        exposures=exposures, totals=totals)
    destination = directory/"covariance_assessment.json"
    destination.write_text(json.dumps(result, indent=2)+"\n")
    print(destination)
    for region, groups in totals.items():
        for group, value in groups.items():
            print(region, group, "higher_minus_lower", value["higher_minus_lower"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    assess(args.directory)
