"""Check physical deprojection and separation of null/coverage diagnostics."""

import copy
import numpy as np

from diagnostics.native_recovery_suite import summarize_run, aggregate


def test_suite_deprojection_mass_ratios_and_null_have_distinct_meanings():
    report = {
        "truth_vref_rturn_theta": [100., .2, .9],
        "truth_inclination_rad": np.pi/6, "fixed_inclination_rad": np.pi/3, "seed": 1,
        "best_by_setting": [{"fitted_vref_rturn_theta": [100., .2, .9],
            "grid": 17, "strength": 0., "stationary": True, "boundary_hits": []}],
        "comparisons": [
            {"grid": 17, "strength": 0., "reference": "null",
             "twice_objective_difference": 10., "valid_optimization": True},
            {"grid": 17, "strength": 0., "reference": "truth_speed",
             "twice_objective_difference": 2., "valid_optimization": True}],
    }
    row, = summarize_run(report, "inclination", "test")
    # Same projected curve with a wrong inclination changes the mass by 1/3.
    np.testing.assert_allclose(row["rotation_only_mass_ratio_profile"], 1/3)
    assert row["nominal_3p84_contains_truth"]
    # A nonrotator's best fit need not be zero. This is not a detection or a
    # mass ratio, and the ordinary one-parameter threshold is not applied.
    null = copy.deepcopy(report)
    null["truth_vref_rturn_theta"][0] = 0.
    null["comparisons"] = null["comparisons"][:1]
    null_row, = summarize_run(null, "null", "test")
    assert null_row["rotation_only_mass_ratio_profile"] is None
    assert null_row["nominal_3p84_contains_truth"] is None
    assert null_row["vref"] == 100
    invalid = copy.deepcopy(report)
    invalid["comparisons"][1]["valid_optimization"] = False
    invalid["best_by_setting"][0]["stationary"] = False
    invalid_row, = summarize_run(invalid, "inclination", "test2")
    summary, = aggregate([row, invalid_row])
    assert summary["n"] == 2
    assert summary["stationary"] == 1
    assert summary["nominal_truth_checks_valid"] == 1
    assert summary["nominal_truth_checks_included"] == 1
