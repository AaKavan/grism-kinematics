import pytest

from diagnostics.native_simple_refine import selected_starts


def fit(artifact,velocity,objective,sigma=20.):
    return dict(artifact=artifact,model='exponential',variant='baseline',fixed_sigma_kms=sigma,
        vrot_0p5arcsec_kms=velocity,objective=objective,theta_east_to_north_deg=30.,
        center_east_north_arcsec=[0.,0.],stationarity_passed=True)


def test_refinement_retains_a_distinct_stored_basin_when_available():
    pool=[fit('best',150.,1.),fit('duplicate',150.01,1.001),fit('other',250.,10.),
          fit('wrong_sigma',400.,.1,40.)]
    selected,distinct=selected_starts(pool,'exponential','baseline',20.)
    assert [f['artifact'] for f in selected]==['best','other']
    assert distinct


def test_repeated_minima_are_explicitly_flagged_and_both_kept():
    pool=[fit('one',150.,1.),fit('two',150.01,1.001)]
    selected,distinct=selected_starts(pool,'exponential','baseline',20.)
    assert len(selected)==2
    assert not distinct
    with pytest.raises(ValueError,match='two saved starts'):
        selected_starts(pool[:1],'exponential','baseline',20.)
