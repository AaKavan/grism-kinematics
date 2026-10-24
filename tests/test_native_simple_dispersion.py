"""Checks for the fixed-coordinate scan and its honest candidate selection."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diagnostics.native_simple_dispersion import (
    pack_free, unpack_free, choose_best, starting_branches, snapshot_inputs)
from dingo.real_workflow import sha256


def test_sigma_removed_from_optimizer_not_clipped_at_boundary():
    full = np.arange(14, dtype=float)
    free = pack_free(full)
    restored = np.asarray(unpack_free(jnp.asarray(free), jnp.asarray(0.)))
    np.testing.assert_array_equal(np.delete(restored, 5), free)
    assert restored[5] == 0
    assert len(free) == len(full)-1
    jac = np.asarray(jax.jacfwd(lambda p: unpack_free(p, jnp.asarray(20.)))(jnp.asarray(free)))
    np.testing.assert_array_equal(jac[5], 0)
    np.testing.assert_array_equal(np.delete(jac,5,axis=0),np.eye(len(free)))
    assert np.asarray(unpack_free(jnp.asarray(free),jnp.asarray(20.)))[5] == pytest.approx(.2)


def test_lower_failed_stationarity_candidate_remains_visible():
    base = dict(model='exponential',variant='baseline',fixed_sigma_kms=20.)
    fits = [base | dict(objective=5.,stationarity_passed=True),
            base | dict(objective=4.,stationarity_passed=False),
            base | dict(model='flexible',objective=3.,stationarity_passed=True)]
    best = choose_best(fits)
    assert len(best) == 2
    assert best[0]['objective'] == 4
    assert best[0]['stationarity_passed'] is False


def branch(artifact, velocity, objective, angle=0.):
    return dict(artifact=artifact,mode='positive',objective=objective,stationarity_passed=True,
                vrot_0p5arcsec_kms=velocity,theta_east_to_north_deg=angle,
                center_east_north_arcsec=[0.,0.])


def test_branch_selection_rejects_repeats_and_retains_distinct_basin():
    fits=[branch('best',200.,1.),branch('repeat',200.01,1.001),branch('other',400.,10.)]
    assert [f['artifact'] for f in starting_branches(fits,'positive')] == ['best','other']
    with pytest.raises(ValueError,match='no separated'):
        starting_branches(fits[:2],'positive')


def test_changed_snapshot_is_not_silently_replaced(tmp_path):
    source=tmp_path/'input.json'
    source.write_text('{"source_id": 21636}')
    output=tmp_path/'run'
    paths,hashes=snapshot_inputs(output,{str(source):sha256(source)})
    (output/paths[str(source)]).write_text('changed')
    with pytest.raises(ValueError,match='snapshot changed'):
        snapshot_inputs(output,{str(source):sha256(source)})
