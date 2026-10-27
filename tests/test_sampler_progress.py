"""Progress display must leave warmup, retained draws, and checkpoints intact."""
from contextlib import nullcontext

import numpy as np
import pytest

from dingo.fitting import BaseFitter


class GaussianFitter(BaseFitter):
    """A tiny analytic posterior; no astronomical data or model fitting."""

    def __init__(self, checkpoint):
        self.device = "cpu"
        self.config = {"mcmc": {
            "nwalkers": 8, "nburn": 3, "nsteps": 5, "seed": 7,
            "checkpoint": str(checkpoint), "checkpoint_every": 2,
            "resume": False, "reseed_stuck": False,
        }}
        self.mcmc_param_names = ["x", "y"]

    def _setup_data(self):
        pass

    def loss(self):
        raise AssertionError("This test should sample only the analytic posterior.")

    def prepare_mcmc(self, stage=None):
        return {"theta0": np.zeros(2), "ndim": 2}

    def log_prior(self, theta):
        return 0.0

    def log_probability(self, theta):
        return -0.5 * float(np.dot(theta, theta))

    def mcmc_fingerprint(self):
        return {"posterior": "unit Gaussian"}


@pytest.mark.parametrize("progress", [False, True, "notebook"])
def test_progress_preserves_sampling_and_checkpoints(tmp_path, monkeypatch, progress):
    emcee = pytest.importorskip("emcee")
    bars = []

    class Bar:
        def update(self, count):
            pass

    def capture_bar(enabled, total, **kwargs):
        bars.append((enabled, total, kwargs))
        return nullcontext(Bar())

    monkeypatch.setattr(emcee.ensemble, "get_progress_bar", capture_bar)
    checkpoint = tmp_path / "checkpoint.npz"
    fitter = GaussianFitter(checkpoint)
    # Exercise the omitted argument too: existing callers must stay quiet.
    options = {} if progress is False else {"progress": progress}
    chain, log_probability = fitter.fit_MCMC(**options)

    assert [(enabled, total) for enabled, total, _ in bars] == [
        (progress, 3), (progress, 5),
    ]
    if progress:
        assert [kwargs["desc"] for _, _, kwargs in bars] == ["Warmup", "MCMC draws"]
    assert chain.shape == (5 * 8, 2)
    assert log_probability.shape == (5 * 8,)
    with np.load(checkpoint, allow_pickle=False) as saved:
        assert saved["chain"].shape == (8, 8, 2)
        assert int(saved["nburn"]) == 3
        assert int(saved["nsteps_done"]) == 8


def test_seed_reproduces_fresh_sampler_without_changing_global_rng(tmp_path):
    pytest.importorskip("emcee")
    before = np.random.get_state()
    first = GaussianFitter(tmp_path / "first.npz")
    first.fit_MCMC()
    second = GaussianFitter(tmp_path / "second.npz")
    second.fit_MCMC()
    after = np.random.get_state()

    np.testing.assert_array_equal(first._mcmc_raw_chain, second._mcmc_raw_chain)
    np.testing.assert_array_equal(first._mcmc_raw_log_prob, second._mcmc_raw_log_prob)
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
