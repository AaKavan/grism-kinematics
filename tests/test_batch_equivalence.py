"""The batched likelihood must agree with the scalar one it replaces.

Run:  python tests/test_batch_equivalence.py        (exit 0 = pass)

ID21636 is the important case, not ID15665. Its frozen variance map contains 2 zero-variance
pixels, and an earlier version of log_likelihood_batch divided by var_map before applying the
mask -- 1/0 = inf, inf*0 = NaN -- so every walker came back -inf and the chain silently froze.
ID15665 has no such pixels and passed regardless, which is exactly why it was not enough.
"""
import os
import sys

os.environ.setdefault('GRISM_CAL_DIR', '/home/aakavan/kinematics/grism_cal')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')          # CPU: deterministic and enough
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import logging
import warnings

import numpy as np
import torch

warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)
from dingo import fitting                                   # noqa: E402

SOURCES = ('ID21636', 'ID15665')     # with / without zero-variance pixels
TOL = 1e-6


def _fitter(sid):
    os.chdir(f'{REPO}/data')
    f = fitting.KinematicsFitter(config_path=f'{sid}/config_kinematics.yaml')
    f.fit_all()
    f.setup_noise_model()
    return f


def check(sid):
    torch.set_num_threads(2)
    f = _fitter(sid)
    # Never resume or write a checkpoint: the test must not touch real results, and a
    # leftover checkpoint with a different nburn makes fit_MCMC refuse to start.
    mc = f.config.setdefault('mcmc', {})
    mc['checkpoint'] = None
    mc['backend'] = None
    mc['resume'] = False
    f.fit_MCMC(nwalkers=32, nsteps=4, nburn=0, vectorize=False)
    nzero = int((f.mcmc_var_map <= 0).sum())
    theta = np.asarray(f.mcmc_chain)[-12:]

    scalar = np.array([f.log_probability(t) for t in theta])
    batch = np.asarray(f.log_probability_batch(theta))

    same_support = np.array_equal(np.isfinite(scalar), np.isfinite(batch))
    finite = np.isfinite(scalar) & np.isfinite(batch)
    rel = (np.abs(batch[finite] - scalar[finite])
           / np.maximum(np.abs(scalar[finite]), 1e-12)).max() if finite.any() else np.inf

    ok = same_support and finite.sum() == len(theta) and rel < TOL
    print(f'{sid}: var_map<=0 in {nzero} px | finite {finite.sum()}/{len(theta)} | '
          f'-inf pattern identical {same_support} | max rel diff {rel:.2e} | '
          f'{"PASS" if ok else "FAIL"}')
    return ok


if __name__ == '__main__':
    results = [check(s) for s in SOURCES]
    print('\nALL PASS' if all(results) else '\nFAILED')
    sys.exit(0 if all(results) else 1)


def test_vsini_reparameterisation_is_exact():
    """Sampling V_sini must give a bit-identical velocity field to V_rot*sin(inc).

    The whole point of the reparameterisation is that it changes only which
    coordinate the sampler walks in, never the model. If this drifts, every chain
    fitted in one parameterisation stops being comparable with the other.
    """
    import numpy as np
    import torch
    from dingo import kinematics as K

    T = torch.tensor
    y, x = torch.meshgrid(torch.arange(81.), torch.arange(81.), indexing='ij')
    for V, inc in ((618.4, 0.456), (223.3, 1.0906), (1000.0, 0.12), (300.0, 1.4)):
        kw = dict(R_v=T(4.686), x0_v=T(42.3), y0_v=T(40.0),
                  theta_v=T(2.36), inc_v=T(inc))
        a = K.arctangent_disk_velocity_model(x, y, V_rot=T(V), **kw)
        b = K.arctangent_disk_velocity_model(x, y, V_sini=T(V * np.sin(inc)), **kw)
        # float32: 1e-7 relative on a ~300 km/s field
        assert torch.allclose(a, b, atol=1e-4), (V, inc, (a - b).abs().max().item())
