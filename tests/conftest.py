'''Shared fixtures.

Everything here exists to undo a side effect the old single test file had at
IMPORT scope, where it applied to the whole pytest session rather than to the
test that wanted it:

  logging.disable(logging.WARNING)      silenced every later test's warnings
  warnings.filterwarnings('ignore')     same, for warnings
  sys.path.insert(0, REPO)              hid whether the package was installed
  os.chdir(f'{REPO}/data')              never restored
  GRISM_CAL_DIR = '/home/aakavan/...'   one user's home, hard-coded

Each is now a scoped fixture that restores itself, and the calibration directory
is discovered rather than assumed, so the suite skips cleanly on a machine that
does not have it instead of failing with a confusing I/O error.
'''
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pytest_configure(config):
    config.addinivalue_line('markers', 'data: needs staged FITS under data/')


@pytest.fixture(scope='session')
def repo():
    return REPO


@pytest.fixture(scope='session', autouse=True)
def _deterministic_cpu():
    '''CPU, single thread: the likelihood must be reproducible to the bit for the
    golden comparisons, and scatter_add_ on CUDA is only deterministic by
    construction, not by default.'''
    import torch
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture(scope='session')
def grism_cal():
    '''Locate the WFSS calibration tables, or skip.

    Order: an explicit GRISM_CAL_DIR, then the sibling checkout layout that the
    scripts all assume. Nothing is hard-coded to a home directory.
    '''
    for cand in (os.environ.get('GRISM_CAL_DIR'),
                 os.path.join(os.path.dirname(REPO), 'grism_cal'),
                 '/data/grism_cal'):
        if cand and os.path.isdir(cand):
            os.environ['GRISM_CAL_DIR'] = cand
            return cand
    pytest.skip('no grism calibration directory; set GRISM_CAL_DIR')


@pytest.fixture
def in_data_dir(grism_cal, monkeypatch):
    '''Run inside data/, where the configs' relative `path:` entries resolve.

    monkeypatch.chdir restores the previous directory at teardown; the bare
    os.chdir this replaces did not, so test order silently mattered.
    '''
    monkeypatch.chdir(os.path.join(REPO, 'data'))
    return os.path.join(REPO, 'data')


@pytest.fixture(scope='session')
def staged():
    """Skip a test whose source has no staged FITS.

    The cutouts are not in the repository -- they are SAPPHIRES EDR products and
    are gitignored -- so a fresh clone can run the pure tests but not the
    data-backed ones. Those must SKIP with a clear reason rather than fail with
    a FileNotFoundError that looks like a bug in the code.
    """
    def _require(sid):
        d = os.path.join(REPO, 'data', sid)
        missing = [f for f in ('R.fits', 'C.fits') if not os.path.exists(os.path.join(d, f))]
        if missing:
            pytest.skip(f'{sid}: no staged {", ".join(missing)} (data is not in the repo)')
        return d
    return _require


@pytest.fixture
def fitter(in_data_dir, staged):
    '''Build a KinematicsFitter for a source id, without fitting.

    Construction is dominated by load_nircam_wfss_model (ASCII tables + splines),
    not by the fit, so a test that pins theta explicitly costs ~2 s rather than
    the ~30 s a fit_all() would add.
    '''
    from dingo import fitting as _f

    def _make(sid):
        staged(sid)
        return _f.KinematicsFitter(config_path=f'{sid}/config_kinematics.yaml')
    return _make
