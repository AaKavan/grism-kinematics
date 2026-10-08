'''Config provenance: every config claims to come from catalogs.make_config().

Until this module existed that claim could not be checked, and the header
comment it carries warns about the exact failure it was meant to prevent --
ID15665 once carried ID3700's redshift because a config was copied by hand.
'''
import glob
import os

import pytest
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = sorted(glob.glob(os.path.join(REPO, 'data', '*', 'config_kinematics*.yaml')))


@pytest.fixture(scope='module')
def catalogs():
    cat = pytest.importorskip('dingo.catalogs')
    try:
        cat.find_cat_dir()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    return cat


@pytest.mark.parametrize('cfg_path', CONFIGS, ids=lambda p: os.path.basename(os.path.dirname(p)))
def test_config_matches_the_catalogue(catalogs, cfg_path):
    '''z and position must come from the EDR catalogue, not from a copied file.

    The redshift matters twice over: it sets the observed wavelength, and the
    cutout position is derived from it, so a wrong z moves the cutout (it moved
    ~73 px for ID15665) as well as mis-scaling the velocities.
    '''
    s = yaml.safe_load(open(cfg_path))['summary']
    sid = int(''.join(c for c in s['name'] if c.isdigit()))
    rec = catalogs.lookup_source(sid)
    assert rec['in_spec_cat'], f'{sid} is not in the spectroscopic catalogue'
    assert s['z'] == pytest.approx(rec['zspec'], abs=1e-4), 'redshift differs from the catalogue'
    assert s['ra'] == pytest.approx(rec['ra'], abs=3e-5), 'RA differs from the catalogue'
    assert s['dec'] == pytest.approx(rec['dec'], abs=3e-5), 'Dec differs from the catalogue'


def test_lookup_source_rejects_an_unknown_id(catalogs):
    with pytest.raises(Exception):
        catalogs.lookup_source(999999999, require_z=True)
