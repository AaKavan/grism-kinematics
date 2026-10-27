"""A recovery fixture must not borrow real pixels or leak mutable inputs."""
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from dingo.simple_simulation import make_mock, load_mock


def test_same_seed_reproduces_images_and_noise_is_independent(tmp_path, grism_cal):
    first = make_mock(tmp_path/"first", noise_sigma=.02, seed=17)
    second = make_mock(tmp_path/"second", noise_sigma=.02, seed=17)
    different = make_mock(tmp_path/"different", noise_sigma=.02, seed=18)
    source = lambda mock, pupil: fits.getdata(mock.data_dir/f"ID{mock.source_id}"/f"{pupil}.fits")
    with np.load(first.directory/"truth.npz") as saved:
        residuals = []
        for pupil in ("R", "C"):
            np.testing.assert_array_equal(source(first, pupil), source(second, pupil))
            assert not np.array_equal(source(first, pupil), source(different, pupil))
            residuals.append(source(first, pupil)-saved[f"noiseless_{pupil}"])
    for noise in residuals:
        assert np.std(noise) == pytest.approx(.02, rel=.05)
    assert abs(np.corrcoef([a.ravel() for a in residuals])[0, 1]) < .05
    assert first.metadata["psf"] == "none"
    assert first.metadata["intrinsic_dispersion_kms"] == 0
    assert first.truth["velocity.V_rot"] == 200.


def test_nonrotating_source_follows_constant_calibrated_translation(tmp_path, grism_cal):
    """A zero-velocity image must translate by the trace, not rotate/shear."""
    import torch
    from dingo import grism, utils

    mock = make_mock(tmp_path/"null", velocity=0., noise_sigma=0., effective_radius=4.)
    with np.load(mock.directory/"truth.npz") as saved:
        assert np.count_nonzero(saved["velocity_los"]) == 0
        source = saved["image"]
        yy, xx = np.indices(source.shape)
        centroid = lambda a: np.array([(a*xx).sum(), (a*yy).sum()])/a.sum()
        for pupil in ("R", "C"):
            _, spatial, dispersion = grism.load_nircam_wfss_model(pupil, "B", "F444W")
            forward = utils.get_grism_model_torch(spatial, dispersion, pupil, 1024, 1024,
                                                 direction="forward", device="cpu")
            wavelength = torch.tensor([1.875], dtype=torch.float32)*(1+1.1231)
            at_middle = forward(torch.tensor(40.), torch.tensor(40.), wavelength)
            origin = np.array([int(at_middle[0])-40, int(at_middle[1])-40])
            displacement = np.array([float(v) for v in forward(torch.tensor(0.), torch.tensor(0.), wavelength)])
            expected_shift = origin-displacement
            actual = saved[f"noiseless_{pupil}"]
            np.testing.assert_allclose(centroid(actual)-centroid(source), expected_shift, atol=2e-5)
            assert actual.sum() == pytest.approx(source.sum(), rel=2e-6)
            working = fits.getdata(mock.data_dir/f"ID{mock.source_id}"/f"{pupil}.fits")
            np.testing.assert_array_equal(working, actual)


def test_changed_inputs_are_rejected_and_invalid_overwrite_preserves_case(tmp_path, grism_cal):
    mock = make_mock(tmp_path/"case")
    original_record = (mock.directory/"truth.json").read_bytes()
    with pytest.raises(ValueError, match="nonnegative"):
        make_mock(mock.directory, noise_sigma=-1., overwrite=True)
    assert (mock.directory/"truth.json").read_bytes() == original_record
    path = mock.data_dir/f"ID{mock.source_id}"/"R.fits"
    with fits.open(path, mode="update") as hdus:
        hdus[0].data[0, 0] += 1.
    with pytest.raises(ValueError, match="input changed"):
        load_mock(mock.directory)


def test_unmarked_or_real_data_directory_cannot_be_overwritten(tmp_path):
    from dingo.simple_simulation import REPO
    unrelated = tmp_path/"unrelated"
    unrelated.mkdir()
    sentinel = unrelated/"keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(ValueError, match="without a simulation record"):
        make_mock(unrelated, overwrite=True)
    assert sentinel.read_text() == "keep"
    with pytest.raises(ValueError, match="real source data"):
        make_mock(REPO/"data"/"ID3700", overwrite=True)


def test_saved_config_points_to_final_inputs_without_real_data(tmp_path, grism_cal):
    import yaml
    mock = make_mock(tmp_path/"case")
    folder = mock.data_dir/f"ID{mock.source_id}"
    config = yaml.safe_load((folder/"config_kinematics.yaml").read_text())
    for pupil in ("R", "C"):
        assert Path(config["image"][pupil]["path"][0]) == folder/f"{pupil}.fits"
    assert load_mock(mock.directory).truth == mock.truth
    assert mock.metadata["noise_sigma"] == .02
