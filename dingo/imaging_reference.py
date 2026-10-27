"""A calibrated F150W PySersic reference for the simple DINGO notebook.

This is an imaging-only operation. It neither prepares a kinematic experiment
nor changes its starting values or priors. The single-Sersic sampler, progress,
checkpoints, and source-level cache are supplied by ``dingo.morphology``.
"""
from pathlib import Path

import numpy as np

from .morphology import fit_morphology
from .real_workflow import DEFAULT_BUNDLE, source_number


def _export_cutout(result):
    """Save the actual linear science/error cutout as WCS-bearing FITS files."""
    from astropy.io import fits

    path = result.path / "cutout.npz"
    if not path.exists():
        return
    metadata = result._check_inputs(result._report("full_pixel"))
    with np.load(path, allow_pickle=False) as saved:
        header = fits.Header.fromstring(str(saved["wcs_header"].item()), sep="")
        header["FILTER"] = "F150W"
        header["SOURCEID"] = result.source_id
        if metadata.get("unit"):
            header["BUNIT"] = metadata["unit"]
        for name, key in (("F150W_image.fits", "image"), ("F150W_error.fits", "error")):
            fits.writeto(result.path / name, saved[key], header, overwrite=True)


def fit_reference(source_id, *, data_dir, output_root, edr_dir=None,
                  cutout_size=101, fit_radius=1.4, warmup=1000, draws=2000,
                  device="cuda", refit=False, seed=735, psf=None, bundle=None,
                  native_reference=None, display=True, show_center_values=False):
    """Fit/show a single Sersic component before manually choosing DINGO settings.

    Parameters
    ----------
    source_id : int
        Source to extract from the calibrated F150W mosaics. Coordinates are
        read from ``data_dir/ID<source_id>/config_kinematics.yaml``.
    data_dir, output_root : path-like
        The same paths used by ``simple_mcmc.prepare``. Imaging is independent
        of any kinematic run and goes under
        ``output_root/<ID>/morph_fits/F150W_reference/fit_NNN``.
    edr_dir : path-like, optional
        Survey root containing ``sapphires_edr_nircam_sci`` and
        ``sapphires_edr_nircam_err``. Defaults to ``~/EDR_data``.
    cutout_size : int
        Odd width of the native mosaic cutout in pixels (101 by default).
        The pixel scale comes from its WCS, not from the R/C stacks.
    fit_radius : float
        Circular fitting radius in arcseconds within that cutout. Lower it
        to exclude an edge neighbour; the source center is not masked.
    warmup, draws : int
        Discarded warmup and retained samples per chain. The existing PySersic
        worker runs four chains with progress and checkpoints. Completion is
        not a convergence guarantee; diagnostics remain visible.
    refit : bool
        False reuses a matching saved fit. True creates a new fit directory,
        preserving earlier results. Set True after changing imaging settings.
    psf, bundle : path-like, optional
        A calibrated F150W empirical PSF, or the calibration bundle containing
        its archive. The PSF is resampled to the actual mosaic pixel scale.
    native_reference : path-like, optional
        Native detector exposure with a SCI WCS, for an approximate DINGO
        center reference. When absent, try a matching exposure in the local
        bundle. This is a catalog-anchored estimate, not measured registration
        of the legacy R/C stacks; it never changes their center settings.
    display : bool
        Show the input, image/model panels, PySersic's standard corner, and
        morphology and inclination tables.
        No residual panel is displayed. The image stretch is display-only.
    show_center_values : bool
        False hides standalone center numbers on the image/model panels and
        printed tables, so imaging pixels cannot be mistaken for DINGO pixels.
        The cyan marker and complete native corner plot remain. Center samples
        and coordinate-reference files are still saved for later inspection.

    Returns
    -------
    ImagingResults
        Saved PySersic results, including completed nonconverged fits for
        inspection. This reference does not impose a kinematic prior and does
        not infer a rotation velocity from imaging.
    """
    source_id = source_number(source_id)
    data_dir = Path(data_dir).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    edr_dir = Path(edr_dir or Path.home() / "EDR_data").expanduser().resolve()
    config_path = data_dir / f"ID{source_id}" / "config_kinematics.yaml"
    local_bundle = edr_dir / "magnif_copy/sapphires_kinematics_inputs_2026-10-13"
    bundle = Path(bundle or (local_bundle if local_bundle.exists() else DEFAULT_BUNDLE))
    directory = output_root / str(source_id) / "morph_fits/F150W_reference"

    result = fit_morphology(
        source_id, directory=directory, config_path=config_path,
        science=edr_dir / "sapphires_edr_nircam_sci/4750_F150W_v05_sci.fits",
        error=edr_dir / "sapphires_edr_nircam_err/4750_F150W_v05_err.fits",
        psf=psf, bundle=bundle, cutout_size=cutout_size, fit_radius=fit_radius,
        warmup=warmup, draws=draws, seed=seed, device=device, refit=refit,
        display=display, center_reference=True, show_center_values=show_center_values,
    )
    _export_cutout(result)

    if (result.path / "full_pixel/samples.npz").exists():
        from .imaging_reference_coordinates import center_reference, find_native_reference

        if native_reference is None:
            native_reference = find_native_reference(config_path, bundle)
        table = center_reference(result, config_path=config_path,
                                 native_reference=native_reference)
        destination = result.path / f"pysersic_{source_id}_center.cat"
        destination.write_text(table.to_string(float_format=lambda value: f"{value:.8g}")
                               + "\n\n" + table.attrs.get("note", "")
                               + f"\nNative detector WCS: {table.attrs.get('native_reference')}\n")
        if display and show_center_values:
            from IPython.display import display as show

            show(table)
            print(table.attrs.get("note", ""))
    print(f"PySersic reference saved in: {result.path}", flush=True)
    print("Imaging is a reference for your manual settings; no DINGO values or priors were changed.",
          flush=True)
    return result
