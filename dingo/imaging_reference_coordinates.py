"""Coordinate references for a saved PySersic imaging fit.

Imaging pixels, sky coordinates, and nominal DINGO coordinates are kept
distinct. A native detector WCS supplies the orientation and scale of an
optional nominal DINGO reference; it does not establish registration of the
old R/C coadds or equate stellar and gas centers. This module only reads data.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.io import fits
from astropy.wcs import WCS


def _source_config(config_path):
    """Read the catalog origin and check that its sky coordinates are finite."""
    config = yaml.safe_load(Path(config_path).expanduser().read_text())
    summary = config["summary"]
    if not np.isfinite([summary["ra"], summary["dec"]]).all():
        raise ValueError("Source config needs finite catalog RA and Dec.")
    catalog = SkyCoord(summary["ra"], summary["dec"], unit="deg")
    return config, catalog


def _native_wcs(path, config, catalog, *, margin=0.):
    """Read a matching SCI detector WCS without loading its image array."""
    with fits.open(path, memmap=True) as hdus:
        if "SCI" not in hdus:
            raise ValueError("Native reference must contain a SCI extension.")
        header = hdus["SCI"].header
        primary = hdus[0].header
        expected_filter = str(config["summary"]["filter"]).upper()
        actual_filter = str(primary.get("FILTER", header.get("FILTER", ""))).upper()
        if actual_filter != expected_filter:
            raise ValueError("Native reference filter does not match the source config.")
        expected_module = str(config["image"]["R"]["module"]).upper()
        actual_module = str(primary.get("MODULE", header.get("MODULE", ""))).upper()
        if actual_module != expected_module:
            raise ValueError("Native reference module does not match the R stack config.")
        wcs = WCS(header, fix=False).celestial
        if not wcs.has_celestial:
            raise ValueError("Native reference lacks a celestial SCI WCS.")
        x, y = np.asarray(wcs.world_to_pixel(catalog), dtype=float)
        width, height = header["NAXIS1"], header["NAXIS2"]
        if (not np.isfinite([x, y]).all() or x-margin < -.5 or y-margin < -.5
                or x+margin > width-.5 or y+margin > height-.5):
            raise ValueError("Source is outside the native reference's usable coverage.")
    return wcs


def find_native_reference(config_path, bundle):
    """Find an optional detector WCS for a *nominal* DINGO center reference.

    Search the supplied bundle's direct/<filter>_imaging/obsnum* directories
    for the R stack's module. Return the first sorted matching SCI exposure
    covering the full source stamp, or None if no suitable exposure exists.
    No images are extracted, registered, or modified by this discovery step.
    """
    config, catalog = _source_config(config_path)
    summary = config["summary"]
    module = str(config["image"]["R"]["module"]).lower()
    root = Path(bundle).expanduser()/"direct"/f"{summary['filter']}_imaging"
    for path in sorted(root.glob(f"obsnum*/*nrc{module}long*_crf.fits")):
        try:
            _native_wcs(path, config, catalog, margin=float(summary["r_fit"]))
        except (OSError, KeyError, ValueError):
            continue
        return path.resolve()
    return None


def center_reference(result, *, config_path, native_reference=None):
    """Summarize a saved imaging centroid in explicitly named coordinate frames.

    Parameters
    ----------
    result : ImagingResults
        Saved single-Sérsic result with ``path/cutout.npz`` and paired ``xc``
        and ``yc`` draws from ``_physical_samples('full_pixel')``. This function
        neither fits nor decides whether that posterior has converged.
    config_path : path-like
        Source configuration supplying catalog RA/Dec and the DINGO ``r_fit``.
    native_reference : path-like or None
        Optional matching direct SCI exposure. Its detector WCS rotates and
        scales each centroid offset, anchoring the catalog at (r_fit,r_fit).
        This is an approximate reference, not measured R/C stack registration.

    Returns
    -------
    pandas.DataFrame
        Rows give median, 16th/84th percentiles, their offsets, units, and an
        explanation. The table's ``attrs['notes']`` records coordinate caveats.
        Without a saved celestial cutout WCS only the native imaging pixel
        rows are returned; missing sky information is never filled with zeros.

    Notes
    -----
    Every coordinate transformation uses paired posterior draws. Intervals
    propagate centroid sampling uncertainty only, excluding astrometric
    systematics and any difference between stellar and gas centers. All pixel
    coordinates are zero-based (x=column, y=row).
    """
    config, catalog = _source_config(config_path)
    source_id = getattr(result, "source_id", None)
    if source_id is not None and config["summary"].get("name") != f"ID{source_id}":
        raise ValueError("Imaging result and source configuration identify different sources.")
    samples = result._physical_samples("full_pixel")
    x, y = np.asarray(samples["xc"], dtype=float), np.asarray(samples["yc"], dtype=float)
    if x.shape != y.shape or not x.size or not np.isfinite([x, y]).all():
        raise ValueError("Imaging xc and yc need finite paired posterior draws of matching shape.")
    x, y = x.ravel(), y.ravel()
    rows = []

    def add(name, values, unit, meaning, *, wrap_ra=False):
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite transformed coordinates for {name}.")
        low, median, high = np.percentile(values, [16, 50, 84])
        if wrap_ra:
            # Preserve small intervals across RA=0 while displaying the median
            # in the usual [0,360) range. Endpoints may straddle that seam.
            shift = median - median % 360.
            low, median, high = low-shift, median-shift, high-shift
        rows.append(dict(parameter=name, median=median, minus_1sigma=median-low,
                         plus_1sigma=high-median, p16=low, p84=high,
                         unit=unit, meaning=meaning))

    add("image_xc", x, "pixel", "F150W cutout x center; not a DINGO pixel coordinate")
    add("image_yc", y, "pixel", "F150W cutout y center; not a DINGO pixel coordinate")
    notes = ["Zero-based imaging pixels: x=column, y=row. Do not copy them directly into DINGO.",
             "Intervals include centroid posterior uncertainty only; astrometric systematics and stellar/gas offsets are excluded."]
    metadata = dict(source_id=source_id, config_path=str(Path(config_path).resolve()),
                    native_reference=None, notes=notes)

    def table():
        frame = pd.DataFrame(rows).set_index("parameter")
        frame.attrs.update(metadata)
        frame.attrs["note"] = "\n".join(notes)
        return frame

    with np.load(Path(result.path)/"cutout.npz", allow_pickle=False) as cutout:
        raw_header = cutout["wcs_header"].item() if "wcs_header" in cutout else None
    if raw_header is None:
        notes.append("Sky and nominal DINGO coordinates unavailable: saved cutout has no WCS header.")
        return table()
    if isinstance(raw_header, bytes):
        raw_header = raw_header.decode("ascii")
    imaging_wcs = WCS(fits.Header.fromstring(str(raw_header)), fix=False).celestial
    if not imaging_wcs.has_celestial:
        notes.append("Sky and nominal DINGO coordinates unavailable: saved cutout WCS is not celestial.")
        return table()
    sky = imaging_wcs.pixel_to_world(x, y)
    ra = catalog.ra.deg + Angle(sky.ra-catalog.ra).wrap_at(180*u.deg).deg
    east, north = catalog.spherical_offsets_to(sky)
    add("ra", ra, "deg", "Fitted centroid right ascension", wrap_ra=True)
    add("dec", sky.dec.deg, "deg", "Fitted centroid declination")
    add("east_offset", east.to_value(u.arcsec), "arcsec", "East of source catalog position")
    add("north_offset", north.to_value(u.arcsec), "arcsec", "North of source catalog position")

    if native_reference is not None:
        native_reference = Path(native_reference).expanduser().resolve()
        native_wcs = _native_wcs(native_reference, config, catalog)
        catalog_x, catalog_y = native_wcs.world_to_pixel(catalog)
        fitted_x, fitted_y = native_wcs.world_to_pixel(sky)
        anchor = float(config["summary"]["r_fit"])
        if not np.isfinite(anchor):
            raise ValueError("Source r_fit must be finite for nominal DINGO coordinates.")
        add("nominal_dingo_x0", anchor+fitted_x-catalog_x, "pixel",
            "Approximate DINGO x0; catalog anchored at cutout midpoint")
        add("nominal_dingo_y0", anchor+fitted_y-catalog_y, "pixel",
            "Approximate DINGO y0; catalog anchored at cutout midpoint")
        metadata["native_reference"] = str(native_reference)
        notes.append(f"Nominal DINGO coordinates anchor the catalog at ({anchor:g},{anchor:g}) using a native detector WCS; no R/C stack registration uncertainty is included.")
    return table()
