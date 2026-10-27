"""Read, inspect, and explicitly repair user-selected input-image regions.

Masks in this module use ``True`` for excluded pixels. They are attached to
the original input grid, not to the velocity-dependent corrected images.
Reading or displaying a mask does not modify the FITS files or run a fit.
Only ``replace_region_with_background`` edits an input, preserving its backup.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import shutil
import tempfile

import numpy as np


def _physical_coordinates(x, y, header):
    """Convert one-based image centers to DS9 physical pixel coordinates."""
    header = {} if header is None else header
    matrix = np.array([
        [header.get("LTM1_1", 1.), header.get("LTM1_2", 0.)],
        [header.get("LTM2_1", 0.), header.get("LTM2_2", 1.)],
    ], dtype=float)
    offset = np.array([header.get("LTV1", 0.), header.get("LTV2", 0.)], dtype=float)
    if not np.isfinite(matrix).all() or not np.isfinite(offset).all():
        raise ValueError("The FITS physical-coordinate transform is not finite.")
    try:
        physical = np.linalg.solve(matrix, np.stack([x.ravel(), y.ravel()]) - offset[:, None])
    except np.linalg.LinAlgError as exc:
        raise ValueError("The FITS physical-coordinate transform is singular.") from exc
    return physical[0].reshape(x.shape), physical[1].reshape(y.shape)


def _polygon_inside(x, y, vertices):
    """Even–odd polygon fill, including pixel centers on the boundary."""
    inside = np.zeros(x.shape, dtype=bool)
    boundary = np.zeros(x.shape, dtype=bool)
    for (x1, y1), (x2, y2) in zip(vertices, np.roll(vertices, -1, axis=0)):
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            continue
        cross = (x - x1) * dy - (y - y1) * dx
        along = (x - x1) * dx + (y - y1) * dy
        boundary |= ((np.abs(cross) <= 1e-10 * max(1., np.hypot(dx, dy)))
                     & (along >= -1e-10) & (along <= dx * dx + dy * dy + 1e-10))
        if dy != 0:
            crossing = ((y1 > y) != (y2 > y)) & (x < x1 + (y - y1) * dx / dy)
            inside ^= crossing
    return inside | boundary


def _shape_inside(name, values, x, y):
    """Rasterize a supported DS9 shape at pixel centers."""
    if name == "polygon":
        if len(values) < 6 or len(values) % 2:
            raise ValueError("A polygon needs at least three x,y vertex pairs.")
        return _polygon_inside(x, y, np.asarray(values).reshape(-1, 2))
    expected = {"circle": 3, "ellipse": 5, "box": 5}
    if name not in expected:
        raise ValueError(f"Unsupported DS9 shape {name!r}; use circle, ellipse, box, or polygon.")
    if len(values) != expected[name]:
        raise ValueError(f"DS9 {name} requires {expected[name]} values.")
    dx, dy = x - values[0], y - values[1]
    if name == "circle":
        if values[2] <= 0:
            raise ValueError("Circle radius must be positive.")
        return dx * dx + dy * dy <= values[2] ** 2
    if min(values[2:4]) <= 0:
        raise ValueError(f"{name.capitalize()} sizes must be positive.")
    angle = np.deg2rad(values[4])
    major = np.cos(angle) * dx + np.sin(angle) * dy
    minor = -np.sin(angle) * dx + np.cos(angle) * dy
    if name == "ellipse":
        return (major / values[2]) ** 2 + (minor / values[3]) ** 2 <= 1.
    # DS9 boxes use full widths, whereas ellipses use semiaxes.
    return (np.abs(major) <= values[2] / 2.) & (np.abs(minor) <= values[3] / 2.)


def read_ds9_exclusion(path, shape, header=None):
    """Read a DS9 region file as an original-image exclusion mask.

    Parameters
    ----------
    path : str or pathlib.Path
        A region file whose marked shapes identify unwanted measurements.
        Supported shapes are circle, ellipse, box and polygon, in ``image``
        or ``physical`` pixel coordinates. Pixel centers inside any marked
        shape are excluded. Sky-coordinate regions and unsupported syntax
        raise an error rather than being silently misinterpreted.
    shape : (int, int)
        FITS array shape in NumPy order, ``(height, width)``.
    header : astropy.io.fits.Header or mapping, optional
        FITS header used to interpret physical coordinates through LTM/LTV.
        Without these keywords, physical and image pixel coordinates agree.

    Returns
    -------
    numpy.ndarray
        Boolean array with ``True`` for excluded input pixels. DS9's
        one-based coordinates are converted to NumPy's zero-based indexing.

    Notes
    -----
    The file describes exclusions for DINGO, so ordinary DS9 shapes (with
    ``include=1``) mark pixels to exclude. DS9 negative regions / holes are
    deliberately rejected; draw separate positive shapes for each unwanted
    region. The default coordinate system, as in DS9, is ``physical``.
    This function does not alter the source image or the fitting likelihood.
    """
    shape = tuple(shape)
    if len(shape) != 2 or any(not isinstance(n, (int, np.integer)) or n <= 0 for n in shape):
        raise ValueError("shape must contain two positive integer dimensions.")
    y, x = np.indices(shape, dtype=float)
    x += 1.
    y += 1.
    coordinates = {"image": (x, y)}
    system = "physical"
    excluded = np.zeros(shape, dtype=bool)
    count = 0
    path = Path(path).expanduser()
    for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
        body, _, properties = raw.partition("#")
        if not body.strip():
            continue
        if re.search(r"\binclude\s*=\s*0\b", body + " " + properties):
            raise ValueError(f"{path}:{line_number}: DS9 include=0 / holes are not supported.")
        for statement in body.split(";"):
            statement = statement.strip()
            if not statement or statement.lower().startswith("global "):
                continue
            if statement.lower() in ("image", "physical"):
                system = statement.lower()
                continue
            match = re.fullmatch(r"([+-]?)([A-Za-z]+)\s*\(([^()]*)\)", statement)
            if match is None:
                raise ValueError(f"{path}:{line_number}: unsupported DS9 statement {statement!r}; "
                                 "save pixel regions using image or physical coordinates.")
            sign, name, arguments = match.groups()
            if sign == "-":
                raise ValueError(f"{path}:{line_number}: negative DS9 regions / holes are not supported.")
            try:
                values = [float(value.strip()) for value in arguments.split(",")]
                if not np.isfinite(values).all():
                    raise ValueError("Region coordinates must be finite.")
                if system not in coordinates:
                    coordinates[system] = _physical_coordinates(x, y, header)
                excluded |= _shape_inside(name.lower(), values, *coordinates[system])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            count += 1
    if count == 0:
        raise ValueError(f"No supported exclusion regions were found in {path}.")
    if not excluded.any():
        raise ValueError(f"No pixel centers fall inside the regions in {path}; check the image and coordinates.")
    if excluded.all():
        raise ValueError(f"The regions in {path} exclude the entire input image.")
    return excluded


def plot_input_masks(source_dir, masks):
    """Show original image/R/C and masked previews in a red–orange heatmap.

    ``masks`` maps ``"R"`` and/or ``"C"`` to original-grid Boolean arrays,
    with True meaning exclude. An optional ``"image"`` mask can also be
    displayed. Missing entries mean no exclusion. The upper row outlines
    excluded pixels in cyan; the lower row leaves excluded pixels white.
    Each column uses exactly the same intensity scaling before and after
    masking. Axes use zero-based NumPy pixel centers, with origin at bottom
    left. The function returns a Matplotlib figure and does not display,
    save, or mutate data. It does not itself apply masks to a fit.
    """
    from astropy.io import fits
    import matplotlib.pyplot as plt
    from matplotlib.colors import AsinhNorm

    source_dir = Path(source_dir)
    unknown = set(masks) - {"image", "R", "C"}
    if unknown:
        raise ValueError(f"Unknown input mask names: {sorted(unknown)}")
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    cmap = plt.get_cmap("afmhot").copy()
    cmap.set_bad("white")
    for column, name in enumerate(("image", "R", "C")):
        array = np.asarray(fits.getdata(source_dir / f"{name}.fits"), dtype=float)
        if array.ndim != 2:
            raise ValueError(f"{name}.fits must contain a two-dimensional image.")
        mask = np.asarray(masks.get(name, np.zeros(array.shape, dtype=bool)))
        if mask.shape != array.shape or mask.dtype != np.bool_:
            raise ValueError(f"The {name} exclusion must be a Boolean mask matching {array.shape}.")
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            raise ValueError(f"{name}.fits has no finite pixels to display.")
        low, high = np.quantile(finite, [.01, .995])
        span = max(float(high - low), np.finfo(float).eps)
        norm = AsinhNorm(linear_width=span / 20., vmin=float(low), vmax=float(low) + span)
        original = axes[0, column].imshow(array, origin="lower", cmap=cmap, norm=norm,
                                          interpolation="nearest")
        if mask.any() and not mask.all():
            axes[0, column].contour(mask.astype(float), levels=[.5], colors=["cyan"], linewidths=1.5)
        preview = np.ma.array(array, mask=mask | ~np.isfinite(array))
        axes[1, column].imshow(preview, origin="lower", cmap=cmap, norm=norm,
                               interpolation="nearest")
        axes[0, column].set_title(f"{name}.fits — original")
        axes[1, column].set_title(f"{name} — {int(mask.sum())} pixels excluded")
        for row in range(2):
            axes[row, column].set_xlabel("x [pixel, zero based]")
            axes[row, column].set_ylabel("y [pixel, zero based]")
        figure.colorbar(original, ax=axes[:, column], shrink=.82, label="Input intensity", extend="both")
    figure.suptitle(f"{source_dir.name}: input exclusions\nCyan outlines / white areas mark excluded pixels")
    return figure


def _file_sha256(path):
    """Hash a file without loading the entire FITS into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, record):
    """Replace one report atomically on its own filesystem."""
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.stem}_", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def replace_region_with_background(fits_path, region_path, *, border_width=10):
    """Fill a user-selected region with the measured image-background median.

    This is an explicit input-data edit, not a likelihood exclusion. Only the
    marked primary-image pixel centers are replaced. Every other primary-image
    value, all other HDU data, and ordinary headers are preserved. Existing
    FITS checksums are updated when necessary.

    The first call preserves an immutable-by-this-function sibling backup,
    ``<stem>_original_before_region_fill.fits``. Every subsequent call derives
    its result from that backup, including when the region is edited. A saved
    ``<stem>_background_fill.json`` records original/filled hashes, the exact
    region text, fill statistics, and time. If the working FITS was independently
    changed, the function refuses to overwrite it. Repeating an unchanged call
    leaves the files byte-identical and returns ``already_applied=True``.

    ``border_width`` specifies the width of the outer image border used to
    estimate the background. Invalid values and the selected region are removed
    first, followed by 3-sigma clipping (up to ten iterations). At least twenty
    surviving background pixels are required. The clipped median is the fill
    value; ``background_sigma`` is their sample standard deviation.

    Returns a dictionary containing ``backup_path``, ``filled_sha256``,
    ``pixels_replaced``, ``background_median``, ``background_sigma``, and the
    remaining provenance. No fitting or sampling is performed.
    """
    from astropy.io import fits
    from astropy.stats import sigma_clip

    fits_path = Path(fits_path).expanduser().absolute()
    region_path = Path(region_path).expanduser().absolute()
    backup = fits_path.with_name(f"{fits_path.stem}_original_before_region_fill{fits_path.suffix}")
    report_path = fits_path.with_name(f"{fits_path.stem}_background_fill.json")
    if any(path.is_symlink() for path in (fits_path, backup, report_path)):
        raise ValueError("Background replacement requires regular FITS, backup, and report paths, not symlinks.")
    if not isinstance(border_width, (int, np.integer)) or isinstance(border_width, bool) or border_width <= 0:
        raise ValueError("border_width must be a positive integer.")
    current_sha = _file_sha256(fits_path)
    region_bytes = region_path.read_bytes()
    region_sha = hashlib.sha256(region_bytes).hexdigest()
    previous = None
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        if previous.get("schema") != "dingo-region-background-fill-v1" or not backup.exists():
            raise ValueError("The background-fill backup/report is incomplete or unrecognized; preserve it for inspection.")
    original_path = backup if backup.exists() else fits_path
    original_sha = _file_sha256(original_path)
    allowed_current = {original_sha}
    if previous is not None:
        if previous.get("original_sha256") != original_sha:
            raise ValueError("The original FITS backup changed; refusing to overwrite the working image.")
        allowed_current.add(previous.get("filled_sha256"))
        # A prepared transaction is recoverable if execution stopped before
        # installing the new FITS; its preceding file hash was also recorded.
        if previous.get("status") == "prepared":
            allowed_current.add(previous.get("previous_sha256"))
        elif previous.get("status") != "complete":
            raise ValueError("The background-fill report has an unrecognized status.")
    if current_sha not in allowed_current:
        raise ValueError("The working FITS changed independently of the recorded background fill; refusing to overwrite it.")
    if (previous is not None and previous.get("status") == "complete"
            and current_sha == previous.get("filled_sha256")
            and region_sha == previous.get("region_sha256")
            and int(border_width) == previous.get("border_width")):
        return dict(previous, already_applied=True)

    with fits.open(original_path, memmap=False) as opened:
        hdus = fits.HDUList([hdu.copy() for hdu in opened])
    try:
        data = hdus[0].data
        if data is None or data.ndim != 2 or not np.issubdtype(data.dtype, np.floating):
            raise ValueError("Background replacement requires a two-dimensional floating-point primary FITS image.")
        if 2 * border_width >= min(data.shape):
            raise ValueError("border_width must be smaller than half the shortest image dimension.")
        excluded = read_ds9_exclusion(region_path, data.shape, hdus[0].header)
        if region_path.read_bytes() != region_bytes:
            raise ValueError("The DS9 region changed while it was being read; rerun the cell.")
        border = np.zeros(data.shape, dtype=bool)
        border[:border_width, :] = border[-border_width:, :] = True
        border[:, :border_width] = border[:, -border_width:] = True
        selected = border & ~excluded & np.isfinite(data)
        clipped = sigma_clip(np.asarray(data[selected], dtype=float), sigma=3., maxiters=10).compressed()
        if clipped.size < 20:
            raise ValueError("Fewer than twenty usable background-border pixels remain after clipping.")
        median = float(np.median(clipped))
        sigma = float(np.std(clipped, ddof=1))
        filled = data.copy()
        filled[excluded] = median
        hdus[0].data = filled
        # Preserve ordinary headers; refresh only checksum information that
        # was already present, rather than adding new metadata to every HDU.
        for hdu in hdus:
            if "CHECKSUM" in hdu.header or "DATASUM" in hdu.header:
                hdu.add_checksum()

        if _file_sha256(fits_path) != current_sha:
            raise ValueError("The working FITS changed during preparation; refusing to overwrite it.")
        if not backup.exists():
            descriptor, temporary_backup = tempfile.mkstemp(prefix=f".{backup.stem}_", dir=backup.parent)
            os.close(descriptor)
            try:
                shutil.copy2(fits_path, temporary_backup)
                if _file_sha256(temporary_backup) != original_sha:
                    raise ValueError("The working FITS changed while backing it up; no replacement was made.")
                # Hard-link creation is exclusive: never overwrite a backup,
                # even if another process created it during this call.
                os.link(temporary_backup, backup)
            finally:
                Path(temporary_backup).unlink(missing_ok=True)

        descriptor, temporary_fits = tempfile.mkstemp(prefix=f".{fits_path.stem}_", suffix=".fits", dir=fits_path.parent)
        os.close(descriptor)
        try:
            hdus.writeto(temporary_fits, overwrite=True)
            shutil.copymode(fits_path, temporary_fits)
            filled_sha = _file_sha256(temporary_fits)
            report = dict(
                schema="dingo-region-background-fill-v1", status="prepared",
                fits_path=str(fits_path), backup_path=str(backup), report_path=str(report_path),
                region_path=str(region_path), region_text=region_bytes.decode("utf-8"),
                original_sha256=original_sha, filled_sha256=filled_sha,
                previous_sha256=current_sha, region_sha256=region_sha,
                pixels_replaced=int(excluded.sum()), border_width=int(border_width),
                background_pixels_before_clipping=int(selected.sum()),
                background_pixels_used=int(clipped.size), background_median=median,
                stored_fill_value=float(filled[excluded][0]), background_sigma=sigma,
                sigma_clip=3., sigma_clip_maxiters=10,
                created_utc=datetime.now(timezone.utc).isoformat(),
            )
            if _file_sha256(fits_path) != current_sha:
                raise ValueError("The working FITS changed during preparation; refusing to overwrite it.")
            _atomic_json(report_path, report)
            os.replace(temporary_fits, fits_path)
            report["status"] = "complete"
            _atomic_json(report_path, report)
        finally:
            Path(temporary_fits).unlink(missing_ok=True)
    finally:
        hdus.close()
    return dict(report, already_applied=False)


def plot_background_replacement(fits_path, report):
    """Show the original backup and actual filled FITS using one red–orange scale.

    The cyan outline marks the filled region. The right panel shows the
    replacement intensities themselves, never white masking or invented NaNs.
    Hash checks prevent accidentally displaying another run's files or a region
    edited after the replacement. The figure is returned without being shown.
    """
    from astropy.io import fits
    import matplotlib.pyplot as plt
    from matplotlib.colors import AsinhNorm

    fits_path = Path(fits_path)
    backup = Path(report["backup_path"])
    region_path = Path(report["region_path"])
    if _file_sha256(fits_path) != report["filled_sha256"] or _file_sha256(backup) != report["original_sha256"]:
        raise ValueError("The before/after FITS files do not match this replacement report.")
    if _file_sha256(region_path) != report["region_sha256"]:
        raise ValueError("The region changed after filling; rerun background replacement before plotting.")
    before, header = fits.getdata(backup, header=True)
    after = fits.getdata(fits_path)
    mask = read_ds9_exclusion(region_path, before.shape, header)
    values = np.asarray(before)[np.isfinite(before)]
    low, high = np.quantile(values, [.01, .995])
    span = max(float(high - low), np.finfo(float).eps)
    norm = AsinhNorm(linear_width=span / 20., vmin=float(low), vmax=float(low) + span)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.8), layout="constrained")
    for axis, data, label in zip(axes, (before, after), ("Before — original backup", "After — actual working FITS")):
        plotted = axis.imshow(data, origin="lower", cmap="afmhot", norm=norm, interpolation="nearest")
        axis.contour(mask.astype(float), levels=[.5], colors=["cyan"], linewidths=1.3)
        axis.set(title=label, xlabel="x [pixel, zero based]", ylabel="y [pixel, zero based]")
    figure.colorbar(plotted, ax=axes, shrink=.85, label="Input intensity", extend="both")
    figure.suptitle(f"{fits_path.name}: {report['pixels_replaced']} pixels filled with background median "
                   f"{report['stored_fill_value']:.5g}")
    return figure
