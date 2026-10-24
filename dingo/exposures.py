"""Native SAPPHIRES detector cutouts for an experimental joint forward fit.

No rotation, interpolation, stacking, continuum subtraction, or error rescaling
is applied to the data. EMLINE is retained only for inspection: ERR belongs to
SCI. Positions are zero-based (the convention in the extraction notebook).
"""

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import hashlib
import json

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import ascii, fits
from astropy.wcs import WCS
import yaml

from . import grism


@dataclass(frozen=True)
class TraceCalibration:
    pupil: str
    module: str
    trace: np.ndarray
    dispersion: np.ndarray
    provenance: dict

    def project(self, x, y, wavelength_um):
        """Direct position -> dispersed position, using the full polynomials."""
        x, y, wavelength_um = np.broadcast_arrays(x, y, wavelength_um)
        d = grism.fit_disp_order32(np.array([x, y, wavelength_um]), *self.dispersion)
        t = grism.fit_disp_order23(np.array([x, y, d]), *self.trace)
        return (x + d, y + t) if self.pupil == "R" else (x + t, y + d)


@lru_cache(maxsize=16)
def load_trace_calibration(calibration_root, module, pupil, filt="F444W"):
    """Explicit calibration location; does not change GRISM_CAL_DIR."""
    if module not in ("A", "B") or pupil not in ("R", "C") or filt != "F444W":
        raise ValueError("Native exposure support currently requires F444W, A/B, R/C")
    root = Path(calibration_root)
    paths = [root / f"dy_dx_tracing_model/DISP_{filt}_mod{module}_grism{pupil}.dat",
             root / f"dx_wave_dispersion_model/DISPL_mod{module}_grism{pupil}.dat"]
    values = [np.asarray(ascii.read(p)["col0"], dtype=float) for p in paths]
    if values[0].shape != (30,) or values[1].shape != (16,):
        raise ValueError("Unexpected trace/dispersion coefficient count")
    if not all(np.isfinite(v).all() for v in values):
        raise ValueError("Nonfinite calibration coefficients")
    return TraceCalibration(pupil, module, *values, {
        str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def native_valid_mask(science, error, dq, reject_bits=3, extra_mask=None):
    """Fixed mask: finite SCI, positive finite ERR, no rejected DQ bits.

    Default 3 rejects DO_NOT_USE | SATURATED. JUMP_DET alone is not a bad
    rate measurement. Additional flags/contaminants must be chosen explicitly.
    extra_mask=True means exclude, and must not depend on trial parameters.
    """
    if not 0 <= reject_bits <= np.iinfo(np.uint32).max:
        raise ValueError("reject_bits must fit an unsigned 32-bit DQ word")
    valid = (np.isfinite(science) & np.isfinite(error) & (error > 0)
             & ((np.asarray(dq, dtype=np.uint32) & reject_bits) == 0))
    if extra_mask is not None:
        if np.shape(extra_mask) != valid.shape:
            raise ValueError("extra_mask shape differs from the exposure")
        valid &= ~np.asarray(extra_mask, dtype=bool)
    return valid


def _read_window(hdu, origin_xy, size, fill, dtype=float):
    """Read only the required FITS section, padding off-detector pixels."""
    x, y = origin_xy
    height, width = hdu.shape
    x1, x2, y1, y2 = max(0, x), min(width, x + size), max(0, y), min(height, y + size)
    out = np.full((size, size), fill, dtype=dtype)
    if x2 > x1 and y2 > y1:
        out[y1-y:y2-y, x1-x:x2-x] = hdu.section[y1:y2, x1:x2]
    return out


@dataclass
class NativeExposure:
    path: Path
    pupil: str
    module: str
    origin_xy: tuple
    direct_xy: tuple
    line_xy: tuple
    science: np.ndarray
    error: np.ndarray
    dq: np.ndarray
    valid: np.ndarray
    emline: np.ndarray
    wcs: WCS
    calibration: TraceCalibration
    metadata: dict

    def source_detector_xy(self, east_arcsec, north_arcsec):
        """Shared sky offsets mapped into this exposure's direct-image frame.

        Anchor at its recorded STATS position. Retain the difference from the
        absolute WCS/catalog position in metadata for registration diagnostics;
        do not silently assume that the two astrometric solutions agree.
        """
        anchor = self.wcs.pixel_to_world(*self.direct_xy)
        # Float32 angles trigger float32 trigonometry inside spherical_offsets_by.
        # At subarcsecond offsets, cancellation can move the result by ~0.8
        # detector pixel. GPU tensors may be float32; WCS setup must not be.
        coords = anchor.spherical_offsets_by(np.asarray(east_arcsec, dtype=np.float64) * u.arcsec,
                                             np.asarray(north_arcsec, dtype=np.float64) * u.arcsec)
        return np.stack(self.wcs.world_to_pixel(coords), axis=-1)


@dataclass
class NativeTarget:
    source_id: int
    wavelength_um: float
    summary: dict
    exposures: list[NativeExposure]
    excluded_exposures: list[dict] = field(default_factory=list)

    def save(self, path):
        """Portable, pickle-free inspection artifact; input FITS remain intact."""
        metadata = {"source_id": self.source_id, "wavelength_um": self.wavelength_um,
                    "summary": self.summary,
                    "exposures": [e.metadata for e in self.exposures],
                    "excluded_exposures": self.excluded_exposures,
                    "noise_for": "SCI only; EMLINE uncertainty is not propagated"}
        np.savez_compressed(path, metadata_json=json.dumps(metadata),
                            **{key: np.stack([getattr(e, key) for e in self.exposures])
                               for key in ("science", "error", "dq", "valid", "emline",
                                           "origin_xy", "direct_xy", "line_xy")})


def load_native_target(bundle_root, config_path, size=81, reject_bits=3):
    """Locate exact exposures through the separate R/C STATS tables.

    The configured spectroscopic redshift selects the cutout wavelength. The
    photometric redshift in the old spectrum's PRIMARY header is not used.
    """
    if size < 3 or size % 2 != 1:
        raise ValueError("size must be odd and >= 3")
    root = Path(bundle_root).resolve()
    config_path = Path(config_path).resolve()
    summary = yaml.safe_load(config_path.read_text())["summary"]
    source_id = int(str(summary["name"]).removeprefix("ID"))
    wavelength = float(summary["lambda_rest"]) * (1 + float(summary["z"]))
    if summary["filter"] != "F444W" or not 3.8 <= wavelength <= 5.1:
        raise ValueError("Expected an F444W line within 3.8--5.1 microns")
    exposures, excluded, seen = [], [], set()
    for pupil in ("R", "C"):
        stats_path = root / (f"grism/target_spectra/GRISM{pupil}/"
                             f"spec_2d_M0416_F444W_ID{source_id}_{pupil}.fits")
        with fits.open(stats_path) as hdul:
            rows = hdul["STATS"].data.copy()
            coord = SkyCoord(hdul[0].header["RA0"], hdul[0].header["DEC0"], unit="deg")
        if len(rows) == 0:
            raise ValueError(f"No GRISM{pupil} associations for ID{source_id}")
        for row in rows:
            stem, module = str(row["name"]).strip(), str(row["module"]).strip()
            if Path(stem).name != stem or stem in seen or str(row["pupil"]).strip() != pupil:
                raise ValueError(f"Invalid/duplicate exposure association: {stem}")
            seen.add(stem)
            path = root / f"grism/GRISM{pupil}/rate_lv1.5/{stem}_rate_lv1.5.fits"
            calibration = load_trace_calibration(str(root / "calibration/grism_cal"), module, pupil)
            direct_xy = (float(row["x0"]), float(row["y0"]))
            line_xy = tuple(float(v) for v in calibration.project(*direct_xy, wavelength))
            origin = tuple(int(np.floor(v + 0.5)) - size // 2 for v in line_xy)
            with fits.open(path, memmap=False) as hdul:
                primary = hdul[0].header
                if (primary["PUPIL"], primary["MODULE"], primary["FILTER"]) != (f"GRISM{pupil}", module, "F444W"):
                    raise ValueError(f"STATS and exposure header disagree: {path}")
                for name in ("SCI", "ERR"):
                    if hdul[name].header.get("BUNIT", "").strip() != "DN/s":
                        raise ValueError(f"Unexpected {name} unit in {path}")
                science = _read_window(hdul["SCI"], origin, size, np.nan)
                error = _read_window(hdul["ERR"], origin, size, np.nan)
                dq = _read_window(hdul["DQ"], origin, size, 1, np.uint32)
                emline = _read_window(hdul["EMLINE"], origin, size, np.nan)
                wcs = WCS(hdul["SCI"].header, fix=False).celestial
                catalog_xy = np.array(wcs.world_to_pixel(coord), dtype=float)
                valid = native_valid_mask(science, error, dq, reject_bits)
                metadata = {
                    "path": str(path), "stats_path": str(stats_path),
                    "config_path": str(config_path), "pupil": pupil, "module": module,
                    "origin_xy": origin, "direct_xy": direct_xy, "line_xy": line_xy,
                    "wcs_minus_stats_xy": (catalog_xy - direct_xy).tolist(),
                    "dq_reject_bits": reject_bits, "valid_pixels": int(valid.sum()),
                    "total_pixels": int(valid.size), "unit": "DN/s",
                    "effective_exposure_s": float(row["EFFEXPTM"]),
                    "calibration_sha256": calibration.provenance,
                    "sci_wcs_header": wcs.to_header(relax=True).tostring(),
                    "warnings": ["EMLINE has no propagated uncertainty",
                                 "Contamination has not yet been masked/modelled",
                                 "SCI processing/background noise has not been validated"],
                }
            if not valid.any():
                excluded.append({**metadata, "reason": "No valid SCI/ERR pixels in the systemic-line cutout"})
                continue
            exposures.append(NativeExposure(path, pupil, module, origin, direct_xy, line_xy,
                                            science, error, dq, valid, emline, wcs,
                                            calibration, metadata))
    if any(not any(e.pupil == p for e in exposures) for p in ("R", "C")):
        raise ValueError(f"ID{source_id}: at least one grism has no usable systemic-line cutouts; excluded={excluded}")
    return NativeTarget(source_id, wavelength, summary, exposures, excluded)
