'''Save DPC results to HDF5 (NeXus) and/or image files (PNG / TIFF).'''

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# HDF5 / NeXus
# ---------------------------------------------------------------------------

def save_hdf5(
    path: str | Path,
    *,
    dx: np.ndarray,
    dy: np.ndarray,
    grad_norm: np.ndarray,
    scan_step_x: float,
    scan_step_y: float,
    beam_energy: float,
    detector_distance: float,
    pixel_size: float,
    method: str,
    crop_size: int,
    mask_mode: str,
    phase: Optional[np.ndarray] = None,
    pixel_mask: Optional[np.ndarray] = None,
    absorption: Optional[np.ndarray] = None,
    scatter: Optional[np.ndarray] = None,
):
    '''Write DPC results to a NeXus-flavoured HDF5 file.

    Parameters
    ----------
    path : path-like
        Output file path.  Will be overwritten if it exists.
    dx, dy : ndarray, shape (scan_y, scan_x)
        Phase gradients in x and y, in rad/m.
    grad_norm : ndarray
        Euclidean norm of the phase gradient.
    scan_step_x, scan_step_y : float
        Scan step sizes in metres.
    beam_energy : float
        Incident beam energy in keV.
    detector_distance : float
        Sample-to-detector distance in mm.
    pixel_size : float
        Detector pixel size in metres.
    method : str
        Phase retrieval method name.
    crop_size : int
        Detector crop size used.
    mask_mode : str
        Pixel masking mode (``'quick'`` or ``'full'``).
    phase : ndarray, optional
        Retrieved phase in rad.
    pixel_mask : ndarray of bool, optional
        Bad-pixel mask (True = bad).
    absorption : ndarray, optional
        Absorption signal map.
    scatter : ndarray, optional
        Scatter signal map.
    '''
    nr, nc = dx.shape
    x_axis = np.arange(nc) * scan_step_x
    y_axis = np.arange(nr) * scan_step_y

    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        # NeXus convention: definition, program_name, start_time are datasets,
        # not attributes, so NeXus validators and browsers find them as fields.
        entry.create_dataset("definition",   data="NXdpc")
        entry.create_dataset("program_name", data="dpc")
        entry.create_dataset("start_time",
                             data=datetime.now(timezone.utc).isoformat())

        # /instrument  [NXinstrument]
        instr = entry.create_group("instrument")
        instr.attrs["NX_class"] = "NXinstrument"
        _scalar(instr, "beam_energy",       beam_energy,       units="keV")
        _scalar(instr, "detector_distance", detector_distance, units="mm")
        _scalar(instr, "pixel_size",        pixel_size,        units="m")

        # /scan  [NXcollection]
        scan = entry.create_group("scan")
        scan.attrs["NX_class"] = "NXcollection"
        _scalar(scan, "scan_step_x", scan_step_x, units="m")
        _scalar(scan, "scan_step_y", scan_step_y, units="m")

        # /processing  [NXnote]
        proc = entry.create_group("processing")
        proc.attrs["NX_class"] = "NXnote"
        proc.create_dataset("method",    data=method)
        proc.create_dataset("crop_size", data=crop_size)
        proc.create_dataset("mask_mode", data=mask_mode)

        # /results  [NXdata]
        # @signal and @axes follow the NeXus 3 convention so silx / h5web /
        # NeXpy resolve the axes automatically without any configuration.
        signal_name = "phase" if phase is not None else "grad_norm"
        results = entry.create_group("results")
        results.attrs["NX_class"] = "NXdata"
        results.attrs["signal"]   = signal_name
        results.attrs["axes"]     = ["y", "x"]

        # 1-D axis datasets — marked as dimension scales so HDF5-aware viewers
        # attach them to the correct dimension of every 2-D dataset.
        ds_x = results.create_dataset("x", data=x_axis.astype(np.float32))
        ds_x.attrs["units"]     = "m"
        ds_x.attrs["long_name"] = "scan position x"
        ds_x.make_scale("x")

        ds_y = results.create_dataset("y", data=y_axis.astype(np.float32))
        ds_y.attrs["units"]     = "m"
        ds_y.attrs["long_name"] = "scan position y"
        ds_y.make_scale("y")

        # 2-D result datasets with dimension scales attached
        _image(results, "dx",        dx,        units="rad/m",
               long_name="Phase gradient dx",        dim_scales=(ds_y, ds_x))
        _image(results, "dy",        dy,        units="rad/m",
               long_name="Phase gradient dy",        dim_scales=(ds_y, ds_x))
        _image(results, "grad_norm", grad_norm,
               long_name="Phase gradient norm",      dim_scales=(ds_y, ds_x))

        if phase is not None:
            _image(results, "phase", phase, units="rad",
                   long_name="Retrieved phase",      dim_scales=(ds_y, ds_x))

        # /auxiliary  [NXcollection]
        aux = entry.create_group("auxiliary")
        aux.attrs["NX_class"] = "NXcollection"

        if pixel_mask is not None:
            ds = aux.create_dataset("pixel_mask", data=pixel_mask,
                                    compression="gzip")
            ds.attrs["long_name"] = "Bad-pixel mask (True = bad)"

        if absorption is not None:
            _image(aux, "absorption", absorption,
                   long_name="Absorption signal log(I0/I)")

        if scatter is not None:
            _image(aux, "scatter", scatter, long_name="Scatter signal")


def _scalar(group, name, value, units=None):
    ds = group.create_dataset(name, data=np.float32(value))
    if units:
        ds.attrs["units"] = units


def _image(group, name, data, units=None, long_name=None, dim_scales=None):
    ds = group.create_dataset(name, data=data.astype(np.float32),
                              compression="gzip")
    if units:
        ds.attrs["units"] = units
    if long_name:
        ds.attrs["long_name"] = long_name
    if dim_scales is not None:
        for i, scale in enumerate(dim_scales):
            ds.dims[i].attach_scale(scale)


# ---------------------------------------------------------------------------
# Image files (PNG / TIFF)
# ---------------------------------------------------------------------------

_SAVE_FIELDS = {
    "phase":     "Retrieved phase",
    "grad_norm": "Phase gradient norm",
    "dx":        "Phase gradient dx",
    "dy":        "Phase gradient dy",
    "absorption":"Absorption",
    "scatter":   "Scatter",
}


def save_images(
    base_path: str | Path,
    fmt: str,
    *,
    dx: np.ndarray,
    dy: np.ndarray,
    grad_norm: np.ndarray,
    phase: Optional[np.ndarray] = None,
    absorption: Optional[np.ndarray] = None,
    scatter: Optional[np.ndarray] = None,
):
    '''Save DPC result arrays as image files.

    Each 2D array is normalised to [0, 255] and saved as an 8-bit image.

    Parameters
    ----------
    base_path : path-like
        Base path/stem for output files.  The field name is appended before
        the extension, e.g. ``results/scan_phase.png``.
    fmt : str
        Image format: ``'png'`` or ``'tif'`` / ``'tiff'``.
    dx, dy, grad_norm : ndarray
        Always saved.
    phase, absorption, scatter : ndarray, optional
        Saved when provided.
    '''
    from PIL import Image

    base_path = Path(base_path)
    ext = fmt.lstrip(".")
    base_path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {
        "dx": dx,
        "dy": dy,
        "grad_norm": grad_norm,
    }
    if phase is not None:
        arrays["phase"] = phase
    if absorption is not None:
        arrays["absorption"] = absorption
    if scatter is not None:
        arrays["scatter"] = scatter

    saved = []
    for field, arr in arrays.items():
        out = base_path.parent / f"{base_path.stem}_{field}.{ext}"
        norm = _normalise_u8(arr)
        Image.fromarray(norm).save(out)
        saved.append(out)

    return saved


def _normalise_u8(arr: np.ndarray) -> np.ndarray:
    '''Linearly scale array to uint8 [0, 255].'''
    a = arr.astype(float)
    lo, hi = a.min(), a.max()
    if hi == lo:
        return np.zeros_like(a, dtype=np.uint8)
    return ((a - lo) / (hi - lo) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# ScientificDataset → HDF5
# ---------------------------------------------------------------------------

def save_scientific(
    path: str | Path,
    ds,
    compression: str = "gzip",
):
    '''Save a :class:`~io_schema.ScientificDataset` to a NeXus HDF5 file.

    Axes are written as 1-D datasets and attached as HDF5 Dimension Scales
    to the main data array so any NeXus-aware viewer (silx, h5web, NeXpy)
    assigns correct physical axis labels automatically.

    Parameters
    ----------
    path : path-like
        Output file path.
    ds : ScientificDataset
        Dataset to save.
    compression : str
        HDF5 compression filter.  Default ``'gzip'``.
    '''
    from datetime import datetime, timezone

    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.create_dataset("definition",   data="NXscientific")
        entry.create_dataset("program_name", data="dpc")
        entry.create_dataset("start_time",
                             data=datetime.now(timezone.utc).isoformat())
        entry.create_dataset("signal_type",  data=ds.signal_type)

        # metadata scalars
        if ds.metadata:
            meta = entry.create_group("metadata")
            meta.attrs["NX_class"] = "NXcollection"
            for k, v in ds.metadata.items():
                try:
                    meta.create_dataset(k, data=v)
                except TypeError:
                    meta.create_dataset(k, data=str(v))

        # primary NXdata group
        nxdata = entry.create_group("data")
        nxdata.attrs["NX_class"] = "NXdata"
        nxdata.attrs["signal"]   = "data"
        nxdata.attrs["axes"]     = [ax.name for ax in ds.axes]

        # write each axis as a dimension scale
        ax_datasets = []
        for ax in ds.axes:
            ax_ds = nxdata.create_dataset(ax.name, data=ax.axis.astype(np.float32))
            ax_ds.attrs["units"]     = ax.units
            ax_ds.attrs["long_name"] = ax.name
            ax_ds.attrs["navigate"]  = ax.navigate
            ax_ds.make_scale(ax.name)
            ax_datasets.append(ax_ds)

        # write main data and attach dimension scales
        data_ds = nxdata.create_dataset(
            "data", data=ds.data, compression=compression
        )
        for i, ax_ds in enumerate(ax_datasets):
            data_ds.dims[i].attach_scale(ax_ds)
