'''Load DPC datasets from HDF5/NeXus files using a declarative field mapping.

The mapping is a plain dict (or a YAML file containing one) that translates
HDF5 dataset paths to fields of DPCDataset.  Any key that does not match a
named DPCDataset field is stored in ``metadata``.

Example YAML mapping (e.g. config/i14_mapping.yaml)
----------------------------------------------------
frames:            /entry/instrument/merlin/data
beam_energy:       /entry/instrument/monochromator/energy
detector_distance: /entry/instrument/merlin/distance
pixel_size:        /entry/instrument/merlin/pixel_size
scan_step_x:       /entry/scan/sample_x/step_size
scan_step_y:       /entry/scan/sample_y/step_size
'''

from __future__ import annotations

import h5py
import numpy as np
import yaml
from pathlib import Path

from io_schema import DPCDataset

# Fields that map directly onto DPCDataset constructor parameters.
_SCHEMA_FIELDS = {
    f for f in DPCDataset.__dataclass_fields__
    if f not in ("frames", "metadata")
}


def load_mapping(mapping_path: str | Path) -> dict:
    '''Load a YAML field-mapping file.

    Parameters
    ----------
    mapping_path : path-like
        Path to a YAML file whose top-level keys are DPCDataset field names
        and whose values are HDF5 dataset paths.

    Returns
    -------
    mapping : dict
    '''
    with open(mapping_path) as f:
        return yaml.safe_load(f)


def load_dpc(
    hdf5_path: str | Path,
    mapping: dict | str | Path,
    lazy: bool = False,
) -> DPCDataset:
    '''Load a DPC dataset from an HDF5/NeXus file.

    Parameters
    ----------
    hdf5_path : path-like
        Path to the HDF5 or NeXus file.
    mapping : dict or path-like
        Either a dict of ``{schema_field: hdf5_path}`` pairs, or a path to a
        YAML file containing one.  Keys matching DPCDataset fields are mapped
        directly; any remaining keys are stored in ``metadata``.
    lazy : bool, optional
        If True the ``frames`` array is returned as a live ``h5py.Dataset``
        rather than being loaded into memory.  The caller is then responsible
        for keeping the file open.  Default False.

    Returns
    -------
    DPCDataset
    '''
    if not isinstance(mapping, dict):
        mapping = load_mapping(mapping)

    scalar_kwargs: dict = {}
    extra: dict = {}
    frames = None

    f = h5py.File(hdf5_path, "r")
    try:
        for field, hpath in mapping.items():
            node = f[hpath]
            if field == "frames":
                frames = node if lazy else node[()]
            elif field in _SCHEMA_FIELDS:
                scalar_kwargs[field] = float(np.asarray(node))
            else:
                extra[field] = float(np.asarray(node))
    finally:
        if not lazy:
            f.close()

    if frames is None:
        raise KeyError("mapping must contain a 'frames' entry")

    return DPCDataset(frames=frames, **scalar_kwargs, metadata=extra)
