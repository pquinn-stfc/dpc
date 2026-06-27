'''HDF5 loader — generic parts re-exported from nexus_io; DPC wrapper here.

Generic classes (:class:`HDF5Loader`, :func:`bind`, :class:`FieldSpec`,
:func:`register_transform`) live in :mod:`nexus_io.hdf5_loader`.
Import directly from there in new code::

    from nexus_io import HDF5Loader, bind
'''

from __future__ import annotations

from pathlib import Path
from typing import Optional

from nexus_io import HDF5Loader, bind, FieldSpec, register_transform  # noqa: F401


# ---------------------------------------------------------------------------
# DPC-specific convenience wrapper
# ---------------------------------------------------------------------------

def load_dpc(
    hdf5_path: str | Path,
    mapping,
    lazy: bool = False,
    overrides: Optional[dict] = None,
):
    '''Load a DPC dataset from an HDF5/NeXus file.

    Convenience wrapper around :class:`~nexus_io.HDF5Loader` and
    :func:`~nexus_io.bind` that returns a :class:`~io_schema.DPCDataset`.

    Parameters
    ----------
    hdf5_path : path-like
    mapping : dict or path-like
        Field mapping (dict or path to YAML).
    lazy : bool
        Return ``frames`` as a live ``h5py.Dataset``.
    overrides : dict, optional
    '''
    from io_schema import DPCDataset

    loader = HDF5Loader(mapping, overrides=overrides)
    data   = loader.load(hdf5_path, lazy=lazy)
    return bind(data, DPCDataset)
