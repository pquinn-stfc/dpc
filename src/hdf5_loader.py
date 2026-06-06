'''Generic HDF5/NeXus loader driven by a declarative field mapping.

The loader is completely decoupled from any specific data schema.  It
resolves a mapping into a plain ``dict`` of field names → values; a
separate :func:`bind` helper populates any dataclass from that dict.

Quick start
-----------
::

    from hdf5_loader import HDF5Loader, bind
    from io_schema import DPCDataset

    loader = HDF5Loader("config/i14_mapping.yaml")
    data   = loader.load("scan.nxs")          # plain dict
    ds     = bind(data, DPCDataset)           # typed dataclass

Or use the DPC-specific convenience function::

    from hdf5_loader import load_dpc
    ds = load_dpc("scan.nxs", "config/i14_mapping.yaml")

Field spec syntax
-----------------
Each entry in the mapping can be:

* A bare HDF5 path string::

      beam_energy: /entry/instrument/monochromator/energy

* A dict with any combination of ``path``, ``value``, ``default``,
  ``scale``, and ``transform``::

      # unit conversion via a named transform
      detector_distance:
        path: /entry/instrument/merlin/distance
        transform: mm_to_m

      # hard-coded override — HDF5 file is not consulted
      pixel_size:
        value: 55.0e-6

      # path with a fallback if the key is absent in the file
      scan_step_x:
        path: /entry/scan/sample_x/step_size
        default: 50.0e-9

      # infer step size from an array of equally-spaced positions
      scan_step_x:
        path: /entry/scan/sample_x/value
        transform: step_from_positions_um

Built-in named transforms
-------------------------
``mm_to_m``, ``um_to_m``, ``nm_to_m``, ``pm_to_m``
    Length unit conversions to metres.
``ev_to_kev``, ``kev_to_ev``
    Energy unit conversions.
``deg_to_rad``, ``rad_to_deg``
    Angle conversions.
``step_from_positions``
    Reduce a 1D position array to its mean step size (native units).
``step_from_positions_mm``, ``step_from_positions_um``, ``step_from_positions_nm``
    As above, with built-in unit conversion to metres.

Custom transforms can be registered via :func:`register_transform`.
'''

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import h5py
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Named transform registry
# ---------------------------------------------------------------------------

_TRANSFORMS: dict[str, Callable] = {
    # length → metres
    "mm_to_m":  lambda x: np.asarray(x) * 1e-3,
    "um_to_m":  lambda x: np.asarray(x) * 1e-6,
    "nm_to_m":  lambda x: np.asarray(x) * 1e-9,
    "pm_to_m":  lambda x: np.asarray(x) * 1e-12,
    # energy
    "ev_to_kev":  lambda x: np.asarray(x) * 1e-3,
    "kev_to_ev":  lambda x: np.asarray(x) * 1e3,
    # angle
    "deg_to_rad": lambda x: np.deg2rad(x),
    "rad_to_deg": lambda x: np.rad2deg(x),
    # array → scalar: infer step size from equally-spaced positions
    "step_from_positions":     lambda x: float(np.diff(np.asarray(x)).mean()),
    "step_from_positions_mm":  lambda x: float(np.diff(np.asarray(x)).mean()) * 1e-3,
    "step_from_positions_um":  lambda x: float(np.diff(np.asarray(x)).mean()) * 1e-6,
    "step_from_positions_nm":  lambda x: float(np.diff(np.asarray(x)).mean()) * 1e-9,
}


def register_transform(name: str, fn: Callable) -> None:
    '''Register a named transform for use in YAML mapping files.

    The function receives the raw value as a numpy array and must return
    either a numpy array or a scalar.  Use a reducing transform (one that
    collapses an array to a scalar) for fields that are stored as 1-D
    position lists in the HDF5 file.

    Parameters
    ----------
    name : str
        Key to use in the YAML ``transform`` field.
    fn : callable
        ``fn(np.ndarray) -> np.ndarray | float``

    Example
    -------
    >>> register_transform("mrad_to_rad", lambda x: np.asarray(x) * 1e-3)
    '''
    _TRANSFORMS[name] = fn


# ---------------------------------------------------------------------------
# FieldSpec
# ---------------------------------------------------------------------------

@dataclass
class FieldSpec:
    '''Specification for how to obtain a single value from an HDF5 file.

    Parameters
    ----------
    path : str, optional
        HDF5 dataset path.  Omit when using ``value``.
    value : scalar, optional
        Hard-coded override; the HDF5 file is not consulted for this field.
    default : scalar, optional
        Fallback used when ``path`` is given but absent from the file.
    scale : float, optional
        Multiply the value by this factor after reading and transforming.
        Default 1.0.
    transform : str, optional
        Name of a registered transform applied to the raw numpy array before
        scaling.  May reduce an array to a scalar (e.g.
        ``step_from_positions``).
    keep_array : bool, optional
        If True, skip the scalar-enforcement step and return the value as a
        numpy array.  Use for fields such as ``frames`` that are inherently
        multidimensional.  Default False.
    '''
    path: Optional[str] = None
    value: Optional[Any] = None
    default: Optional[Any] = None
    scale: float = 1.0
    transform: Optional[str] = None
    keep_array: bool = False

    def resolve(self, f: h5py.File) -> Optional[Any]:
        '''Resolve the field value from an open HDF5 file.

        Resolution pipeline:

        1. Read raw value as a numpy array (preserving shape).
        2. Apply named ``transform`` (may reduce array → scalar).
        3. Multiply by ``scale``.
        4. Cast to float scalar (unless ``keep_array=True``).

        Returns
        -------
        float, ndarray, or None
            None when the field cannot be resolved (absent path, no default).

        Raises
        ------
        KeyError
            If ``transform`` names an unregistered function.
        ValueError
            If the resolved value is still non-scalar and ``keep_array`` is
            False, guiding the caller to add a reducing transform.
        '''
        # 1. Obtain raw numpy array
        if self.value is not None:
            raw = np.asarray(self.value)
        elif self.path is not None:
            if self.path in f:
                raw = np.asarray(f[self.path])
            elif self.default is not None:
                raw = np.asarray(self.default)
            else:
                return None
        elif self.default is not None:
            raw = np.asarray(self.default)
        else:
            return None

        # 2. Named transform
        if self.transform is not None:
            if self.transform not in _TRANSFORMS:
                raise KeyError(
                    f"Unknown transform {self.transform!r}. "
                    f"Available: {sorted(_TRANSFORMS)}"
                )
            raw = np.asarray(_TRANSFORMS[self.transform](raw))

        # 3. Scale (only for numeric types)
        if np.issubdtype(raw.dtype, np.number):
            result = raw * self.scale
        else:
            result = raw

        # 4. Return
        if self.keep_array:
            return result

        if result.ndim != 0:
            raise ValueError(
                f"Field at path {self.path!r} resolved to a non-scalar array "
                f"of shape {result.shape}. Add a reducing transform such as "
                f"'step_from_positions', or set keep_array=true in the mapping."
            )
        # return native Python scalar
        return result.item()


def _parse_field_spec(raw) -> FieldSpec:
    '''Convert a raw YAML value into a FieldSpec.

    Accepts:
    - a bare string  → ``FieldSpec(path=raw)``
    - an int/float   → ``FieldSpec(value=raw)``
    - a dict         → ``FieldSpec(**raw)``
    '''
    if isinstance(raw, str):
        return FieldSpec(path=raw)
    if isinstance(raw, (int, float)):
        return FieldSpec(value=float(raw))
    if isinstance(raw, dict):
        return FieldSpec(**raw)
    raise TypeError(f"Cannot parse field spec from {raw!r}")


# ---------------------------------------------------------------------------
# HDF5Loader — generic, schema-agnostic
# ---------------------------------------------------------------------------

class HDF5Loader:
    '''Generic HDF5/NeXus loader driven by a declarative field mapping.

    The loader knows nothing about any particular data schema.  It resolves
    each field spec and returns a plain ``dict``.  Use :func:`bind` to
    populate a typed dataclass from that dict.

    Parameters
    ----------
    mapping : dict or path-like
        Field mapping as a dict or path to a YAML file.
    overrides : dict, optional
        Additional field specs merged on top of ``mapping``.  Useful for
        runtime overrides without editing the YAML file.

    Examples
    --------
    Load into a plain dict::

        loader = HDF5Loader("config/i14_mapping.yaml")
        data = loader.load("scan.nxs")

    Load into a typed dataclass::

        from io_schema import DPCDataset
        ds = bind(loader.load("scan.nxs"), DPCDataset)

    Runtime override::

        data = loader.load("scan.nxs",
                           overrides={"pixel_size": {"value": 110e-6}})
    '''

    def __init__(
        self,
        mapping: dict | str | Path,
        overrides: Optional[dict] = None,
    ):
        raw = mapping if isinstance(mapping, dict) else _load_yaml(mapping)
        if overrides:
            raw = {**raw, **overrides}
        self._specs: dict[str, FieldSpec] = {
            k: _parse_field_spec(v) for k, v in raw.items()
        }

    def load(
        self,
        hdf5_path: str | Path,
        lazy: bool = False,
        overrides: Optional[dict] = None,
    ) -> dict:
        '''Load fields from an HDF5 file into a plain dict.

        Parameters
        ----------
        hdf5_path : path-like
            Path to the HDF5 or NeXus file.
        lazy : bool, optional
            When True, array fields with ``keep_array=True`` are returned as
            live ``h5py.Dataset`` objects rather than numpy arrays.  The
            caller is responsible for keeping the file open.  Default False.
        overrides : dict, optional
            Per-call field overrides (same format as the mapping).

        Returns
        -------
        dict
            ``{field_name: resolved_value}`` for every field that could be
            resolved.  Fields that cannot be resolved (no path, no default,
            no value) are silently omitted.
        '''
        specs = self._specs
        if overrides:
            specs = {**specs, **{k: _parse_field_spec(v)
                                 for k, v in overrides.items()}}

        result: dict = {}
        f = h5py.File(hdf5_path, "r")
        try:
            for field_name, spec in specs.items():
                if lazy and spec.keep_array and spec.path and spec.path in f:
                    result[field_name] = f[spec.path]
                else:
                    resolved = spec.resolve(f)
                    if resolved is not None:
                        result[field_name] = resolved
        finally:
            if not lazy:
                f.close()

        return result


# ---------------------------------------------------------------------------
# bind — attach a plain dict to any dataclass
# ---------------------------------------------------------------------------

def bind(data: dict, schema) -> Any:
    '''Populate a dataclass from a plain dict.

    Fields present in ``data`` that match a constructor parameter of
    ``schema`` are passed directly.  Any remaining fields are collected into
    ``schema.metadata`` if that field exists, otherwise they are silently
    dropped.

    Parameters
    ----------
    data : dict
        Field dict as returned by :meth:`HDF5Loader.load`.
    schema : dataclass type
        Target dataclass.

    Returns
    -------
    An instance of ``schema``.

    Example
    -------
    ::

        from io_schema import DPCDataset
        ds = bind(loader.load("scan.nxs"), DPCDataset)
    '''
    schema_fields = {f.name for f in dataclasses.fields(schema)}
    kwargs  = {k: v for k, v in data.items() if k in schema_fields
                                              and k != "metadata"}
    extra   = {k: v for k, v in data.items() if k not in schema_fields}

    if "metadata" in schema_fields:
        kwargs["metadata"] = extra

    return schema(**kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)




# ---------------------------------------------------------------------------
# DPC-specific convenience wrapper
# ---------------------------------------------------------------------------

def load_dpc(
    hdf5_path: str | Path,
    mapping: dict | str | Path,
    lazy: bool = False,
    overrides: Optional[dict] = None,
):
    '''Load a DPC dataset from an HDF5/NeXus file.

    Convenience wrapper around :class:`HDF5Loader` and :func:`bind` that
    returns a :class:`~io_schema.DPCDataset`.

    Parameters
    ----------
    hdf5_path : path-like
        Path to the HDF5 or NeXus file.
    mapping : dict or path-like
        Field mapping (dict or path to YAML).
    lazy : bool, optional
        Return ``frames`` as a live ``h5py.Dataset``.  Default False.
    overrides : dict, optional
        Per-call field overrides.

    Returns
    -------
    DPCDataset
    '''
    from io_schema import DPCDataset

    loader = HDF5Loader(mapping, overrides=overrides)
    data   = loader.load(hdf5_path, lazy=lazy)
    return bind(data, DPCDataset)
