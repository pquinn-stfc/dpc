'''Load DPC datasets from HDF5/NeXus files using a declarative field mapping.

Each field in the mapping can be specified as:

1. A bare HDF5 path string — read directly::

       beam_energy: /entry/instrument/monochromator/energy

2. A dict with any combination of ``path``, ``value``, ``default``,
   ``scale``, and ``transform``::

       # unit conversion via a named transform
       detector_distance:
         path: /entry/instrument/merlin/distance
         transform: mm_to_m

       # raw value in microns, apply a scale factor
       pixel_size:
         path: /entry/instrument/merlin/pixel_size
         scale: 1.0e-6

       # hard-coded override — HDF5 file is not consulted
       pixel_size:
         value: 55.0e-6

       # path with a fallback if the key is absent in the file
       scan_step_x:
         path: /entry/scan/sample_x/step_size
         default: 50.0e-9

       # path + default in the raw unit + transform
       scan_step_x:
         path: /entry/scan/sample_x/step_size
         default: 50.0
         transform: nm_to_m

Built-in named transforms
-------------------------
``mm_to_m``, ``um_to_m``, ``nm_to_m``, ``pm_to_m``
    Length unit conversions to metres.
``ev_to_kev``, ``kev_to_ev``
    Energy unit conversions.
``deg_to_rad``, ``rad_to_deg``
    Angle conversions.

Custom transforms can be registered at runtime via ``register_transform``.
'''

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import h5py
import numpy as np
import yaml
from pathlib import Path

from io_schema import DPCDataset

# ---------------------------------------------------------------------------
# Named transform registry
# ---------------------------------------------------------------------------

_TRANSFORMS: dict[str, Callable[[float], float]] = {
    # length → metres
    "mm_to_m":  lambda x: x * 1e-3,
    "um_to_m":  lambda x: x * 1e-6,
    "nm_to_m":  lambda x: x * 1e-9,
    "pm_to_m":  lambda x: x * 1e-12,
    # energy
    "ev_to_kev": lambda x: x * 1e-3,
    "kev_to_ev": lambda x: x * 1e3,
    # angle
    "deg_to_rad": lambda x: np.deg2rad(x),
    "rad_to_deg": lambda x: np.rad2deg(x),
}


def register_transform(name: str, fn: Callable[[float], float]) -> None:
    '''Register a named transform for use in YAML mapping files.

    Parameters
    ----------
    name : str
        Name to use in the YAML ``transform`` key.
    fn : callable
        Function that takes a single float and returns a float.

    Example
    -------
    >>> register_transform("mrad_to_rad", lambda x: x * 1e-3)
    '''
    _TRANSFORMS[name] = fn


# ---------------------------------------------------------------------------
# FieldSpec — one entry in the mapping
# ---------------------------------------------------------------------------

@dataclass
class FieldSpec:
    '''Specification for how to obtain a single dataset field.

    Parameters
    ----------
    path : str, optional
        HDF5 dataset path inside the file.  If absent, ``value`` must be set.
    value : scalar, optional
        Hard-coded override.  When set, the HDF5 file is not consulted for
        this field and ``path`` / ``default`` are ignored.
    default : scalar, optional
        Fallback value used when ``path`` is given but the key is absent from
        the file.  If neither the path nor a default is available the field is
        omitted (and lands in ``DPCDataset.metadata`` if it is not a required
        schema field).
    scale : float, optional
        Multiply the raw value by this factor after reading.  Applied before
        ``transform``.  Default 1.0.
    transform : str, optional
        Name of a registered transform function to apply after ``scale``.
    '''
    path: Optional[str] = None
    value: Optional[Any] = None
    default: Optional[Any] = None
    scale: float = 1.0
    transform: Optional[str] = None

    def resolve(self, f: h5py.File) -> Optional[float]:
        '''Read and process the field value from an open HDF5 file.

        Parameters
        ----------
        f : h5py.File
            Open file handle.

        Returns
        -------
        float or None
            Resolved scalar value, or None if the field cannot be resolved.

        Raises
        ------
        KeyError
            If ``transform`` names an unregistered function.
        '''
        # Hard-coded override — never touch the file
        if self.value is not None:
            raw = float(self.value)
        elif self.path is not None:
            if self.path in f:
                raw = float(np.asarray(f[self.path]))
            elif self.default is not None:
                raw = float(self.default)
            else:
                return None
        elif self.default is not None:
            raw = float(self.default)
        else:
            return None

        # Apply scale then named transform
        result = raw * self.scale
        if self.transform is not None:
            if self.transform not in _TRANSFORMS:
                raise KeyError(
                    f"Unknown transform {self.transform!r}. "
                    f"Available: {sorted(_TRANSFORMS)}"
                )
            result = _TRANSFORMS[self.transform](result)

        return result


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
# Schema field set
# ---------------------------------------------------------------------------

_SCHEMA_FIELDS = {
    f for f in DPCDataset.__dataclass_fields__
    if f not in ("frames", "metadata")
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_mapping(mapping_path: str | Path) -> dict:
    '''Load a YAML field-mapping file.

    Parameters
    ----------
    mapping_path : path-like
        Path to a YAML mapping file.

    Returns
    -------
    dict
        Raw mapping dict (values not yet parsed into ``FieldSpec`` objects).
    '''
    with open(mapping_path) as f:
        return yaml.safe_load(f)


def load_dpc(
    hdf5_path: str | Path,
    mapping: dict | str | Path,
    lazy: bool = False,
    overrides: Optional[dict] = None,
) -> DPCDataset:
    '''Load a DPC dataset from an HDF5/NeXus file.

    Parameters
    ----------
    hdf5_path : path-like
        Path to the HDF5 or NeXus file.
    mapping : dict or path-like
        Field mapping as a dict or path to a YAML file.  Each value may be:

        * a bare HDF5 path string
        * a dict with keys ``path``, ``value``, ``default``, ``scale``,
          ``transform`` (all optional, see module docstring for details)

    lazy : bool, optional
        If True, ``frames`` is returned as a live ``h5py.Dataset`` and the
        file is left open.  The caller is responsible for closing it.
        Default False.
    overrides : dict, optional
        Additional field specs (same format as ``mapping``) merged on top of
        the mapping *after* loading.  Useful for setting or overriding values
        at call time without editing the YAML file, e.g.::

            load_dpc(path, mapping, overrides={"pixel_size": {"value": 55e-6}})

    Returns
    -------
    DPCDataset

    Raises
    ------
    KeyError
        If no ``frames`` entry is found or resolved.
    '''
    if not isinstance(mapping, dict):
        mapping = load_mapping(mapping)

    # Merge overrides on top of mapping
    if overrides:
        mapping = {**mapping, **overrides}

    # Parse every entry into a FieldSpec
    specs: dict[str, FieldSpec] = {}
    frames_spec: Optional[FieldSpec] = None
    for field_name, raw in mapping.items():
        spec = _parse_field_spec(raw)
        if field_name == "frames":
            frames_spec = spec
        else:
            specs[field_name] = spec

    if frames_spec is None:
        raise KeyError("mapping must contain a 'frames' entry")

    scalar_kwargs: dict = {}
    extra: dict = {}
    frames = None

    f = h5py.File(hdf5_path, "r")
    try:
        # Frames — always read as array or lazy dataset
        if frames_spec.value is not None:
            raise ValueError("'frames' cannot use a hard-coded value override")
        if frames_spec.path is None or frames_spec.path not in f:
            raise KeyError(
                f"'frames' path {frames_spec.path!r} not found in {hdf5_path}"
            )
        frames = f[frames_spec.path] if lazy else f[frames_spec.path][()]

        # Scalar fields
        for field_name, spec in specs.items():
            resolved = spec.resolve(f)
            if resolved is None:
                # field absent and no default — skip silently
                continue
            if field_name in _SCHEMA_FIELDS:
                scalar_kwargs[field_name] = resolved
            else:
                extra[field_name] = resolved

    finally:
        if not lazy:
            f.close()

    return DPCDataset(frames=frames, **scalar_kwargs, metadata=extra)
