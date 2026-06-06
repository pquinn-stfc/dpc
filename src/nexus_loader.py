'''NeXus file loader and NXData utility functions.

Loads NeXus/HDF5 files into :class:`~io_schema.NXData` objects without
requiring a field-mapping file, by following standard NeXus conventions
(``@signal``, ``@axes``, ``@default``, HDF5 Dimension Scales).

A YAML selection file lets you choose which ``NXdata`` groups to load from
a multi-modal scan and override their signal type, navigate flags, or units.

Utility functions
-----------------
These operate on :class:`~io_schema.NXData` objects directly:

``coord_to_index(ax, coord)``
    Nearest array index for a physical coordinate.
``slice_by_coords(ds, **kwargs)``
    Slice by physical coordinate range.
``on_same_nav_grid(a, b)``
    Check two datasets share the same navigation grid.
``apply_nav(ds, fn)``
    Map a function over every navigation position.

Quick start
-----------
::

    from nexus_loader import NeXusLoader

    loader = NeXusLoader("scan.nxs")

    # one dataset (follows @default chain)
    ds = loader.load()

    # specific NXdata group
    ds = loader.load("/entry/results")

    # everything in the file
    all_ds = loader.load_all()

    # selected groups via YAML
    datasets = loader.load_from_yaml("config/i14_nexus_selection.yaml")
'''

from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from io_schema import AxisInfo, NXData


# ---------------------------------------------------------------------------
# NeXusLoader
# ---------------------------------------------------------------------------

class NeXusLoader:
    '''Load :class:`~io_schema.NXData` objects from a NeXus file.

    Parameters
    ----------
    path : path-like
        Path to the HDF5/NeXus file.
    '''

    def __init__(self, path: str | Path):
        self.path = Path(path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, path: Optional[str] = None) -> NXData:
        '''Load one :class:`~io_schema.NXData` from the file.

        Parameters
        ----------
        path : str, optional
            HDF5 path to a specific ``NXdata`` group.  If omitted the
            default group is used (following ``@default`` attributes).

        Returns
        -------
        NXData
        '''
        with h5py.File(self.path, "r") as f:
            grp = f[path] if path is not None else _find_default_nxdata(f)
            if grp is None:
                raise ValueError(
                    f"No NXdata group found in {self.path}. "
                    "Use list_nxdata() to see available groups."
                )
            return _nxdata_to_nxdata(grp)

    def load_all(self) -> dict[str, NXData]:
        '''Load every ``NXdata`` group in the file.

        Returns
        -------
        dict
            ``{hdf5_path: NXData}`` for every NXdata group found.
        '''
        results: dict[str, NXData] = {}
        with h5py.File(self.path, "r") as f:
            f.visititems(_collect_nxdata(results))
        return results

    def load_from_yaml(
        self,
        yaml_path,
        overrides: Optional[dict] = None,
    ) -> dict[str, NXData]:
        '''Load a named selection of NXdata groups defined in a YAML file.

        Each top-level key in the YAML becomes the label in the returned dict.
        The value is either a bare HDF5 path string or a dict:

        .. code-block:: yaml

            frames:
              path: /entry/instrument/merlin/data
              signal_type: DPC
              navigate:
                scan_y: true
                scan_x: true

            xrf_fe:
              path: /entry/xrf/Fe_Ka
              signal_type: XRF
              navigate:
                scan_y: true
                scan_x: true
              axis_units:
                energy: eV

        When ``navigate`` is provided it is **exhaustive**: axes not listed
        default to ``False`` (signal), overriding the heuristic.

        Parameters
        ----------
        yaml_path : path-like or file-like
            Path to the YAML file, or a ``StringIO`` / open file object.
        overrides : dict, optional
            Entries merged on top of the YAML at call time.

        Returns
        -------
        dict
            ``{label: NXData}``
        '''
        import yaml
        if isinstance(yaml_path, (str, Path)):
            with open(yaml_path) as f:
                raw = yaml.safe_load(f) or {}
        else:
            raw = yaml.safe_load(yaml_path) or {}

        if overrides:
            raw = {**raw, **overrides}

        results: dict[str, NXData] = {}
        with h5py.File(self.path, "r") as f:
            for label, spec in raw.items():
                if isinstance(spec, str):
                    spec = {"path": spec}

                hdf5_path   = spec.get("path")
                signal_type = spec.get("signal_type")
                nav_flags   = spec.get("navigate", {})
                unit_flags  = spec.get("axis_units", {})

                if hdf5_path is None:
                    raise KeyError(f"Entry {label!r} has no 'path' key.")
                if hdf5_path not in f:
                    raise KeyError(
                        f"HDF5 path {hdf5_path!r} not found in {self.path}. "
                        f"Available: {self.list_nxdata()}"
                    )

                ds = _nxdata_to_nxdata(f[hdf5_path])

                # apply overrides
                new_axes = []
                for ax in ds.axes:
                    navigate = (nav_flags.get(ax.name, False)
                                if nav_flags else ax.navigate)
                    units    = unit_flags.get(ax.name, ax.units)
                    new_axes.append(AxisInfo(ax.name, ax.values,
                                             navigate=navigate, units=units))

                results[label] = NXData(
                    data        = ds.data,
                    axes        = new_axes,
                    signal_type = signal_type if signal_type is not None
                                  else ds.signal_type,
                    metadata    = ds.metadata,
                )

        return results

    def list_nxdata(self) -> list[str]:
        '''Return the HDF5 paths of all NXdata groups in the file.'''
        paths: list[str] = []
        with h5py.File(self.path, "r") as f:
            f.visititems(lambda name, obj: _visit_nxdata(name, obj, paths))
        return paths

    def summary(self) -> str:
        '''Human-readable overview of all NXdata groups in the file.'''
        lines = [f"NeXus file: {self.path}"]
        with h5py.File(self.path, "r") as f:
            groups: list[str] = []
            f.visititems(lambda name, obj: _visit_nxdata(name, obj, groups))
            if not groups:
                lines.append("  (no NXdata groups found)")
            for g in groups:
                try:
                    ds = _nxdata_to_nxdata(f[g])
                    lines.append(f"  {g}")
                    lines.append(f"    {ds!r}")
                    for ax in ds.axes:
                        lines.append(f"      {ax!r}")
                except Exception as exc:
                    lines.append(f"  {g}  [could not load: {exc}]")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def coord_to_index(ax: AxisInfo, coord: float) -> int:
    '''Find the nearest array index for a physical coordinate.

    Uses ``numpy.searchsorted`` for irregular axes and exact arithmetic for
    uniform ones.

    Parameters
    ----------
    ax : AxisInfo
    coord : float

    Returns
    -------
    int
        Clamped to ``[0, ax.size - 1]``.
    '''
    vals = ax.values
    if ax.is_uniform:
        step = vals[1] - vals[0] if len(vals) > 1 else 1.0
        idx  = int(round((coord - float(vals[0])) / step))
        return max(0, min(idx, ax.size - 1))
    idx = int(np.searchsorted(vals, coord))
    if idx >= ax.size:
        return ax.size - 1
    if idx == 0:
        return 0
    if abs(vals[idx - 1] - coord) < abs(vals[idx] - coord):
        return idx - 1
    return idx


def slice_by_coords(ds: NXData, **kwargs) -> NXData:
    '''Slice an NXData by physical coordinate ranges.

    Keyword arguments are ``axis_name=(lo, hi)`` pairs in the physical
    units of that axis.  Unspecified axes are returned in full.

    Returns a new :class:`~io_schema.NXData` with trimmed coordinate arrays.

    Example
    -------
    ::

        roi  = slice_by_coords(ds, scan_x=(0, 500e-9), scan_y=(0, 500e-9))
        low_e = slice_by_coords(xrf, energy=(7100, 7130))
    '''
    slices   = []
    new_axes = []
    for ax in ds.axes:
        if ax.name in kwargs:
            lo, hi = kwargs[ax.name]
            i_lo   = coord_to_index(ax, lo)
            i_hi   = coord_to_index(ax, hi) + 1
            slices.append(slice(i_lo, i_hi))
            new_axes.append(AxisInfo(ax.name, ax.values[i_lo:i_hi],
                                     ax.navigate, ax.units))
        else:
            slices.append(slice(None))
            new_axes.append(ax)

    return NXData(
        data        = ds.data[tuple(slices)],
        axes        = new_axes,
        signal_type = ds.signal_type,
        metadata    = ds.metadata.copy(),
    )


def on_same_nav_grid(a: NXData, b: NXData, rtol: float = 1e-3) -> bool:
    '''Return True if two NXData objects share the same navigation grid.

    Compares navigation axis names, sizes, and coordinate arrays element-wise.
    Useful for verifying that an XRF map and a DPC map came from the same scan.

    Parameters
    ----------
    a, b : NXData
    rtol : float
        Relative tolerance for coordinate comparison.  Default 1e-3.
    '''
    nav_a = [ax for ax in a.axes if ax.navigate]
    nav_b = [ax for ax in b.axes if ax.navigate]
    if len(nav_a) != len(nav_b):
        return False
    for ax_a, ax_b in zip(nav_a, nav_b):
        if ax_a.name != ax_b.name or ax_a.size != ax_b.size:
            return False
        if not np.allclose(ax_a.values, ax_b.values, rtol=rtol):
            return False
    return True


def apply_nav(ds: NXData, fn, dtype=None, sig_shape=None) -> NXData:
    '''Apply a function to the signal slice at every navigation position.

    Returns a new :class:`~io_schema.NXData` with navigation axes preserved
    and new signal axes inferred from the function output.

    Parameters
    ----------
    ds : NXData
    fn : callable
        ``fn(signal_slice: ndarray) -> ndarray or scalar``
    dtype : numpy dtype, optional
    sig_shape : tuple, optional

    Example
    -------
    ::

        # sum each detector frame → scalar map
        intensity = apply_nav(ds, lambda frame: frame.sum())

        # centre-of-mass → (2,) per position
        com_map = apply_nav(ds, centre_of_mass, sig_shape=(2,))
    '''
    nav_shape = tuple(ax.size for ax in ds.axes if ax.navigate)
    sig_shape_in = tuple(ax.size for ax in ds.axes if not ax.navigate)
    flat_nav  = int(np.prod(nav_shape)) if nav_shape else 1

    first  = np.asarray(fn(ds.data.reshape(flat_nav, *sig_shape_in)[0]))
    if sig_shape is None:
        sig_shape = first.shape
    if dtype is None:
        dtype = first.dtype

    out      = np.empty((*nav_shape, *sig_shape), dtype=dtype)
    flat_in  = ds.data.reshape(flat_nav, *sig_shape_in)
    flat_out = out.reshape(flat_nav, *sig_shape)
    for i in range(flat_nav):
        flat_out[i] = fn(flat_in[i])

    nav_axes = [ax for ax in ds.axes if ax.navigate]
    new_sig  = [AxisInfo(f"result_{i}", np.arange(s), navigate=False)
                for i, s in enumerate(sig_shape)]

    return NXData(
        data        = out,
        axes        = [*nav_axes, *new_sig],
        signal_type = ds.signal_type,
        metadata    = ds.metadata.copy(),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _nx_class(obj) -> str:
    cls = obj.attrs.get("NX_class", "")
    return cls.decode() if isinstance(cls, bytes) else str(cls)


def _visit_nxdata(name: str, obj, out: list) -> None:
    if isinstance(obj, h5py.Group) and _nx_class(obj) == "NXdata":
        out.append("/" + name)


def _collect_nxdata(results: dict):
    def visitor(name, obj):
        if isinstance(obj, h5py.Group) and _nx_class(obj) == "NXdata":
            try:
                results["/" + name] = _nxdata_to_nxdata(obj)
            except Exception:
                pass
    return visitor


def _find_default_nxdata(f: h5py.File) -> Optional[h5py.Group]:
    '''Follow @default chain to find the preferred NXdata group.'''
    def _decode(v):
        return v.decode() if isinstance(v, (bytes, np.bytes_)) else str(v)

    entry = None
    d = f.attrs.get("default")
    if d and _decode(d) in f:
        entry = f[_decode(d)]
    if entry is None:
        for obj in f.values():
            if isinstance(obj, h5py.Group) and _nx_class(obj) == "NXentry":
                entry = obj
                break
    if entry is None:
        entry = f

    d = entry.attrs.get("default")
    if d:
        key = _decode(d)
        if key in entry and _nx_class(entry[key]) == "NXdata":
            return entry[key]

    found: list[str] = []
    entry.visititems(lambda name, obj: _visit_nxdata(name, obj, found))
    if found:
        return entry[found[0].lstrip("/")]
    return None


def _read_str(obj, key) -> Optional[str]:
    val = obj.attrs.get(key)
    if val is None:
        return None
    return val.decode() if isinstance(val, (bytes, np.bytes_)) else str(val)


def _read_strlist(obj, key) -> list[str]:
    val = obj.attrs.get(key)
    if val is None:
        return []
    if isinstance(val, (str, bytes, np.bytes_)):
        s = val.decode() if isinstance(val, (bytes, np.bytes_)) else val
        return [s.rstrip('\x00').strip()]
    return [
        (v.decode().rstrip('\x00').strip()
         if isinstance(v, (bytes, np.bytes_)) else str(v).rstrip('\x00').strip())
        for v in val
    ]


def _nxdata_to_nxdata(grp: h5py.Group) -> NXData:
    '''Convert an open NXdata HDF5 group to an NXData object.'''

    # find signal dataset
    signal_name = _read_str(grp, "signal")
    if signal_name is None or signal_name not in grp:
        axes_names = set(_read_strlist(grp, "axes")) - {"."}
        for name, obj in grp.items():
            if isinstance(obj, h5py.Dataset) and name not in axes_names:
                signal_name = name
                break
    if signal_name is None:
        raise ValueError(f"Cannot identify signal dataset in {grp.name!r}")

    signal_ds = grp[signal_name]
    data      = signal_ds[()]
    ndim      = data.ndim

    # signal_type
    signal_type = _read_str(grp, "signal_type") or ""
    if not signal_type and "signal_type" in grp.parent:
        val = grp.parent["signal_type"][()]
        signal_type = val.decode() if isinstance(val, bytes) else str(val)

    # build dim → axis dataset mapping
    # Resolution order (highest priority first):
    #   1. @axes attribute  — explicit per-dim axis names; "." means no axis
    #   2. @AXISNAME_indices — only for axes named in @axes (avoids monitor
    #      channels and positioners that spuriously claim all dims)
    #   3. HDF5 Dimension Scales — last resort
    axes_attr    = _read_strlist(grp, "axes")
    dim_to_axds: dict[int, h5py.Dataset] = {}

    # Step 1 — @axes list (most reliable)
    # Track the intended axis name separately from the resolved dataset path,
    # because @axes entries may be soft links to datasets elsewhere in the file.
    dim_to_axname: dict[int, str] = {}   # dim → preferred axis name
    named_axes: set[str] = set()
    for dim, ax_name in enumerate(axes_attr):
        if ax_name == ".":
            continue                      # "." means no axis for this dim
        named_axes.add(ax_name)
        if ax_name in grp and isinstance(grp[ax_name], h5py.Dataset):
            dim_to_axds[dim]   = grp[ax_name]
            dim_to_axname[dim] = ax_name  # use link name, not target path

    # Step 2 — @AXISNAME_indices only for axes listed in @axes
    for name in named_axes:
        if name not in grp or not isinstance(grp[name], h5py.Dataset):
            continue
        idx = grp.attrs.get(f"{name}_indices")
        if idx is not None:
            for d in np.atleast_1d(idx):
                d = int(d)
                if d not in dim_to_axds:   # don't overwrite @axes result
                    dim_to_axds[d] = grp[name]

    # Step 3 — HDF5 Dimension Scales for remaining dims
    for dim in range(ndim):
        if dim in dim_to_axds:
            continue
        try:
            if len(signal_ds.dims[dim]) > 0:
                dim_to_axds[dim] = signal_ds.dims[dim][0]
        except Exception:
            pass

    # build AxisInfo list
    n_named = len(dim_to_axds)
    axes: list[AxisInfo] = []
    for dim in range(ndim):
        ax_ds = dim_to_axds.get(dim)
        if ax_ds is not None:
            coords   = ax_ds[()]
            # prefer the name from @axes (handles soft links correctly)
            name     = dim_to_axname.get(dim, ax_ds.name.split("/")[-1])
            units    = _read_str(ax_ds, "units") or ""
            nav_attr = ax_ds.attrs.get("navigate")
            if nav_attr is not None:
                navigate = bool(nav_attr)
            else:
                # heuristic: nav axes precede the signal dims
                navigate = dim < (ndim - max(0, ndim - n_named))
            if coords.ndim == 0:
                coords = np.array([float(coords)])
            axes.append(AxisInfo(name, np.asarray(coords, dtype=float),
                                 navigate=navigate, units=units))
        else:
            axes.append(AxisInfo(f"dim_{dim}",
                                 np.arange(data.shape[dim], dtype=float),
                                 navigate=False))

    # metadata: scalars from NXdata group and neighbouring instrument groups
    used = {signal_name} | {ds.name.split("/")[-1]
                             for ds in dim_to_axds.values()}
    metadata: dict = {}
    for name, obj in grp.items():
        if name in used or not isinstance(obj, h5py.Dataset):
            continue
        val = np.asarray(obj[()])
        if val.ndim == 0:
            try:
                metadata[name] = val.item()
            except Exception:
                metadata[name] = str(val)

    for name, obj in grp.parent.items():
        if isinstance(obj, h5py.Dataset) and np.asarray(obj[()]).ndim == 0:
            try:
                metadata[name] = obj[()].item()
            except Exception:
                pass
        elif isinstance(obj, h5py.Group) and _nx_class(obj) in (
            "NXinstrument", "NXsample", "NXsource",
            "NXmonochromator", "NXcollection", "NXnote",
        ):
            _collect_scalars(obj, metadata, prefix=name)

    return NXData(data=data, axes=axes, signal_type=signal_type,
                  metadata=metadata)


def _collect_scalars(grp: h5py.Group, out: dict, prefix: str = "") -> None:
    for name, obj in grp.items():
        key = f"{prefix}.{name}" if prefix else name
        if isinstance(obj, h5py.Dataset):
            val = np.asarray(obj[()])
            if val.ndim == 0:
                try:
                    out[key] = val.item()
                except Exception:
                    out[key] = str(val)
        elif isinstance(obj, h5py.Group):
            _collect_scalars(obj, out, prefix=key)
