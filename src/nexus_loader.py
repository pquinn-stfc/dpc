'''NeXus file loader — introspects HDF5/NeXus files without a mapping YAML.

Unlike :class:`~hdf5_loader.HDF5Loader`, which requires an explicit field
mapping, this loader understands NeXus conventions and can discover the
data structure automatically:

* ``NXentry`` groups are found by walking the file root.
* Within each entry, ``NXdata`` groups are found and their ``@signal``
  and ``@axes`` attributes are used to identify the primary dataset and
  its axis datasets.
* HDF5 Dimension Scales (written by :func:`~io_output.save_scientific`)
  are also honoured as a fallback.
* Axis uniformity is detected automatically; irregular coordinate arrays
  are preserved as :class:`~io_schema.Axis` ``from_array`` objects.

Quick start
-----------
::

    from nexus_loader import NeXusLoader

    # All NXdata groups in the file as ScientificDatasets
    datasets = NeXusLoader("scan.nxs").load_all()

    # The default (first) dataset
    ds = NeXusLoader("scan.nxs").load()

    # A specific NXdata group by path
    ds = NeXusLoader("scan.nxs").load(path="/entry/results")

NeXus conventions supported
----------------------------
* ``/entry/@default`` — preferred NXentry
* ``NXdata/@signal`` — primary dataset name
* ``NXdata/@axes``   — axis dataset names (``"."`` = no axis for that dim)
* ``NXdata/@AXISNAME_indices`` — which dimensions the axis applies to
* ``Axis.attrs["navigate"]``   — written by :func:`~io_output.save_scientific`
* HDF5 Dimension Scales (``DIMENSION_LIST``) — fallback axis discovery
'''

from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from io_schema import Axis, ScientificDataset


# ---------------------------------------------------------------------------
# NeXusLoader
# ---------------------------------------------------------------------------

class NeXusLoader:
    '''Load :class:`~io_schema.ScientificDataset` objects from a NeXus file.

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

    def load(self, path: Optional[str] = None) -> ScientificDataset:
        '''Load one :class:`~io_schema.ScientificDataset` from the file.

        Parameters
        ----------
        path : str, optional
            HDF5 path to a specific ``NXdata`` group, e.g.
            ``"/entry/results"``.  If omitted the default entry/data group
            is used (following ``@default`` attributes where present).

        Returns
        -------
        ScientificDataset

        Raises
        ------
        ValueError
            If no NXdata group is found.
        '''
        with h5py.File(self.path, "r") as f:
            if path is not None:
                grp = f[path]
                return _nxdata_to_dataset(grp)
            # Follow @default chain
            grp = _find_default_nxdata(f)
            if grp is None:
                raise ValueError(
                    f"No NXdata group found in {self.path}. "
                    "Use load_all() to inspect all available groups, "
                    "or supply an explicit path."
                )
            return _nxdata_to_dataset(grp)

    def load_all(self) -> dict[str, ScientificDataset]:
        '''Load every ``NXdata`` group in the file.

        Returns
        -------
        dict
            ``{hdf5_path: ScientificDataset}`` for every NXdata group found.
        '''
        results = {}
        with h5py.File(self.path, "r") as f:
            f.visititems(_collect_nxdata(f, results))
        return results

    def load_from_yaml(
        self,
        yaml_path: str | Path,
        overrides: Optional[dict] = None,
    ) -> dict[str, ScientificDataset]:
        '''Load a named selection of NXdata groups defined in a YAML file.

        The YAML specifies which groups to load and optionally overrides
        their signal type, navigate flags, or axis calibration.

        YAML structure
        --------------
        Each top-level key becomes the label in the returned dict.  The value
        is either a bare HDF5 path string or a dict with these optional keys:

        ``path`` *(required)*
            HDF5 path to the NXdata group.
        ``signal_type`` *(optional)*
            Override the signal type string from the file.
        ``navigate`` *(optional)*
            Dict of ``{axis_name: true|false}`` to override the navigate flag
            for specific axes (useful when a file has no ``navigate`` attribute).
        ``axis_units`` *(optional)*
            Dict of ``{axis_name: unit_string}`` to override axis units.

        Example YAML
        ------------
        ::

            # Load two NXdata groups from a multi-modal scan
            frames:
              path: /entry/instrument/merlin/data
              signal_type: DPC

            xrf_iron:
              path: /entry/xrf/Fe_Ka
              signal_type: XRF
              navigate:
                scan_y: true
                scan_x: true
              axis_units:
                energy: eV

            absorption:
              path: /entry/absorption/data
              # signal_type taken from file

        Parameters
        ----------
        yaml_path : path-like
            Path to the YAML selection file.
        overrides : dict, optional
            Additional entries merged over the YAML (same format).
            Useful for adding or replacing entries at call time.

        Returns
        -------
        dict
            ``{label: ScientificDataset}`` for each entry in the YAML.
        '''
        import yaml, io
        if isinstance(yaml_path, (str, Path)):
            with open(yaml_path) as f:
                raw = yaml.safe_load(f) or {}
        else:
            # accept a file-like or StringIO directly
            raw = yaml.safe_load(yaml_path) or {}
        if overrides:
            raw = {**raw, **overrides}

        results: dict[str, ScientificDataset] = {}
        with h5py.File(self.path, "r") as f:
            for label, spec in raw.items():
                # normalise to dict
                if isinstance(spec, str):
                    spec = {"path": spec}

                hdf5_path   = spec.get("path")
                signal_type = spec.get("signal_type")
                nav_flags   = spec.get("navigate", {})
                unit_flags  = spec.get("axis_units", {})

                if hdf5_path is None:
                    raise KeyError(
                        f"Entry {label!r} in {yaml_path} has no 'path' key."
                    )
                if hdf5_path not in f:
                    raise KeyError(
                        f"HDF5 path {hdf5_path!r} not found in {self.path}. "
                        f"Available NXdata groups: {self.list_nxdata()}"
                    )

                ds = _nxdata_to_dataset(f[hdf5_path])

                # apply overrides
                if signal_type is not None:
                    ds = ScientificDataset(
                        data=ds.data, axes=ds.axes,
                        signal_type=signal_type,
                        metadata=ds.metadata,
                    )
                if nav_flags or unit_flags:
                    new_axes = []
                    for ax in ds.axes:
                        if nav_flags:
                            # If any navigate flags are given, treat the dict
                            # as exhaustive: axes not listed default to False
                            # (signal), not to the heuristic value.
                            navigate = nav_flags.get(ax.name, False)
                        else:
                            navigate = ax.navigate
                        units = unit_flags.get(ax.name, ax.units)
                        if ax._values is not None:
                            new_ax = Axis.from_array(
                                ax.name, ax._values,
                                navigate=navigate, units=units,
                            )
                        else:
                            new_ax = Axis(
                                ax.name, ax.size, navigate=navigate,
                                scale=ax.scale, offset=ax.offset, units=units,
                            )
                        new_axes.append(new_ax)
                    ds = ScientificDataset(
                        data=ds.data, axes=new_axes,
                        signal_type=ds.signal_type,
                        metadata=ds.metadata,
                    )

                results[label] = ds

        return results

    def list_nxdata(self) -> list[str]:
        '''Return the HDF5 paths of all NXdata groups in the file.'''
        paths = []
        with h5py.File(self.path, "r") as f:
            f.visititems(lambda name, obj: _visit_nxdata(name, obj, paths))
        return paths

    def summary(self) -> str:
        '''Return a human-readable summary of all NXdata groups.'''
        lines = [f"NeXus file: {self.path}"]
        with h5py.File(self.path, "r") as f:
            groups: list[str] = []
            f.visititems(lambda name, obj: _visit_nxdata(name, obj, groups))
            if not groups:
                lines.append("  (no NXdata groups found)")
            for g in groups:
                try:
                    ds = _nxdata_to_dataset(f[g])
                    lines.append(f"  {g}")
                    lines.append(f"    {ds!r}")
                    for ax in ds.axes:
                        lines.append(f"      {ax!r}")
                except Exception as exc:
                    lines.append(f"  {g}  [could not load: {exc}]")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _nx_class(obj) -> str:
    '''Return the NX_class attribute of an HDF5 group, or "".'''
    cls = obj.attrs.get("NX_class", "")
    if isinstance(cls, bytes):
        cls = cls.decode()
    return cls


def _visit_nxdata(name: str, obj, out: list) -> None:
    '''h5py visitor that collects NXdata group paths.'''
    if isinstance(obj, h5py.Group) and _nx_class(obj) == "NXdata":
        out.append("/" + name)


def _collect_nxdata(f: h5py.File, results: dict):
    '''Return a visitor that builds ScientificDatasets for every NXdata.'''
    def visitor(name, obj):
        if isinstance(obj, h5py.Group) and _nx_class(obj) == "NXdata":
            try:
                results["/" + name] = _nxdata_to_dataset(obj)
            except Exception:
                pass   # skip malformed groups silently
    return visitor


def _find_default_nxdata(f: h5py.File) -> Optional[h5py.Group]:
    '''Follow @default attributes to find the preferred NXdata group.

    Falls back to the first NXdata group found if no @default chain exists.
    '''
    # Follow root @default → NXentry
    entry = None
    default_entry = f.attrs.get("default", None)
    if default_entry is not None:
        if isinstance(default_entry, bytes):
            default_entry = default_entry.decode()
        if default_entry in f:
            entry = f[default_entry]

    if entry is None:
        # pick first NXentry
        for name, obj in f.items():
            if isinstance(obj, h5py.Group) and _nx_class(obj) == "NXentry":
                entry = obj
                break

    if entry is None:
        # no NXentry at all — try root
        entry = f

    # Within entry, follow @default → NXdata
    default_data = entry.attrs.get("default", None)
    if default_data is not None:
        if isinstance(default_data, bytes):
            default_data = default_data.decode()
        if default_data in entry:
            candidate = entry[default_data]
            if _nx_class(candidate) == "NXdata":
                return candidate

    # Fall back: first NXdata in the entry (depth-first)
    found = []
    entry.visititems(lambda name, obj: _visit_nxdata(name, obj, found))
    if found:
        return entry[found[0].lstrip("/")]

    return None


def _read_attr_str(obj, key) -> Optional[str]:
    '''Read an HDF5 attribute as a Python string, handling bytes.'''
    val = obj.attrs.get(key, None)
    if val is None:
        return None
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode()
    return str(val)


def _read_attr_strlist(obj, key) -> list[str]:
    '''Read an HDF5 attribute that may be a string or list of strings.'''
    val = obj.attrs.get(key, None)
    if val is None:
        return []
    if isinstance(val, (str, bytes, np.bytes_)):
        v = val.decode() if isinstance(val, (bytes, np.bytes_)) else val
        return [v]
    # array-like
    out = []
    for item in val:
        if isinstance(item, (bytes, np.bytes_)):
            item = item.decode()
        out.append(str(item))
    return out


def _nxdata_to_dataset(grp: h5py.Group) -> ScientificDataset:
    '''Convert an NXdata HDF5 group to a ScientificDataset.

    Resolution order for axes:

    1. ``@axes`` attribute — explicit axis dataset names per dimension.
    2. ``@AXISNAME_indices`` — which dimensions a named axis applies to.
    3. HDF5 Dimension Scales (``DIMENSION_LIST``) — written by
       :func:`~io_output.save_scientific`.
    4. Fallback: integer index axes with ``scale=1``.
    '''
    # ---- find the primary signal dataset ----
    signal_name = _read_attr_str(grp, "signal")
    if signal_name is None or signal_name not in grp:
        # try the first non-axis dataset
        axes_names = set(_read_attr_strlist(grp, "axes")) - {"."}
        for name, obj in grp.items():
            if isinstance(obj, h5py.Dataset) and name not in axes_names:
                signal_name = name
                break
    if signal_name is None:
        raise ValueError(f"Cannot identify signal dataset in {grp.name!r}")

    signal_ds   = grp[signal_name]
    data        = signal_ds[()]
    ndim        = data.ndim
    # signal_type may be on the NXdata group (our convention) or on the
    # parent NXentry (save_scientific writes it there as a dataset)
    signal_type = _read_attr_str(grp, "signal_type") or ""
    if not signal_type:
        parent = grp.parent
        if "signal_type" in parent:
            val = parent["signal_type"][()]
            signal_type = val.decode() if isinstance(val, bytes) else str(val)

    # ---- build axis list ----
    axes_attr = _read_attr_strlist(grp, "axes")  # one name per dimension

    # Resolve "which dataset covers which dimension" from @AXISNAME_indices
    dim_to_axis: dict[int, h5py.Dataset] = {}
    for name, obj in grp.items():
        if not isinstance(obj, h5py.Dataset) or name == signal_name:
            continue
        idx_attr = grp.attrs.get(f"{name}_indices", None)
        if idx_attr is not None:
            for dim in np.atleast_1d(idx_attr):
                dim_to_axis[int(dim)] = obj

    # Fill from @axes where not already covered
    for dim, ax_name in enumerate(axes_attr):
        if ax_name == "." or dim in dim_to_axis:
            continue
        if ax_name in grp and isinstance(grp[ax_name], h5py.Dataset):
            dim_to_axis[dim] = grp[ax_name]

    # Fallback: HDF5 Dimension Scales
    if "DIMENSION_LIST" in signal_ds.attrs or len(dim_to_axis) < ndim:
        for dim in range(ndim):
            if dim in dim_to_axis:
                continue
            try:
                n_scales = len(signal_ds.dims[dim])
                if n_scales > 0:
                    dim_to_axis[dim] = signal_ds.dims[dim][0]
            except Exception:
                pass

    # ---- build Axis objects ----
    axes: list[Axis] = []
    for dim in range(ndim):
        ax_ds = dim_to_axis.get(dim, None)
        if ax_ds is not None:
            coords = ax_ds[()]
            name   = ax_ds.name.split("/")[-1]
            units  = _read_attr_str(ax_ds, "units") or ""

            # navigate flag — written by save_scientific; default heuristic
            # is that axes earlier than the signal axes are navigation
            navigate_attr = ax_ds.attrs.get("navigate", None)
            if navigate_attr is not None:
                navigate = bool(navigate_attr)
            else:
                # heuristic: treat as navigation unless it's the last 1–2 dims
                # of an image/spectrum signal
                navigate = dim < (ndim - _guess_signal_ndim(grp, signal_name))

            if coords.ndim == 0:
                # scalar → treat as scale
                axes.append(Axis(name, size=data.shape[dim],
                                 navigate=navigate, scale=float(coords),
                                 units=units))
            elif coords.ndim == 1 and len(coords) == data.shape[dim]:
                axes.append(Axis.from_array(name, coords,
                                            navigate=navigate, units=units))
            else:
                axes.append(Axis(name, size=data.shape[dim],
                                 navigate=navigate, units=units))
        else:
            # no axis information — use integer indices
            axes.append(Axis(f"dim_{dim}", size=data.shape[dim], navigate=False))

    # ---- metadata: all scalar datasets not used as axes ----
    used = {signal_name} | {ds.name.split("/")[-1] for ds in dim_to_axis.values()}
    metadata: dict = {}
    for name, obj in grp.items():
        if name in used or not isinstance(obj, h5py.Dataset):
            continue
        val = obj[()]
        if np.asarray(val).ndim == 0:
            try:
                metadata[name] = val.item()
            except Exception:
                metadata[name] = str(val)

    # also collect parent group scalar datasets (instrument, scan, etc.)
    parent = grp.parent
    for name, obj in parent.items():
        if isinstance(obj, h5py.Dataset) and np.asarray(obj[()]).ndim == 0:
            try:
                metadata[name] = obj[()].item()
            except Exception:
                pass
        elif isinstance(obj, h5py.Group) and _nx_class(obj) in (
            "NXinstrument", "NXsample", "NXsource", "NXmonochromator",
            "NXcollection", "NXnote",
        ):
            _collect_scalars(obj, metadata, prefix=name)

    return ScientificDataset(
        data        = data,
        axes        = axes,
        signal_type = signal_type,
        metadata    = metadata,
    )


def _guess_signal_ndim(grp: h5py.Group, signal_name: str) -> int:
    '''Heuristic: how many trailing dimensions are signal (not navigation)?

    Uses the number of axes that are *not* in the @axes list but *are* in the
    group, or falls back to 0.
    '''
    axes_names = set(_read_attr_strlist(grp, "axes")) - {"."}
    signal_ds  = grp[signal_name]
    n_axes     = len(axes_names)
    return max(0, signal_ds.ndim - n_axes)


def _collect_scalars(grp: h5py.Group, out: dict, prefix: str = "") -> None:
    '''Recursively collect scalar datasets from a group into a flat dict.'''
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
