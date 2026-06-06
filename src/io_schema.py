'''Data schemas.

The in-memory representation for NeXus data is :class:`NXData` — a thin
container that mirrors a NeXus ``NXdata`` group: a primary data array, a
list of axis coordinate arrays (each tagged as navigation or signal), a
signal-type label, and arbitrary metadata.

Utility functions for coordinate lookup, slicing, and grid comparison live
in :mod:`nexus_loader` alongside the loader that produces ``NXData`` objects.

:class:`DPCDataset` is kept for instrument-specific loading; it can be
converted to ``NXData`` via :meth:`DPCDataset.as_nxdata`.
'''

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import numpy as np


# ---------------------------------------------------------------------------
# AxisInfo — one axis, as simple as possible
# ---------------------------------------------------------------------------

class AxisInfo(NamedTuple):
    '''Coordinate information for one dimension of an :class:`NXData` array.

    Parameters
    ----------
    name : str
        Axis name, e.g. ``"scan_x"``, ``"energy"``.
    values : ndarray, shape (N,)
        Physical coordinate for each point along this dimension.
    navigate : bool
        ``True`` for scan/map (navigation) dimensions; ``False`` for
        detector/spectrum (signal) dimensions.
    units : str
        Physical unit string, e.g. ``"m"``, ``"eV"``.
    '''
    name: str
    values: np.ndarray
    navigate: bool = False
    units: str = ""

    @property
    def size(self) -> int:
        return len(self.values)

    @property
    def is_uniform(self) -> bool:
        '''True if all coordinate steps agree within 0.1 %.'''
        if len(self.values) < 2:
            return True
        steps = np.diff(self.values)
        return bool(np.allclose(steps, steps[0], rtol=1e-3))

    @property
    def step_size(self) -> float:
        '''Mean spacing between coordinate values.'''
        if len(self.values) < 2:
            return 1.0
        return float(np.diff(self.values).mean())

    def __repr__(self) -> str:
        kind = "nav" if self.navigate else "sig"
        tag  = f"uniform step={self.step_size:.3g}" if self.is_uniform \
               else f"irregular mean_step={self.step_size:.3g}"
        return f"AxisInfo({self.name!r}, size={self.size}, {kind}, {tag}, units={self.units!r})"


# ---------------------------------------------------------------------------
# NXData — in-memory NXdata group
# ---------------------------------------------------------------------------

@dataclass
class NXData:
    '''In-memory representation of a NeXus NXdata group.

    Mirrors the NeXus model directly: a primary ``data`` array, one
    :class:`AxisInfo` per dimension, a ``signal_type`` label, and
    free-form ``metadata``.

    Parameters
    ----------
    data : ndarray
        The measurement array, shape ``(*nav_shape, *sig_shape)``.
    axes : list of AxisInfo
        One entry per dimension of ``data``, in array order.  Navigation
        axes come first; signal axes come last.
    signal_type : str
        Free-form label: ``"DPC"``, ``"XRF"``, ``"EELS"``, etc.
    metadata : dict
        Scalar fields from the NeXus file (instrument settings, etc.).
    '''

    data: np.ndarray
    axes: list
    signal_type: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if len(self.axes) != self.data.ndim:
            raise ValueError(
                f"len(axes)={len(self.axes)} must equal data.ndim={self.data.ndim}"
            )

    # ------------------------------------------------------------------
    # Axis access
    # ------------------------------------------------------------------

    @property
    def nav_axes(self) -> list:
        '''Navigation axes in array order.'''
        return [ax for ax in self.axes if ax.navigate]

    @property
    def sig_axes(self) -> list:
        '''Signal axes in array order.'''
        return [ax for ax in self.axes if not ax.navigate]

    @property
    def nav_shape(self) -> tuple:
        return tuple(ax.size for ax in self.nav_axes)

    @property
    def sig_shape(self) -> tuple:
        return tuple(ax.size for ax in self.sig_axes)

    @property
    def signal_dimension(self) -> int:
        return len(self.sig_axes)

    def get_axis(self, name: str) -> AxisInfo:
        '''Return the AxisInfo with the given name.'''
        for ax in self.axes:
            if ax.name == name:
                return ax
        raise KeyError(
            f"No axis {name!r}. Available: {[a.name for a in self.axes]}"
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        nav = " × ".join(str(ax.size) for ax in self.nav_axes) or "–"
        sig = " × ".join(str(ax.size) for ax in self.sig_axes) or "–"
        kind = {0: "ScalarMap", 1: "SpectrumImage", 2: "ImageStack"}.get(
            self.signal_dimension, "Dataset"
        )
        label = f" [{self.signal_type}]" if self.signal_type else ""
        return f"NXData{label} | {kind} | nav={nav} | sig={sig}"


# ---------------------------------------------------------------------------
# DPCDataset — instrument-specific, kept for backward compatibility
# ---------------------------------------------------------------------------

@dataclass
class DPCDataset:
    '''Typed container for a loaded DPC scan.

    Kept for backward compatibility and instrument-specific loading.
    Convert to :class:`NXData` via :meth:`as_nxdata` for generic processing.
    '''

    frames: np.ndarray
    beam_energy: float
    detector_distance: float
    pixel_size: float
    scan_step_x: float
    scan_step_y: float
    metadata: dict = field(default_factory=dict)

    @property
    def wavelength(self) -> float:
        '''Photon wavelength in metres.'''
        hc_eV = 6.626e-34 * 3.0e8 / 1.6e-19
        return hc_eV / (self.beam_energy * 1000.0)

    @property
    def com_scale(self) -> float:
        '''CoM pixel shift → phase gradient scale factor (rad/m).

        2π · pixel_size / (λ · detector_distance).
        '''
        return (2.0 * np.pi * self.pixel_size
                / (self.wavelength * self.detector_distance / 1000.0))

    def as_nxdata(self) -> NXData:
        '''Convert to :class:`NXData` with calibrated axes.'''
        nr, nc = self.frames.shape[:2]
        det_shape = self.frames.shape[2:]

        nav = [
            AxisInfo("scan_y", np.arange(nr) * self.scan_step_y,
                     navigate=True,  units="m"),
            AxisInfo("scan_x", np.arange(nc) * self.scan_step_x,
                     navigate=True,  units="m"),
        ]
        sig = [
            AxisInfo(f"det_{i}", np.arange(s) * self.pixel_size,
                     navigate=False, units="m")
            for i, s in enumerate(det_shape)
        ]
        return NXData(
            data        = self.frames,
            axes        = [*nav, *sig],
            signal_type = "DPC",
            metadata    = {
                "beam_energy":       self.beam_energy,
                "detector_distance": self.detector_distance,
                "com_scale":         self.com_scale,
                **self.metadata,
            },
        )
