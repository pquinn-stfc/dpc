'''Data schemas for scientific datasets.

The core abstraction is :class:`ScientificDataset` — an N-dimensional array
with labelled, calibrated axes and a ``signal_type`` string that describes
the physical meaning of the data.  Navigation axes describe the scan/map
dimensions; signal axes describe the detector or spectral dimension.

Signal dimensionality determines the "kind" of dataset:

=================  ==================  =============================
signal_dimension   kind                Example
=================  ==================  =============================
0                  Scalar map          DPC phase, absorption map
1                  Spectrum image      XRF, EELS, DPC-XANES
2                  Image stack         Raw Merlin frames, ptychography
=================  ==================  =============================

A :class:`DPCDataset` is kept for backward compatibility but is now a
thin wrapper that can convert itself to a :class:`ScientificDataset`.
'''

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Axis
# ---------------------------------------------------------------------------

@dataclass
class Axis:
    '''A single labelled, calibrated axis of a :class:`ScientificDataset`.

    Physical coordinates along the axis are::

        coordinate[i] = offset + i * scale

    Parameters
    ----------
    name : str
        Human-readable axis name, e.g. ``"scan_x"``, ``"energy"``.
    size : int
        Number of points along this axis.
    navigate : bool
        ``True`` for navigation (scan/map) axes; ``False`` for signal
        (detector/spectrum) axes.
    scale : float
        Physical size of one pixel/channel in ``units``.  Default 1.
    offset : float
        Physical coordinate of the first point.  Default 0.
    units : str
        Physical unit string, e.g. ``"m"``, ``"keV"``, ``"px"``.
    '''
    name: str
    size: int
    navigate: bool = False
    scale: float = 1.0
    offset: float = 0.0
    units: str = ""

    @property
    def axis(self) -> np.ndarray:
        '''1-D array of physical coordinates along this axis.'''
        return self.offset + np.arange(self.size) * self.scale

    def __repr__(self) -> str:
        kind = "nav" if self.navigate else "sig"
        return (f"Axis({self.name!r}, size={self.size}, {kind}, "
                f"scale={self.scale}, units={self.units!r})")


# ---------------------------------------------------------------------------
# ScientificDataset
# ---------------------------------------------------------------------------

@dataclass
class ScientificDataset:
    '''Generic N-dimensional scientific dataset with labelled axes.

    The ``axes`` list must have the same length as ``data.ndim``.
    By convention, navigation axes come first (lower indices) and signal
    axes come last — matching the natural array layout::

        data.shape = (*nav_shape, *signal_shape)

    Parameters
    ----------
    data : ndarray
        The measurement data.
    axes : list of Axis
        One :class:`Axis` per dimension of ``data``.
    signal_type : str
        Semantic label for the data kind, e.g. ``"XRF"``, ``"DPC"``,
        ``"EELS"``, ``"ptychography"``.  No controlled vocabulary is
        enforced — use whatever is meaningful to your analysis.
    metadata : dict
        Arbitrary key-value pairs (instrument settings, provenance, etc.).

    Examples
    --------
    Build a spectrum-image (XRF map)::

        ds = ScientificDataset(
            data   = xrf_array,   # shape (scan_y, scan_x, n_channels)
            axes   = [
                Axis("scan_y",  size=scan_y,     navigate=True,  scale=50e-9, units="m"),
                Axis("scan_x",  size=scan_x,     navigate=True,  scale=50e-9, units="m"),
                Axis("energy",  size=n_channels, navigate=False, scale=10.0,  units="eV"),
            ],
            signal_type = "XRF",
        )

    Build a raw DPC image-stack::

        ds = ScientificDataset(
            data   = frames,      # shape (scan_y, scan_x, det_y, det_x)
            axes   = [
                Axis("scan_y", size=scan_y, navigate=True,  scale=50e-9, units="m"),
                Axis("scan_x", size=scan_x, navigate=True,  scale=50e-9, units="m"),
                Axis("det_y",  size=det_y,  navigate=False, scale=55e-6, units="m"),
                Axis("det_x",  size=det_x,  navigate=False, scale=55e-6, units="m"),
            ],
            signal_type = "DPC",
        )
    '''

    data: np.ndarray
    axes: list
    signal_type: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if len(self.axes) != self.data.ndim:
            raise ValueError(
                f"Number of axes ({len(self.axes)}) must match data.ndim "
                f"({self.data.ndim})."
            )
        # Validate that navigation axes precede signal axes
        nav_done = False
        for ax in reversed(self.axes):
            if ax.navigate:
                nav_done = True
            elif nav_done:
                raise ValueError(
                    "Navigation axes must all precede signal axes. "
                    "Got a signal axis after a navigation axis."
                )

    # ------------------------------------------------------------------
    # Axis views
    # ------------------------------------------------------------------

    @property
    def navigation_axes(self) -> list:
        '''Navigation axes in array order (outermost first).'''
        return [ax for ax in self.axes if ax.navigate]

    @property
    def signal_axes(self) -> list:
        '''Signal axes in array order (innermost first for signal dims).'''
        return [ax for ax in self.axes if not ax.navigate]

    @property
    def navigation_shape(self) -> tuple:
        '''Shape of the navigation (scan) space.'''
        return tuple(ax.size for ax in self.navigation_axes)

    @property
    def signal_shape(self) -> tuple:
        '''Shape of the signal (detector / spectral) space.'''
        return tuple(ax.size for ax in self.signal_axes)

    @property
    def signal_dimension(self) -> int:
        '''Number of signal axes (0 = scalar map, 1 = spectrum, 2 = image).'''
        return len(self.signal_axes)

    # ------------------------------------------------------------------
    # Convenience predicates
    # ------------------------------------------------------------------

    @property
    def is_scalar_map(self) -> bool:
        '''True when every axis is a navigation axis (signal_dimension == 0).'''
        return self.signal_dimension == 0

    @property
    def is_spectrum(self) -> bool:
        '''True for spectrum-image datasets (signal_dimension == 1).'''
        return self.signal_dimension == 1

    @property
    def is_image(self) -> bool:
        '''True for image-stack datasets (signal_dimension == 2).'''
        return self.signal_dimension == 2

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        nav = " × ".join(str(s) for s in self.navigation_shape) or "–"
        sig = " × ".join(str(s) for s in self.signal_shape) or "–"
        kind = {0: "ScalarMap", 1: "SpectrumImage", 2: "ImageStack"}.get(
            self.signal_dimension, "Dataset"
        )
        label = f" [{self.signal_type}]" if self.signal_type else ""
        return f"ScientificDataset{label} | {kind} | nav={nav} | sig={sig}"

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_scalar_map(
        cls,
        data: np.ndarray,
        nav_axes: list,
        signal_type: str = "",
        metadata: Optional[dict] = None,
    ) -> "ScientificDataset":
        '''Construct a scalar map — all axes are navigation axes.

        Parameters
        ----------
        data : ndarray, shape (ny, nx[, ...])
        nav_axes : list of Axis
            All axes, each with ``navigate=True``.
        '''
        for ax in nav_axes:
            ax.navigate = True
        return cls(data=data, axes=nav_axes, signal_type=signal_type,
                   metadata=metadata or {})

    @classmethod
    def from_spectrum_image(
        cls,
        data: np.ndarray,
        nav_axes: list,
        energy_axis: "Axis",
        signal_type: str = "",
        metadata: Optional[dict] = None,
    ) -> "ScientificDataset":
        '''Construct a spectrum-image (e.g. XRF, EELS).

        Parameters
        ----------
        data : ndarray, shape (*nav_shape, n_channels)
        nav_axes : list of Axis (navigate=True)
        energy_axis : Axis (navigate=False)
        '''
        for ax in nav_axes:
            ax.navigate = True
        energy_axis.navigate = False
        return cls(data=data, axes=[*nav_axes, energy_axis],
                   signal_type=signal_type, metadata=metadata or {})

    @classmethod
    def from_image_stack(
        cls,
        data: np.ndarray,
        nav_axes: list,
        det_axes: list,
        signal_type: str = "",
        metadata: Optional[dict] = None,
    ) -> "ScientificDataset":
        '''Construct an image-stack (e.g. raw detector frames).

        Parameters
        ----------
        data : ndarray, shape (*nav_shape, det_y, det_x)
        nav_axes : list of Axis (navigate=True)
        det_axes : list of Axis (navigate=False)
        '''
        for ax in nav_axes:
            ax.navigate = True
        for ax in det_axes:
            ax.navigate = False
        return cls(data=data, axes=[*nav_axes, *det_axes],
                   signal_type=signal_type, metadata=metadata or {})


# ---------------------------------------------------------------------------
# DPCDataset — kept for backward compatibility
# ---------------------------------------------------------------------------

@dataclass
class DPCDataset:
    '''Typed container for a loaded DPC scan.

    Kept for backward compatibility.  For new code, prefer
    :class:`ScientificDataset` with ``signal_type="DPC"``.
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
        '''Scale factor: CoM pixel shift → phase gradient (rad/m).

        Defined as 2π·pixel_size / (λ·detector_distance).
        '''
        return (2.0 * np.pi * self.pixel_size
                / (self.wavelength * self.detector_distance / 1000.0))

    def as_scientific(self) -> ScientificDataset:
        '''Convert to a :class:`ScientificDataset` with fully labelled axes.

        Returns a ``signal_type="DPC"`` image-stack where the navigation
        axes carry the physical scan step sizes and the signal axes carry
        the detector pixel size.
        '''
        nr, nc = self.frames.shape[:2]
        det_shape = self.frames.shape[2:]
        nav_axes = [
            Axis("scan_y", size=nr, navigate=True,
                 scale=self.scan_step_y, units="m"),
            Axis("scan_x", size=nc, navigate=True,
                 scale=self.scan_step_x, units="m"),
        ]
        det_axes = [
            Axis(f"det_{i}", size=s, navigate=False,
                 scale=self.pixel_size, units="m")
            for i, s in enumerate(det_shape)
        ]
        return ScientificDataset(
            data        = self.frames,
            axes        = [*nav_axes, *det_axes],
            signal_type = "DPC",
            metadata    = {
                "beam_energy":       self.beam_energy,
                "detector_distance": self.detector_distance,
                "com_scale":         self.com_scale,
                **self.metadata,
            },
        )
