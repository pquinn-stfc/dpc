'''Data schemas.

Generic types (:class:`AxisInfo`, :class:`NXData`) are imported from
:mod:`nexus_io` and re-exported here for backward compatibility.

:class:`DPCDataset` is DPC-specific and defined locally.
'''

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nexus_io import AxisInfo, NXData

__all__ = ['AxisInfo', 'NXData', 'DPCDataset']


@dataclass
class DPCDataset:
    '''Typed container for a loaded DPC scan.

    Kept for instrument-specific loading via :func:`~hdf5_loader.load_dpc`.
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
        hc_eV = 6.626e-34 * 3.0e8 / 1.6e-19
        return hc_eV / (self.beam_energy * 1000.0)

    @property
    def com_scale(self) -> float:
        '''CoM pixel shift → phase gradient scale factor (rad/m).'''
        return (2.0 * np.pi * self.pixel_size
                / (self.wavelength * self.detector_distance / 1000.0))

    def as_nxdata(self) -> NXData:
        nr, nc = self.frames.shape[:2]
        det_shape = self.frames.shape[2:]
        nav = [
            AxisInfo('scan_y', np.arange(nr) * self.scan_step_y,
                     navigate=True,  units='m'),
            AxisInfo('scan_x', np.arange(nc) * self.scan_step_x,
                     navigate=True,  units='m'),
        ]
        sig = [
            AxisInfo(f'det_{i}', np.arange(s) * self.pixel_size,
                     navigate=False, units='m')
            for i, s in enumerate(det_shape)
        ]
        return NXData(
            data        = self.frames,
            axes        = [*nav, *sig],
            signal_type = 'DPC',
            metadata    = {
                'beam_energy':       self.beam_energy,
                'detector_distance': self.detector_distance,
                'com_scale':         self.com_scale,
                **self.metadata,
            },
        )
