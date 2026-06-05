from dataclasses import dataclass, field
import numpy as np


@dataclass
class DPCDataset:
    '''Standard schema for a DPC dataset loaded from HDF5/NeXus.

    Attributes
    ----------
    frames : ndarray
        Raw detector frames, shape (scan_y, scan_x, det_y, det_x).
    beam_energy : float
        Incident beam energy in keV.
    detector_distance : float
        Sample-to-detector distance in mm.
    pixel_size : float
        Physical detector pixel size in metres.
    scan_step_x : float
        Scan step size in x in metres.
    scan_step_y : float
        Scan step size in y in metres.
    metadata : dict
        Any additional scalar values extracted from the file that did not map
        to a named field above.
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
        '''Photon wavelength in metres, derived from beam_energy.'''
        hc_eV = 6.626e-34 * 3.0e8 / 1.6e-19
        return hc_eV / (self.beam_energy * 1000.0)

    @property
    def com_scale(self) -> float:
        '''Scale factor converting centre-of-mass pixel shift to phase gradient.

        Defined as 2π·pixel_size / (λ·detector_distance).
        '''
        return (2.0 * np.pi * self.pixel_size
                / (self.wavelength * self.detector_distance / 1000.0))
