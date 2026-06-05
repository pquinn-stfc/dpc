# dpc

A Python package for extracting differential phase contrast (DPC) images from
pixelated-detector data at scanning X-ray nanoprobe beamlines.

Developed and used at the hard X-ray nanoprobe beamline **I14**, Diamond Light
Source. The methods and their validation against ptychography are described in:

> Quinn, P. D., Cacho-Nerin, F., Gomez-Gonzalez, M. A., Parker, J. E., Poon, T.
> & Walker, J. M. (2023). Differential phase contrast for quantitative imaging
> and spectro-microscopy at a nanoprobe beamline.
> *J. Synchrotron Rad.* **30**, 200–207.
> https://doi.org/10.1107/S1600577522010633

---

## Background

When a focused X-ray beam passes through a sample it is refracted, absorbed,
and scattered. A pixelated detector placed downstream records the full
intensity distribution at each scan point, allowing several contrast modes to
be extracted simultaneously:

| Signal | Physical origin | Extracted as |
|--------|----------------|--------------|
| Absorption | Attenuation of total intensity | log(I₀/I) |
| DPC | Refraction — gradient of the phase shift | Centre of mass (CoM) shift |
| Dark field | Small-angle scattering from sub-beam features | Intensity outside beam |

DPC exploits the fact that the real part of the refractive index δ is much
larger than the absorption β, making it sensitive to light elements that are
largely invisible to XRF or absorption imaging.  The beam deflection angles
θₓ, θᵧ are related to the phase gradient by

$$\theta_x = \frac{1}{k}\frac{\partial\phi}{\partial x}, \quad
  \theta_y = \frac{1}{k}\frac{\partial\phi}{\partial y}$$

and the phase φ is recovered by a single Fourier integration step.  For *N*
scan points, DPC requires *N* CoM calculations and 2 FFTs — compared with
roughly 2*N* FFTs per iteration over hundreds of iterations for ptychography —
so results are available within seconds of scan completion.

### Phase retrieval methods

Four Fourier-based integration methods are provided, plus the SCS direct
method.  They differ mainly in how they handle boundary artefacts and
low-frequency components:

| Method | Key property | Reference |
|--------|-------------|-----------|
| `kottler` (default) | Direct Fourier integration | Kottler *et al.*, 2007 |
| `arnison` | Discrete-frequency denominator | Arnison *et al.*, 2004 |
| `frankot` | Weighted least-squares in Fourier space | Frankot & Chellappa, 1988 |
| `ishizuka` | DCT-based; natural Neumann boundary conditions | Ishizuka & Ishizuka, 2020 |
| `lazic` | Weighted integration with optional high-pass filter | Lazić *et al.*, 2016 |

Anti-symmetric mirroring (extending the gradient maps 4× before the FFT) is
available for all FFT methods and substantially reduces low-frequency
background bands.

---

## Package structure

```
dpc/
├── main.py                  # CLI entry point
├── config/
│   └── i14_mapping.yaml     # Example HDF5 field mapping for Diamond I14
└── src/
    ├── dpc_tools.py          # Beam detection, cropping, centre-of-mass
    ├── dpc_recon.py          # Phase retrieval algorithms
    ├── mask_tools.py         # Pixel masking, circular/radial masks, signal maps
    ├── outlier_removal_tools.py  # Hampel filter, dead-pixel detection
    ├── io_schema.py          # DPCDataset dataclass + per-stage HDF5 schemas
    ├── hdf5_loader.py        # Generic HDF5 loader with YAML field mapping
    └── io_output.py          # Save results to NeXus HDF5, PNG, or TIFF
```

### Module summary

**`dpc_tools`** — Beam identification and masking.
Uses Otsu thresholding (`get_beam`) to locate the direct beam automatically,
`crop_beam` to restrict the calculation to a region around it, and
`get_dark_field_mask` to define the beam and scatter regions.
`centre_of_mass` calculates the weighted CoM over a labelled detector region.

**`dpc_recon`** — Phase retrieval.
Standalone functions (`kottler`, `arnison`, `frankt`, `ishizuka`, `scs`,
`lazic`) and a unified `phase_retrieval(dx, dy, calX, calY, method)` dispatcher
that takes plain numpy arrays.

**`mask_tools`** — Pixel quality and region masks.
`build_pixel_mask` handles both quick (single-frame Hampel) and full
(mean/std/hot-pixel) masking modes.  `create_circular_mask` and
`create_circular_radial_masks` build beam-region and directional scatter
masks.  `absorption_signal` and `scatter_signal` compute the scalar maps from
a masked detector frame.

**`outlier_removal_tools`** — Bad-pixel suppression.
Numba-accelerated Hampel filter for 1D and 2D data, `median_subtraction`,
and `sure_dead_pixel` for detection of stuck pixels across a stack.

**`io_schema` / `hdf5_loader`** — Data loading without HyperSpy.
`DPCDataset` is the standard in-memory schema; `load_dpc` reads any
HDF5/NeXus file using a declarative YAML mapping from HDF5 paths to schema
fields.  `DPCDataset.wavelength` and `DPCDataset.com_scale` derive the photon
wavelength and the CoM-to-phase-gradient conversion factor
(2π · pixel\_size / λ · detector\_distance) automatically.

**`io_output`** — Saving results.
`save_hdf5` writes a NeXus-flavoured file with `NXentry`, `NXinstrument`,
`NXdata` (with attached scan axes for direct loading in silx/h5web), and a
provenance `NXnote`.  `save_images` writes normalised 8-bit PNG or TIFF files.

---

## Installation

```bash
pip install numpy scipy scikit-image numba h5py pyyaml pillow matplotlib
git clone https://github.com/yourname/dpc.git
cd dpc
```

---

## Usage

### Command line

```bash
# Default: Kottler method, HDF5 output
python main.py scan.nxs config/i14_mapping.yaml

# Specify method, output location and formats
python main.py scan.nxs config/i14_mapping.yaml \
    --method arnison \
    --output /data/processed/scan98400 \
    --format hdf5 png

# Full pixel-masking mode (slower, uses whole stack)
python main.py scan.nxs config/i14_mapping.yaml --full-mask

# Skip phase retrieval (gradient norm only)
python main.py scan.nxs config/i14_mapping.yaml --no-phase
```

### Python API

```python
import sys
sys.path.insert(0, 'src')

from hdf5_loader import load_dpc
from dpc_tools import get_beam, crop_beam, get_dark_field_mask, centre_of_mass
from mask_tools import build_pixel_mask
from dpc_recon import phase_retrieval
from io_output import save_hdf5

# Load
ds = load_dpc('scan.nxs', 'config/i14_mapping.yaml')
print(f'Wavelength: {ds.wavelength * 1e10:.4f} Å')
print(f'CoM scale:  {ds.com_scale:.4f} rad/pixel')

# Mask
sample_frame = ds.frames[0, 0].astype(float)
pixel_mask, filtered = build_pixel_mask(sample_frame)

# Find beam
prop = get_beam(filtered)
crop_x, crop_y = crop_beam(filtered, crop_size=256, bbox=prop.bbox)
beam_mask = get_dark_field_mask(filtered, prop.bbox,
                                cropped_x=crop_x, cropped_y=crop_y)
com_label = (~(beam_mask | pixel_mask)).astype(int)

# CoM map → phase gradients
com_map = ... # apply centre_of_mass per frame
dx = (com_map[..., 1] - com_map[..., 1].mean()) * ds.com_scale
dy = (com_map[..., 0] - com_map[..., 0].mean()) * ds.com_scale

# Phase retrieval
phase = phase_retrieval(dx, dy, calX=ds.scan_step_x, calY=ds.scan_step_y,
                        method='kottler')
```

### HDF5 field mapping

Create a YAML file that maps HDF5 dataset paths to `DPCDataset` fields:

```yaml
# config/my_instrument_mapping.yaml
frames:            /entry/instrument/detector/data
beam_energy:       /entry/instrument/monochromator/energy
detector_distance: /entry/instrument/detector/distance
pixel_size:        /entry/instrument/detector/pixel_size
scan_step_x:       /entry/scan/step_x
scan_step_y:       /entry/scan/step_y
```

Any extra keys are stored in `DPCDataset.metadata` without error.

---

## Output HDF5 schema

Results are written as a NeXus-compatible HDF5 file.  The structure follows
the NeXus 3 conventions so any NeXus-aware viewer (silx, h5web, NeXpy) opens
it without configuration.

```
<output>.nxs
└── /entry  [NXentry]
    ├── definition        = "NXdpc"          ← dataset, not attribute
    ├── program_name      = "dpc"
    ├── start_time        (ISO-8601 UTC)
    │
    ├── /instrument  [NXinstrument]
    │   ├── beam_energy          (float, keV)
    │   ├── detector_distance    (float, mm)
    │   └── pixel_size           (float, m)
    │
    ├── /scan  [NXcollection]
    │   ├── scan_step_x          (float, m)
    │   └── scan_step_y          (float, m)
    │
    ├── /processing  [NXnote]
    │   ├── method               (str, e.g. "kottler")
    │   ├── crop_size            (int)
    │   └── mask_mode            (str, "quick" | "full")
    │
    ├── /results  [NXdata]
    │   ├── @signal = "phase"    ← "grad_norm" when phase retrieval is skipped
    │   ├── @axes   = ["y", "x"]
    │   ├── x   (1D float, m)    ← HDF5 dimension scale attached to dim 1
    │   ├── y   (1D float, m)    ← HDF5 dimension scale attached to dim 0
    │   ├── phase      (2D float, rad)    [optional]
    │   ├── grad_norm  (2D float)
    │   ├── dx         (2D float, rad/m)
    │   └── dy         (2D float, rad/m)
    │
    └── /auxiliary  [NXcollection]
        ├── pixel_mask   (2D bool)          [optional]
        ├── absorption   (2D float)         [optional]
        └── scatter      (2D float)         [optional]
```

**NeXus conventions used:**

- `definition`, `program_name`, and `start_time` are written as HDF5 datasets
  under `/entry`, not as group attributes, so NeXus validators find them as
  required fields.
- `x` and `y` are registered as HDF5 Dimension Scales (`make_scale`) and
  attached to dimension 1 and 0 respectively of every 2D dataset in
  `/results`.  This means silx, h5web, and NeXpy assign correct physical axis
  labels and tick values to every plot automatically.
- All 2D arrays are stored with gzip compression.

---

## Quantitative DPC and spectro-microscopy

The CoM shift is converted to a phase gradient using

$$\text{scale} = \frac{2\pi \cdot p}{\lambda \cdot z}$$

where *p* is the detector pixel size, *λ* is the photon wavelength, and *z*
is the sample-to-detector distance.  This factor is computed automatically
from the loaded metadata as `DPCDataset.com_scale`.

DPC is quantitative when absorption is negligible.  Near or above an
absorption edge the measured signal contains a mixture of differential phase
and differential absorption contributions.  These can be separated by fitting
a linear combination of the XRF-XANES signal and its Kramers–Kronig transform
(see Quinn *et al.*, 2023, §4.1).

---

## References

- Arnison *et al.* (2004). *J. Microsc.* **214**, 7–12.
- Frankot & Chellappa (1988). *IEEE TPAMI* **10**, 439–451.
- Ishizuka & Ishizuka (2020). *JEOL News* **55**, 24–31.
- Kottler *et al.* (2007). *Opt. Express* **15**, 1175.
- Lazić *et al.* (2016). *Ultramicroscopy* **160**, 265–280.
- Quinn *et al.* (2023). *J. Synchrotron Rad.* **30**, 200–207.
- Simchony *et al.* (1990). *IEEE TPAMI* **12**, 435–446.

---

## Acknowledgements

Work carried out at Diamond Light Source beamline I14.  We thank Benedikt
Daurer and Darren Batey for assistance with ptychography reconstructions,
Hongchang Wang for discussions on dark-field signals, and Gerald Langer for
coccolith samples.
