'''DPC processing pipeline.

Usage
-----
    python main.py <nexus_file> <mapping_yaml> [options]

Arguments
---------
nexus_file      Path to the HDF5/NeXus input file.
mapping_yaml    Path to a YAML field-mapping file (see config/i14_mapping.yaml).

Options
-------
--method        Phase retrieval method: kottler (default), arnison, frankot,
                ishizuka, scs.
--crop-size     Detector crop size in pixels (default: 256).
--output        Output NeXus file path.  Defaults to <input_stem>_dpc.nxs
                next to the input file.
--full-mask     Use the full pixel-masking mode (requires loading all frames).
                Default is the faster single-frame (quick) mode.
--no-phase      Skip phase retrieval and only compute the gradient norm.
'''

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='Compute DPC images from a NeXus/HDF5 detector dataset.'
    )
    p.add_argument('nexus_file', help='Input HDF5/NeXus file')
    p.add_argument('mapping_yaml', help='YAML field-mapping file')
    p.add_argument('--method', default='kottler',
                   choices=['kottler', 'arnison', 'frankot', 'ishizuka', 'scs'],
                   help='Phase retrieval method (default: kottler)')
    p.add_argument('--crop-size', type=int, default=256,
                   help='Detector crop size in pixels (default: 256)')
    p.add_argument('--output', default=None,
                   help='Output base path, without extension '
                        '(default: <input stem>_dpc next to the input file)')
    p.add_argument('--format', nargs='+',
                   choices=['hdf5', 'png', 'tif'],
                   default=['hdf5'],
                   metavar='FORMAT',
                   help='One or more output formats: hdf5, png, tif '
                        '(default: hdf5)')
    p.add_argument('--full-mask', action='store_true',
                   help='Use full pixel-masking mode instead of quick mode')
    p.add_argument('--no-phase', action='store_true',
                   help='Skip phase retrieval')
    p.add_argument('--debug', action='store_true',
                   help='Save a diagnostic figure showing masks and overlays')
    return p.parse_args(argv)


def run(args):
    # Import here so the module works as a library without heavy imports at
    # the top level.
    sys.path.insert(0, str(Path(__file__).parent))

    from core.hdf5_loader import load_dpc
    from core.dpc_tools import get_beam, crop_beam, get_dark_field_mask, centre_of_mass
    from core.mask_tools import build_pixel_mask
    from core.dpc_recon import phase_grad_norm, phase_retrieval
    from core.outlier_removal_tools import median_subtraction

    # ------------------------------------------------------------------
    # 1. Load dataset (lazy — frames stay on disk until indexed)
    # ------------------------------------------------------------------
    print(f'Loading {args.nexus_file} ...')
    import h5py
    ds = load_dpc(args.nexus_file, args.mapping_yaml, lazy=True)
    frames = ds.frames                  # h5py.Dataset — not yet in RAM
    scan_shape = frames.shape[:2]
    det_shape  = frames.shape[2:]
    frame_bytes = int(np.prod(det_shape)) * frames.dtype.itemsize
    total_gb    = frames.size * frames.dtype.itemsize / 1024**3
    print(f'  frames shape : {frames.shape}  ({total_gb:.2f} GB total, '
          f'{frame_bytes/1024:.0f} KB/frame — lazy)')
    print(f'  beam energy  : {ds.beam_energy:.3f} keV')
    print(f'  wavelength   : {ds.wavelength * 1e10:.4f} Å')
    print(f'  det distance : {ds.detector_distance:.1f} mm')
    print(f'  CoM scale    : {ds.com_scale:.4f} rad/pixel')

    # ------------------------------------------------------------------
    # 2. Build pixel mask from a representative frame
    # ------------------------------------------------------------------
    print('Building pixel mask ...')
    sample_frame = frames[0, 0].astype(float)   # reads one frame

    stack = None
    if args.full_mask:
        print('  Reading all frames for full masking ...')
        stack = frames[()].reshape(-1, *det_shape).astype(float)

    pixel_mask, filtered_for_beam = build_pixel_mask(
        sample_frame,
        quick=not args.full_mask,
        stack=stack,
    )

    # ------------------------------------------------------------------
    # 3. Detect beam, crop, dark-field mask
    # ------------------------------------------------------------------
    print('Detecting beam ...')
    try:
        prop = get_beam(filtered_for_beam)
        bbox = prop.bbox
        cropped_x, cropped_y = crop_beam(filtered_for_beam, args.crop_size, bbox)
    except Exception:
        print('  Beam detection failed — using full detector area.')
        bbox = np.array([0, 0, det_shape[0], det_shape[1]])
        cropped_y = slice(0, det_shape[0])
        cropped_x = slice(0, det_shape[1])

    beam_mask = get_dark_field_mask(
        filtered_for_beam, bbox,
        cropped_x=cropped_x, cropped_y=cropped_y,
    )
    com_mask = np.logical_or(beam_mask, pixel_mask)   # exclude beam + dead pixels

    # ------------------------------------------------------------------
    # 4. Centre-of-mass map
    # ------------------------------------------------------------------
    print('Computing centre-of-mass map ...')
    label   = (~com_mask).astype(int)
    com_map = np.zeros((*scan_shape, 2))
    n_total = scan_shape[0] * scan_shape[1]
    for i in range(scan_shape[0]):
        for j in range(scan_shape[1]):
            idx = i * scan_shape[1] + j
            if idx % max(1, n_total // 10) == 0:
                print(f'  {idx}/{n_total} frames', end='\r', flush=True)
            com_map[i, j] = centre_of_mass(
                frames[i, j].astype(float), labelled_region=label
            )
    print(f'  {n_total}/{n_total} frames')

    # Remove mean offset and apply quantitative scale
    com_map[..., 0] -= com_map[..., 0].mean()
    com_map[..., 1] -= com_map[..., 1].mean()
    com_map *= ds.com_scale

    dy = com_map[..., 0]
    dx = com_map[..., 1]

    # ------------------------------------------------------------------
    # 5. Outlier removal on gradient maps
    # ------------------------------------------------------------------
    print('Removing outliers from gradient maps ...')
    dx, _ = median_subtraction(dx)
    dy, _ = median_subtraction(dy)

    # ------------------------------------------------------------------
    # 6. Gradient norm
    # ------------------------------------------------------------------
    grad_norm = phase_grad_norm(dx, dy)
    grad_norm = np.nan_to_num(grad_norm, nan=0.0)
    print(f'  gradient norm range: {grad_norm.min():.4f} – {grad_norm.max():.4f}')

    # ------------------------------------------------------------------
    # 7. Phase retrieval
    # ------------------------------------------------------------------
    phase = None
    if not args.no_phase:
        print(f'Retrieving phase using {args.method!r} method ...')
        phase = phase_retrieval(
            dx, dy,
            calX=ds.scan_step_x,
            calY=ds.scan_step_y,
            method=args.method,
        )
        print(f'  phase range: {phase.min():.4f} – {phase.max():.4f} rad')

    # ------------------------------------------------------------------
    # 8. Save results
    # ------------------------------------------------------------------
    from core.io_output import save_hdf5, save_images

    if args.output is not None:
        base = Path(args.output)
    else:
        stem = Path(args.nexus_file).stem
        base = Path(args.nexus_file).parent / f'{stem}_dpc'

    formats = [f.lower() for f in args.format]
    mask_mode = 'full' if args.full_mask else 'quick'

    common = dict(
        dx=dx, dy=dy, grad_norm=grad_norm,
        phase=phase, pixel_mask=pixel_mask,
    )
    image_common = dict(
        dx=dx, dy=dy, grad_norm=grad_norm,
        phase=phase,
    )

    if 'hdf5' in formats:
        out = base.with_suffix('.nxs')
        print(f'Saving HDF5 → {out}')
        save_hdf5(
            out,
            **common,
            scan_step_x=ds.scan_step_x,
            scan_step_y=ds.scan_step_y,
            beam_energy=ds.beam_energy,
            detector_distance=ds.detector_distance,
            pixel_size=ds.pixel_size,
            method=args.method,
            crop_size=args.crop_size,
            mask_mode=mask_mode,
        )

    for fmt in ('png', 'tif'):
        if fmt in formats:
            print(f'Saving {fmt.upper()} images → {base}_*.{fmt}')
            saved = save_images(base, fmt, **image_common)
            for p_ in saved:
                print(f'  {p_}')

    # ------------------------------------------------------------------
    # 9. Debug figure
    # ------------------------------------------------------------------
    if args.debug:
        from core.visualisation import save_debug
        save_debug(
            base,
            sample_frame      = sample_frame,
            filtered_frame    = filtered_for_beam,
            pixel_mask        = pixel_mask,
            beam_mask         = beam_mask,
            com_mask          = com_mask,
            bbox              = bbox,
            cropped_x         = cropped_x,
            cropped_y         = cropped_y,
            com_map           = com_map,
            dx                = dx,
            dy                = dy,
            grad_norm         = grad_norm,
            phase             = phase,
            scan_step_x       = ds.scan_step_x,
            scan_step_y       = ds.scan_step_y,
        )

    print('Done.')


if __name__ == '__main__':
    run(parse_args())
