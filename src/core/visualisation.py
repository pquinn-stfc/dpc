'''Debug visualisation for the DPC processing pipeline.

Generates a multi-page diagnostic figure saved to
``<output_base>_debug.png`` showing:

Page 1 — Detector / masking
  - Raw sample frame (log scale)
  - Pixel mask (dead / hot pixels)
  - Filtered frame used for beam detection
  - Dark-field / beam mask
  - CoM mask (beam + dead pixels combined)
  - Masked frame as seen by the CoM calculation

Page 2 — Scan results
  - Phase gradient dx
  - Phase gradient dy
  - Gradient norm
  - Retrieved phase (if available)
  - CoM shift x (raw, before scaling)
  - CoM shift y (raw, before scaling)
'''

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def save_debug(
    base: Path,
    *,
    sample_frame: np.ndarray,
    filtered_frame: np.ndarray,
    pixel_mask: np.ndarray,
    beam_mask: np.ndarray,
    com_mask: np.ndarray,
    bbox,
    cropped_x: slice,
    cropped_y: slice,
    com_map: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    grad_norm: np.ndarray,
    phase: Optional[np.ndarray] = None,
    scan_step_x: float = 1.0,
    scan_step_y: float = 1.0,
):
    '''Generate and save the full debug figure.

    Parameters
    ----------
    base : Path
        Output base path (without extension).  Saves to ``base_debug.png``.
    sample_frame : ndarray
        Raw detector frame (first scan point).
    filtered_frame : ndarray
        Hampel-filtered frame used for beam detection.
    pixel_mask : ndarray of bool
        Bad-pixel mask (True = bad).
    beam_mask : ndarray of bool
        Dark-field / beam region mask.
    com_mask : ndarray of bool
        Combined mask used for CoM (beam + dead pixels).
    bbox : array-like
        Beam bounding box (lowy, lowx, highy, highx).
    cropped_x, cropped_y : slice
        Detector crop slices.
    com_map : ndarray, shape (scan_y, scan_x, 2)
        Raw CoM map in detector pixels ([...,0]=y, [...,1]=x).
    dx, dy : ndarray
        Phase gradients after scaling and outlier removal.
    grad_norm : ndarray
        Phase gradient norm.
    phase : ndarray, optional
        Retrieved phase.
    scan_step_x, scan_step_y : float
        Scan step sizes in metres (for axis labels).
    '''
    out = base.parent / f'{base.stem}_debug.png'
    print(f'Saving debug figure → {out}')

    fig = plt.figure(figsize=(18, 22))
    fig.suptitle(f'DPC debug — {base.stem}', fontsize=14, fontweight='bold')

    # -------------------------------------------------------------------
    # Row 1 — Raw and filtered detector frames
    # -------------------------------------------------------------------
    ax1 = fig.add_subplot(4, 3, 1)
    _show_frame(ax1, sample_frame, 'Raw frame (log scale)', log=True)
    _draw_bbox(ax1, bbox, cropped_x, cropped_y)

    ax2 = fig.add_subplot(4, 3, 2)
    _show_frame(ax2, filtered_frame, 'Filtered frame (Hampel)', log=True)
    _draw_bbox(ax2, bbox, cropped_x, cropped_y)

    ax3 = fig.add_subplot(4, 3, 3)
    _show_mask(ax3, pixel_mask, 'Pixel mask\n(bad / dead pixels)')

    # -------------------------------------------------------------------
    # Row 2 — Masks
    # -------------------------------------------------------------------
    ax4 = fig.add_subplot(4, 3, 4)
    _show_mask(ax4, beam_mask, 'Beam / dark-field mask')
    _draw_bbox(ax4, bbox, cropped_x, cropped_y)

    ax5 = fig.add_subplot(4, 3, 5)
    _show_mask(ax5, com_mask, 'CoM mask\n(beam + dead pixels)')

    ax6 = fig.add_subplot(4, 3, 6)
    # Show the frame with the CoM mask overlaid as a red tint
    _show_masked_frame(ax6, sample_frame, com_mask,
                       'Masked frame\n(white = included in CoM)',
                       cropped_x, cropped_y)

    # -------------------------------------------------------------------
    # Row 3 — CoM map (raw pixel shifts)
    # -------------------------------------------------------------------
    nr, nc = com_map.shape[:2]
    x_nm = np.arange(nc) * scan_step_x * 1e9
    y_nm = np.arange(nr) * scan_step_y * 1e9
    extent_scan = [x_nm[0], x_nm[-1], y_nm[-1], y_nm[0]]

    ax7 = fig.add_subplot(4, 3, 7)
    _show_map(ax7, com_map[..., 1], 'CoM shift x (pixels)',
              extent_scan, cmap='RdBu_r', symmetric=True)

    ax8 = fig.add_subplot(4, 3, 8)
    _show_map(ax8, com_map[..., 0], 'CoM shift y (pixels)',
              extent_scan, cmap='RdBu_r', symmetric=True)

    ax9 = fig.add_subplot(4, 3, 9)
    _show_map(ax9, grad_norm, 'Gradient norm',
              extent_scan, cmap='viridis', symmetric=False)

    # -------------------------------------------------------------------
    # Row 4 — Phase gradients and phase
    # -------------------------------------------------------------------
    ax10 = fig.add_subplot(4, 3, 10)
    _show_map(ax10, dx, 'Phase gradient dx (rad/m)',
              extent_scan, cmap='RdBu_r', symmetric=True)

    ax11 = fig.add_subplot(4, 3, 11)
    _show_map(ax11, dy, 'Phase gradient dy (rad/m)',
              extent_scan, cmap='RdBu_r', symmetric=True)

    ax12 = fig.add_subplot(4, 3, 12)
    if phase is not None:
        _show_map(ax12, phase, 'Phase (rad)',
                  extent_scan, cmap='RdBu_r', symmetric=True)
    else:
        ax12.text(0.5, 0.5, 'Phase retrieval\nnot run',
                  ha='center', va='center', transform=ax12.transAxes,
                  fontsize=12, color='grey')
        ax12.set_title('Phase (rad)')
        ax12.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _show_frame(ax, frame, title, log=False):
    data = np.clip(frame, 1, None) if log else frame
    norm = LogNorm(vmin=data[data > 0].min(), vmax=data.max()) if log else None
    im = ax.imshow(data, cmap='gray', norm=norm, origin='upper', aspect='equal')
    ax.set_title(title, fontsize=9)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _show_mask(ax, mask, title):
    ax.imshow(mask, cmap='RdGy_r', vmin=0, vmax=1,
              origin='upper', aspect='equal')
    ax.set_title(title, fontsize=9)
    ax.axis('off')
    bad  = mpatches.Patch(color='red',  label='masked')
    good = mpatches.Patch(color='white', label='valid')
    ax.legend(handles=[good, bad], loc='lower right',
              fontsize=7, framealpha=0.7)


def _show_masked_frame(ax, frame, mask, title, crop_x, crop_y):
    '''Show the frame cropped to the beam region with masked pixels highlighted.'''
    crop = frame[crop_y, crop_x].copy()
    m_crop = mask[crop_y, crop_x]

    # Normalise crop to 0-1 for display
    lo, hi = np.percentile(crop, (1, 99))
    norm_crop = np.clip((crop - lo) / max(hi - lo, 1e-9), 0, 1)

    # RGB: grey background, red overlay where masked
    rgb = np.stack([norm_crop, norm_crop, norm_crop], axis=-1)
    rgb[m_crop] = [1.0, 0.2, 0.2]   # red where excluded

    ax.imshow(rgb, origin='upper', aspect='equal')
    ax.set_title(title, fontsize=9)
    ax.axis('off')


def _show_map(ax, data, title, extent, cmap, symmetric):
    vmax = np.percentile(np.abs(data), 99)
    kw = dict(extent=extent, origin='upper', cmap=cmap,
              aspect='equal', interpolation='nearest')
    if symmetric:
        im = ax.imshow(data, vmin=-vmax, vmax=vmax, **kw)
    else:
        im = ax.imshow(data, **kw)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('x (nm)', fontsize=7)
    ax.set_ylabel('y (nm)', fontsize=7)
    ax.tick_params(labelsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _draw_bbox(ax, bbox, crop_x, crop_y):
    '''Overlay the beam bounding box and crop rectangle on a detector image.'''
    lowy, lowx, highy, highx = bbox
    # beam bounding box — yellow
    rect_beam = mpatches.Rectangle(
        (lowx, lowy), highx - lowx, highy - lowy,
        linewidth=1.5, edgecolor='yellow', facecolor='none',
        label='beam bbox',
    )
    ax.add_patch(rect_beam)
    # crop region — cyan
    rect_crop = mpatches.Rectangle(
        (crop_x.start, crop_y.start),
        crop_x.stop - crop_x.start,
        crop_y.stop - crop_y.start,
        linewidth=1.5, edgecolor='cyan', facecolor='none',
        linestyle='--', label='crop',
    )
    ax.add_patch(rect_crop)
    ax.legend(loc='upper right', fontsize=7, framealpha=0.7)
