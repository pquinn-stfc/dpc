import numpy as np
from outlier_removal_tools import hampel


# ---------------------------------------------------------------------------
# Geometric masks
# ---------------------------------------------------------------------------

def create_circular_mask(h, w, center=None, radius=None):
    '''Create a circular boolean mask.

    Parameters
    ----------
    h : int
        Image height in pixels.
    w : int
        Image width in pixels.
    center : tuple of float, optional
        (col, row) centre of the circle.  Defaults to the image centre.
    radius : float, optional
        Radius in pixels.  Defaults to the largest circle that fits inside
        the image given the centre.  Clamped so the circle does not exceed
        the image boundary.

    Returns
    -------
    mask : ndarray of bool, shape (h, w)
        True inside the circle.
    '''
    if center is None:
        center = (w / 2, h / 2)

    cx, cy = center
    if radius is None:
        radius = min(cx, cy, w - cx, h - cy)

    # clamp radius so the circle stays within the image
    radius = min(radius, cx, cy, w - cx, h - cy)

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    return dist <= radius


def create_circular_radial_masks(h, w, center=None, radius=None,
                                 no_of_segments=8):
    '''Divide a circular region into equal angular segments.

    Parameters
    ----------
    h : int
        Image height in pixels.
    w : int
        Image width in pixels.
    center : tuple of float, optional
        (col, row) centre.  Defaults to the image centre.
    radius : float, optional
        If given, each segment mask is further restricted to pixels within
        this radius of the centre.
    no_of_segments : int, optional
        Number of equal angular segments.  Default 8.

    Returns
    -------
    mask_list : list of ndarray of bool
        One boolean mask per segment, each of shape (h, w).
    '''
    if center is None:
        center = (w / 2, h / 2)

    cx, cy = center
    Y, X = np.ogrid[:h, :w]
    phi = np.arctan2(Y - cy, X - cx)
    theta = (phi / np.pi + 1) * 180.0          # map to [0, 360)

    boundaries = [i * 360.0 / no_of_segments for i in range(no_of_segments + 1)]

    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    mask_list = []
    for i in range(no_of_segments):
        lo, hi = boundaries[i], boundaries[i + 1]
        segment = np.logical_and(theta >= lo, theta < hi)
        if radius is not None:
            segment = np.logical_and(segment, dist < radius)
        mask_list.append(segment)

    return mask_list


def generate_butterworth_mask(rows, cols, n, d, flip=False):
    '''Generate a 2D Butterworth low-pass (or high-pass) filter mask.

    Parameters
    ----------
    rows : int
        Number of rows in the filter.
    cols : int
        Number of columns in the filter.
    n : int
        Filter order.  Higher values give a sharper roll-off.
    d : float
        Cut-off radius in pixels from the centre.
    flip : bool, optional
        If True return 1 - filter (high-pass).  Default False.

    Returns
    -------
    output : ndarray of float32, shape (rows, cols)
    '''
    cr = rows / 2
    cc = cols / 2
    r_idx, c_idx = np.indices((rows, cols), dtype=float)
    dist = np.sqrt((r_idx - cr) ** 2 + (c_idx - cc) ** 2)
    output = 1.0 / (1.0 + (dist / d) ** (2 * n))
    if flip:
        output = 1.0 - output
    return output.astype(np.float32)


# ---------------------------------------------------------------------------
# Pixel masking pipeline
# ---------------------------------------------------------------------------

def build_pixel_mask(data_im, quick=True, stack=None,
                     hampel_t_quick=6.0, hampel_size_quick=5,
                     hampel_t_beam=20.0, hampel_size_beam=7,
                     hampel_t_full=15.0, hampel_size_full=9,
                     hot_pixel_threshold=20.0):
    '''Build a bad-pixel mask for a detector frame.

    Two modes are available:

    * **quick** — applies a Hampel filter to a single representative frame.
      Fast but less reliable; intended for real-time use.
    * **full** — computes pixel-wise mean and standard deviation over a stack
      of frames and identifies bad pixels from the normalised noise image
      ``std / sqrt(mean)``.  Requires ``stack`` to be provided.

    Parameters
    ----------
    data_im : ndarray, shape (det_y, det_x)
        A single representative detector frame (used in both modes).
    quick : bool, optional
        Use quick mode.  Default True.
    stack : ndarray, shape (N, det_y, det_x), optional
        Full stack of detector frames.  Required when ``quick=False``.
    hampel_t_quick : float
        Hampel threshold for quick bad-pixel detection.
    hampel_size_quick : int
        Hampel window size for quick bad-pixel detection.
    hampel_t_beam : float
        Hampel threshold used when smoothing the frame for beam finding.
    hampel_size_beam : int
        Hampel window size for beam-finding smoothing.
    hampel_t_full : float
        Hampel threshold applied to the normalised noise image in full mode.
    hampel_size_full : int
        Hampel window size in full mode.
    hot_pixel_threshold : float
        Pixels with ``std / sqrt(mean)`` above this value are flagged as hot
        in full mode.

    Returns
    -------
    pixel_mask : ndarray of bool
        True where a pixel is bad.
    filtered_for_beam : ndarray
        Smoothed version of ``data_im`` suitable for beam-centre detection.
    '''
    nr, nc = data_im.shape

    # cross-shaped mask covering the central row/column (Merlin chip gap)
    dead_pixels = np.zeros((nr, nc), dtype=bool)
    mid_r, mid_c = nr // 2 - 1, nc // 2 - 1
    dead_pixels[mid_r - 2:mid_r + 2, :] = True
    dead_pixels[:, mid_c - 2:mid_c + 2] = True

    filtered_for_beam, _ = hampel(data_im, t=hampel_t_beam,
                                  size=hampel_size_beam)

    if quick:
        filtered, invalid_mask = hampel(data_im, t=hampel_t_quick,
                                        size=hampel_size_quick)
        pixel_mask = np.logical_or(dead_pixels, invalid_mask)
    else:
        if stack is None:
            raise ValueError("stack must be provided when quick=False")

        stack = np.asarray(stack)
        mean_image = stack.mean(axis=0)
        std_image = stack.std(axis=0)

        # zero-variance pixels that are bright are certainly dead
        dead_pixels_std = np.logical_and(std_image == 0.0, mean_image > 1000.0)
        dead_pixels = np.logical_or(dead_pixels, dead_pixels_std)

        # hot pixels: anomalously large normalised noise
        with np.errstate(divide='ignore', invalid='ignore'):
            norm_image = np.where(mean_image > 0, std_image / np.sqrt(mean_image), 0.0)
        hot_pixels = norm_image > hot_pixel_threshold
        dead_pixels = np.logical_or(dead_pixels, hot_pixels)

        _, invalid_mask = hampel(norm_image, t=hampel_t_full,
                                 size=hampel_size_full)
        dead_pixels = np.logical_or(dead_pixels, invalid_mask)
        pixel_mask = dead_pixels

    return pixel_mask, filtered_for_beam


# ---------------------------------------------------------------------------
# Derived signal maps
# ---------------------------------------------------------------------------

def absorption_signal(frame, mask=None):
    '''Compute the log-absorption signal for a single detector frame.

    Parameters
    ----------
    frame : ndarray
        Detector frame.
    mask : ndarray of bool, optional
        True for pixels to include.  If None all pixels are used.

    Returns
    -------
    float
        log(1 / sum) of the unmasked pixels, or 0 if the sum is negligible.
    '''
    if mask is not None:
        data = np.ma.array(frame, mask=~mask)
    else:
        data = np.ma.array(frame)
    data = np.ma.masked_invalid(data)
    s = float(data.sum())
    if s <= 1.0e-8 or 1.0 / s <= 1.0e-8:
        return 0.0
    return np.log(1.0 / s)


def scatter_signal(frame, mask):
    '''Sum detector counts outside the masked (beam) region.

    Parameters
    ----------
    frame : ndarray
        Detector frame.
    mask : ndarray of bool
        True for pixels to *exclude* (i.e. the beam region).

    Returns
    -------
    float
        Sum of counts in the unmasked pixels.
    '''
    return float(np.ma.array(frame, mask=mask).sum())
