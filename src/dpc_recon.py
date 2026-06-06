import numpy as np
from scipy.fft import dctn, idctn, dct, idct
import matplotlib.pyplot as plt

def k_grid(nc, nr, calX=1, calY=1):
    '''Build 2D Fourier-space frequency grids

    Parameters
    ----------
    nc : int
        number of columns (x dimension)
    nr : int
        number of rows (y dimension)
    calX : float, optional
        pixel size in x (real-space calibration). Default 1.
    calY : float, optional
        pixel size in y (real-space calibration). Default 1.

    Returns
    -------
    kx_grid : ndarray
        2D array of x spatial frequencies in rad/unit
    ky_grid : ndarray
        2D array of y spatial frequencies in rad/unit
    '''
    # nc is x, nr is y
    # equivalent to fftshift(fftfreq(nc))
    colrng = np.arange(nc) - nc//2
    rowrng = np.arange(nr) - nr//2
    kx = 2*np.pi*colrng / (nc*calX)
    ky = 2*np.pi*rowrng / (nr*calY)
    kx_grid, ky_grid = np.meshgrid(kx, ky)

    return kx_grid, ky_grid

def arnison(dx, dy, calX, calY, approx=False):
    '''Phase retrieval using the Arnison method

    Parameters
    ----------
    dx : ndarray
        phase gradient in x
    dy : ndarray
        phase gradient in y
    calX : float
        pixel size in x
    calY : float
        pixel size in y
    approx : bool, optional
        use the continuous approximation of the denominator. Default False.

    Returns
    -------
    retrieved : ndarray
        complex array of the retrieved phase
    '''

    nc, nr = dx.shape[1], dx.shape[0]

    kx_grid, ky_grid = k_grid(nc, nr)

    gxy = dx + 1j*dy

    # handle the division by zero in the central pixel
    # set the undefined/infinity pixel to 0
    with np.errstate(divide='ignore', invalid='ignore'):
        numerator = np.fft.fftshift(np.fft.fft2(gxy))
        if approx:
            denominator = 4*np.pi*1j*calX*np.sqrt((kx_grid**2+ky_grid**2)*
                                      np.exp(1j*np.arctan2(ky_grid, kx_grid)))
        else:
            denominator = 2j*(np.sin(2*np.pi*calX*kx_grid) +
                              1j*np.sin(2*np.pi*calY*ky_grid))

        res = numerator / denominator

    res = np.nan_to_num(res, nan=0, posinf=0, neginf=0)

    retrieved = np.fft.ifft2(np.fft.ifftshift(res))

    return retrieved

def kottler(dx, dy):
    '''Phase retrieval using the Kottler method

    Parameters
    ----------
    dx : ndarray
        phase gradient in x
    dy : ndarray
        phase gradient in y

    Returns
    -------
    retrieved : ndarray
        complex array of the retrieved phase
    '''
    nc, nr = dx.shape[1], dx.shape[0]

    kx_grid, ky_grid = k_grid(nc, nr)

    # Fourier transform of the centre of mass
    gxy = dx + 1j*dy
    numerator = np.fft.fftshift(np.fft.fft2(gxy))
    denominator = 2*np.pi*1j*(kx_grid + 1j*ky_grid)

    # handle the division by zero in the central pixel
    # set the undefined/infinity pixel to 0
    with np.errstate(divide='ignore', invalid='ignore'):
        res = numerator / denominator
    res = np.nan_to_num(res, nan=0, posinf=0, neginf=0)

    retrieved = np.fft.ifft2(np.fft.ifftshift(res))

    return retrieved

def frankt(dx, dy, calX, calY, pad_width=0, w=0.5):
    '''Phase retrieval using the Frankot-Chellappa method

    Parameters
    ----------
    dx : ndarray
        phase gradient in x
    dy : ndarray
        phase gradient in y
    calX : float
        pixel size in x
    calY : float
        pixel size in y
    pad_width : float, optional
        fractional padding applied to each side to reduce boundary
        discontinuities. 0 disables padding. Default 0.
    w : float, optional
        weighting toward the y gradient (1-w applied to x). Default 0.5.

    Returns
    -------
    phase : ndarray
        complex array of the retrieved phase
    '''
    # Frankt-Chellappa
#    if negate:
#        dx = np.asarray(-dx)
#    else:
#        dx = np.asarray(dx)
#    dy = np.asarray(dy)

    # pad the gradient to avoid discontinuity at boundary
    if pad_width:
        pad_col = dx.shape[1] * pad_width
        pad_row = dx.shape[0] * pad_width
        dx_pad = np.pad(dx, ((pad_row, pad_row), (pad_col, pad_col)),
                        mode='constant', constant_values=0)
        dy_pad = np.pad(dy, ((pad_row, pad_row), (pad_col, pad_col)),
                        mode='constant', constant_values=0)
    else:
        dx_pad, dy_pad = dx, dy
    nr, nc = dx_pad.shape[0], dx_pad.shape[1]

    # construct the Fourier coordinate
    kx_grid, ky_grid = k_grid(nc, nr, calX=calX, calY=calY)
#    colrng = np.arange(nc) - nc//2
#    rowrng = np.arange(nr) - nr//2
#    kx = 2*np.pi*colrng / (nc*calX)
#    ky = 2*np.pi*rowrng / (nr*calY)
#    kx_grid, ky_grid = np.meshgrid(kx, ky)

    # Fourier transform of the (padded) centre of mass
    fx = np.fft.fftshift(np.fft.fft2(dx_pad))
    fy = np.fft.fftshift(np.fft.fft2(dy_pad))

    # get the numerator and denominator of the phase retrieval equation
    wx, wy = 1-w, w
    numerator = -1j*(wx*kx_grid*fx + wy*ky_grid*fy)
    denominator = wx*kx_grid**2 + wy*ky_grid**2

    # handle the division by zero in the central pixel
    with np.errstate(divide='ignore', invalid='ignore'):
        res = numerator / denominator

    # set the undefined/infinity pixel to 0
    res = np.nan_to_num(res, nan=0, posinf=0, neginf=0)
    retrieved = np.fft.ifft2(np.fft.ifftshift(res))

    if pad_width:
        phase = retrieved[pad_row:-pad_row, pad_col:-pad_col]
    else:
        phase = retrieved

    return phase




def _zero_pad_2d(data):
    '''Zero-pad a 2D array to the next power of two (capped at 512).

    Parameters
    ----------
    data : ndarray
        2D real-valued array.

    Returns
    -------
    datapad : ndarray
        Zero-padded array of shape (order, order).
    '''
    n, m = data.shape
    order = int(2 ** np.ceil(np.log2(2 * max(n, m))))
    order = min(512, order)
    datapad = np.zeros((order, order), dtype=float)
    datapad[:n, :m] = data
    return datapad


def lazic(dx, dy, calX=1, calY=1, high_pass_filter=False, highpass=0.0,
          d=5, n=1, mirroring=True, anti=True, zeropad=True):
    '''Phase retrieval using the Lazic method with optional Butterworth high-pass filter.

    Implements the weighted Fourier integration from Lazic et al. with
    optional boundary-artefact reduction via mirroring and zero-padding.

    Parameters
    ----------
    dx : ndarray
        Phase gradient in x.
    dy : ndarray
        Phase gradient in y.
    calX : float, optional
        Pixel calibration in x.  Default 1.
    calY : float, optional
        Pixel calibration in y.  Default 1.
    high_pass_filter : bool, optional
        Apply a Butterworth high-pass filter to the Fourier-transformed
        gradients before integration.  Default False.
    highpass : float, optional
        Additional Tikhonov-style regularisation parameter added to the
        denominator as ``highpass * max(k²)``.  Default 0.
    d : float, optional
        Cut-off radius for the Butterworth filter (pixels).  Default 5.
    n : int, optional
        Butterworth filter order.  Default 1.
    mirroring : bool, optional
        Mirror the gradients to reduce boundary discontinuities.  Default True.
    anti : bool, optional
        Anti-symmetric mirroring mode.  Default True.
    zeropad : bool, optional
        Zero-pad to the next power of two before FFT.  Default True.

    Returns
    -------
    retrieved : ndarray
        Complex array of the retrieved phase (take `.real`).

    References
    ----------
    Lazic, I. et al., 2016. Phase contrast STEM for thin samples: Integrated
    differential phase contrast. Ultramicroscopy 160, pp.265-280.
    '''
    from mask_tools import generate_butterworth_mask

    dx = dx.copy() - dx.mean()
    dy = dy.copy() - dy.mean()
    raw_shape = dx.shape

    if zeropad:
        dx = _zero_pad_2d(dx)
        dy = _zero_pad_2d(dy)

    if mirroring:
        Ax, Bx = dx, np.flip(dx, axis=1)
        Cx, Dx = np.flip(dx, axis=0), np.flip(dx)
        Ay, By = dy, np.flip(dy, axis=1)
        Cy, Dy = np.flip(dy, axis=0), np.flip(dy)
        if anti:
            dx = np.bmat([[Ax, Bx], [-Cx, -Dx]]).A
            dy = np.bmat([[Ay, -By], [Cy, -Dy]]).A
        else:
            dx = np.bmat([[Ax, Bx], [Cx, Dx]]).A
            dy = np.bmat([[Ay, By], [Cy, Dy]]).A

    nr, nc = dx.shape
    kx_grid, ky_grid = k_grid(nc, nr, calX=calX, calY=calY)
    k2 = kx_grid ** 2 + ky_grid ** 2
    k2[k2 == 0.0] = 1.0

    ft_dx = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(dx)))
    ft_dy = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(dy)))

    if high_pass_filter:
        mask = generate_butterworth_mask(nr, nc, n, d, flip=True)
        ft_dx *= mask
        ft_dy *= mask

    numerator = ft_dx * kx_grid + ft_dy * ky_grid
    denominator = 2 * np.pi * 1j * (k2 + highpass * k2.max())

    with np.errstate(divide='ignore', invalid='ignore'):
        res = numerator / denominator
    res = np.nan_to_num(res, nan=0, posinf=0, neginf=0)

    retrieved = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(res)))

    if mirroring:
        m, n_col = retrieved.shape
        retrieved = retrieved[:m // 2, :n_col // 2]
    if zeropad:
        retrieved = retrieved[:raw_shape[0], :raw_shape[1]]

    return retrieved


def _make_window_scs(height, width):
    '''Build the frequency-domain window for the SCS phase retrieval method.

    Parameters
    ----------
    height : int
    width : int

    Returns
    -------
    sin_u : ndarray
    sin_v : ndarray
    window : ndarray
        Complex integration kernel; DC component is zeroed.
    '''
    ulist = np.arange(width) / width
    vlist = np.arange(height) / height
    u, v = np.meshgrid(ulist, vlist)
    sin_u = np.sin(2 * np.pi * u)
    sin_v = np.sin(2 * np.pi * v)
    sin_u2 = np.sin(np.pi * u) ** 2
    sin_v2 = np.sin(np.pi * v) ** 2
    window = sin_u2 + sin_v2
    window[0, 0] = 1.0
    window = 1.0 / (4j * window)
    window[0, 0] = 0.0
    return sin_u, sin_v, window


def scs(dx, dy, pad=0, pad_mode="linear_ramp", correct_negative=True,
        window=None):
    '''Phase retrieval using the Simchony-Chellappa-Shao (SCS) method.

    Note: the DC component (mean value) of the reconstructed phase is
    undefined because the DC component of the FFT window is zero.

    Parameters
    ----------
    dx : ndarray
        Phase gradient in x.
    dy : ndarray
        Phase gradient in y.
    pad : int, optional
        Width of padding applied to each edge before reconstruction, removed
        afterwards.  Default 0 (no padding).
    pad_mode : str, optional
        Padding mode passed to ``numpy.pad``.  Default ``'linear_ramp'``.
    correct_negative : bool, optional
        If True, shift the result so the minimum value is non-negative.
        Default True.
    window : tuple of ndarray, optional
        Pre-computed (sin_u, sin_v, window) from ``_make_window_scs``.
        Computed from the (padded) array shape if not supplied.

    Returns
    -------
    retrieved : ndarray of float32
        Retrieved phase.

    References
    ----------
    Simchony, T., Chellappa, R. and Shao, M., 1990. Direct analytical methods
    for solving Poisson equations in computer vision problems. IEEE TPAMI,
    12(5), pp.435-446. https://doi.org/10.1109/34.55103
    '''
    if dx.shape != dy.shape:
        raise ValueError("dx and dy must have the same shape")

    if pad:
        dx = np.pad(dx, pad, mode=pad_mode)
        dy = np.pad(dy, pad, mode=pad_mode)

    height, width = dx.shape
    if window is None:
        sin_u, sin_v, win = _make_window_scs(height, width)
    else:
        if len(window) != 3:
            raise ValueError("window must be a 3-tuple (sin_u, sin_v, window)")
        sin_u, sin_v, win = window
        if win.shape != dx.shape:
            raise ValueError(
                f"window shape {win.shape} does not match array shape {dx.shape}"
            )

    fmat = sin_u * np.fft.fft2(dx) + sin_v * np.fft.fft2(dy)
    retrieved = np.real(np.fft.ifft2(fmat * win))

    if pad:
        retrieved = retrieved[pad:-pad, pad:-pad]

    if correct_negative:
        nmin = retrieved.min()
        if nmin < 0.0:
            retrieved -= 2 * nmin

    return retrieved.astype(np.float32)


def phase_grad_norm(dx, dy):
    '''Compute the norm of the phase gradient

    Parameters
    ----------
    dx : ndarray
        phase gradient in x
    dy : ndarray
        phase gradient in y

    Returns
    -------
    gradientNorm : ndarray
        Euclidean norm of the gradient after shifting both components
        to be non-negative
    '''

    mintoAdd = min(np.amin(dx),np.amin(dy))

    dxPlus = dx + mintoAdd
    dyPlus = dy + mintoAdd

    gradientNorm = np.sqrt(dxPlus**2 + dyPlus**2)

    return gradientNorm

def _get_laplacian(dx, dy, calX, calY):
    '''Compute the Laplacian from two phase gradient arrays

    Uses a finite-difference divergence with modified Neumann boundary
    conditions, after subtracting the mean to satisfy the compatibility
    criterion.

    Parameters
    ----------
    dx : ndarray
        phase gradient in x
    dy : ndarray
        phase gradient in y
    calX : float
        pixel size in x
    calY : float
        pixel size in y

    Returns
    -------
    laplacian : ndarray
        the Laplacian of the phase
    '''

    dx = dx.copy()
    dy = dy.copy()

    # compatibility criteria (integrate it becomes zero, include source and boundary)
    dx -= dx.mean()
    dy -= dy.mean()

    # construct Laplacian from 1st derivative, modified Neumann boundary condition
    ddx = np.roll(dx, -1, axis=1) - np.roll(dx, 1, axis=1)
    ddx[:, 0] = dx[:, 0] + dx[:, 1]
    ddx[:, -1] = -(dx[:, -2] + dx[:, -1])
    ddx /= 2*calX

    ddy = np.roll(dy, -1, axis=0) - np.roll(dy, 1, axis=0)
    ddy[0, :] = dy[0, :] + dy[1, :]
    ddy[-1, :] = -(dy[-2, :] + dy[-1, :])
    ddy /= 2*calY

    laplacian = ddx + ddy

    return laplacian

def ishizuka(dx, dy, calX, calY, approx=False):
    '''Phase retrieval using the Ishizuka method

    Uses the discrete cosine transform (DCT) to solve the phase from
    the Laplacian, with exact or approximate Fourier eigenvalues.

    Parameters
    ----------
    dx : ndarray
        phase gradient in x
    dy : ndarray
        phase gradient in y
    calX : float
        pixel size in x
    calY : float
        pixel size in y
    approx : bool, optional
        use continuous (approximate) eigenvalues instead of the exact
        discrete ones. Default False.

    Returns
    -------
    retrieved : ndarray
        real-valued retrieved phase
    '''

    nr, nc = dx.shape[0], dx.shape[1]

    # get the Laplacian
    L = _get_laplacian(dx, dy, calX, calY)

    # construct the Fourier coordinate
    # kx_grid, ky_grid = k_grid(nc, nr, calX=calX, calY=calY)
    # kx_grid /= 2*np.pi
    # ky_grid /= 2*np.pi
    # kx = kx_grid * np.pi * calX / 2
    # ky = ky_grid * np.pi * calY / 2

    # get denominator and numerator
    colrng = np.arange(nc) - nc//2
    rowrng = np.arange(nr) - nr//2
    if not approx:
        lambda_k = -4*np.sin((colrng*np.pi) / (2*nc))**2
        lambda_l = -4*np.sin((rowrng*np.pi) / (2*nr))**2
        numerator = np.fft.fftshift(dctn(L, type=2, norm='ortho'))*calX*calY
        denominator = lambda_k[None, :] + lambda_l[:, None]
    else:
        kx = colrng / (nc*calX)
        ky = rowrng / (nr*calY)
        numerator = np.fft.fftshift(dctn(L, type=2, norm='ortho'))
        denominator = -np.pi**2*(kx[None, :]**2 + ky[:, None]**2)


    # denominator = 4*np.sin(kx)[None,:]**2 + 4*np.sin(ky)[:,None]**2
    # denominator = np.pi**2 * (kx_grid**2 + ky_grid**2)

    with np.errstate(divide='ignore', invalid='ignore'):
        res = numerator / denominator
    res = np.nan_to_num(res, nan=0, posinf=0, neginf=0)

    retrieved = idctn(np.fft.ifftshift(res), type=2, norm='ortho')

    return retrieved.real

def phase_retrieval(dx, dy, calX=1.0, calY=1.0, method="kottler",
                    mirroring=False, mirror_flip=False):
    """Retrieve the phase from two orthogonal phase gradients.

    Parameters
    ----------
    dx : ndarray, shape (scan_y, scan_x)
        Phase gradient in x, in rad/m.
    dy : ndarray, shape (scan_y, scan_x)
        Phase gradient in y, in rad/m.
    calX : float, optional
        Scan step size in x (metres).  Used by the Arnison and Ishizuka
        methods.  Default 1.
    calY : float, optional
        Scan step size in y (metres).  Default 1.
    method : str, optional
        the formula to use: 'kottler'[1], 'arnison'[2], 'frankot'[3],
        'ishizuka', or 'scs'[4]. The default is 'kottler'.
    mirroring : bool, optional
        Mirror the phase gradients before the Fourier transform to reduce
        boundary artefacts. The default is False.
    mirror_flip : bool, optional
        Only active when ``mirroring`` is True. Flip the sign convention of
        the mirroring. If the retrieved phase looks wrong with mirroring
        enabled, try setting this to True. The default is False.

    Raises
    ------
    ValueError
        if the method is not recognised

    Returns
    -------
    retrieved : ndarray
        the phase retrieved.

    References
    ----------
    .. [1] Kottler, C., David, C., Pfeiffer, F. and Bunk, O., 2007. A
    two-directional approach for grating based differential phase contrast
    imaging using hard x-rays. Optics Express, 15(3), p.1175. (Equation 4)
    .. [2] Arnison, M., Larkin, K., Sheppard, C., Smith, N. and
    Cogswell, C., 2004. Linear phase imaging using differential
    interference contrast microscopy. Journal of Microscopy, 214(1),
    pp.7-12. (Equation 6)
    .. [3] Frankot, R. and Chellappa, R., 1988. A method for enforcing
    integrability in shape from shading algorithms.
    IEEE Transactions on Pattern Analysis and Machine Intelligence,
    10(4), pp.439-451. (Equation 21)
    .. [4] Simchony, T., Chellappa, R. and Shao, M., 1990. Direct analytical
    methods for solving Poisson equations in computer vision problems.
    IEEE TPAMI, 12(5), pp.435-446. https://doi.org/10.1109/34.55103
    """
    method = method.lower()
    if method not in ("kottler", "arnison", "frankot", "ishizuka", "scs"):
        raise ValueError(
            "Method '{}' not recognised. 'kottler', 'arnison', 'frankot',"
            " 'ishizuka' and 'scs' are available.".format(method)
        )

    # compatibility criteria — remove mean so integration has a unique solution
    dx = np.asarray(dx, dtype=float).copy()
    dy = np.asarray(dy, dtype=float).copy()
    dx -= dx.mean()
    dy -= dy.mean()

    # attempt to reduce boundary effect
    if mirroring and method != "ishizuka":
        Ax = dx
        Bx = np.flip(dx, axis=1)
        Cx = np.flip(dx, axis=0)
        Dx = np.flip(dx)

        Ay = dy
        By = np.flip(dy, axis=1)
        Cy = np.flip(dy, axis=0)
        Dy = np.flip(dy)

        # the -ve depends on the direction of derivatives
        if not mirror_flip:
            dx = np.bmat([[Ax, -Bx], [Cx, -Dx]]).A
            dy = np.bmat([[Ay, By], [-Cy, -Dy]]).A
        else:
            dx = np.bmat([[Ax, Bx], [-Cx, -Dx]]).A
            dy = np.bmat([[Ay, -By], [Cy, -Dy]]).A

    nc, nr = dx.shape[1], dx.shape[0]

    # construct Fourier-space grids
    kx = (2 * np.pi) * np.fft.fftshift(np.fft.fftfreq(nc))
    ky = (2 * np.pi) * np.fft.fftshift(np.fft.fftfreq(nr))
    kx_grid, ky_grid = np.meshgrid(kx, ky)

    if method == "kottler":
        # dx, dy are in rad/m; kx_grid is in rad/scan-pixel.
        # Multiply by scan step (m/pixel) to convert gradients to rad/pixel
        # so numerator and denominator carry consistent units → phase in rad.
        gxy = (dx * calX) + 1j * (dy * calY)
        numerator = np.fft.fftshift(np.fft.fft2(gxy))
        denominator = 2 * np.pi * 1j * (kx_grid + 1j * ky_grid)
    elif method == "arnison":
        gxy = dx + 1j * dy
        numerator = np.fft.fftshift(np.fft.fft2(gxy))
        denominator = 2j * (
            np.sin(2 * np.pi * calX * kx_grid)
            + 1j * np.sin(2 * np.pi * calY * ky_grid)
        )
    elif method == "frankot":
        kx_grid /= calX
        ky_grid /= calY
        fx = np.fft.fftshift(np.fft.fft2(dx))
        fy = np.fft.fftshift(np.fft.fft2(dy))
        # weights in x,y directins, currently hardcoded, but easy to extend if required
        wx, wy = 0.5, 0.5

        numerator = -1j * (wx * kx_grid * fx + wy * ky_grid * fy)
        denominator = wx * kx_grid ** 2 + wy * ky_grid ** 2
    elif method == "ishizuka":
        # get the Laplacian
        L = _get_laplacian(dx, dy, calX, calY)

        colrng = np.arange(nc) - nc//2
        rowrng = np.arange(nr) - nr//2
        lambda_k = -4*np.sin((colrng*np.pi) / (2*nc))**2
        lambda_l = -4*np.sin((rowrng*np.pi) / (2*nr))**2
        numerator = np.fft.fftshift(dctn(L, type=2, norm='ortho'))*calX*calY
        denominator = lambda_k[None, :] + lambda_l[:, None]
    elif method == "scs":
        return scs(dx, dy)

    # handle the division by zero in the central pixel
    # set the undefined/infinity pixel to 0
    with np.errstate(divide="ignore", invalid="ignore"):
        res = numerator / denominator
    res = np.nan_to_num(res, nan=0, posinf=0, neginf=0)

    if method != "ishizuka":
        retrieved = np.fft.ifft2(np.fft.ifftshift(res)).real
    else:
        retrieved = idctn(np.fft.ifftshift(res), type=2, norm='ortho').real

    # get 1/4 of the result if mirroring
    if mirroring and method != "ishizuka":
        M, N = retrieved.shape
        retrieved = retrieved[: M // 2, : N // 2]

    return retrieved


# ---------------------------------------------------------------------------
# NXData-aware convenience wrapper
# ---------------------------------------------------------------------------

def phase_retrieval_from(gradient_ds, method="kottler", mirroring=False,
                         mirror_flip=False):
    '''Retrieve phase from a gradient :class:`~io_schema.NXData`.

    Reads ``calX`` and ``calY`` directly from the navigation axes so they
    do not need to be supplied manually.  The dataset must have at least two
    navigation axes (y, x) and a signal axis of size 2 carrying [dy, dx]
    at each scan point.

    Parameters
    ----------
    gradient_ds : NXData
        Shape ``(scan_y, scan_x, 2)`` where ``data[..., 0]`` is ``dy`` and
        ``data[..., 1]`` is ``dx``.
    method : str
        Phase retrieval method.  Default ``'kottler'``.
    mirroring : bool
        Apply anti-symmetric mirroring.  Default False.
    mirror_flip : bool
        Flip mirroring sign convention.  Default False.

    Returns
    -------
    NXData
        Scalar map of the retrieved phase in rad with the same navigation
        axes as the input.
    '''
    from io_schema import AxisInfo, NXData

    nav = [ax for ax in gradient_ds.axes if ax.navigate]
    if len(nav) < 2:
        raise ValueError(
            "gradient_ds must have at least 2 navigation axes (y, x). "
            f"Got: {[a.name for a in nav]}"
        )
    calY = nav[0].step_size
    calX = nav[1].step_size

    dy = gradient_ds.data[..., 0]
    dx = gradient_ds.data[..., 1]

    phase = phase_retrieval(dx, dy, calX=calX, calY=calY,
                            method=method, mirroring=mirroring,
                            mirror_flip=mirror_flip)

    return NXData(
        data        = phase.real.astype(np.float32),
        axes        = [AxisInfo(ax.name, ax.values, navigate=True, units=ax.units)
                       for ax in nav],
        signal_type = "DPC_phase",
        metadata    = gradient_ds.metadata.copy(),
    )
