import numpy as np
from scipy.special import erfinv
from scipy.ndimage import generic_filter

# Consistent with a Gaussian distribution: K = 1 / (sqrt(2) * erfinv(0.5))
_K = 1 / (np.sqrt(2) * erfinv(0.5))


def _window_slice_1d(centre, ws, size):
    '''Return a slice for a 1D window centred at index, capped at array edges.'''
    hw = ws // 2
    t1 = max(0, centre - hw)
    t2 = min(size, centre + (ws - hw))
    return slice(t1, t2)


def _window_slice_2d(r, c, ws, nrows, ncols):
    '''Return row and column slices for a 2D window, capped at array edges.'''
    hw = ws // 2
    r1 = max(0, r - hw)
    r2 = min(nrows, r + (ws - hw))
    c1 = max(0, c - hw)
    c2 = min(ncols, c + (ws - hw))
    return slice(r1, r2), slice(c1, c2)


def _MAD_scale_estimator(values, median):
    '''MAD-based scale estimate consistent with Gaussian standard deviation.

    Parameters
    ----------
    values : ndarray
        data values in the local window
    median : float
        median of values

    Returns
    -------
    Sk : float
        K * MAD, where K normalises for a Gaussian distribution
    '''
    MAD = np.median(np.abs(values - median))
    return _K * MAD


def _hampel_2d(img, t=5, size=3):
    '''2D Hampel filter.

    Parameters
    ----------
    img : ndarray
        2D input image
    t : float, optional
        threshold multiplier. Default 5.
    size : int, optional
        total window size. Default 3.

    Returns
    -------
    imgc : ndarray
        filtered copy of img
    invalid : ndarray of int32
        1 where a pixel was replaced, 0 elsewhere
    '''
    nrows, ncols = img.shape
    invalid = np.zeros(img.shape, dtype=np.int32)
    imgc = img.copy()

    for r in range(nrows):
        for c in range(ncols):
            sr, sc = _window_slice_2d(r, c, size, nrows, ncols)
            region = img[sr, sc].ravel()
            mk = np.median(region)
            Sk = _MAD_scale_estimator(region, mk)
            if np.abs(img[r, c] - mk) > t * Sk:
                imgc[r, c] = mk
                invalid[r, c] = 1

    return imgc, invalid


def _hampel_1d(y, t=5, size=3):
    '''1D Hampel filter.

    Parameters
    ----------
    y : ndarray
        1D input array
    t : float, optional
        threshold multiplier. Default 5.
    size : int, optional
        total window size. Default 3.

    Returns
    -------
    yc : ndarray
        filtered copy of y
    invalid : ndarray of int32
        1 where a value was replaced, 0 elsewhere
    '''
    n = len(y)
    invalid = np.zeros(n, dtype=np.int32)
    yc = y.copy()

    for centre in range(n):
        st = _window_slice_1d(centre, size, n)
        region = y[st]
        mk = np.median(region)
        Sk = _MAD_scale_estimator(region, mk)
        if np.abs(y[centre] - mk) > t * Sk:
            yc[centre] = mk
            invalid[centre] = 1

    return yc, invalid


def hampel(data, t=5, size=3):
    '''Hampel filter to remove outliers.

    Pearson, R.K., Neuvo, Y., Astola, J. et al.
    Generalized Hampel Filters. EURASIP J. Adv. Signal Process. 2016, 87 (2016)

    Parameters
    ----------
    data : array-like
        1D series or 2D image.
    t : float, optional
        Tuning parameter; smaller values are more aggressive. Default 5.
    size : int, optional
        Total size of the moving window. Default 3.

    Returns
    -------
    ret : ndarray
        Filtered copy of data (not in-place).
    invalid : ndarray of bool
        True where values were replaced.
    '''
    data = np.asarray(data, dtype=float)
    size = int(size)

    if data.ndim == 2:
        ret, invalid = _hampel_2d(data, t, size)
    elif data.ndim == 1:
        ret, invalid = _hampel_1d(data, t, size)
    else:
        raise ValueError(f'hampel supports 1D and 2D arrays; got ndim={data.ndim}')

    return ret, invalid.astype(bool)


def sure_dead_pixel(stack, tol=0):
    '''Find dead pixels by comparing values across a stack of data.

    Parameters
    ----------
    stack : array-like
        Stack of 1D or 2D data.
    tol : float
        Tolerance for considering pixels identical across the stack. Default 0.

    Returns
    -------
    dead_mask : ndarray of bool
        True at positions of sure dead pixels.
    '''
    stack = np.asarray(stack)
    if stack.ndim == 1:
        raise ValueError('It needs to be a stack of images.')
    elif stack.ndim == 2:
        dead_mask = (np.abs(np.diff(stack, axis=1)) <= tol).all(axis=1)
    elif stack.ndim == 3:
        dead_mask = (np.abs(np.diff(stack, axis=2)) <= tol).all(axis=2)

    return dead_mask


def median_subtraction(data, factor=1, size=3, sure_dead=None):
    '''Outlier removal by comparing data to a median-filtered version.

    Parameters
    ----------
    data : array-like
        1D series or 2D image.
    factor : float, optional
        Multiplier on the filtered image's standard deviation to set the
        threshold. Smaller is more aggressive. Default 1.
    size : int, optional
        Moving window size. Default 3.
    sure_dead : ndarray, optional
        Boolean mask of known dead pixel positions.

    Returns
    -------
    ret : ndarray
        Filtered copy (not in-place).
    invalid : ndarray of bool
        True where values were replaced.
    '''
    data = np.asarray(data, dtype=float)
    # t=0 in Hampel reduces to a plain median filter
    filtered, _ = hampel(data, t=0, size=size)
    thresh = factor * np.std(filtered)

    invalid = np.abs(data - filtered) > thresh
    if sure_dead is not None:
        invalid = invalid | sure_dead

    ret = data.copy()
    ret[invalid] = filtered[invalid]

    return ret, invalid
