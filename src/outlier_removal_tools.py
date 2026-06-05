import numpy as np
from scipy.special import erfinv
from numba import njit

_K = 1/(np.sqrt(2) * erfinv(0.5))

@njit()
def _get_window_value_1d(centre, ws, data_size):
    '''Return the slice for a 1D window centred at index, capped at array edges

    Parameters
    ----------
    centre : int
        index of the central element
    ws : int
        total window size
    data_size : tuple
        shape of the 1D array (single-element tuple)

    Returns
    -------
    st : slice
        slice spanning the window
    '''

    if ws % 2 == 1:
        hw = (ws//2, ws//2)
    else:
        hw = (ws//2, ws//2 - 1)

    t1 = centre - hw[0]
    t2 = centre + hw[1] + 1
    if t1 < 0:
        t1 = 0
    if t2 > data_size[0]:
        t2 = data_size[0]

    st = slice(t1, t2)

    return st

@njit()
def _get_window_value_2d(centre, ws, data_size):
    '''Return row and column slices for a 2D window centred at a pixel, capped at array edges

    Parameters
    ----------
    centre : tuple of int
        (row, col) index of the central pixel
    ws : int
        total window size (same for both dimensions)
    data_size : tuple of int
        shape of the 2D array

    Returns
    -------
    sr : slice
        row slice spanning the window
    sc : slice
        column slice spanning the window
    '''

    if ws % 2 == 1:
        hw = (ws//2, ws//2)
    else:
        hw = (ws//2, ws//2 - 1)

    r1 = centre[0] - hw[0]
    r2 = centre[0] + hw[1] + 1
    c1 = centre[1] - hw[0]
    c2 = centre[1] + hw[1] + 1
    if r1 < 0:
        r1 = 0
    if r2 > data_size[0]:
        r2 = data_size[0]
    if c1 < 0:
        c1 = 0
    if c2 > data_size[1]:
        c2 = data_size[1]

    sr = slice(r1,r2)
    sc = slice(c1, c2)

    return sr, sc

@njit()
def _MAD_scale_estimator(values, median):
    '''Compute the MAD-based scale estimate (consistent with Gaussian std dev)

    Parameters
    ----------
    values : ndarray
        the data values in the local window
    median : float
        the median of values

    Returns
    -------
    Sk : float
        scale estimate: K * MAD, where K normalises for a Gaussian distribution
    '''
    MAD = np.median(np.abs(values - median))
    Sk = _K * MAD

    return Sk

@njit()
def _hampel_2d(img, t=5, size=3):
    '''Numba-accelerated 2D Hampel filter

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

    img_size = img.shape
    invalid = np.zeros(img_size, dtype=np.int32)
    imgc = img.copy()

    for r in range(img_size[0]):
        for c in range(img_size[1]):
            centre = (r, c)
            xk = img[centre]
            sr, sc = _get_window_value_2d(centre, size, img_size)
            region = img[sr,sc]

            mk = np.median(region)
            Sk = _MAD_scale_estimator(region, mk)

            if np.abs(xk - mk) > t*Sk:
                imgc[centre] = mk
                invalid[centre] = 1

    return imgc, invalid

@njit()
def _hampel_1d(y, t=5, size=3):
    '''Numba-accelerated 1D Hampel filter

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

    y_size = y.shape
    invalid = np.zeros(y_size, dtype=np.int32)
    yc = y.copy()

    for centre, xk in enumerate(yc):
        st = _get_window_value_1d(centre, size, y_size)
        region = y[st]

        mk = np.median(region)
        Sk = _MAD_scale_estimator(region, mk)

        if np.abs(xk - mk) > t*Sk:
            yc[centre] = mk
            invalid[centre] = 1

    return yc, invalid

def hampel(data, t=5, size=3):
    '''Hampel filter to remove outliers
    Pearson, R.K., Neuvo, Y., Astola, J. et al.
    Generalized Hampel Filters. EURASIP J. Adv. Signal Process. 2016, 87 (2016)

    Parameter
    ---------
    data : array-like
        can be a 1D series data or 2D image
    t : float, optional
        the tuning parameter of Hampel filter, the smaller the t, the more
        aggressive the filter is. Default to 5.
    size : int, optional
        the total size of the moving window. Default to 3.

    Returns
    -------
    ret : ndarray
        the filtered data. The filter is not in-place, a new filtered copy is
        returned
    invalid : ndarray
        a boolean array where the altered values' positions are True
    '''

    data = np.asarray(data)
    size = int(size)
    nd = data.ndim

    # delegate to respective numba optimised functions
    if nd == 2:
        ret, invalid = _hampel_2d(data, t, size)
    elif nd == 1:
        ret, invalid = _hampel_1d(data, t, size)
    invalid = invalid.astype(bool)

    return ret, invalid

def sure_dead_pixel(stack, tol=0):
    '''Find dead pixel by comparing data values across a stack of data

    Parameters
    ----------
    stack : array-like
        the stack of 1D or 2D data
    tol : float
        the tolerance of determining whether the pixels across the stack are
        the same. Default to 0.

    Returns
    -------
    dead_mask : ndarray
        a boolean array where the positions of sure dead pixels are True
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
    '''Outlier removal by subtracting median-filtered data with original one
    and replace them if larger than a threshold

    Parameters
    ----------
    data : array-like
        can be a 1D series data or 2D image
    factor : float, optional
        a factor that multiplies the median-filtered image's standard deviation
        to determine the threshold. The smaller the value, the more aggressive
        the removal is. Default to 1.
    size : int, optional
        the total size of the moving window. Default to 3.
    sure_dead : ndarray, optional
        a boolean array where the positions of sure dead pixels are True.
        Default as None.

    Returns
    -------
    ret : ndarray
        the filtered data. The filter is not in-place, a new filtered copy is
        returned
    invalid : ndarray
        a boolean array where the altered values' positions are True
    '''

    # filtere the data by median filter
    # t=0 in Hampel filter reduces to median filter
    filtered, _ = hampel(data, t=0, size=size)
    thresh = factor * np.std(filtered)

    # determine the positions of invalid pixels
    d = np.abs(data - filtered)
    invalid = d > thresh
    if sure_dead is not None:
        invalid = invalid | sure_dead

    # replace invalid pixels with median values
    ret = data.copy()
    ret[invalid] = filtered[invalid]

    return ret, invalid
