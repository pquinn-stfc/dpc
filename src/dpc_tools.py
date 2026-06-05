import numpy as np
from skimage import filters
from skimage.measure import regionprops

def get_beam(img):
    '''Apply OTSU filter to identify the beam position

    Parameter
    ---------
    img : ndarray
        the image containing the beam

    Returns
    -------
    prop : RegionProperties
        contains description of the labelled beam, access through attributes

    '''

    threshold_value = filters.threshold_otsu(img)
    labelled_foreground = (img > threshold_value).astype(int)

    properties = regionprops(labelled_foreground, intensity_image=img)
    prop = properties[0]

    return prop


def get_dark_field_mask(img, bbox, frac=0.95, cropped_x=None, cropped_y=None):
    '''Define the dark field mask of the beam

    Parameters
    ----------
    img : ndarray
        the beam image
    bbox : array-like
        the boundary box of the beam, in order of lowy, lowx, highy, highx
    frac : float, optional
        the fraction of the dark field mask to the beam. Default as 0.95.

    Returns
    -------
    mask : ndarray
        the dark field mask
    '''

    # min_row, min_col, max_row, max_col
    lowy, lowx, highy, highx = bbox

    # dark field index
    lowx_df = int(lowx + ((highx-lowx) * (1 - frac)) + 0.5)
    highx_df = int(lowx + ((highx-lowx) * frac) + 0.5)
    lowy_df = int(lowy + ((highy-lowy) * (1 - frac)) + 0.5)
    highy_df = int(lowy + (highy-lowy) * frac + 0.5)

    # dark field mask initialisation
    mask = np.zeros(img.shape, dtype=bool)

    # inside the beam
    mask[lowy_df:highy_df, lowx_df:highx_df] = True

    # outside the beam
    if cropped_x is not None:
        crop_lowx = cropped_x.start
        crop_highx = cropped_x.stop
        mask[:, 0:crop_lowx] = True
        mask[:, crop_highx:img.shape[1]] = True
    if cropped_y is not None:
        crop_lowy = cropped_y.start
        crop_highy = cropped_y.stop
        mask[0:crop_lowy, :] = True
        mask[crop_highy:img.shape[0], :] = True

    return mask

def crop_beam(img, crop_size, bbox):
    '''Crop out the beam

    Parameters
    ----------
    img : ndarray
        the beam image
    crop_size : integer
        the size of the cropping, this value is for both y and x dimensions
    bbox : ndarray
        the boundary box of the beam, in order of lowy, lowx, highy, highx

    Returns
    -------
    cropped_x : slice
        column slice defining the cropped region
    cropped_y : slice
        row slice defining the cropped region
    '''

    # min_row, min_col, max_row, max_col
    lowy, lowx, highy, highx = bbox

    box_cx = int(0.5*(lowx+highx)+0.5)
    box_cy = int(0.5*(lowy+highy)+0.5)

    # define box size
    box_size_y = crop_size
    box_size_x = crop_size


    if box_cy < 0.5*box_size_y:
        crop_lowy = 0
        crop_highy = box_size_y
    elif box_cy + 0.5 *box_size_y > img.shape[1]:
        crop_lowy = img.shape[1] - box_size_y
        crop_highy = img.shape[1]
    else:
        crop_lowy = int(box_cy-0.5*box_size_y)
        crop_highy = int(box_cy+0.5*box_size_y)

    if box_cx < 0.5*box_size_x:
        crop_lowx = 0
        crop_highx = box_size_x
    elif box_cx + 0.5 *box_size_x > img.shape[0]:
        crop_lowx = img.shape[0]-box_size_x
        crop_highx = img.shape[0]
    else:
        crop_lowx = int(box_cx-0.5*box_size_x+0.5)
        crop_highx = int(box_cx+0.5*box_size_x+0.5)

    cropped_x = slice(crop_lowx, crop_highx)
    cropped_y = slice(crop_lowy, crop_highy)

    return cropped_x, cropped_y


def centre_of_mass(img, labelled_region=None):
    '''Calculate the centre of mass of a labelled region of a beam

    Parameters
    ----------
    img : array-like
        the beam image
    labelled_region : array-like, optional
        the label for the region where the centre of mass is to be calculated,
        inactive region should be set to 0 and active region an integer. Default
        to None and it is set to the whole beam image.

    Returns
    -------
    w_com : ndarray
        the centre of mass of the labelled region of the beam
    '''
    try:
        if labelled_region is None:
            labelled_region = np.ones(img.shape, dtype=int)

        labelled_region = np.asarray(labelled_region).astype(int)

        # centre of mass of masked pixels
        properties = regionprops(labelled_region, intensity_image=img)
        prop = properties[0]

        w_com = np.asarray(prop.weighted_centroid)

        return w_com
    except:
        return np.array([0.0,0.0])

#def replace_invalid_pixels(data, tolerance=10, size=2, sure_dead=None):
#
#    # for the main image
#    blurred = median_filter(data, size=size)
#    threshold = tolerance * np.std(blurred)
##    print('The threshold is at {:.4f}'.format(threshold))
#
#    # handle edge
#    edge_filter_size = size + 1
#    cut = size // 2
#    left = generic_filter(data[:,:size], np.nanmedian, size=edge_filter_size,
#                          mode='constant', cval=np.nan)[:,:cut]
#    right = generic_filter(data[:,-size:], np.nanmedian, size=edge_filter_size,
#                          mode='constant', cval=np.nan)[:,-cut:]
#    bottom = generic_filter(data[:size,:], np.nanmedian, size=edge_filter_size,
#                          mode='constant', cval=np.nan)[:cut, :]
#    top = generic_filter(data[-size:,:], np.nanmedian, size=edge_filter_size,
#                          mode='constant', cval=np.nan)[-cut:, :]
#    blurred[:,:cut] = left
#    blurred[:,-cut:] = right
#    blurred[:cut, :] = bottom
#    blurred[-cut:,:] = top
#
#    # determine the positions of invalid pixels
#    difference = np.abs(data - blurred)
#    invalid = difference > threshold
#    if sure_dead is not None:
#        invalid = invalid | sure_dead
#
#    # replace invalid pixels with filtered/median value
#    img = data.copy()
#    img[invalid] = blurred[invalid]
#
#    return img, invalid
