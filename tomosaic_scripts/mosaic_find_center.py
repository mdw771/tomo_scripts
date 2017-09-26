import tomosaic
import tomopy
import glob, os
import numpy as np
import dxchange
try:
    from mosaic_meta import *
except:
    reader = open(os.path.join('tomosaic_misc', 'meta'), 'rb')
    prefix, file_grid, x_shift, y_shift = pickle.load(reader)
    reader.close()
from mosaic_util import *

# ==========================================
center_st = 5250
center_end = 5280
row_st = 0
row_end = 1
method = 'vo' # 'manual' or 'vo'
mode = 'discrete' # 'discrete' or 'merged' or 'single'
in_tile_pos = 1000
ds = 1
dest_folder = 'center'
# merged:
slice = 1000
# discrete:
source_folder = 'data_raw_1x'
data_format = 'aps_13bm'
# merged:
fname = 'fulldata_flatcorr_1x/fulldata_flatcorr_1x.h5'
# single:
sino_name = 'sino_4810.tiff'
preprocess_single = False
# ==========================================

import tomopy
import h5py
import dxchange
import time
import numpy as np
from scipy import ndimage
import pyfftw
import dxchange
from scipy.optimize import minimize
from skimage.feature import register_translation
from tomopy.misc.corr import circ_mask
from tomopy.misc.morph import downsample
from tomopy.recon.algorithm import recon
import tomopy.util.dtype as dtype
import os.path
import logging
from scipy.misc import imresize

logger = logging.getLogger(__name__)
PI = 3.14159265359

def find_center_vo(tomo, ind=None, smin=-50, smax=50, srad=6, step=0.5,
                   ratio=0.5, drop=20):
    """
    Find rotation axis location using Nghia Vo's method. :cite:`Vo:14`.
    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    ind : int, optional
        Index of the slice to be used for reconstruction.
    smin, smax : int, optional
        Coarse search radius. Reference to the horizontal center of the sinogram.
    srad : float, optional
        Fine search radius.
    step : float, optional
        Step of fine searching.
    ratio : float, optional
        The ratio between the FOV of the camera and the size of object.
        It's used to generate the mask.
    drop : int, optional
        Drop lines around vertical center of the mask.
    Returns
    -------
    float
        Rotation axis location.
    """
    tomo = dtype.as_float32(tomo)

    if ind is None:
        ind = tomo.shape[1] // 2
    _tomo = tomo[:, ind, :]

    # Enable cache for FFTW.
    pyfftw.interfaces.cache.enable()

    # Reduce noise by smooth filters. Use different filters for coarse and fine search
    _tomo_cs = ndimage.filters.gaussian_filter(_tomo, (3, 1))
    _tomo_fs = ndimage.filters.median_filter(_tomo, (2, 2))

    # Coarse and fine searches for finding the rotation center.
    if _tomo.shape[0] * _tomo.shape[1] > 4e6:  # If data is large (>2kx2k)
        _tomo_coarse = downsample(np.expand_dims(_tomo_cs,1), level=2)[:, 0, :]
        init_cen = _search_coarse(_tomo_coarse, smin/4, smax/4, ratio, drop)
        fine_cen = _search_fine(_tomo_fs, srad, step, init_cen*4, ratio, drop)
    else:
        init_cen = _search_coarse(_tomo_cs, smin, smax, ratio, drop)
        fine_cen = _search_fine(_tomo_fs, srad, step, init_cen, ratio, drop)

    logger.debug('Rotation center search finished: %i', fine_cen)
    return fine_cen


def _search_coarse(sino, smin, smax, ratio, drop):
    """
    Coarse search for finding the rotation center.
    """
    (Nrow, Ncol) = sino.shape
    print(Nrow, Ncol)
    centerfliplr = (Ncol - 1.0) / 2.0

    # Copy the sinogram and flip left right, the purpose is to
    # make a full [0;2Pi] sinogram
    _copy_sino = np.fliplr(sino[1:])

    # This image is used for compensating the shift of sinogram 2
    temp_img = np.zeros((Nrow - 1, Ncol), dtype='float32')
    temp_img[:] = np.flipud(sino)[1:]

    # Start coarse search in which the shift step is 1
    listshift = np.arange(smin, smax + 1)
    print('listshift', listshift)
    listmetric = np.zeros(len(listshift), dtype='float32')
    mask = _create_mask(2 * Nrow - 1, Ncol, 0.5 * ratio * Ncol, drop)
    for i in listshift:
        _sino = np.roll(_copy_sino, i, axis=1)
        if i >= 0:
            _sino[:, 0:i] = temp_img[:, 0:i]
        else:
            _sino[:, i:] = temp_img[:, i:]
        listmetric[i - smin] = np.sum(np.abs(np.fft.fftshift(
            pyfftw.interfaces.numpy_fft.fft2(
                np.vstack((sino, _sino))))) * mask)
    minpos = np.argmin(listmetric)
    print('coarse return', centerfliplr + listshift[minpos] / 2.0)
    return centerfliplr + listshift[minpos] / 2.0


def _search_fine(sino, srad, step, init_cen, ratio, drop):
    """
    Fine search for finding the rotation center.
    """
    Nrow, Ncol = sino.shape
    centerfliplr = (Ncol + 1.0) / 2.0 - 1.0
    # Use to shift the sinogram 2 to the raw CoR.
    shiftsino = np.int16(2 * (init_cen - centerfliplr))
    _copy_sino = np.roll(np.fliplr(sino[1:]), shiftsino, axis=1)
    if init_cen <= centerfliplr:
        lefttake = np.int16(np.ceil(srad + 1))
        righttake = np.int16(np.floor(2 * init_cen - srad - 1))
    else:
        lefttake = np.int16(np.ceil(
            init_cen - (Ncol - 1 - init_cen) + srad + 1))
        righttake = np.int16(np.floor(Ncol - 1 - srad - 1))
    Ncol1 = righttake - lefttake + 1
    mask = _create_mask(2 * Nrow - 1, Ncol1, 0.5 * ratio * Ncol, drop)
    numshift = np.int16((2 * srad) / step) + 1
    listshift = np.linspace(-srad, srad, num=numshift)
    listmetric = np.zeros(len(listshift), dtype='float32')
    factor1 = np.mean(sino[-1, lefttake:righttake])
    factor2 = np.mean(_copy_sino[0,lefttake:righttake])
    _copy_sino = _copy_sino * factor1 / factor2
    num1 = 0
    for i in listshift:
        _sino = ndimage.interpolation.shift(
            _copy_sino, (0, i), prefilter=False)
        sinojoin = np.vstack((sino, _sino))
        listmetric[num1] = np.sum(np.abs(np.fft.fftshift(
            pyfftw.interfaces.numpy_fft.fft2(
                sinojoin[:, lefttake:righttake + 1]))) * mask)
        num1 = num1 + 1
    minpos = np.argmin(listmetric)
    return init_cen + listshift[minpos] / 2.0


def _create_mask(nrow, ncol, radius, drop):
    du = 1.0 / ncol
    dv = (nrow - 1.0) / (nrow * 2.0 * PI)
    centerrow = np.int16(np.ceil(nrow / 2) - 1)
    centercol = np.int16(np.ceil(ncol / 2) - 1)
    mask = np.zeros((nrow, ncol), dtype='float32')
    for i in range(nrow):
        num1 = np.round(((i - centerrow) * dv / radius) / du)
        (p1, p2) = np.int16(np.clip(np.sort(
            (-int(num1) + centercol, num1 + centercol)), 0, ncol - 1))
        mask[i, p1:p2 + 1] = np.ones(p2 - p1 + 1, dtype='float32')
    if drop < centerrow:
        mask[centerrow - drop:centerrow + drop + 1,
             :] = np.zeros((2 * drop + 1, ncol), dtype='float32')
    mask[:,centercol-1:centercol+2] = np.zeros((nrow, 3), dtype='float32')
    return mask

try:
    shift_grid = tomosaic.util.file2grid("shifts.txt")
    shift_grid = tomosaic.absolute_shift_grid(shift_grid, file_grid)
except:
    shift_grid = tomosaic.start_shift_grid(file_grid, x_shift, y_shift)

shift_grid = shift_grid / ds
in_tile_pos = in_tile_pos / ds

log = open('center_pos.txt', 'a')
center_pos = []

if mode == 'merged':
    f = h5py.File(fname)
    for row in range(row_st, row_end):
        sino = slice + shift_grid[row, 0, 0]
        sino = f['exchange/data'][:, sino:sino+1, :]
        if method == 'manual':
            tomopy.write_center(sino, os.path.join('center', str(row)), cen_range=(center_st, center_end))
        elif method == 'vo':
            mid = sino.shape[2] / 2
            smin = (center_st - mid) * 2
            smax = (center_end - mid) * 2
            center = find_center_vo(sino, smin=smin, smax=smax)
            print('For {} center is {}.'.format(row, center))
            log.write('{} {}\n'.format(row, center))
    log.close()
elif mode == 'discrete':
    for row in range(row_st, row_end):
        print('Row {}'.format(row))
        slice = int(shift_grid[row, 0, 0] + in_tile_pos)
        # create sinogram
        try:
            sino = dxchange.read_tiff('center_temp/sino/sino_{:05d}.tiff'.format(slice))
            print('center_temp/sino/sino_{:05d}.tiff'.format(slice))
            print(sino.shape)
            sino = sino.reshape([sino.shape[0], 1, sino.shape[1]])
        except:
            center_vec = [center_st] * file_grid.shape[0]
            center_vec = np.array(center_vec)
            tomosaic.recon_block(file_grid, shift_grid, source_folder, 'center_temp', (slice, slice+1), 1,
                                 center_vec, algorithm='gridrec', test_mode=True, ds_level=0, save_sino=True,
                                 blend_method='pyramid', data_format=data_format)
            sino = dxchange.read_tiff('center_temp/sino/sino_{:05d}.tiff'.format(slice))
            sino = sino.reshape([sino.shape[0], 1, sino.shape[1]])
        sino = tomopy.remove_stripe_ti(sino, alpha=4)
        if method == 'manual':
            tomopy.write_center(sino, tomopy.angles(sino.shape[0]), dpath='center/{}'.format(row),
                                cen_range=(center_st, center_end))
        elif method == 'vo':
            mid = sino.shape[2] / 2
            smin = (center_st - mid) * 2
            smax = (center_end - mid) * 2
            center = find_center_vo(sino, smin=smin, smax=smax)
            print('For {} center is {}.'.format(row, center))
            log.write('{} {}\n'.format(row, center))
    log.close()
elif mode == 'single':
    sino = dxchange.read_tiff(sino_name)
    sino = sino.reshape([sino.shape[0], 1, sino.shape[1]])
    if preprocess_single:
        sino = tomosaic.preprocess(np.copy(sino))
    if method == 'manual':
        tomopy.write_center(sino, tomopy.angles(sino.shape[0]), dpath='center', cen_range=(center_st, center_end))
    elif method == 'vo':
        mid = sino.shape[2] / 2
        smin = (center_st - mid) * 2
        smax = (center_end - mid) * 2
        center = find_center_vo(sino, smin=smin, smax=smax)
        print('Center is {}.'.format(center))
        log.write('{}\n'.format(center))
        log.close()

#for row in range(file_grid.shape[0]):
#    center = tomosaic.misc.minimum_entropy(os.path.join('center', str(row)))
#    log.writelines(str(center) + '\n')
#    center_pos.append(center)

log.close()