# -*- coding: utf-8 -*-

import tomopy
import dxchange
import numpy as np
import h5py
import os


def sino_360_to_180(data, overlap=0, rotation='left'):
    """
    Converts 0-360 degrees sinogram to a 0-180 sinogram.
    If the number of projections in the input data is odd, the last projection
    will be discarded.
    Parameters
    ----------
    data : ndarray
        Input 3D data.
    overlap : scalar, optional
        Overlapping number of pixels.
    rotation : string, optional
        Left if rotation center is close to the left of the
        field-of-view, right otherwise.
    Returns
    -------
    ndarray
        Output 3D data.
    """
    dx, dy, dz = data.shape

    overlap = int(np.round(overlap))

    lo = overlap//2
    ro = overlap - lo
    n = dx//2

    out = np.zeros((n, dy, 2*dz-overlap), dtype=data.dtype)

    if rotation == 'left':
        out[:, :, -(dz-lo):] = data[:n, :, lo:]
        out[:, :, :-(dz-lo)] = data[n:2*n, :, ro:][:, :, ::-1]
    elif rotation == 'right':
        out[:, :, :dz-lo] = data[:n, :, :-lo]
        out[:, :, dz-lo:] = data[n:2*n, :, :-ro][:, :, ::-1]

    return out


def write_center_360(file_name, output_path, slice_no, center_st, center_end, medfilt_size=1, level=0, debug=1):

    try:
        prj0, flat, dark, theta = dxchange.read_aps_32id(file_name, sino=(slice_no, slice_no+1))
    except:
        prj0, flat, dark = dxchange.read_aps_32id(file_name, sino=(slice_no, slice_no+1))

    f = h5py.File(file_name, "r")

    theta = tomopy.angles(f['exchange/data'].shape[0], ang1=0, ang2=360)

    if debug:
        print('## Debug: after reading data:')
        print('\n** Shape of the data:'+str(np.shape(prj0)))
        print('** Shape of theta:'+str(np.shape(theta)))
        print('\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj0), np.max(prj0)))

    prj0 = tomopy.normalize(prj0, flat, dark)
    print('\n** Flat field correction done!')

    for center in range(center_st, center_end):

        overlap =  2 * (f['exchange/data'].shape[2] - center)

        prj = np.copy(prj0)

        prj = sino_360_to_180(prj, overlap=overlap, rotation='right')
        print('\n** Sinogram converted!')

        if debug:
            print('## Debug: after normalization:')
            print('\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))

        prj = tomopy.minus_log(prj)
        print('\n** minus log applied!')

        if debug:
            print('## Debug: after minus log:')
            print('\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))

        prj = tomopy.misc.corr.remove_neg(prj, val=0.001)
        prj = tomopy.misc.corr.remove_nan(prj, val=0.001)
        prj[np.where(prj == np.inf)] = 0.001

        if debug:
            print('## Debug: after cleaning bad values:')
            print('\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))

        prj = tomopy.remove_stripe_ti(prj, 4)
        print('\n** Stripe removal done!')
        if debug:
            print('## Debug: after remove_stripe:')
            print('\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))

        prj = tomopy.median_filter(prj, size=medfilt_size)
        print('\n** Median filter done!')
        if debug:
            print('## Debug: after nedian filter:')
            print('\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))


        if level>0:
            prj = tomopy.downsample(prj, level=level)
            print('\n** Down sampling done!\n')
        if debug:
            print('## Debug: after down sampling:')
            print('\n** Min and max val in prj before recon: %0.5f, %0.3f'  % (np.min(prj), np.max(prj)))

        rec = tomopy.recon(prj, theta, center=center, algorithm='gridrec')
        print('\nReconstruction done!\n')

        dxchange.write_tiff(rec, fname=os.path.join(output_path, '{0:.2f}'.format(center)), dtype='float32')

