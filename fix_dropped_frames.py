"""
Interpolate missing frames in HDF5 files based on UniqueId
"""

import h5py
import numpy as np
from itertools import izip


def fix_h5(fname, target_nframe=None):

    print(fname)
    f = h5py.File(fname)
    id = f['defaults/NDArrayUniqueId'].value
    id_jump = np.nonzero(np.roll(id, -1) - id != 1)[0]
    id_jump = id_jump[:-1][::-1]
    n_jump = [id[i + 1] - id[i] - 1 for i in id_jump]
    n_tot = len(n_jump)
    dset = f['exchange/data']
    print('Total frame loss is {:d}.'.format(n_tot))

    if target_nframe is not None:
        if target_nframe > dset.shape[0]:
            print('Incorrect dataset shape encountered. Attempting to expand')
            dset2 = f['exchange'].create_dataset('temp', [target_nframe] + list(dset.shape[1:]), dtype=dset.dtype)
            for i in range(0, dset.shape[0], 50):
                end = min([i + 50, dset.shape[0]])
                dset2[i:end, :, :] = dset[i:end, :, :]
    del f['exchange/data']
    dset = dset2

    move_st = -n_tot - 1
    count = 0
    for i, (this_id, this_n) in enumerate(izip(id_jump, n_jump)):
        img1 = dset[this_id, :, :]
        img2 = dset[this_id + 1, :, :]

        move_step = n_tot - count
        move_end = move_st
        if move_end < 0:
            move_end = dset.shape[0] + move_end
        move_st = this_id
        print(move_st, move_end, move_step)
        if move_st >= move_end:
            for ii in range(move_end, dset.shape[0]):
                dset[ii, :, :] = dset[move_end, :, :]
        else:
            for ii in range(move_end, move_st, -1):
                dset[ii + move_step, :, :] = dset[ii, :, :]
            interp_seq = interpolate_projs(img1, img2, this_n)
            for ii in range(this_n):
                dset[this_id+move_step-this_n+ii+1, :, :] = interp_seq[ii].astype(dset.dtype)
        count += this_n


def interpolate_projs(img1, img2, n):

    res = []
    for i in range(n):
        img = img1 + (img2 - img1) * (i + 1.) / (n + 1.)
        res.append(img)
    return res


if __name__ == '__main__':

    fix_h5('h5_fix_test/test0.h5')