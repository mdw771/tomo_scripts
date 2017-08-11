# Fix stage tilt.

import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import h5py
import os
import glob
from mpi4py import MPI
import time

# ==========================================
pattern = '*5x*4k*'
dest_folder = 'tilt_fixed'
angle = -0.45 # degree; positive = anticlockwise, negative = clockwise
chunk_size = 50
# ==========================================

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

def allocate_mpi_subsets(n_task, size, task_list=None):

    if task_list is None:
        task_list = range(n_task)
    sets = []
    for i in range(size):
        sets.append(task_list[i:n_task:size])
    return sets

def write_data(dset_f, dset_o):

    task_list = range(0, dset_o.shape[0], chunk_size)
    sets = allocate_mpi_subsets(len(task_list), size, task_list)
    for i in sets[rank]:
        print('Block starting {:d}'.format(i))
        end = min([i+chunk_size, dset_o.shape[0]])
        dset = dset_o[i:end, :, :]
        dset = rotate(dset, angle, axes=(1, 2), reshape=False, mode='nearest')
        dset = dset.astype(np.uint16)
        dset_f[i:end, :, :] = dset
    comm.Barrier()

folderlist = []
for i in glob.glob(pattern):
    if os.path.isdir(i):
        folderlist.append(i)
root = os.getcwd()

print(folderlist)
time.sleep(10)

for folder in folderlist:
    print(folder)
    os.chdir(folder)
    filelist = glob.glob('*.h5')

    for fname in filelist:

        new_fname = os.path.join(os.path.splitext(fname)[0] + '_tiltfixed.h5')

        if rank == 0:
            o = h5py.File(fname, 'r')
            f = h5py.File(new_fname)
        comm.Barrier()
        if rank != 0:
            o = h5py.File(fname, 'r')
            f = h5py.File(new_fname)

        o_data = o['exchange/data']
        o_flat = o['exchange/data_white']
        o_dark = o['exchange/data_dark']
        o_theta = o['exchange/theta']

        grp = f.create_group('exchange')
        f_data = grp.create_dataset('data', o_data.shape, dtype=np.uint16)
        f_flat = grp.create_dataset('data_white', o_flat.shape, dtype=np.uint16)
        f_dark = grp.create_dataset('data_dark', o_dark.shape, dtype=np.uint16)
        f_theta = grp.create_dataset('theta', o_theta.shape, dtype=np.uint16)
        comm.Barrier()

        write_data(f_data, o_data)
        write_data(f_flat, o_flat)
        write_data(f_dark, o_dark)

        f_theta[...] = o_theta[...]

        f.close()
        o.close()

        print('Done!')
        comm.Barrier()

    os.chdir(root)
