from scipy.ndimage import rotate

import numpy as np
import dxchange
import glob, os
import time
from tomopy import downsample
from scipy.ndimage import uniform_filter
from itertools import izip
import pickle
import re
try:
    from mpi4py import MPI
except:
    pass

# ==============================
angle = 0
crop = None # (x_start, y_start, x_length, y_length; same as ImageJ convention)
min = -0.0005
max = 0.0011
bitdepth = 8
saturation = 35
mean_filter = None # radius
mean3d_x = 1 # window full size
mean3d_y = 1
mean3d_z = 1
ds_factor = 1
chunk_size = 30
src_folder = 'recon_flatcorr_4x/recon'
dest_folder = 'recon_flatcorr_4x/recon_crop_8'
# ==============================

def barrier():
    try:
        comm.Barrier()
    except:
        pass

def allocate_mpi_subsets(n_task, size, task_list=None):

    if task_list is None:
        task_list = range(n_task)
    sets = []
    max_len = 0
    for i in range(size):
        selection = range(i, n_task, size)
        sets.append(np.take(task_list, selection).tolist())
        if len(sets[i]) > max_len:
            max_len = len(sets[i])
    for i in range(size):
        for j in range(len(sets[i]), max_len):
            sets[i].append(None)
    return sets

def divide_chunks(filelist, continuous=True):

    chunks = []
    counter = 0
    subset = []
    end_file = filelist[-1]
    for i, f in enumerate(filelist):
        counter += 1
        subset.append(f)
        break_judger = False
        if continuous and i+1 < len(filelist):
            this_ind = int(re.search('\d+', f).group(0))
            next_ind = int(re.search('\d+', filelist[i+1]).group(0))
            if next_ind - this_ind > 1:
                break_judger = True
        if counter == chunk_size or f == end_file or break_judger:
            chunks.append(subset)
            counter = 0
            subset = []
    return chunks

def mean3d(stack):

    kernel_size = [mean3d_z, mean3d_y, mean3d_x]
    stack = uniform_filter(stack, size=kernel_size, mode='nearest')
    return stack


raw_folder = os.getcwd()
os.chdir(src_folder)
filelist = glob.glob('*.tif*')
filelist.sort()
os.chdir(raw_folder)

try:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    name = MPI.Get_processor_name()
except:
    rank = 0
    size = 1

if rank == 0:
    try:
        os.mkdir(dest_folder)
    except:
        pass
barrier()

os.chdir(dest_folder)
templist = []
for f in filelist:
    if not os.path.isfile(f):
        templist.append(f)
filelist = templist[:]
os.chdir(raw_folder)
if len(filelist) > 0:
    chunks = divide_chunks(filelist)
    sets = allocate_mpi_subsets(len(chunks), size, task_list=chunks)
    border_files = []

    if mean_filter is not None:
        try:
            import cv2
        except:
            import opencv as cv2

    for subset in sets[rank]:

        if subset is not None:
            # load chunk
            img = []
            for fname in subset:
                print(fname, rank)
                slice = dxchange.read_tiff(os.path.join(src_folder, fname))
                img.append(slice)
            img = np.asarray(img)

            # rotate and crop
            if not np.isclose(angle, 0):
                img = rotate(img, -angle, reshape=False, axes=(1, 2))
            if crop is not None:
                img = img[:, crop[1]:crop[1]+crop[3], crop[0]:crop[0]+crop[2]]

            # downsample
            if ds_factor > 1:
                if img.shape[0] > 1:
                    img = downsample(img, level=int(np.log2(ds_factor)), axis=0)
                img = downsample(img, level=int(np.log2(ds_factor)), axis=1)
                img = downsample(img, level=int(np.log2(ds_factor)), axis=2)

            # 2d mean filter
            if mean_filter is not None:
                filt_size = mean_filter * 2 + 1
                for i in range(img.shape[0]):
                    img[i] = cv2.blur(img[i], (filt_size, filt_size))

            # 3d mean filter
            if mean3d_x is not None:
                img = mean3d(img)

            # scale intensity
            if min is not None:
                bit_full = 2 ** bitdepth - 1
                img = (img - min) / (max - min) * bit_full
                img[img < 0] = 0
                img = np.clip(img, 0, bit_full)

            # save
            for slice, fname in izip(img, subset):
                dxchange.write_tiff(slice, os.path.join(dest_folder, fname), dtype='uint{:d}'.format(bitdepth))

            border_files.append(subset[0])
            border_files.append(subset[-1])

        barrier()
        if rank != 0:
            comm.send(border_files, dest=0)
            border_files = []
        else:
            for src in range(1, size):
                temp = comm.recv(source=src)
                border_files += temp

        barrier()
        if rank == 0:
            border_saver = open(os.path.join(dest_folder, 'borders'), 'w')
            pickle.dump(border_files, border_saver)
            border_saver.close()
        barrier()

# second pass to fix mean3d borders

barrier()
if mean3d_x is not None:
    print('Second pass...', rank)
    border_saver = open(os.path.join(dest_folder, 'borders'), 'r')
    border_files = pickle.load(border_saver)
    # remove adjacent redundancies
    deletion_mark = []
    for i, f in enumerate(border_files[:-1]):
        this_ind = int(re.search('\d+', f).group(0))
        next_ind = int(re.search('\d+', border_files[i + 1]).group(0))
        if next_ind == this_ind + 1:
            deletion_mark.append(i)
    for i in deletion_mark[::-1]:
        del border_files[i]
    print border_files
    sets = allocate_mpi_subsets(len(border_files), size, task_list=border_files)
    rad = int(mean3d_z / 2) + 1
    os.chdir(src_folder)
    filelist = glob.glob('*.tif*')
    filelist.sort()
    os.chdir(raw_folder)

    for border_slice in sets[rank]:
        img = []
        border_ind = np.searchsorted(filelist, border_slice)
        if border_ind-rad >= 0:
            subset = filelist[border_ind-rad*2:border_ind+rad*2+1]
        else:
            subset = filelist[0:border_ind+rad*2+1]
        for fname in subset:
            print(fname, '2nd, {}'.format(rank))
            slice = dxchange.read_tiff(os.path.join(dest_folder, fname))
            img.append(slice)
        img = np.asarray(img)
        img = mean3d(img)

        for slice, fname in izip(img[rad:-rad], subset[rad:-rad]):
            dxchange.write_tiff(slice, os.path.join(dest_folder, fname), dtype='uint{:d}'.format(bitdepth), overwrite=True)