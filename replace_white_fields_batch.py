import numpy as np
import h5py
import dxchange
import glob
import os

# ================================================
folder_pattern = 'Cox*'
source_file = '../flat_fields_tiltfixed.tiff'
pattern = '*.h5'
# ================================================

temp = glob.glob(folder_pattern)
folder_list = []
for i in temp:
    if os.path.isdir(i):
        folder_list.append(i)

root = os.getcwd()

for folder in folder_list:
    os.chdir(folder)
    filelist = glob.glob(pattern)
    os.chdir(root)

    flat_good = dxchange.read_tiff(source_file)
    if flat_good.ndim == 2:
        flat_good = flat_good[np.newaxis, :, :]

    for fname in filelist:
        print(fname)
        fname = os.path.join(folder, fname)
        f = h5py.File(fname)
        dset = f['exchange/data_white']
        if dset.shape[0] < flat_good.shape[0]:
            dset[...] = flat_good[:dset.shape[0], :, :]
        else:
            for i in range(0, dset.shape[0], flat_good.shape[0]):
                end = min([i+flat_good.shape[0], dset.shape[0]])
                dset[i:end, :, :] = flat_good[:end-i, :, :]
        f.close()