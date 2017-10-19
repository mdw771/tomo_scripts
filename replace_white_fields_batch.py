import numpy as np
import h5py
import dxchange
import glob
import os

# ================================================
folder = '.'
source_file = '../flat_fields_tiltfixed.tiff'
pattern = '*.h5'
# ================================================

root = os.getcwd()
os.chdir(folder)
filelist = glob.glob(pattern)
os.chdir(root)

flat_good = dxchange.read_tiff(source_file)

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