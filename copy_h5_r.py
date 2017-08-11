"""
Recursively copy an HDF5 file.
"""
import h5py
import numpy as np
import os
from itertools import izip

# =============================================
original_name = 'WholeBrainMRI_phase35cm_5x_2k_gap31_exp30_newfocus_y4_x4.h5'
new_name = os.path.basename(original_name) + '_copy.h5'
# =============================================

# def copy_items(obj_o, parent_o, parent_f):
#
#     try:
#         subs = obj_o.items()
#         isdset = False
#     except:
#         isdset = True
#         subs = None
#     if isdset:
#         obj_f = parent_f.create_dataset(obj_o.name, obj_o.shape, obj_o.dtype)
#         obj_f[...] = obj_o[...]
#     else:
#         for i in subs:


o = h5py.File(original_name, 'r')
f = h5py.File(new_name)

f.create_group('exchange')
do = o['exchange/data']
df = f['exchange'].create_dataset('data', do.shape, do.dtype)
for i in range(0, df.shape[0], 50):
    end = min([i + 50, df.shape[0]])
    df[i:end, :, :] = do[i:end, :, :]

do = o['exchange/data_white']
df = f['exchange'].create_dataset('data_white', do.shape, do.dtype)
for i in range(0, df.shape[0], 50):
    end = min([i + 50, df.shape[0]])
    df[i:end, :, :] = do[i:end, :, :]

do = o['exchange/data_dark']
df = f['exchange'].create_dataset('data_dark', do.shape, do.dtype)
for i in range(0, df.shape[0], 50):
    end = min([i + 50, df.shape[0]])
    df[i:end, :, :] = do[i:end, :, :]

