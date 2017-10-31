import h5py
import glob
import os

correct_theta = 'data_raw_1x/Ming_Charcoal_25kev_lens10x_dfocus12cm_180_y2_x0.h5'
folder = 'data_raw_2x'

o = h5py.File(correct_theta)
theta = o['exchange/theta'].value

filelist = glob.glob(folder)
for fname in filelist:
    print(fname)
    f = h5py.File(fname)
    f['exchange/theta'][...] = theta
    f.close()
