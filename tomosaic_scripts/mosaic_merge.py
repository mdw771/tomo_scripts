import tomosaic
import glob, os
import numpy as np
import pickle
import h5py
import dxchange
try:
    from mosaic_meta import *
except:
    reader = open(os.path.join('tomosaic_misc', 'meta'), 'rb')
    prefix, file_grid, x_shift, y_shift = pickle.load(reader)
    reader.close()
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()


# ==========================================
blend_options = {'depth': 7,
                 'blur': 0.4}
source_folder = 'data_raw_1x'
ds = 1
# ==========================================



shift_grid = tomosaic.util.file2grid("shifts.txt")
shift_grid = tomosaic.absolute_shift_grid(shift_grid, file_grid)
shift_grid = shift_grid / ds

t0 = time.time()
tomosaic.total_fusion(source_folder, 'fulldata_flatcorr_1x', 'fulldata_flatcorr_1x.h5', file_grid,
                           shift_grid, blend_method='pyramid', blend_method2='overlay',
                           color_correction=False, blend_options=blend_options, dtype='float16')

f = h5py.File(os.path.join('fulldata_flatcorr_1x', 'fulldata_flatcorr_1x.h5'), 'r')
dat = f['exchange/data']
dxchange.write_tiff(dat[0], os.path.join('fulldata_flatcorr_1x', 'proj_0'), dtype='float32', overwrite=True)
last_ind = dat.shape[0] - 1
dxchange.write_tiff(dat[last_ind], os.path.join('fulldata_flatcorr_1x', 'proj_{}'.format(last_ind)), dtype='float32', overwrite=True)
f.close()

print('Rank {}: total time: {} s.'.format(rank, time.time() - t0))