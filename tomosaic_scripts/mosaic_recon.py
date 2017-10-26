import tomosaic
import glob, os
import numpy as np
import time
import tomopy
import pickle
try:
    from mosaic_meta import *
except:
    reader = open(os.path.join('tomosaic_misc', 'meta'), 'rb')
    prefix, file_grid, x_shift, y_shift = pickle.load(reader)
    reader.close()
from mosaic_util import *


# ==========================================
slice_st =  0	
slice_end = 13781
slice_step = 1
mode = 'merged'
center_vec = np.array([5952, 5954, 5960, 5967, 5975, 5976, 5986, 5991, 5994, 5996, 6003, 6001, 6007, 6012, 6021, 6027])
dest_folder = 'recon_flatcorr_1x'
ds = 1
chunk_size = 5
# discrete ------------------------
source_folder = 'data_raw_2x'
# merged --------------------------
fname = 'fulldata_flatcorr_1x/fulldata_flatcorr_1x.h5'
# single --------------------------
sino_name = ''
preprocess_single = False
center_single = 1000
dest_fname = 'recon.tiff'
# ==========================================

try:
    shift_grid = tomosaic.util.file2grid("shifts.txt")
    shift_grid = tomosaic.absolute_shift_grid(shift_grid, file_grid)
except:
    shift_grid = tomosaic.start_shift_grid(file_grid, x_shift, y_shift)

shift_grid = shift_grid / float(ds)
center_vec = center_vec / float(ds)

t0 = time.time()
if mode == 'merged':
    tomosaic.recon.recon_hdf5(fname, dest_folder, (slice_st, slice_end),
                              slice_step, shift_grid, center_vec=center_vec, chunk_size=chunk_size,
                              dtype='float32', save_sino=False)
elif mode == 'discrete':
    tomosaic.recon_block(file_grid, shift_grid, source_folder, dest_folder, (slice_st, slice_end), 1,
                         center_vec, algorithm='gridrec', test_mode=False, ds_level=0, save_sino=True,
                         blend_method='pyramid', data_format=data_format)
elif mode == 'single':
    sino = dxchange.read_tiff(sino_name)
    sino = sino.reshape([sino.shape[0], 1, sino.shape[1]]) 
    if preprocess_single:
        sino = tomosaic.preprocess(np.copy(sino))
    theta = tomopy.angles(sino.shape[0])
    rec = tomopy.recon(sino, theta, center=center_single, algorithm='gridrec')
    dxchange.write_tiff(rec, dest_fname, dtype='float32')

print('Rank {}: total time: {} s.'.format(rank, time.time() - t0))