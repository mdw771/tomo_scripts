import tomosaic
import tomopy
import glob, os
import numpy as np
import dxchange
try:
    from mosaic_meta import *
except:
    reader = open(os.path.join('tomosaic_misc', 'meta'), 'rb')
    prefix, file_grid, x_shift, y_shift = pickle.load(reader)
    reader.close()
from mosaic_util import *

# ==========================================
center_st = 5250
center_end = 5280
center_step = 1
row_st = 0
row_end = 1
method = 'vo' # 'manual' or 'vo'
mode = 'discrete' # 'discrete' or 'merged' or 'single'
in_tile_pos = 1000
ds = 1
dest_folder = 'center'
# merged:
slice = 1000
# discrete:
source_folder = 'data_raw_1x'
# merged:
fname = 'fulldata_flatcorr_1x/fulldata_flatcorr_1x.h5'
# single:
sino_name = 'sino_4810.tiff'
preprocess_single = False
# ==========================================

import time
import logging
from scipy.misc import imresize
from tomosaic.center import *

logger = logging.getLogger(__name__)

try:
    shift_grid = tomosaic.util.file2grid("shifts.txt")
    shift_grid = tomosaic.absolute_shift_grid(shift_grid, file_grid)
except:
    shift_grid = tomosaic.start_shift_grid(file_grid, x_shift, y_shift)

shift_grid = shift_grid / ds
in_tile_pos = in_tile_pos / ds

t0 = time.time()
if mode == 'merged':
    find_center_merged(fname, shift_grid, (row_st, row_end), (center_st, center_end), center_step, slice=slice,
                       method=method)
elif mode == 'discrete':
    find_center_discrete(source_folder, file_grid, shift_grid, (row_st, row_end), (center_st, center_end), center_step,
                         slice=slice, method=method)
elif mode == 'single':
    find_center_single(sino_name, (center_st, center_end), center_step, preprocess_single=preprocess_single,
                       method=method)
print('Rank {}: total time = {} s.'.format(rank, time.time() - t0))