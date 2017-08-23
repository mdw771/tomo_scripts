import tomosaic
import tomopy
import glob, os
import numpy as np
import dxchange
from mosaic_meta import *
from mosaic_util import *

# ==========================================
center_st = 5950/2
center_end = 6020/2
row_st = 0
row_end = 16
mode = 'discrete' # 'discrete' or 'merged' or 'single'
in_tile_pos = 600
ds = 2
dest_folder = 'center'
# merged:
slice = 4810
# discrete:
source_folder = 'data_raw_2x'
# merged:
fname = 'fulldata_flatcorr_1x/fulldata_flatcorr_1x.h5'
# single:
sino_name = 'sino_4810.tiff'
preprocess_single = False
# ==========================================


try:
    shift_grid = tomosaic.util.file2grid("shifts.txt")
    shift_grid = tomosaic.absolute_shift_grid(shift_grid, file_grid)
except:
    shift_grid = tomosaic.start_shift_grid(file_grid, x_shift, y_shift)

shift_grid = shift_grid / ds
in_tile_pos = in_tile_pos / ds

log = open('center_pos.txt', 'w')
center_pos = []

if mode == 'merged':
    for row in range(file_grid.shape[0]):
        sino = slice + shift_grid[row, 0, 0]
        tomopy.write_center(fname, os.path.join('center', str(row)), sino, center_st, center_end)
elif mode == 'discrete':
    for row in range(row_st, row_end):
        print('Row {}'.format(row))
        slice = shift_grid[row, 0, 0] + in_tile_pos
        center_vec = [center_st] * file_grid.shape[0]
        center_vec = np.array(center_vec)
        tomosaic.recon_block(file_grid, shift_grid, source_folder, dest_folder, (slice, slice+1), 1,
                             center_vec, algorithm='gridrec', test_mode=True, ds_level=0, save_sino=True,
                             blend_method='pyramid')
        sino = dxchange.read_tiff('center_temp/sino/sino_{:05d}.tiff'.format(slice))
        sino = sino.reshape([sino.shape[0], 1, sino.shape[1]])
        tomopy.write_center(sino, tomopy.angles(sino.shape[0]), dpath='center/{}'.format(row),
                            cen_range=(center_st, center_end))
elif mode == 'single':
    sino = dxchange.read_tiff(sino_name)
    sino = sino.reshape([sino.shape[0], 1, sino.shape[1]]) 
    if preprocess_single:
        sino = tomosaic.preprocess(np.copy(sino))
    tomopy.write_center(sino, tomopy.angles(sino.shape[0]), dpath='center', cen_range=(center_st, center_end))

#for row in range(file_grid.shape[0]):
#    center = tomosaic.misc.minimum_entropy(os.path.join('center', str(row)))
#    log.writelines(str(center) + '\n')
#    center_pos.append(center)

log.close()
