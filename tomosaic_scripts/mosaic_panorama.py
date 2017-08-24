import tomosaic
from tomosaic import *
from tomosaic.misc.misc import read_data_adaptive
import glob, os
import numpy as np
from mosaic_meta import *
import dxchange

# ==========================================
frame = 0
method = 'pyramid'
margin=50
src_folder = 'data_raw_1x'
blend_options = {'depth': 7,
                 'blur': 0.4}
# ==========================================

def preprocess(dat, blur=None):

    dat[np.abs(dat) < 2e-3] = 2e-3
    dat[dat > 1] = 1
    dat = -np.log(dat)
    dat[np.where(np.isnan(dat) == True)] = 0
    if blur is not None:
        dat = gaussian_filter(dat, blur)

    return dat

root = os.getcwd()
os.chdir(src_folder)
shift_grid = tomosaic.start_shift_grid(file_grid, x_shift, y_shift)
last_none = False
buff = np.zeros([1, 1])
for (y, x), value in np.ndenumerate(file_grid):
    if value != None:
        prj, flt, drk = read_data_adaptive(value, proj=(frame, frame + 1))
        prj = tomopy.normalize(prj, flt, drk)
        prj = preprocess(np.copy(prj))
        buff = blend(buff, np.squeeze(prj), shift_grid[y, x, :], method=method)
        print(y, x)

os.chdir(root)

dxchange.write_tiff(buff, 'panos/{}_norm'.format(frame), dtype='float32', overwrite=True)
