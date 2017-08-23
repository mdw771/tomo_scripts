import tomosaic
from tomosaic import *
import glob, os
import numpy as np
from mosaic_meta import *
import dxchange

# ==========================================
frame = 3600
method = 'pyramid'
margin=50
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

shift_grid = tomosaic.start_shift_grid(file_grid, x_shift, y_shift)
last_none = False
buff = np.zeros([1, 1])
for (y, x), value in np.ndenumerate(file_grid):
    if value != None:
        prj, flt, drk = read_aps_32id_adaptive('data_raw_1x/' + value, proj=(frame, frame + 1))
        #prj = tomopy.normalize(prj, flt, drk)
        #prj = dxchange.read_tiff('partial_projections/y{}x{}_180.tiff'.format(y, x))
        #prj = preprocess(np.copy(prj))
        #t0 = time.time()
        prj[np.isnan(prj)] = 0
        buff = blend(buff, np.squeeze(prj), shift_grid[y, x, :], method=method, **blend_options)
dxchange.write_tiff(buff, 'panos/{}_nonorm'.format(frame), dtype='float32', overwrite=True)
