from mosaic_meta import *
from tomosaic.misc.misc import read_data_adaptive
import tomopy
import numpy as np
import dxchange

# ===============================================
folder = 'data_raw_1x'
# ===============================================

for (y, x), f in np.ndenumerate(file_grid):
    print(y, x)
    dat, flt, drk, _ = read_data_adaptive(os.path.join(folder, f), proj=(0, 1), data_format=data_format)
    dat = tomopy.normalize(dat, flt, drk)
    dxchange.write_tiff(dat, 'first_frames/y{}x{}'.format(y, x), dtype='float32', overwrite=True)