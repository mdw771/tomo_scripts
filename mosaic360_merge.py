import tomosaic
import glob, os
import numpy as np
from mosaic360_meta import *
try:
    from mpi4py import MPI
except:
    from tomosaic.util.pseudo import pseudo_comm


try:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    name = MPI.Get_processor_name()
except:
    comm = pseudo_comm()
    rank = 0
    size = 1

# ==========================================
blend_options = {'depth': 7,
                 'blur': 0.4}
# ==========================================


shift_grid = tomosaic.util.file2grid("shifts.txt")
shift_grid = tomosaic.absolute_shift_grid(shift_grid, file_grid)
print shift_grid

tomosaic.util.total_fusion('.', 'fulldata_flatcorr_1x', 'fulldata_flatcorr_1x.h5', file_grid,
                           shift_grid.astype('int'), blend_method='pyramid', blend_method2='overlay',
                           color_correction=False, blend_options=blend_options, dtype='float16')