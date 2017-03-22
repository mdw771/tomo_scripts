import tomosaic
import glob, os
import numpy as np
from mosaic360_meta import *
from mosaic360_util import *
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
slice = 500
center_st = 9999
center_end = 9999
fname = 'fulldata_flatcorr_1x/fulldata_flatcorr_1x.h5'
# ==========================================


shift_grid = tomosaic.util.file2grid("shifts.txt")
shift_grid = tomosaic.absolute_shift_grid(shift_grid, file_grid)
print shift_grid

log = open('center_pos.txt', 'w')
center_pos = []

for row in range(file_grid.shape[0]):
    sino = slice + shift_grid[row, 0, 0]
    print(sino)
    write_center_360(fname, os.path.join('center', str(row)), sino, center_st, center_end)

for row in range(file_grid.shape[0]):
    center = tomosaic.misc.minimum_entropy(os.path.join('center', str(row)))
    log.writelines(str(center) + '\n')
    center_pos.append(center)

log.close()