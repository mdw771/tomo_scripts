import tomosaic
import glob
import os
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

shift_grid = tomosaic.start_shift_grid(file_grid, x_shift, y_shift)
refined_shift = tomosaic.refine_shift_grid(file_grid, shift_grid)
if rank == 0:
    np.savetxt('shifts.txt', refined_shift, fmt=str('%4.2f'))

