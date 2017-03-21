import tomosaic
import glob
import os
import numpy as np
import mpi4py as MPI
from mosaic360_meta import *


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

shift_grid = tomosaic.start_shift_grid(file_grid, x_shift, y_shift)
refined_shift = tomosaic.refine_shift_grid(file_grid, shift_grid)
if rank == 0:
    np.savetxt('shifts.txt', refined_shift, fmt=str('%4.2f'))

