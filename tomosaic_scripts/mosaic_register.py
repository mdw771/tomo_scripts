import tomosaic
import glob
import os
import numpy as np
from mpi4py import MPI
from mosaic_meta import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

print file_grid
root = os.getcwd()
os.chdir('data_raw_1x')
shift_grid = tomosaic.start_shift_grid(file_grid, x_shift, y_shift)
refined_shift = tomosaic.refine_shift_grid(file_grid, shift_grid, motor_readout=(y_shift, x_shift))
os.chdir(root)
np.savetxt('shifts.txt', refined_shift, fmt=str('%4.2f'))

