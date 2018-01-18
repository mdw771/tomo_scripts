import tomosaic
import glob
import os
import numpy as np
from mpi4py import MPI
import pickle
import time
try:
    from mosaic_meta import *
except:
    reader = open(os.path.join('tomosaic_misc', 'meta'), 'rb')
    prefix, file_grid, x_shift, y_shift = pickle.load(reader)
    reader.close()

# =============================================
method = 'pc'
src_folder = 'data_raw_1x'
# =============================================

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

t0 = time.time()
root = os.getcwd()
shift_grid = tomosaic.start_shift_grid(file_grid, x_shift, y_shift)
if method == 'pc':
    refined_shift = tomosaic.refine_shift_grid(file_grid, shift_grid, src_folder=src_folder, motor_readout=(y_shift, x_shift))
elif method == 'reslice':
    refined_shift = tomosaic.refine_shift_grid_reslice(file_grid, shift_grid, '.', center_search_range=(240, 260))
print('Rank {}: total time: {} s.'.format(rank, time.time() - t0))
