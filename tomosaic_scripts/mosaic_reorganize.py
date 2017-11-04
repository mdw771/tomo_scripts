import tomosaic
import time
from mpi4py import MPI
from mosaic_meta import *
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

t0 = time.time()
# os.chdir('.')
tomosaic.reorganize_dir(file_list, raw_ds=(1, 4))

#file_list = ['WholeBrainMRI_phase35cm_5x_2k_gap31_exp30_newfocus_y0_x6.h5']
#tomosaic.util.reorganize_dir(file_list, raw_ds=(2,))

print('Total time: {}s'.format(time.time() - t0))