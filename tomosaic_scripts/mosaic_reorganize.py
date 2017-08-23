import tomosaic
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

prefix = 'WholeBrainMRI_phase35cm_5x_2k_gap31_exp30_newfocus'
file_list = tomosaic.get_files('data_raw_1x', prefix, type='h5')
file_list.sort()
print(file_list)

#tomosaic.util.reorganize_dir(file_list, raw_ds=(1, 2, 4, 8, 16, 32))

file_list = ['WholeBrainMRI_phase35cm_5x_2k_gap31_exp30_newfocus_y0_x6.h5']
tomosaic.util.reorganize_dir(file_list, raw_ds=(2,))
