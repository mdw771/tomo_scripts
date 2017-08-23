from scipy.ndimage import rotate
import numpy as np
import dxchange
import glob, os
import time
try:
    from mpi4py import MPI
except:
    pass

# ==============================
angle = 34
crop = (540*2, 2380*2, 8240*2, 5820*2) # (x_start, y_start, x_length, y_length; same as ImageJ convention)
min = 0
max = 0.0015
bitdepth = 8
saturation = 35
src_folder = 'recon'
dest_folder = 'recon_crop_rot_8'
# ==============================

def allocate_mpi_subsets(n_task, size, task_list=None):

    if task_list is None:
        task_list = range(n_task)
    sets = []
    for i in range(size):
        selection = range(i, n_task, size)
        sets.append(np.array(np.take(task_list, selection).tolist()))
    return sets

raw_folder = os.getcwd()
os.chdir(src_folder)
filelist = glob.glob('*.tif*')
filelist.sort()
filelist = filelist[200:]
os.chdir(raw_folder)

try:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    name = MPI.Get_processor_name()
except:
    rank = 0
    size = 1

try:
    os.chdir(dest_folder)
except:
    os.mkdir(dest_folder)
    os.chdir(dest_folder)
templist = []
for f in filelist:
    if not os.path.isfile(f):
        templist.append(f)
filelist = templist[:]
os.chdir(raw_folder)

sets = allocate_mpi_subsets(len(filelist), size, task_list=filelist)

print sets[rank]

for fname in sets[rank]:

    print(fname)
    img = dxchange.read_tiff(os.path.join(src_folder, fname))
    img = rotate(img, -angle, reshape=False)
    img = img[crop[1]:crop[1]+crop[3], crop[0]:crop[0]+crop[2]]

    bit_full = 2 ** bitdepth - 1
    img = (img - min) / (max - min) * bit_full
    img[img<0] = 0
    img = np.clip(img, 0, bit_full)

    dxchange.write_tiff(img, os.path.join(dest_folder, fname), dtype='uint{:d}'.
format(bitdepth))