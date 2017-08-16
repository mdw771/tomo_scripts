import tomosaic
from tomosaic.merge import blend
from tomosaic.recon import load_sino
import tomopy
import glob
import dxchange
import os
import pickle
import numpy as np
from mpi4py import MPI

# ===========================================
center_wrt_midtile = 948.25
tile_with_axis = 3
row_st = 0
row_end = 16
half_range = 5
x_shift = 1674
y_shift = 842
# ===========================================

def save_checkpoint(shift_grid, row, tile_covered, center, sino):
    f = open(os.path.join('checkpoints', 'checkpoint'), 'wb')
    pickle.dump([shift_grid, row, tile_covered, center], f)
    np.save(os.path.join('checkpoints', 'current_sino.npy'), sino)
    f.close()

def load_checkpoint():
    f = open(os.path.join('checkpoints', 'checkpoint'), 'rb')
    sino = np.load(os.path.join('checkpoints', 'current_sino.npy'))
    ret = pickle.load(f)
    ret.append(sino)
    return ret


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

prefix = 'WholeBrainMRI_phase35cm_5x_2k_gap31_exp30_newfocus'
file_list = tomosaic.get_files('data_raw_1x', prefix, type='h5')
file_grid = tomosaic.start_file_grid(file_list, pattern=1)

root = os.getcwd()
os.chdir('data_raw_1x')
shift_grid = tomosaic.start_shift_grid(file_grid, x_shift, y_shift)

# try loading checkpoint
try:
    shift_grid, row_st, tile_covered, center_pos, buffer = load_checkpoint()
except:
    row = 0
    tile_covered = [False] * shift_grid.shape[1]
    tile_covered[tile_with_axis] = True
    center_pos = center_wrt_midtile
    buffer = None

tile_list = []
i = 1
while len(tile_list) < file_grid.shape[1]:
    tile_list.append(tile_with_axis - i)
    if len(tile_list) < file_grid.shape[1]:
        tile_list.append(tile_with_axis + i)
    i += 1

for row in range(row_st, row_end):

    print('At row: {}'.format(row))
    shift_old = shift_grid[row, :, :]
    axis_tile_shift = shift_old[tile_with_axis]
    i_slice = int(shift_grid[row, 0, 0] + y_shift / 2)
    # read in proper sinograms
    pix_shift_grid = np.ceil(shift_grid)
    pix_shift_grid[pix_shift_grid < 0] = 0
    grid_lines = np.zeros(file_grid.shape[1], dtype=np.int)
    # compensate vertical shifts
    slice_in_tile = np.zeros(file_grid.shape[1], dtype=np.int)
    for col in range(file_grid.shape[1]):
        bins = pix_shift_grid[:, col, 0]
        grid_lines[col] = int(np.squeeze(np.digitize(i_slice, bins)) - 1)
        if grid_lines[col] == -1:
            print(
                "WARNING: The specified starting slice number does not allow for full sinogram construction. Trying next slice...")
            mod_start_slice = 1
            break
        else:
            mod_start_slice = 0
        slice_in_tile[col] = i_slice - bins[grid_lines[col]]

    sinos = [None] * file_grid.shape[1]
    for col in range(file_grid.shape[1]):
        try:
            sinos[col] = load_sino(file_grid[grid_lines[col], col], slice_in_tile[col], normalize=True)
        except:
            pass
    if buffer is None:
        buffer = sinos[tile_with_axis]

    for col in tile_list:
        if tile_covered[col]:
            continue
        sino = sinos[col]
        current_shift = shift_old[row, 1]
        for offset in range(-half_range, half_range):
            new_shift = current_shift + offset
            relative_shift = new_shift - axis_tile_shift
            if relative_shift < 0:
                sino = blend(sino, buffer, [0, -relative_shift], method='pyramid')
                center_pos -= relative_shift
            else:
                sino = blend(buffer, sino, [0, relative_shift], method='pyramid')
            sino_feed = np.exp(-sino)
            sino_feed = tomopy.normalize_bg(sino_feed[:, np.newaxis, :])
            sino_feed = tomopy.minus_log(sino_feed)
            rec = tomopy.recon(sino_feed, tomopy.angles(sino_feed.shape[0]), algorithm='gridrec', center=center_pos)
            dxchange.write_tiff('bf_shift/{}/{}/{}'.format(row, col, offset), dtype='float32', overwrite=True)
        best_offset = raw_input('Examine shifts now and enter the best offset value:\n')
        new_shift = shift_old[row, 1] + best_offset
        shift_grid[row, col, 1] = new_shift
        relative_shift = new_shift - axis_tile_shift
        if relative_shift < 0:
            sino = blend(sino, buffer, [0, -relative_shift], method='pyramid')
            center_pos -= relative_shift
        else:
            sino = blend(buffer, sino, [0, relative_shift], method='pyramid')
        tile_covered[col] = True
        save_checkpoint(shift_grid, row, tile_covered, center_pos, sino)
        print('Checkpoint saved.')
