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
center_vec = np.array([5952, 5954, 5960, 5967, 5975, 5976, 5986, 5991, 5994, 5996, 6003, 6001, 6007, 6012, 6021, 6027])
tile_with_axis = 3
row_st = 1
row_end = 16
half_range = 5
x_shift = 1674
y_shift = 842
source_folder = 'data_raw_1x'
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
file_list = tomosaic.get_files(source_folder, prefix, type='h5')
file_grid = tomosaic.start_file_grid(file_list, pattern=1)
# try read shifts
try:
    shift_grid = tomosaic.util.file2grid("shifts.txt")
    shift_grid = tomosaic.absolute_shift_grid(shift_grid, file_grid)
except:
    shift_grid = tomosaic.start_shift_grid(file_grid, x_shift, y_shift)

center_wrt_mid = np.array(center_vec) - shift_grid[:, tile_with_axis, 1]

# try loading checkpoint
try:
    shift_grid, row_st, tile_covered, center_pos, buffer = load_checkpoint()
except:
    row = 0
    tile_covered = [False] * shift_grid.shape[1]
    tile_covered[tile_with_axis] = True
    center_pos = center_wrt_mid[row]
    buffer = None

tile_list = []
i = 1
while len(tile_list) < file_grid.shape[1]:
    if tile_with_axis - i >= 0:
        tile_list.append(tile_with_axis - i)
    if len(tile_list) < file_grid.shape[1] and tile_with_axis + i < file_grid.shape[1]:
        tile_list.append(tile_with_axis + i)
    i += 1

for row in range(row_st, row_end):

    print('At row: {}'.format(row))
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
            sinos[col] = load_sino(os.path.join(source_folder, file_grid[grid_lines[col], col]), slice_in_tile[col], normalize=True)
        except:
            pass
    if buffer is None:
        buffer = sinos[tile_with_axis]

    for col in tile_list:
        # if this is the first tested tile in this row, set center_pos to the value
        # with regards to middle (axis) tile
        if col == tile_list[0]:
            center_pos = center_wrt_mid[row]
        print('Row: {} Col: {}'.format(row, col))
        shift_old = shift_grid[row, :, :]
        axis_tile_shift = shift_old[tile_with_axis][1]
        if tile_covered[col]:
            continue
        sino = sinos[col]
        current_shift = shift_old[col, 1]
        ref_shift = shift_old[col+1, 1] if col < tile_with_axis else shift_old[col-1, 1]
        if col < tile_with_axis:
            max_len = int(buffer.shape[1] + ref_shift - np.floor(current_shift) + half_range)
        else:
            max_len = int(buffer.shape[1] + np.ceil(current_shift) - ref_shift + half_range)
        max_len += 1024
        for offset in range(-half_range, half_range):
            print('    Offset: {}'.format(offset))
            center_temp = center_pos
            temp_buffer = np.copy(buffer)
            new_shift = current_shift + offset
            relative_shift = new_shift - ref_shift
            if relative_shift < 0:
                print(sino.shape, temp_buffer.shape, -relative_shift)
                temp_buffer = blend(sino, temp_buffer, [0, -relative_shift], method='pyramid')
                center_temp -= relative_shift
            else:
                temp_buffer = blend(temp_buffer, sino, [0, relative_shift], method='pyramid')
            print('    Center before padding: {}'.format(center_temp))
            sino_feed = np.exp(-temp_buffer[:, np.newaxis, :])
            #dxchange.write_tiff(np.squeeze(sino_feed), 'bf_shift/test/bf', dtype='float32')
            sino_feed = tomopy.normalize_bg(sino_feed)
            #dxchange.write_tiff(np.squeeze(sino_feed), 'bf_shift/test/af', dtype='float32')
            sino_feed = tomopy.minus_log(sino_feed)
            sino_feed = np.pad(sino_feed, ((0, 0), (0, 0), (512, 512)), mode='constant')
            center_temp += 512
            dxchange.write_tiff(np.squeeze(sino_feed), 'bf_shift/test/af', dtype='float32')
            rec = tomopy.recon(sino_feed, tomopy.angles(sino_feed.shape[0]), algorithm='gridrec', center=center_temp)
            rec = np.squeeze(rec)
            out = np.zeros([max_len, max_len])
            out[:rec.shape[0], :rec.shape[1]] = rec
            dxchange.write_tiff(out, 'bf_shift/{}/{}/{}'.format(row, col, offset), dtype='float32', overwrite=True)
        best_offset = raw_input('Examine shifts now and enter the best offset value:\n')
        new_shift = shift_old[row, 1, 1] + best_offset
        shift_grid[row, col, 1] = new_shift
        relative_shift = new_shift - axis_tile_shift
        if relative_shift < 0:
            buffer = blend(sino, buffer, [0, -relative_shift], method='pyramid')
            center_pos -= relative_shift
        else:
            buffer = blend(buffer, sino, [0, relative_shift], method='pyramid')
        tile_covered[col] = True
        save_checkpoint(shift_grid, row, tile_covered, center_pos, buffer)
        print('Checkpoint saved.')
