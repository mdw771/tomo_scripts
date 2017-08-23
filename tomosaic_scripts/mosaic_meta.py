import tomosaic
import pickle
import os

prefix = 'WholeBrainMRI_phase35cm_5x_2k_gap31_exp30_newfocus'
file_list = tomosaic.get_files('data_raw_1x', prefix, type='h5')
file_grid = tomosaic.start_file_grid(file_list, pattern=1)
x_shift = 1674
y_shift = 842

try:
    os.mkdir('tomosaic_misc')
except:
    pass
writer = open(os.path.join('tomosaic_misc', 'meta'), 'wb')
pickle.dump([prefix, file_grid, x_shift, y_shift], writer)
writer.close()