import tomosaic


prefix = 'booooooooooooooom'
file_list = tomosaic.get_files('.', prefix, type='h5')
file_grid = tomosaic.start_file_grid(file_list, pattern=1)
x_shift = 9999
y_shift = 9999
