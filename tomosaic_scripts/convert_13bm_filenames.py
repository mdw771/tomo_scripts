import re
import os
import glob


# ====================================
prefix = 'WholeBrain03_'
suffix_main = '_2.nc' # use suffix of main data file
suffix_flat1 = '_1.nc'
suffix_flat2 = '_3.nc'
suffix_setup = '_.setup'
n_col = 5
flip = True
# ====================================


# main data files

filelist = glob.glob(prefix + '*' + suffix_main)
print filelist
for f in filelist:
    index = int(re.findall((prefix + '(\d)' + suffix_main), f)[0]) - 1
    y = int(index / n_col)
    x = index % int(n_col)
    if flip:
        x = n_col - 1 - x
    newname = prefix + 'y{:02d}_x{:02d}.nc'.format(y, x)
    os.rename(f, newname)

# flat 1

filelist = glob.glob(prefix + '*' + suffix_flat1)
print filelist
for f in filelist:
    index = int(re.findall((prefix + '(\d)' + suffix_flat1), f)[0]) - 1
    y = int(index / n_col)
    x = index % int(n_col)
    if flip:
        x = n_col - 1 - x
    newname = prefix + 'y{:02d}_x{:02d}_flat1.nc'.format(y, x)
    os.rename(f, newname)

# flat 2

filelist = glob.glob(prefix + '*' + suffix_flat2)
print filelist
for f in filelist:
    index = int(re.findall((prefix + '(\d)' + suffix_flat2), f)[0]) - 1
    y = int(index / n_col)
    x = index % int(n_col)
    if flip:
        x = n_col - 1 - x
    newname = prefix + 'y{:02d}_x{:02d}_flat2.nc'.format(y, x)
    os.rename(f, newname)

# setup

filelist = glob.glob(prefix + '*' + suffix_setup)
print filelist
for f in filelist:
    index = int(re.findall((prefix + '(\d)' + suffix_setup), f)[0]) - 1
    y = int(index / n_col)
    x = index % int(n_col)
    if flip:
        x = n_col - 1 - x
    newname = prefix + 'y{:02d}_x{:02d}.setup'.format(y, x)
    os.rename(f, newname)
