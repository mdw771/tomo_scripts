import tomopy
import dxchange
import numpy as np

# ===================================================
fname = 'sino_4810.tiff'
algorithm = 'fista' # 'gridrec', 'fista'
center = 5976
extra_options = {'tv_reg':0.000005,'bmin':0.0,'print_progress':True}
n_iterations = 200
out_name = 'iter'
# ===================================================

sino = dxchange.read_tiff(fname)
sino = sino[:, np.newaxis, :]
theta = tomopy.angles(sino.shape[0])

if algorithm == 'gridrec':
    rec = tomopy.recon(sino, theta, center, algorithm='gridrec')
elif algorithm == 'fista':
    import tvtomo
    import astra
    astra.plugin.register(tvtomo.plugin)
    print astra.plugin.get_help('TV-FISTA')
    rec = tomopy.recon(sino, theta, center=center, algorithm=tomopy.astra,
                       options={'method':'TV-FISTA', 'proj_type':'cuda', 'num_iter':n_iterations})
else:
    rec = None
    print('You f*ed it up')

dxchange.write_tiff(rec, out_name, dtype='float32')
