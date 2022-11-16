"""
Visualize the sampling volume
"""

from mayavi import mlab
from read_opt_res import read_opt_res

INP_PATH = '/home/andzhda/storage/Data/MEGSim/2022-06-22_paper_RC2_full_run/run_thick/out'
params, sens_array, interm_res, opt_res, iter_indx = read_opt_res(INP_PATH, max_n_samp=3)

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
rmags = sens_array._sampling_locs_rmags
mlab.clf(fig)
mlab.points3d(rmags[:, 0], rmags[:, 1], rmags[:, 2], resolution=32, scale_factor=0.005, color=(0, 0, 1))
mlab.show()
