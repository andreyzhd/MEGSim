"""
Visualize the sampling volumes, 2D and 3D
"""

from mayavi import mlab
from figutils import read_opt_res

INP_PATH_3D = '/home/andrey/storage/Data/MEGSim/2022-06-22_paper_RC2_full_run/run_thick/out'
INP_PATH_2D = '/home/andrey/storage/Data/MEGSim/2022-12-20_paper_RC2_full_run/run_thin/out'

sens_array_thick = read_opt_res(INP_PATH_3D, max_n_samp=3)[1]
sens_array_thin = read_opt_res(INP_PATH_2D, max_n_samp=3)[1]

fig_thick = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
rmags = sens_array_thick._sampling_locs_rmags
mlab.clf(fig_thick)
mlab.points3d(rmags[:, 0], rmags[:, 1], rmags[:, 2], resolution=32, scale_factor=0.005, color=(0, 0, 1))

fig_thin = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
rmags = sens_array_thin._sampling_locs_rmags
mlab.clf(fig_thin)
mlab.points3d(rmags[:, 0], rmags[:, 1], rmags[:, 2], resolution=32, scale_factor=0.005, color=(0, 0, 1))

mlab.sync_camera(fig_thick, fig_thin)

mlab.show()

