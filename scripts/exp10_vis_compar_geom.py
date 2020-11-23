"""
Iteratively optimize sensor array locations -- compare the resulting sensor geometries
"""

#%% Inits
import pickle
import numpy as np
from mayavi import mlab

FNAME_TEMPL_SL = '/home/andrey/storage/Data/MEGSim/2020-11-19_iterative/iter_opt_chin_%0.3frad.pkl'
FNAME_TEMPL_DL_5000 = '/home/andrey/storage/Data/MEGSim/2020-11-23_iterative_double_layer/iter_opt_chin_dl_5000_%0.3frad.pkl'
FNAME_TEMPL_DL_10000 = '/home/andrey/storage/Data/MEGSim/2020-11-23_iterative_double_layer/iter_opt_chin_dl_10000_%0.3frad.pkl'

CHIN_STRAP_ANGLE = 3/16 * np.pi

def _read_n_plot(fname, title, scaling):
    fl = open(fname, 'rb')
    dt = pickle.load(fl)
    fl.close()

    x_helm = dt['x_sphere'][dt['helm_indx']] * scaling
    y_helm = dt['y_sphere'][dt['helm_indx']] * scaling
    z_helm = dt['z_sphere'][dt['helm_indx']] * scaling
    sens_indx = dt['sens_indx']

    fig = mlab.figure('%s %i' % (title, len(x_helm)), bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
    mlab.clf(fig)
    mlab.points3d(x_helm, y_helm, z_helm, scale_factor=0.005, color=(0, 1, 0), resolution=32, opacity=0.3)
    mlab.points3d(x_helm[sens_indx], y_helm[sens_indx], z_helm[sens_indx], scale_factor=0.01, color=(0, 0, 1), resolution=32)
    return fig


#%% Read the data
fig_sl = _read_n_plot(FNAME_TEMPL_SL % CHIN_STRAP_ANGLE, 'Single layer', 0.15)
fig_dl_5 = _read_n_plot(FNAME_TEMPL_DL_5000 % CHIN_STRAP_ANGLE, 'Double layer', 1)
fig_dl_10 = _read_n_plot(FNAME_TEMPL_DL_10000 % CHIN_STRAP_ANGLE, 'Double layer', 1)

mlab.sync_camera(fig_sl, fig_dl_5)
mlab.sync_camera(fig_sl, fig_dl_10)
mlab.show()