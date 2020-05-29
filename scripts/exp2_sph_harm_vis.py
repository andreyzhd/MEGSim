#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 15:28:17 2020

@author: andrey

Visualize/compare the magnetic fields comuted with Biot-Savart and spherical
harmonics expansion (if everything works correctly, the fields should be very
similar). Read the fields from a file computed with another script.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab


DATA_FNAME = '/tmp/vsh.pkl'

def decomp_B(B, phi_nsamp, theta_nsamp):
    """Return x, y, z components of B in phi_nsamp-by-theta_nsamp matrices."""
    return B[:,0].reshape(phi_nsamp, theta_nsamp), B[:,1].reshape(phi_nsamp, theta_nsamp), B[:,2].reshape(phi_nsamp, theta_nsamp)
                                                                                                            

f = open(DATA_FNAME, 'rb')
B_bs, B_vsh, x, y, z, alphas = pickle.load(f)
B_vsh = np.real(B_vsh)
f.close()

phi_nsamp, theta_nsamp = x.shape


##-----------------------------------------------------------------------------
# Plot 3D
#
s = np.linalg.norm(B_bs, axis=1)

fig1 = mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
mlab.clf(fig1)
mlab.mesh(x, y, z, scalars=s.reshape((phi_nsamp, theta_nsamp)), colormap='viridis', figure=fig1)
fig1.name='Biot Savart'

s = np.linalg.norm(np.real(B_vsh), axis=1)

fig2 = mlab.figure(2, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
mlab.clf(fig2)
mlab.mesh(x, y, z, scalars=s.reshape((phi_nsamp, theta_nsamp)), colormap='viridis', figure=fig2)
fig2.name='VSH'

mlab.sync_camera(fig1, fig2)

##-----------------------------------------------------------------------------
# Plot 2D
#
def _plot_comp(axis, B, maxval, title):
    axis.imshow(np.transpose(B),
                cmap='viridis',
                vmin=-maxval, vmax=maxval,
                interpolation='lanczos',
                origin='lower',
                extent=[0, 2*np.pi, 0, np.pi])
    axis.set_xlabel(r'$\phi$, rad')
    axis.set_ylabel(r'$\theta$, rad')
    axis.set_title(title)

B_bs_x, B_bs_y, B_bs_z = decomp_B(B_bs, phi_nsamp, theta_nsamp)
B_vsh_x, B_vsh_y, B_vsh_z = decomp_B(B_vsh, phi_nsamp, theta_nsamp)
maxval = np.max(np.abs(np.stack((B_bs, B_vsh))))

fig, axs = plt.subplots(2, 3)

_plot_comp(axs[0,0], B_bs_x, maxval, r'$B_x$, Biot-Savart')
_plot_comp(axs[0,1], B_bs_y, maxval, r'$B_y$, Biot-Savart')
_plot_comp(axs[0,2], B_bs_z, maxval, r'$B_z$, Biot-Savart')

_plot_comp(axs[1,0], B_vsh_x, maxval, r'$B_x$, VSH')
_plot_comp(axs[1,1], B_vsh_y, maxval, r'$B_y$, VSH')
_plot_comp(axs[1,2], B_vsh_z, maxval, r'$B_z$, VSH')