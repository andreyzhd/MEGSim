#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 12:45:25 2020

@author: andrey

Try to fit a dipole with arbitrary sensor array using brute-force search
"""

import time
import pickle
import numpy as np
from mayavi import mlab

from megsimutils.utils import spherepts_golden
from megsimutils import dipfld_sph
from megsimutils.dipole_fit import bf_dipole_fit

N_COILS = 100
R = 0.15
ANGLE = 4*np.pi/3

Q = np.array([100, 0, 0]) * 1e-9    # Dipole moment
RQ = np.array([0, 0, 0.13])         # Dipole location

# search domain for dipole fitting
R_MIN = R*0.1
R_MAX = R*0.95
THETA_MAX = np.arccos(1 - ANGLE/(2*np.pi))  # Theta corresponding to the area
                                            # covered by the sensors. Only
                                            # valid for ANGLE <= 2*pi

N_R = 100
N_THETA = 100
N_PHI = 4*N_THETA

OUTFILE = '/home/andrey/scratch/dipole_bf.pkl'
        
#%% Do the work         
t_start = time.time()

# Create sensor array
rmags = spherepts_golden(N_COILS, angle=ANGLE) * R
cosmags = spherepts_golden(N_COILS, angle=ANGLE) # radial sensor orientation

# Create a dipole
field = dipfld_sph(Q, RQ, rmags, np.zeros(3))
data = (field*cosmags).sum(axis=1)

pos, ori, resid, locs, resids = bf_dipole_fit(rmags, cosmags, data, {'rmin' : R_MIN,
                                                       'rmax' : R_MAX,
                                                       'theta_max' : THETA_MAX,
                                                       'n_r' : N_R,
                                                       'n_theta' : N_THETA,
                                                       'n_phi' : N_PHI},
                                                       debug_data=True)

#%% Visualise the results

# Original dipole's field
fig1 = mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))

## Visualize the points
pts = mlab.points3d(rmags[:,0], rmags[:,1], rmags[:,2], data, scale_mode='none', scale_factor=0.01)

## Plot the original dipole
scQ = Q / np.linalg.norm(Q)
mlab.quiver3d(RQ[0], RQ[1], RQ[2], scQ[0], scQ[1], scQ[2], scale_factor=0.05)

## Plot the reconstructed dipole
sc_ori = ori / np.linalg.norm(ori)
mlab.quiver3d(pos[0], pos[1], pos[2], sc_ori[0], sc_ori[1], sc_ori[2], scale_factor=0.05, color=(0,0,0))

mlab.points3d(0, 0, 0, resolution=32, scale_factor=0.01, color=(0,1,0), mode='axes')

## Create and visualize the mesh
mesh = mlab.pipeline.delaunay2d(pts)
surf = mlab.pipeline.surface(mesh)

mlab.title('Original dipole\'s field', size=0.5)

# Reconstructed dipole's field
rec_field = dipfld_sph(sc_ori*np.linalg.norm(Q), pos, rmags, np.zeros(3))
rec_data = (rec_field*cosmags).sum(axis=1)

fig2 = mlab.figure(2, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))

## Visualize the points
pts = mlab.points3d(rmags[:,0], rmags[:,1], rmags[:,2], rec_data, scale_mode='none', scale_factor=0.01)

## Plot the original dipole
scQ = Q / np.linalg.norm(Q)
mlab.quiver3d(RQ[0], RQ[1], RQ[2], scQ[0], scQ[1], scQ[2], scale_factor=0.05)

## Plot the reconstructed dipole
sc_ori = ori / np.linalg.norm(ori)
mlab.quiver3d(pos[0], pos[1], pos[2], sc_ori[0], sc_ori[1], sc_ori[2], scale_factor=0.05, color=(0,0,0))

mlab.points3d(0, 0, 0, resolution=32, scale_factor=0.01, color=(0,1,0), mode='axes')

## Create and visualize the mesh
mesh = mlab.pipeline.delaunay2d(pts)
surf = mlab.pipeline.surface(mesh)

mlab.title('Reconstructed dipole\'s field', size=0.5)

mlab.sync_camera(fig1, fig2)

print('Discrepancey between the true field and field from the reconstructed dipole is %0.3f%%' % (np.linalg.norm(data-rec_data) / np.linalg.norm(data) * 100))
print('The execution took %i seconds' % (time.time()-t_start))

# Save the results
fl = open(OUTFILE, 'wb')
pickle.dump((locs, resids), fl)
fl.close()

input('Press enter to finish ...')