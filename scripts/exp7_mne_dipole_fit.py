#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 12:45:25 2020

@author: andrey

Try to fit a dipole with arbitrary sensor array using MNE-Python dipole fitting
"""

import numpy as np
from mayavi import mlab

import mne
from mne.io.constants import FIFF
from megsimutils.utils import spherepts_golden, sensordata_to_ch_dicts
from megsimutils import dipfld_sph

N_COILS = 100
R = 0.15
ANGLE = 4*np.pi/3

Q = np.array([100, 0, 0]) * 1e-9    # Dipole moment
RQ = np.array([0, 0, 0.13])         # Dipole location


def _create_sim_info(rmags, cosmags):
        assert len(rmags) == len(cosmags)
    
        n_coils = len(rmags)
        ch_names = ['MEG%03i' % i for i in range(n_coils)]
        info = mne.create_info(ch_names, 1, ch_types='mag', verbose=None)

        coil_type = FIFF.FIFFV_COIL_POINT_MAGNETOMETER
        coil_types = n_coils * [coil_type]
    
        # apply no rotations to integration points
        Iprot = np.zeros(n_coils)
        sensors_ = list(sensordata_to_ch_dicts(rmags, cosmags, Iprot, coil_types))
        info['chs'] = sensors_
        info['nchan'] = len(sensors_)
        info['ch_names'] = [ch['ch_name'] for ch in info['chs']]

        return info

class SimEvoked(mne.Evoked):
    """A variation of Evoked class for simulated data and arbitrary sensor geometry."""
    
    def __init__(self, rmags, cosmags, data):
        """
        Parameters
        ----------
        rmags : N-by-3 matrix of sensor locations
        cosmags : N-by-3 matrix of sensor orientations
        data : N-length vector of sensor readings

        Returns
        -------
        None.

        """
        assert len(rmags) == len(data)
    
        self.info = _create_sim_info(rmags, cosmags)
        self.nave = 1   # number of averages
        #self._aspect_kind = 100 # WTF is this?
        self.comment = 'Simulated evoked file'
        self.times = np.zeros(1)
        self.data = data[:, np.newaxis]
        self._update_first_last()
        self.verbose = None
        self.preload = True
        
                
# Create sensor array
rmags = spherepts_golden(N_COILS, angle=ANGLE) * R
cosmags = spherepts_golden(N_COILS, angle=ANGLE) # radial sensor orientation

# Create a dipole
field = dipfld_sph(Q, RQ, rmags, np.zeros(3))
data = (field*cosmags).sum(axis=1)

sim_evoked = SimEvoked(rmags, cosmags, data)
cov = mne.make_ad_hoc_cov(_create_sim_info(rmags, cosmags), std=1)

dip, res = mne.fit_dipole(sim_evoked, cov, mne.make_sphere_model())

#%% Visualise the results

# Original dipole's field
fig1 = mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))

## Visualize the points
pts = mlab.points3d(rmags[:,0], rmags[:,1], rmags[:,2], data, scale_mode='none', scale_factor=0.01)

## Plot the original dipole
scQ = Q / np.linalg.norm(Q)
mlab.quiver3d(RQ[0], RQ[1], RQ[2], scQ[0], scQ[1], scQ[2], scale_factor=0.05)

## Plot the reconstructed dipole
sc_ori = dip.ori[0] / np.linalg.norm(dip.ori[0])
mlab.quiver3d(dip.pos[0][0], dip.pos[0][1], dip.pos[0][2], sc_ori[0], sc_ori[1], sc_ori[2], scale_factor=0.05, color=(0,0,0))

mlab.points3d(0, 0, 0, resolution=32, scale_factor=0.01, color=(0,1,0), mode='axes')

## Create and visualize the mesh
mesh = mlab.pipeline.delaunay2d(pts)
surf = mlab.pipeline.surface(mesh)

mlab.title('Original dipole\'s field', size=0.5)

# Reconstructed dipole's field
rec_field = dipfld_sph(sc_ori*np.linalg.norm(Q), dip.pos[0], rmags, np.zeros(3))
rec_data = (rec_field*cosmags).sum(axis=1)

fig2 = mlab.figure(2, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))

## Visualize the points
pts = mlab.points3d(rmags[:,0], rmags[:,1], rmags[:,2], rec_data, scale_mode='none', scale_factor=0.01)

## Plot the original dipole
scQ = Q / np.linalg.norm(Q)
mlab.quiver3d(RQ[0], RQ[1], RQ[2], scQ[0], scQ[1], scQ[2], scale_factor=0.05)

## Plot the reconstructed dipole
sc_ori = dip.ori[0] / np.linalg.norm(dip.ori[0])
mlab.quiver3d(dip.pos[0][0], dip.pos[0][1], dip.pos[0][2], sc_ori[0], sc_ori[1], sc_ori[2], scale_factor=0.05, color=(0,0,0))

mlab.points3d(0, 0, 0, resolution=32, scale_factor=0.01, color=(0,1,0), mode='axes')

## Create and visualize the mesh
mesh = mlab.pipeline.delaunay2d(pts)
surf = mlab.pipeline.surface(mesh)

mlab.title('Reconstructed dipole\'s field', size=0.5)

mlab.sync_camera(fig1, fig2)