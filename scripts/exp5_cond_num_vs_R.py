#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 02:46:46 2020

@author: andrey

Compute condition number vs l and radius
"""
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from mne.preprocessing.maxwell import _sss_basis
from megsimutils.utils import spherepts_golden, xyz2pol, pol2xyz, local_axes

R_RANGE = np.linspace(0.05, 1, 100)
N_COILS = 300
ANGLE = 4*np.pi/3
COSMAGS_DIR = 2 # 0 means e_r, 1 - e_theta, and 2 - e_phi

def _build_slicemap(bins, n_coils):
    
    assert np.all(np.equal(np.mod(bins, 1), 0)) # all the values should be integers
    assert np.unique(bins).shape == (n_coils,)

    slice_map = {}
    
    for coil_ind in np.unique(bins):
        inds = np.argwhere(bins==coil_ind)[:,0]
        assert inds[-1] - inds[0] == len(inds) - 1 # indices should be contiguous
        slice_map[coil_ind] = slice(inds[0], inds[-1]+1)
        
    return slice_map  

#%% Compute condition numbers
bins = np.arange(N_COILS, dtype=np.int64)
mag_mask = np.ones(N_COILS, dtype=np.bool)
slice_map = _build_slicemap(bins, N_COILS)

sss_origin = np.array([0.0, 0.0, 0.0])  # origin of device coords
l_max = np.int64(np.floor(np.sqrt(N_COILS+1) - 1)) # Largest l for which we still have more measurements than components
cond_nums = np.zeros((l_max, len(R_RANGE)))

for l in range(1, l_max+1):
    print('Computing condition numbers for l = %i ...' % l)
    for i in range(len(R_RANGE)):
        rmags = spherepts_golden(N_COILS, angle=ANGLE) * R_RANGE[i]
        r, theta, phi = xyz2pol(rmags[:,0], rmags[:,1], rmags[:,2])
        e_r, e_theta, e_phi = local_axes(theta, phi)

        cosmags = [e_r, e_theta, e_phi][COSMAGS_DIR]
        allcoils = (rmags, cosmags, bins, N_COILS, mag_mask, slice_map)

        exp = {'origin': sss_origin, 'int_order': l, 'ext_order': 0}
    
        S = _sss_basis(exp, allcoils)
        cond_nums[l-1, i] = np.linalg.cond(S)
        
#%% Plot condition numbers

ls, rs = np.meshgrid(range(1, l_max+1), R_RANGE, indexing='ij')

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(ls, rs, np.log10(cond_nums), cmap='viridis', edgecolor='none')
ax.set_xlabel('l')
ax.set_ylabel('R')
ax.set_zlabel(r'$\log_{10}(R_{cond})$')

