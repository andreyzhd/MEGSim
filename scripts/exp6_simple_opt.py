#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 02:46:46 2020

@author: andrey

Compute condition number vs l and radius
"""
#%% Inits
import time
import pickle
import numpy as np
from scipy.optimize import least_squares
from mne.preprocessing.maxwell import _sss_basis
from megsimutils.utils import spherepts_golden, xyz2pol, pol2xyz, local_axes

R = 0.15
N_COILS = 100
ANGLE = 4*np.pi/3
L = 5
DATA_FNAME = '/tmp/opt.pkl'


def _build_slicemap(bins, n_coils):
    
    assert np.all(np.equal(np.mod(bins, 1), 0)) # all the values should be integers
    assert np.unique(bins).shape == (n_coils,)

    slice_map = {}
    
    for coil_ind in np.unique(bins):
        inds = np.argwhere(bins==coil_ind)[:,0]
        assert inds[-1] - inds[0] == len(inds) - 1 # indices should be contiguous
        slice_map[coil_ind] = slice(inds[0], inds[-1]+1)
        
    return slice_map


def _cond_num(inp, r, l, bins, n_coils, mag_mask, slice_map):
    theta = inp[:np.int64(len(inp)/2)]
    phi = inp[np.int64(len(inp)/2):]
    
    x, y, z = pol2xyz(r, theta, phi)
    e_r, e_theta, e_phi = local_axes(theta, phi)
    cosmags = e_r
    sss_origin = np.array([0.0, 0.0, 0.0])  # origin of device coords

    allcoils = (np.stack((x,y,z), axis=1), cosmags, bins, N_COILS, mag_mask, slice_map)
    exp = {'origin': sss_origin, 'int_order': l, 'ext_order': 0}
    
    S = _sss_basis(exp, allcoils)
    S /= np.linalg.norm(S, axis=0)
    return np.linalg.cond(S)


#%% Run the optimization
t_start = time.time()
assert L**2 + 2*L <= N_COILS

bins = np.arange(N_COILS, dtype=np.int64)
mag_mask = np.ones(N_COILS, dtype=np.bool)
slice_map = _build_slicemap(bins, N_COILS)

rmags0 = spherepts_golden(N_COILS, angle=ANGLE) * R
r0, theta0, phi0 = xyz2pol(rmags0[:,0], rmags0[:,1], rmags0[:,2])
x0 = np.concatenate((theta0, phi0)) # Note that x0 has nothing to do with the x axis!

low_bound = np.concatenate((-np.pi/2 * np.ones(N_COILS), -np.Inf * np.ones(N_COILS)))
upp_bound = np.concatenate((np.pi/2 * np.ones(N_COILS), np.Inf * np.ones(N_COILS)))
res = least_squares(_cond_num, x0, method='trf', bounds=(low_bound, upp_bound), args = (R, L, bins, N_COILS, mag_mask, slice_map))

# "Fold" the polar coordinates of the result to [0, pi], [0, 2*pi]
theta = res.x[:np.int64(len(res.x)/2)]
phi = res.x[np.int64(len(res.x)/2):]
x, y, z = pol2xyz(R, theta, phi)
r, theta, phi = xyz2pol(x, y, z)

cond_num0 = np.log10(_cond_num(x0, R, L, bins, N_COILS, mag_mask, slice_map))
cond_num = np.log10(_cond_num(np.concatenate((theta, phi)), R, L, bins, N_COILS, mag_mask, slice_map))

print('The optimization took %i seconds' % (time.time()-t_start))
print('Initial condition number is 10^%0.3f' % cond_num0)
print('Final condition number is 10^%0.3f' % cond_num)

#%% Save the results
f = open(DATA_FNAME, 'wb')
pickle.dump((rmags0, x, y, z, cond_num0, cond_num), f)
f.close()


