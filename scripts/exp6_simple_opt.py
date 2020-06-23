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
import itertools
import numpy as np
import scipy.optimize
from mne.preprocessing.maxwell import _sss_basis
from megsimutils.utils import spherepts_golden, xyz2pol, pol2xyz, local_axes

R = 0.15
N_COILS = 100
ANGLE = 4*np.pi/3
L = 9
DATA_FNAME = 'E:/opt.pkl'

# Parameters controling penalty-based constraints
THETA_BOUND = np.pi / 2 # theta is not allowed to be larger than THETA_BOUND
PENALTY_SHARPNESS = 5   # Controls how steeply the penalty increases as we
                        # approach THETA_BOUND. The larger the value, the
                        # steeper the increase. Probably, the values in the
                        # range [1, 5] are OK.
PENALTY_MAX = 1e20      # The maximum penalty, after we reach this value the
                        # penalty flattens out.


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
    theta = inp[:n_coils]
    phi = inp[n_coils:2*n_coils]
    theta_cosmags = inp[2*n_coils:3*n_coils]
    phi_cosmags = inp[3*n_coils:4*n_coils]
    
    x, y, z = pol2xyz(r, theta, phi)
    x_cosmags, y_cosmags, z_cosmags = pol2xyz(1, theta_cosmags, phi_cosmags)
    sss_origin = np.array([0.0, 0.0, 0.0])  # origin of device coords

    allcoils = (np.stack((x,y,z), axis=1), np.stack((x_cosmags,y_cosmags,z_cosmags), axis=1), bins, N_COILS, mag_mask, slice_map)
    exp = {'origin': sss_origin, 'int_order': l, 'ext_order': 0}
    
    S = _sss_basis(exp, allcoils)
    S /= np.linalg.norm(S, axis=0)
    return np.linalg.cond(S)


def _constraint(inp, n_coils):
    """ Compute the constraint penalty"""
    theta = inp[:n_coils]
    
    theta_max = THETA_BOUND - (THETA_BOUND/PENALTY_SHARPNESS/PENALTY_MAX)
    if (theta >= theta_max).any():
        return PENALTY_MAX
    else:
        return (THETA_BOUND / PENALTY_SHARPNESS / (theta_max - theta)).max()
    
    
def _objective(inp, r, l, bins, n_coils, mag_mask, slice_map):
    assert len(inp) == n_coils*4
    
    return _cond_num(inp, r, l, bins, n_coils, mag_mask, slice_map) + \
           _constraint(inp, n_coils)

def _callback(x, f, accept):
    print('Time = %i, f = %f, accept = %s' % (time.time(), f, ('False', 'True')[accept]))


#%% Run the optimization
t_start = time.time()
assert L**2 + 2*L <= N_COILS

bins = np.arange(N_COILS, dtype=np.int64)
mag_mask = np.ones(N_COILS, dtype=np.bool)
slice_map = _build_slicemap(bins, N_COILS)

rmags0 = spherepts_golden(N_COILS, angle=ANGLE) * R
cosmags0 = spherepts_golden(N_COILS, angle=ANGLE)

r0, theta0, phi0 = xyz2pol(rmags0[:,0], rmags0[:,1], rmags0[:,2])
x0 = np.concatenate((theta0, phi0, theta0, phi0)) # Note that x0 has nothing to do with the x axis!

"""
low_bound = np.concatenate((-np.pi/2 * np.ones(N_COILS), -np.Inf * np.ones(N_COILS)))
upp_bound = np.concatenate((np.pi/2 * np.ones(N_COILS), np.Inf * np.ones(N_COILS)))
opt_res = scipy.optimize.least_squares(_cond_num, x0, method='trf', bounds=(low_bound, upp_bound), args=(R, L, bins, N_COILS, mag_mask, slice_map))
"""

"""
opt_res = scipy.optimize.least_squares(_cond_num, x0, method='trf', args=(R, L, bins, N_COILS, mag_mask, slice_map))
"""

opt_res = scipy.optimize.basinhopping(_objective, x0, niter=1, callback=_callback, minimizer_kwargs={'args':(R, L, bins, N_COILS, mag_mask, slice_map)})

"""
bounds = list(itertools.repeat((0, np.pi), N_COILS)) + list(itertools.repeat((0, 2*np.pi), N_COILS))
#opt_res = scipy.optimize.differential_evolution(_cond_num, bounds, args = (R, L, bins, N_COILS, mag_mask, slice_map), workers=-1)
opt_res = scipy.optimize.shgo(_cond_num, bounds, args = (R, L, bins, N_COILS, mag_mask, slice_map))
"""

# Fold the polar coordinates of the result to [0, pi], [0, 2*pi]
theta = opt_res.x[:N_COILS]
phi = opt_res.x[N_COILS:2*N_COILS]
theta_cosmags = opt_res.x[2*N_COILS:3*N_COILS]
phi_cosmags = opt_res.x[3*N_COILS:4*N_COILS]

x, y, z = pol2xyz(R, theta, phi)
r, theta, phi = xyz2pol(x, y, z)

x_cosmags, y_cosmags, z_cosmags = pol2xyz(1, theta_cosmags, phi_cosmags)
r_cosmags, theta_cosmags, phi_cosmags = xyz2pol(x_cosmags, y_cosmags, z_cosmags)

cond_num0 = np.log10(_cond_num(x0, R, L, bins, N_COILS, mag_mask, slice_map))
cond_num = np.log10(_cond_num(np.concatenate((theta, phi, theta_cosmags, phi_cosmags)), R, L, bins, N_COILS, mag_mask, slice_map))

print('The optimization took %i seconds' % (time.time()-t_start))
print('Initial condition number is 10^%0.3f' % cond_num0)
print('Final condition number is 10^%0.3f' % cond_num)

#%% Save the results
f = open(DATA_FNAME, 'wb')
pickle.dump((rmags0, cosmags0, x, y, z, x_cosmags, y_cosmags, z_cosmags, cond_num0, cond_num, opt_res), f)
f.close()


