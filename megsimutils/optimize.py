#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 15:14:09 2020

@author: andrey
"""

import numpy as np
from mne.preprocessing.maxwell import _sss_basis

from megsimutils.utils import pol2xyz


# Parameters controling penalty-based constraints
PENALTY_SHARPNESS = 5   # Controls how steeply the penalty increases as we
                        # approach THETA_BOUND. The larger the value, the
                        # steeper the increase. Probably, the values in the
                        # range [1, 5] are OK.
PENALTY_MAX = 1e15      # The maximum penalty, after we reach this value the
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


class CondNumber:
    def __init__(self, r, l, bins, n_coils, mag_mask):
        self._r = r
        self._l = l
        self._bins = bins
        self._n_coils = n_coils
        self._mag_mask = mag_mask
        self._slice_map = _build_slicemap(bins, n_coils)
        
    def compute(self, inp):
        theta = inp[:self._n_coils]
        phi = inp[self._n_coils:2*self._n_coils]
        theta_cosmags = inp[2*self._n_coils:3*self._n_coils]
        phi_cosmags = inp[3*self._n_coils:4*self._n_coils]
    
        x, y, z = pol2xyz(self._r, theta, phi)
        x_cosmags, y_cosmags, z_cosmags = pol2xyz(1, theta_cosmags, phi_cosmags)
        sss_origin = np.array([0.0, 0.0, 0.0])  # origin of device coords

        allcoils = (np.stack((x,y,z), axis=1), np.stack((x_cosmags,y_cosmags,z_cosmags), axis=1), 
                    self._bins, self._n_coils, self._mag_mask, self._slice_map)
        exp = {'origin': sss_origin, 'int_order': self._l, 'ext_order': 0}
    
        S = _sss_basis(exp, allcoils)
        S /= np.linalg.norm(S, axis=0)
        return np.linalg.cond(S)


class Constraint:
    def __init__(self, n_coils, theta_bound):
        self._n_coils = n_coils
        self._theta_bound = theta_bound
        self._theta_max = self._theta_bound - (self._theta_bound/PENALTY_SHARPNESS/PENALTY_MAX)
        assert self._theta_bound > self._theta_max # make sure there are no numerical issues
        
    def compute(self, inp):
        """ Compute the constraint penalty"""
        theta = (inp[:self._n_coils]).abs() # we care only about the absolute value of theta
    
        current_max = theta.max()
        if current_max >= self._theta_max:
            return PENALTY_MAX * current_max / self._theta_max
        else:
            return self._theta_bound / PENALTY_SHARPNESS / (self._theta_max - current_max)
        

class Objective:
    def __init__(self, r, l, bins, n_coils, mag_mask, theta_bound):
        self._cond_num = CondNumber(r, l, bins, n_coils, mag_mask)
        self._constraint = Constraint(n_coils, theta_bound)
        self._n_coils = n_coils
        self._counter = 0
        
    def compute(self, inp):
        assert len(inp) == self._n_coils*4
        self._counter += 1
        if self._counter % 1000 == 0:
            print('Objective function has been called %i times' % self._counter)
            
        return self._cond_num.compute(inp) + self._constraint.compute(inp)

