#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 16:04:48 2020

@author: andrey
"""

#%% Inits
import pickle
import pathlib
from math import isclose
import numpy as np
#from mayavi import mlab
#import matplotlib.pyplot as plt
#from megsimutils.utils import pol2xyz, fold_uh
#from megsimutils.optimize import ArrayFixedLoc

from megsimutils import dipfld_sph
from megsimutils.dipole_fit import bf_dipole_fit
from megsimutils.utils import pol2xyz, fold_uh


INP_PATH = '/home/andrey/storage/Data/MEGSim/2020-08-15_fixed_locations/out_L06_192'
FINAL_FNAME = 'final.pkl'
INTERM_PREFIX = 'iter'
START_FNAME = 'start.pkl'

# Dipole parameters
Q = np.array([100, 0, 0]) * 1e-9    # Dipole moment, A*m
RQ = np.array([0, 0, 0.9])          # Dipole location, multiplied by the sensor sphere radius

# search domain for dipole fitting
R_MIN = 0.1                         # Relative to the sensor sphere radius
R_MAX = 0.95                        # Relative to the sensor sphere radius
ANGLE = 4*np.pi/2                           # Should be compatible with the angle used to generate the sensor array
THETA_MAX = np.arccos(1 - ANGLE/(2*np.pi))  # Theta corresponding to the area
                                            # covered by the sensors. Only
                                            # valid for ANGLE <= 2*pi
N_R = 20
N_THETA = 20
N_PHI = 4*N_THETA

# Noise
NOISE_STD = 0.1                     # Relative to signal

RAND_SEED = 42

def _v2cosmags(v):
    n_coils = len(v) // 2
    theta, phi = fold_uh(v[:n_coils], v[n_coils:])
    
    return np.stack(pol2xyz(1, theta, phi), axis=1)


#%% Read the data
# Read the starting timestamp, etc
fl = open('%s/%s' % (INP_PATH, START_FNAME), 'rb')
params, t_start, rmags, v0, cond_num_comp = pickle.load(fl)
fl.close()

# Read the intermediate results
interm_res = []
interm_res.append((v0, cond_num_comp.compute(v0), False, t_start))

for fname in sorted(pathlib.Path(INP_PATH).glob('%s*.pkl' % INTERM_PREFIX)):
    print('Reading %s ...' % fname)
    fl = open (fname, 'rb')
    v, f, accept, tstamp = pickle.load(fl)
    fl.close()
    assert isclose(cond_num_comp.compute(v), f)
    assert len(v) == params['n_coils'] * 2
    interm_res.append((v, f, accept, tstamp))
    
assert len(interm_res) > 1  # should have at least one intermediate result
    
# Try to read the final result
try:
    fl = open('%s/%s' % (INP_PATH, FINAL_FNAME), 'rb')
    v, opt_res, tstamp = pickle.load(fl)
    fl.close()
except:
    print('Could not find the final result, using the last intermediate result instead')
       
    v = interm_res[-1][0]   
    tstamp = interm_res[-1][-1]



#%% Generate the dipole
rng = np.random.default_rng(RAND_SEED) # initialize the random number generator

r = np.mean(np.linalg.norm(rmags, axis=1))
assert np.max(np.abs(np.linalg.norm(rmags, axis=1) - r)) < 1e-10 # all vectors should have length very similar to r

# Create a dipole
field = dipfld_sph(Q, RQ*r, rmags, np.zeros(3))

# Generate the noise using the initial array's signal level as reference
cosmags = _v2cosmags(v0)
data = (field*cosmags).sum(axis=1)
noise = rng.standard_normal(data.shape) * data.std() * NOISE_STD


#%% Fit the dipole with the original array
cosmags = _v2cosmags(v0)
data = (field*cosmags).sum(axis=1)
pos0, ori0, resid0 = bf_dipole_fit(rmags, cosmags, data+noise, {'rmin' : R_MIN,
                                                             'rmax' : R_MAX,
                                                             'theta_max' : THETA_MAX,
                                                             'n_r' : N_R,
                                                             'n_theta' : N_THETA,
                                                             'n_phi' : N_PHI})


#%% Fit the dipole with the final array
cosmags = _v2cosmags(v)
data = (field*cosmags).sum(axis=1)
pos, ori, resid = bf_dipole_fit(rmags, cosmags, data+noise, {'rmin' : R_MIN,
                                                             'rmax' : R_MAX,
                                                             'theta_max' : THETA_MAX,
                                                             'n_r' : N_R,
                                                             'n_theta' : N_THETA,
                                                             'n_phi' : N_PHI})


print('For the original array, the difference is %0.1f mm' % (np.linalg.norm(pos0-RQ*r)*1000))
print('For the final array, the difference is %0.1f mm' % (np.linalg.norm(pos-RQ*r)*1000))




