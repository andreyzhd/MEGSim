#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 02:46:46 2020

@author: andrey

Run the sensor array optimization for fixed locations
"""
#%% Inits
import time
import pickle
import numpy as np
import scipy.optimize
from megsimutils.utils import spherepts_golden, xyz2pol, pol2xyz
from megsimutils.optimize import CondNumberFixedLoc

PARAMS = {'R' : 0.15,                   # Radius of the sensor sphere, meters
          'n_coils' : 100,
          'L' : 6}

ANGLE = 4*np.pi/2
OUT_PATH = '/home/andrey/scratch/out'
FINAL_FNAME = 'final.pkl'
INTERM_PREFIX = 'iter'
START_FNAME = 'start.pkl'

NITER = 100            # Number of iterations for the optimization algorithm

class _counter:
    cnt = 0

def _callback(x, f, accept):
    
    tstamp = time.time()
    fl = open('%s/%s%06i.pkl' % (OUT_PATH, INTERM_PREFIX, _counter.cnt), 'wb')
    pickle.dump((x, f, accept, tstamp), fl)
    fl.close()
    
    print('Saved intermediate results in %s/%s%06i.pkl' % (OUT_PATH, INTERM_PREFIX, _counter.cnt))
    _counter.cnt += 1


#%% Prepare for running the optimization
assert PARAMS['L']**2 + 2*PARAMS['L'] <= PARAMS['n_coils']

t_start = time.time()

bins = np.arange(PARAMS['n_coils'], dtype=np.int64)
mag_mask = np.ones(PARAMS['n_coils'], dtype=np.bool)

# evenly spread the sensors
rmags = spherepts_golden(PARAMS['n_coils'], angle=ANGLE) * PARAMS['R']

# start with radial sensor orientation
cosmags0 = spherepts_golden(PARAMS['n_coils'], angle=ANGLE)
x_cosmags0, y_cosmags0, z_cosmags0 = cosmags0[:,0], cosmags0[:,1], cosmags0[:,2]
theta0, phi0 = xyz2pol(x_cosmags0, y_cosmags0, z_cosmags0)[1:3]
v0 = np.concatenate((theta0, phi0)) # initial guess

cond_num_comp = CondNumberFixedLoc(PARAMS['L'], bins, PARAMS['n_coils'], mag_mask, rmags)

# Save the starting time, other params
fl = open('%s/%s' % (OUT_PATH, START_FNAME), 'wb')
pickle.dump((PARAMS, t_start, rmags, v0, cond_num_comp), fl)
fl.close()

#%% Run the optimization
# Basinhopping
opt_res = scipy.optimize.basinhopping(lambda inp : cond_num_comp.compute(inp), v0, niter=NITER, callback=_callback, disp=True, minimizer_kwargs={'method' : 'Nelder-Mead'})


#%% Postprocess and save the results
# Fold the polar coordinates of the result to [0, pi], [0, 2*pi]
theta = opt_res.x[:PARAMS['n_coils']]
phi = opt_res.x[PARAMS['n_coils']:]

x_cosmags, y_cosmags, z_cosmags = pol2xyz(1, theta, phi)
theta, phi = xyz2pol(x_cosmags, y_cosmags, z_cosmags)[1:3]
v = np.concatenate((theta, phi))

tstamp = time.time()
print('The optimization took %i seconds' % (tstamp-t_start))
print('Initial condition number is 10^%0.3f' % np.log10(cond_num_comp.compute(v0)))
print('Final condition number is 10^%0.3f' % np.log10(cond_num_comp.compute(v)))

# Save the results
fl = open('%s/%s' % (OUT_PATH, FINAL_FNAME), 'wb')
pickle.dump((v, opt_res, tstamp), fl)
fl.close()


