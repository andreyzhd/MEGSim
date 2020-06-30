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
from megsimutils.utils import spherepts_golden, xyz2pol, pol2xyz
from megsimutils.optimize import Objective, CondNumber

PARAMS = {'R' : 0.15,                   # Radius of the sensor sphere, meters
          'n_coils' : 100,
          'L' : 6,
          'theta_bound' : np.pi / 2}    # abs(theta) is not allowed to be larger than theta_bound

ANGLE = 4*np.pi/3
OUT_PATH = '/tmp/out'
FINAL_FNAME = 'final.pkl'
INTERM_PREFIX = 'iter'
START_FNAME = 'start.pkl'

NITER = 100             # Number of iterations for the optimization algorithm

    
class _counter:
    cnt = 0

def _callback(x, f, accept):
    
    tstamp = time.time()
    fl = open('%s/%s%03i.pkl' % (OUT_PATH, INTERM_PREFIX, _counter.cnt), 'wb')
    pickle.dump((x, f, accept, tstamp), fl)
    fl.close()
    
    print('Saved intermediate results in %s/%s%03i.pkl' % (OUT_PATH, INTERM_PREFIX, _counter.cnt))
    _counter.cnt += 1


#%% Run the optimization
assert PARAMS['L']**2 + 2*PARAMS['L'] <= PARAMS['n_coils']

# Save the starting time
t_start = time.time()

fl = open('%s/%s' % (OUT_PATH, START_FNAME), 'wb')
pickle.dump((PARAMS, t_start), fl)
fl.close()

bins = np.arange(PARAMS['n_coils'], dtype=np.int64)
mag_mask = np.ones(PARAMS['n_coils'], dtype=np.bool)

objective = Objective(PARAMS['R'], PARAMS['L'], bins, PARAMS['n_coils'], mag_mask, PARAMS['theta_bound'])

rmags0 = spherepts_golden(PARAMS['n_coils'], angle=ANGLE) * PARAMS['R']
cosmags0 = spherepts_golden(PARAMS['n_coils'], angle=ANGLE)

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

opt_res = scipy.optimize.basinhopping(lambda inp : objective.compute(inp), x0, niter=NITER, callback=_callback, disp=True)

"""
bounds = list(itertools.repeat((0, np.pi), N_COILS)) + list(itertools.repeat((0, 2*np.pi), N_COILS))
#opt_res = scipy.optimize.differential_evolution(_cond_num, bounds, args = (R, L, bins, N_COILS, mag_mask, slice_map), workers=-1)
opt_res = scipy.optimize.shgo(_cond_num, bounds, args = (R, L, bins, N_COILS, mag_mask, slice_map))
"""

# Fold the polar coordinates of the result to [0, pi], [0, 2*pi]
theta = opt_res.x[:PARAMS['n_coils']]
phi = opt_res.x[PARAMS['n_coils']:2*PARAMS['n_coils']]
theta_cosmags = opt_res.x[2*PARAMS['n_coils']:3*PARAMS['n_coils']]
phi_cosmags = opt_res.x[3*PARAMS['n_coils']:4*PARAMS['n_coils']]

x, y, z = pol2xyz(PARAMS['R'], theta, phi)
r, theta, phi = xyz2pol(x, y, z)

x_cosmags, y_cosmags, z_cosmags = pol2xyz(1, theta_cosmags, phi_cosmags)
r_cosmags, theta_cosmags, phi_cosmags = xyz2pol(x_cosmags, y_cosmags, z_cosmags)
cond_num_comp = CondNumber(PARAMS['R'], PARAMS['L'], bins, PARAMS['n_coils'], mag_mask)
cond_num0 = np.log10(cond_num_comp.compute(x0))
cond_num = np.log10(cond_num_comp.compute(np.concatenate((theta, phi, theta_cosmags, phi_cosmags))))

tstamp = time.time()
print('The optimization took %i seconds' % (tstamp-t_start))
print('Initial condition number is 10^%0.3f' % cond_num0)
print('Final condition number is 10^%0.3f' % cond_num)

#%% Save the results
fl = open('%s/%s' % (OUT_PATH, FINAL_FNAME), 'wb')
pickle.dump((rmags0, cosmags0, x, y, z, x_cosmags, y_cosmags, z_cosmags, cond_num0, cond_num, opt_res, tstamp), fl)
fl.close()


