#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 02:46:46 2020

@author: andrey

Run the sensor array optimization for fixed locations. Run in parallel in 
separate processes, each process for different parameters.
"""
#%% Inits
import time
import pickle
from pathlib import Path
from multiprocessing import Process
import numpy as np
import scipy.optimize
from megsimutils.utils import spherepts_golden, xyz2pol, pol2xyz
from megsimutils.optimize import ArrayFixedLoc

R = '0.15'                              # Radius of the sensor sphere, meters
L = range(6, 14)
N_COILS = map((lambda l: 2*l*(l+2)), L)

ANGLE = 4*np.pi/2
OUT_PATH_PREFIX = '/home/andrey/scratch/out'
FINAL_FNAME = 'final.pkl'
INTERM_PREFIX = 'iter'
START_FNAME = 'start.pkl'

NITER = 100            # Number of iterations for the optimization algorithm

class Callback:
    def __init__(self, out_path):
        self._out_path = out_path
        self._cnt = 0

    def call(self, x, f, accept):    
        tstamp = time.time()
        fl = open('%s/%s%06i.pkl' % (self._out_path, INTERM_PREFIX, self._cnt), 'wb')
        pickle.dump((x, f, accept, tstamp), fl)
        fl.close()
    
        print('Saved intermediate results in %s/%s%06i.pkl' % (self._out_path, INTERM_PREFIX, self._cnt))
        self._cnt += 1


def run_opt(params, out_path):
    print('Starting optimization, L=%i, n_coils=%i' % (params['L'], params['n_coils']))
    # Prepare for running the optimization
    assert params['L']**2 + 2*params['L'] <= params['n_coils']
    
    t_start = time.time()
    
    bins = np.arange(params['n_coils'], dtype=np.int64)
    mag_mask = np.ones(params['n_coils'], dtype=np.bool)
    
    # evenly spread the sensors
    rmags = spherepts_golden(params['n_coils'], angle=ANGLE) * params['R']
    
    # start with radial sensor orientation
    cosmags0 = spherepts_golden(params['n_coils'], angle=ANGLE)
    x_cosmags0, y_cosmags0, z_cosmags0 = cosmags0[:,0], cosmags0[:,1], cosmags0[:,2]
    theta0, phi0 = xyz2pol(x_cosmags0, y_cosmags0, z_cosmags0)[1:3]
    v0 = np.concatenate((theta0, phi0)) # initial guess
    
    sens_array = ArrayFixedLoc(bins, params['n_coils'], mag_mask, rmags)
    
    # Save the starting time, other params
    fl = open('%s/%s' % (out_path, START_FNAME), 'wb')
    pickle.dump((params, t_start, rmags, v0, sens_array), fl)
    fl.close()
    
    cb = Callback(out_path)
    
    # Run the optimization
    # Basinhopping
    opt_res = scipy.optimize.basinhopping(lambda inp : sens_array.compute_cond_num(inp, params['L']), v0, niter=NITER, callback=(lambda x, f, accept : cb.call(x, f, accept)), disp=True, minimizer_kwargs={'method' : 'Nelder-Mead'})
    
    
    # Postprocess and save the results
    # Fold the polar coordinates of the result to [0, pi], [0, 2*pi]
    theta = opt_res.x[:params['n_coils']]
    phi = opt_res.x[params['n_coils']:]
    
    x_cosmags, y_cosmags, z_cosmags = pol2xyz(1, theta, phi)
    theta, phi = xyz2pol(x_cosmags, y_cosmags, z_cosmags)[1:3]
    v = np.concatenate((theta, phi))
    
    tstamp = time.time()
    print('The optimization took %i seconds' % (tstamp-t_start))
    print('Initial condition number is 10^%0.3f' % np.log10(sens_array.compute_cond_num(v0, params['L'])))
    print('Final condition number is 10^%0.3f' % np.log10(sens_array.compute_cond_num(v, params['L'])))
    
    # Save the results
    fl = open('%s/%s' % (out_path, FINAL_FNAME), 'wb')
    pickle.dump((v, opt_res, tstamp), fl)
    fl.close()
    

#%% Run the optimizations
procs = []
for l, n_coils in zip(L, N_COILS):
    out_fldr_name = '%s_L%02i' % (OUT_PATH_PREFIX, l)
    Path(out_fldr_name).mkdir(parents=True, exist_ok=True)
    
    params = {'R' : 0.15, 'n_coils' : n_coils, 'L' : l}
    
    p = Process(target=run_opt, args=(params, out_fldr_name))
    p.start()
    
    procs.append(p)

for p in procs:
    p.join()