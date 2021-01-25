#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 17:34:20 2021

@author: andrey
"""

#%% Imports
import time
import pickle
import numpy as np
import scipy.optimize

from megsimutils.optimize import BarbuteArray, ConstraintPenalty

#%% Parameter definitions
PARAMS = {'R_inner' : 0.15,
          'R_outer' : 0.25,
          'height_lower' : 0.15,
          'n_coils' : 30,
          'L' : 3}
OUT_PATH = '/home/andrey/scratch/out'
NITER = 1000
USE_CONSTR = True


#%% Init
class _Callback:
    def __init__(self, out_path):
        self._out_path = out_path
        self._cnt = 0

    def call(self, x, f, accept):    
        tstamp = time.time()
        fl = open('%s/iter%06i.pkl' % (self._out_path, self._cnt), 'wb')
        pickle.dump((x, f, accept, tstamp), fl)
        fl.close()
    
        print('Saved intermediate results in %s/iter%06i.pkl' % (self._out_path, self._cnt))
        self._cnt += 1


#%% Prepare the optimization
assert PARAMS['L']**2 + 2*PARAMS['L'] <= PARAMS['n_coils']
t_start = time.time()

sens_array = BarbuteArray(PARAMS['n_coils'], PARAMS['L'],
                          R_inner=PARAMS['R_inner'], R_outer=PARAMS['R_outer'], height_lower=PARAMS['height_lower'])

if USE_CONSTR:
    constraint_penalty = ConstraintPenalty(sens_array.get_bounds())
    func = (lambda v : sens_array.comp_fitness(v) + constraint_penalty.compute(v))
else:
    constraint_penalty = None
    func = (lambda v : sens_array.comp_fitness(v))

v0 = 0.9*sens_array.get_init_vector() + 0.1*np.mean(sens_array.get_bounds(), axis=1)

# Save the starting time, other params
fl = open('%s/start.pkl' % OUT_PATH, 'wb')
pickle.dump((PARAMS, t_start, v0, sens_array, constraint_penalty), fl)
fl.close()

cb = _Callback(OUT_PATH)

#%% Run the optimization
opt_res = scipy.optimize.basinhopping(func, v0, niter=NITER, callback=cb.call, disp=True, minimizer_kwargs={'method' : 'Nelder-Mead'})
#opt_res = scipy.optimize.dual_annealing(func, sens_array.get_bounds(), x0=v0,  callback=cb.call)


#%% Postprocess / save
tstamp = time.time()
print('The optimization took %i seconds' % (tstamp-t_start))
print('Initial condition number is 10^%0.3f' % np.log10(sens_array.comp_fitness(v0)))
print('Final condition number is 10^%0.3f' % np.log10(sens_array.comp_fitness(opt_res.x)))
    
# Save the results
fl = open('%s/final.pkl' % OUT_PATH, 'wb')
pickle.dump((opt_res, tstamp), fl)
fl.close()