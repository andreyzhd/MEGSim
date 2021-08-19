#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 17:34:20 2021

@author: andrey
"""

#%% Imports
import time
import sys
import os
import pickle
import numpy as np
import scipy.optimize

from megsimutils.optimize import BarbuteArraySL, ConstraintPenalty

#%% Parameter definitions
PARAMS = {'n_sens' : 576,
          'R_inner' : 0.15, 
          'R_outer' : 0.25,
          'l_int' : 16,
          'n_samp_layers' : 2,
          'n_samp_per_layer' : 100,
          'kwargs' : {#'Re' : 0.2,               # Radius for energy-based normalization
                      'height_lower' : 0.15,
                      'l_ext' : 0,
                      'opm' : False,
                      'origin' : np.array([[0., 0., 0.],]),
                      #'ellip_sc' : np.array([1.2, 1., 1.1])
                      'ellip_sc' : np.array([1., 1., 1.])
                      }
          }
NITER = 1000
USE_CONSTR = False

# Read the output folder
assert len(sys.argv) == 2
out_path = sys.argv[-1]

#%% Init
class _Callback:
    def __init__(self, out_path):
        self._out_path = out_path
        self._cnt = 0

    def call(self, x, f, accept):    
        tstamp = time.time()
        fname = '%s/iter%06i.pkl' % (self._out_path, self._cnt)
        assert not os.path.exists(fname)
        fl = open(fname, 'wb')
        pickle.dump((x, f, accept, tstamp), fl)
        fl.close()
    
        print('Saved intermediate results in %s/iter%06i.pkl' % (self._out_path, self._cnt))
        self._cnt += 1


#%% Prepare the optimization
t_start = time.time()

sens_array = BarbuteArraySL(PARAMS['n_sens'], PARAMS['l_int'], R_inner=PARAMS['R_inner'], R_outer=PARAMS['R_outer'], n_samp_layers=PARAMS['n_samp_layers'], n_samp_per_layer=PARAMS['n_samp_per_layer'], **PARAMS['kwargs'])

if USE_CONSTR:
    constraint_penalty = ConstraintPenalty(sens_array.get_bounds())
    func = (lambda v : sens_array.comp_fitness(v) + constraint_penalty.compute(v))
else:
    constraint_penalty = None
    func = (lambda v : sens_array.comp_fitness(v))

#v0 = 0.9*sens_array.get_init_vector() + 0.1*np.mean(sens_array.get_bounds(), axis=1)
v0 = sens_array.get_init_vector()

# Save the starting time, other params
fname = '%s/start.pkl' % out_path
assert not os.path.exists(fname)
fl = open(fname, 'wb')
pickle.dump((PARAMS, t_start, v0, sens_array, constraint_penalty), fl)
fl.close()

cb = _Callback(out_path)

#%% Run the optimization
#opt_res = scipy.optimize.basinhopping(func, v0, niter=NITER, callback=cb.call, disp=True, minimizer_kwargs={'method' : 'Nelder-Mead'})
opt_res = scipy.optimize.dual_annealing(func, sens_array.get_bounds(), x0=v0,  callback=cb.call, maxiter=NITER)


#%% Postprocess / save
tstamp = time.time()
print('The optimization took %i seconds' % (tstamp-t_start))
print('Initial condition number is 10^%0.3f' % np.log10(sens_array.comp_fitness(v0)))
print('Final condition number is 10^%0.3f' % np.log10(sens_array.comp_fitness(opt_res.x)))
    
# Save the results
fname = '%s/final.pkl' % out_path
assert not os.path.exists(fname)
fl = open(fname, 'wb')
pickle.dump((opt_res, tstamp), fl)
fl.close()