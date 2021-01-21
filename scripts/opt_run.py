#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 17:34:20 2021

@author: andrey
"""

#%% Inits
import time
import pickle
import numpy as np
import scipy.optimize

from megsimutils.optimize import BarbuteArray, constraint_penaly

PARAMS = {'R_inner' : 0.15,
          'R_outer' : 0.25,
          'n_coils' : 288,
          'L' : 16}
OUT_PATH = '/home/andrey/scratch/out'
NITER = 100

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

sens_array = BarbuteArray(PARAMS['n_coils'], PARAMS['L'], R_inner=PARAMS['R_inner'], R_outer=PARAMS['R_outer'])
v0 = sens_array.get_init_vector()

# Save the starting time, other params
fl = open('%s/start.pkl' % OUT_PATH, 'wb')
pickle.dump((PARAMS, t_start, v0), fl)
fl.close()

cb = _Callback(OUT_PATH)

#%% Run the optimization
#opt_res = scipy.optimize.basinhopping(lambda x : sens_array.comp_fitness(x) + constraint_penaly(x, sens_array.get_bounds()), v0, niter=NITER, callback=cb.call, disp=True, minimizer_kwargs={'method' : 'Nelder-Mead'})
opt_res = scipy.optimize.dual_annealing(sens_array.comp_fitness, sens_array.get_bounds(), x0=v0,  callback=cb.call)


#%% Postprocess / save
tstamp = time.time()
print('The optimization took %i seconds' % (tstamp-t_start))
print('Initial condition number is 10^%0.3f' % np.log10(sens_array.comp_fitness(v0)))
print('Final condition number is 10^%0.3f' % np.log10(sens_array.comp_fitness(opt_res.x)))
    
# Save the results
fl = open('%s/final.pkl' % OUT_PATH, 'wb')
pickle.dump((opt_res, tstamp), fl)
fl.close()