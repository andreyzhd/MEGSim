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

from megsimutils.optimize2 import FixedLocSpherArray

PARAMS = {'R' : 0.15,
          'n_coils' : 96,
          'L' : 6,
          'array_sangle' : 4*np.pi/2}
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

sens_array = FixedLocSpherArray(PARAMS['n_coils'], PARAMS['array_sangle'], PARAMS['L'], PARAMS['R'])
v0 = sens_array.get_init_vector()

# Save the starting time, other params
fl = open('%s/start.pkl' % OUT_PATH, 'wb')
pickle.dump((PARAMS, t_start, v0, sens_array), fl)
fl.close()

cb = _Callback(OUT_PATH)

#%% Run the optimization
opt_res = scipy.optimize.basinhopping(sens_array.comp_fitness, v0, niter=NITER, callback=cb.call, disp=True, minimizer_kwargs={'method' : 'Nelder-Mead'})

#%% Postprocess / save
tstamp = time.time()
print('The optimization took %i seconds' % (tstamp-t_start))
print('Initial condition number is 10^%0.3f' % np.log10(sens_array.comp_fitness(v0)))
print('Final condition number is 10^%0.3f' % np.log10(sens_array.comp_fitness(opt_res.x)))
    
# Save the results
fl = open('%s/final.pkl' % OUT_PATH, 'wb')
pickle.dump((opt_res, tstamp), fl)
fl.close()