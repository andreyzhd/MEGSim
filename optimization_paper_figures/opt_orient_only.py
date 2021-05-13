#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 12:30:16 2021

@author: andrey
"""

#%% Imports
import time
import os
import pickle
import multiprocessing
import numpy as np
import scipy.optimize

from megsimutils.optimize import FixedBarbuteArraySL


#%% Parameter definitions
PARAMS = {'R' : 1,
          'height_lower' : 1,
          'L_INT' : 16,
          'L_EXT' : 3,
          'OPM' : False,
          'ellip_sc' : np.array([0.0994945 , 0.08377065, 0.09113235]),
          'origin' : np.array([[0.02100584,  0.00433862,  0.0040068], 
                               [-0.0071708 , 0.00042154, -0.00346818],])}

N_SENS_RANGE = range(PARAMS['L_INT']*(PARAMS['L_INT']+2) + PARAMS['L_EXT']*(PARAMS['L_EXT']+2), 3 * (PARAMS['L_INT']*(PARAMS['L_INT']+2) + PARAMS['L_EXT']*(PARAMS['L_EXT']+2)) + 1, 100)
NITER = 1000

OUT_PATH = 'opt_orient_only_ellip'


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


def _run_opt(n_sens):
    params = PARAMS.copy()
    params['n_sens'] = n_sens
    
    t_start = time.time()
    sens_array = FixedBarbuteArraySL(n_sens, PARAMS['L_INT'], l_ext=PARAMS['L_EXT'],
                                     R_inner=PARAMS['R'], height_lower=PARAMS['height_lower'],
                                     opm=PARAMS['OPM'], origin=PARAMS['origin'], ellip_sc=PARAMS['ellip_sc'])
    v0 = sens_array.get_init_vector()
    
    os.mkdir('%s/%03i_sens' % (OUT_PATH, n_sens))
    fl = open('%s/%03i_sens/start.pkl' % (OUT_PATH, n_sens), 'wb')
    pickle.dump((params, t_start, v0, sens_array, None), fl)
    fl.close()
    
    cb = _Callback('%s/%03i_sens' % (OUT_PATH, n_sens))

    # Run the optimization
    opt_res = scipy.optimize.dual_annealing(sens_array.comp_fitness, sens_array.get_bounds(), x0=v0,  callback=cb.call, maxiter=NITER)

    # Save the results
    tstamp = time.time()
    fl = open('%s/%03i_sens/final.pkl' % (OUT_PATH, n_sens), 'wb')
    pickle.dump((opt_res, tstamp), fl)
    fl.close()


#%% Do the job
assert PARAMS['L_INT']*(PARAMS['L_INT']+2) + PARAMS['L_EXT']*(PARAMS['L_EXT']+2) <= np.min(N_SENS_RANGE) 

pool = multiprocessing.Pool(len(N_SENS_RANGE))
pool.map(_run_opt, N_SENS_RANGE)