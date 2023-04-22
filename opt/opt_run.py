#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 17:34:20 2021

@author: andrey
"""

import time
import os
import pickle
import scipy.optimize

from megsimutils.arrays import BarbuteArraySL


class _Callback:
    def __init__(self, out_path):
        self.__out_path = out_path
        self.__cnt = 1

    def call(self, x, f, accept):    
        tstamp = time.time()
        
        # Save everything except x
        fname = '%s/iter%06i.pkl' % (self.__out_path, self.__cnt)
        assert not os.path.exists(fname)
        fl = open(fname, 'wb')
        pickle.dump((x, f, accept, tstamp), fl)
        fl.close()
                
        print('Saved intermediate results in %s/iter%06i' % (self.__out_path, self.__cnt))
        self.__cnt += 1


def opt_run(params, niter, out_path):
    """
    Run the optimization process for maximum number of iterations niter;
    save the results to out_path. params specifies various parameters on
    the sensor array, etc.
    """
    # Prepare the optimization
    t_start = time.time()
    sens_array = BarbuteArraySL(params['n_sens'],
                                params['l_int'],
                                params['l_ext'],
                                R_inner=params['R_inner'],
                                R_outer=params['R_outer'],
                                n_samp_layers=params['n_samp_layers'],
                                n_samp_per_layer=params['n_samp_per_layer'],
                                debug_fldr=out_path,
                                **params['kwargs'])
    v0 = sens_array.get_init_vector(params['init_depth'])

    # Save the starting time, other params
    fname = '%s/start.pkl' % out_path
    assert not os.path.exists(fname)
    with open(fname, 'wb') as fl:
        pickle.dump((params, t_start, v0, sens_array), fl)

    cb = _Callback(out_path)

    # Run the optimization
    #opt_res = scipy.optimize.basinhopping(sens_array.comp_fitness, v0, niter=niter, callback=cb.call, disp=True, minimizer_kwargs={'method' : 'Nelder-Mead'})
    opt_res = scipy.optimize.dual_annealing(sens_array.comp_fitness, sens_array.get_bounds(), x0=v0,  callback=cb.call, maxiter=niter)

    # Postprocess / save
    tstamp = time.time()
    print('The optimization took %i seconds' % (tstamp-t_start))
    print('Initial fitness value is %0.3f' % sens_array.comp_fitness(v0))
    print('Final fitness value is %0.3f' % sens_array.comp_fitness(opt_res.x))

    # Save the results
    fname = '%s/final.pkl' % out_path
    assert not os.path.exists(fname)
    with open(fname, 'wb') as fl:
        pickle.dump((opt_res, tstamp), fl)
