#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: andrey
"""

import pickle
import pathlib
from math import isclose

from megsimutils.arrays import BarbuteArraySL, BarbuteArraySLGrid, noise_max, noise_mean

def read_opt_res(inp_path):
    """Read the results of optimization run"""

    # Read the starting timestamp, etc
    fl = open('%s/start.pkl' % inp_path, 'rb')
    params, t_start, v0, sens_array, constraint_penalty = pickle.load(fl)
    fl.close()

    if constraint_penalty is None:
        func = (lambda v : sens_array.comp_fitness(v))
    else:
        func = (lambda v : sens_array.comp_fitness(v) + constraint_penalty.compute(v))
        
    # Read the intermediate results
    interm_res = []
    interm_res.append((v0, func(v0), False, t_start))

    for fname in sorted(pathlib.Path(inp_path).glob('iter*.pkl')):
        print('Reading %s ...' % fname)
        fl = open (fname, 'rb')
        v, f, accept, tstamp = pickle.load(fl)
        fl.close()
        assert isclose(func(v), f, rel_tol=1e-4)
        interm_res.append((v, f, accept, tstamp))
    
    assert len(interm_res) > 1  # should have at least one intermediate result
    
    # Try to read the final result
    try:
        fl = open('%s/final.pkl' % inp_path, 'rb')
        opt_res, tstamp = pickle.load(fl)
        fl.close()
    
        interm_res.append((opt_res.x, func(opt_res.x), True, tstamp))
    except:
        opt_res = None
        print('Could not find the final result, using the last intermediate result instead')

    return params, sens_array, interm_res, opt_res
