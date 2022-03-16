#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: andrey
"""

import pickle
import pathlib
from math import isclose, inf
import sys
from megsimutils.utils import subsample


def read_opt_res(inp_path, max_n_samp=inf):
    """Read the results of optimization run"""
    interm_res = []

    # Read the starting timestamp, etc
    fl = open('%s/start.pkl' % inp_path, 'rb')
    params, t_start, v0, sens_array, constraint_penalty = pickle.load(fl)
    fl.close()

    if constraint_penalty is None:
        func = (lambda v : sens_array.comp_fitness(v))
    else:
        func = (lambda v : sens_array.comp_fitness(v) + constraint_penalty.compute(v))

    interm_res.append((v0, func(v0), False, t_start))

    # Try to read the final result
    try:
        fl = open('%s/final.pkl' % inp_path, 'rb')
        opt_res, final_tstamp = pickle.load(fl)
        fl.close()
    except:
        opt_res = None
        print('Could not find the final result, using the last intermediate result instead')
        
    # Read the intermediate results
    file_list = sorted(pathlib.Path(inp_path).glob('iter*.pkl'))
    sys.setrecursionlimit(min(len(file_list), max_n_samp) + 1000)
    indx = subsample(len(file_list) + (opt_res is not None) + 1, max_n_samp)

    if opt_res is None:
        file_indx = (i-1 for i in indx[1:])
    else:
        file_indx = (i - 1 for i in indx[1:-1])

    for i in file_indx:
        fname = file_list[i]
        print('Reading %s ...' % fname)
        fl = open (fname, 'rb')
        v, f, accept, tstamp = pickle.load(fl)
        fl.close()
        assert isclose(func(v), f, rel_tol=1e-4)
        interm_res.append((v, f, accept, tstamp))
    
    assert len(interm_res) > 1  # should have at least one intermediate result

    if opt_res is not None:
        interm_res.append((opt_res.x, func(opt_res.x), True, final_tstamp))

    return params, sens_array, interm_res, opt_res, indx
