#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 12:10:34 2021

@author: andrey
"""
import pickle
import pathlib
import numpy as np
import matplotlib.pyplot as plt

INP_FNAME = 'fig_01.pkl'
OPT_FLDR_NAME = 'opt_orient_only'


#%% Read the optimization data
n_sens_range_opt = []
r_conds_opt = []

for fldr in sorted(pathlib.Path(OPT_FLDR_NAME).glob('*_sens')):
    fl = open('%s/start.pkl' % fldr, 'rb')
    params, t_start, v0, sens_array, constraint_penalty = pickle.load(fl)
    fl.close()
    
    n_sens_range_opt.append(params['n_sens'])
    
    fl_list = sorted(fldr.glob('iter*.pkl'))
    assert len(fl_list) > 0 # should have at least one intermediate result

    print('Reading %s ...' % fl_list[-1])
    fl = open (fl_list[-1], 'rb')
    v, f, accept, tstamp = pickle.load(fl)
    fl.close()

    r_conds_opt.append(f)

#%% Read the non-optimized data
fl = open(INP_FNAME, 'rb')
r_conds, r_conds_radial, r_conds_rand, n_sens_range = pickle.load(fl)
fl.close()

#%% Plot
plt.semilogy(n_sens_range, r_conds)
plt.semilogy(n_sens_range, r_conds_radial)

r_conds_rand = np.array(r_conds_rand)
plt.semilogy(n_sens_range, r_conds_rand.min(axis=1))
plt.semilogy(n_sens_range, r_conds_rand.max(axis=1))

plt.semilogy(n_sens_range_opt, r_conds_opt)

plt.legend(['surface normal', 'radial', 'random (min)', 'random (max)', 'optimized'])

plt.xlabel('Number of sensors')
plt.ylabel('Condition number')