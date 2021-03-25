#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 20:07:42 2021

@author: andrey
"""

import numpy as np
import matplotlib.pyplot as plt
from megsimutils.optimize import BarbuteArraySL

#%% Parameter definitions
R = 0.15
HEIGHT_LOWER = 0.15
L_INT = 16
L_EXT = 3
N_SENS_RANGE = range(L_INT*(L_INT+2) + L_EXT*(L_EXT+2), 3 * (L_INT*(L_INT+2) + L_EXT*(L_EXT+2)) + 1, 10)
IS_OPM = False
N_ITER_RAND = 100

r_conds = []
r_conds_radial = []
r_conds_rand = []

for n_sens in N_SENS_RANGE:
    print('Processing array of size %i' % n_sens)
    sens_array = BarbuteArraySL(n_sens, L_INT, L_EXT, R_inner=R, height_lower=HEIGHT_LOWER, opm=IS_OPM)
    r_conds.append(sens_array.comp_fitness(sens_array.evenly_spaced_radial_v(False)))
    r_conds_radial.append(sens_array.comp_fitness(sens_array.evenly_spaced_radial_v(True)))
    
    r_conds_cur = []
    for i in range(N_ITER_RAND):
        r_conds_cur.append(sens_array.comp_fitness(sens_array.evenly_spaced_rand_v()))
        
    r_conds_rand.append(r_conds_cur)
    
plt.semilogy(N_SENS_RANGE, r_conds)
plt.semilogy(N_SENS_RANGE, r_conds_radial)

r_conds_rand = np.array(r_conds_rand)
plt.semilogy(N_SENS_RANGE, r_conds_rand.min(axis=1))
plt.semilogy(N_SENS_RANGE, r_conds_rand.max(axis=1))

plt.legend(['surface normal', 'radial', 'random (min)', 'random (max)'])