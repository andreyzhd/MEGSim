#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 20:07:42 2021

@author: andrey
"""

import numpy as np
import matplotlib.pyplot as plt
from megsimutils.optimize import BarbuteArraySL, GoldenRatioError

#%% Parameter definitions
R = 0.15
HEIGHT_LOWER = 0.15
L_INT = 16
L_EXT = 3
N_SENS_RANGE = range(L_INT*(L_INT+2) + L_EXT*(L_EXT+2), 4 * (L_INT*(L_INT+2) + L_EXT*(L_EXT+2)) + 1)
IS_OPM = False

r_conds = []

for n_sens in N_SENS_RANGE:
    try:
        sens_array = BarbuteArraySL(n_sens, L_INT, L_EXT, R_inner=R, height_lower=HEIGHT_LOWER, opm=IS_OPM)
        r_conds.append(sens_array.comp_fitness(sens_array.get_init_vector()))
    except GoldenRatioError:
        r_conds.append(-1)
        
r_conds = np.array(r_conds)
indx = r_conds != -1
print("Failed to create arrays for %i out of %i configurations" % (np.count_nonzero(r_conds==-1), len(r_conds)))
r_conds = r_conds[indx]
n_sens_range = np.array(N_SENS_RANGE)[indx]
    
    
plt.semilogy(n_sens_range, r_conds)