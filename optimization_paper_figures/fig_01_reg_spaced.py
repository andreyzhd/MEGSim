#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 20:07:42 2021

@author: andrey
"""

import pickle
import numpy as np
from megsimutils.optimize import BarbuteArraySL

#%% Parameter definitions
#R = 0.15
#HEIGHT_LOWER = 0.15
#ELLIP_SC = np.array([1.,1.,1.])
#ORIGIN = np.array([[0.,0.,0.],])

R = 1
HEIGHT_LOWER = 1
ELLIP_SC = np.array([0.0994945 , 0.08377065, 0.09113235])
ORIGIN = np.array([[0.02100584,  0.00433862,  0.0040068], 
                   [-0.0071708 , 0.00042154, -0.00346818],])

L_INT = 16
L_EXT = 3
N_SENS_RANGE = range(L_INT*(L_INT+2) + L_EXT*(L_EXT+2), 3 * (L_INT*(L_INT+2) + L_EXT*(L_EXT+2)) + 1, 10)
IS_OPM = False
RE = 0.2            # Radius for energy-based normlization, unaffected by ELLIP_SC
N_ITER_RAND = 100

OUT_FNAME = 'fig_01_ellip_enorm.pkl'

r_conds = []
r_conds_radial = []
r_conds_rand = []

for n_sens in N_SENS_RANGE:
    print('Processing array of size %i' % n_sens)
    sens_array = BarbuteArraySL(n_sens, L_INT, L_EXT, R_inner=R, height_lower=HEIGHT_LOWER, opm=IS_OPM, ellip_sc=ELLIP_SC, origin=ORIGIN, Re=RE)
    r_conds.append(sens_array.comp_fitness(sens_array.evenly_spaced_radial_v(False)))
    r_conds_radial.append(sens_array.comp_fitness(sens_array.evenly_spaced_radial_v(True)))
    
    r_conds_cur = []
    for i in range(N_ITER_RAND):
        r_conds_cur.append(sens_array.comp_fitness(sens_array.evenly_spaced_rand_v()))
        
    r_conds_rand.append(r_conds_cur)
    
    
# Save the results
fl = open(OUT_FNAME, 'wb')
pickle.dump((r_conds, r_conds_radial, r_conds_rand, N_SENS_RANGE), fl)
fl.close()
