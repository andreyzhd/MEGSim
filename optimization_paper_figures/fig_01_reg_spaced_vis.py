#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 12:10:34 2021

@author: andrey
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt

INP_FNAME = 'fig_01.pkl'


fl = open(INP_FNAME, 'rb')
r_conds, r_conds_radial, r_conds_rand, n_sens_range = pickle.load(fl)
fl.close()

plt.semilogy(n_sens_range, r_conds)
plt.semilogy(n_sens_range, r_conds_radial)

r_conds_rand = np.array(r_conds_rand)
plt.semilogy(n_sens_range, r_conds_rand.min(axis=1))
plt.semilogy(n_sens_range, r_conds_rand.max(axis=1))

plt.legend(['surface normal', 'radial', 'random (min)', 'random (max)'])