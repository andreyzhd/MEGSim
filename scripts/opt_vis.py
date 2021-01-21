#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 18:22:47 2021

@author: andrey
"""

import pickle
import pathlib
from math import isclose
import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab

from megsimutils.optimize import BarbuteArray


INP_PATH = '/home/andrey/scratch/out'

#%% Read the data
# Read the starting timestamp, etc
fl = open('%s/start.pkl' % INP_PATH, 'rb')
params, t_start, v0, sens_array = pickle.load(fl)
fl.close()

# Read the intermediate results
interm_res = []
interm_res.append((v0, sens_array.comp_fitness(v0), False, t_start))

for fname in sorted(pathlib.Path(INP_PATH).glob('iter*.pkl')):
    print('Reading %s ...' % fname)
    fl = open (fname, 'rb')
    v, f, accept, tstamp = pickle.load(fl)
    fl.close()
    if not isclose(sens_array.comp_fitness(v), f):
        print('Warning! The function values reported by the optimization algorithm and does not match the parameters vector.')
        print('The optimization algorithm reports f = %f' %  f)
        print('The parameters yield sens_array.comp_fitness(v) = %f' % sens_array.comp_fitness(v))
        assert(False)
    interm_res.append((v, sens_array.comp_fitness(v), accept, tstamp))
    
assert len(interm_res) > 1  # should have at least one intermediate result
    
# Try to read the final result
try:
    fl = open('%s/final.pkl' % INP_PATH, 'rb')
    opt_res, tstamp = pickle.load(fl)
    v_final = opt_res.x
    fl.close()
except:
    print('Could not find the final result, using the last intermediate result instead')
       
    v_final = interm_res[-1][0]   
    tstamp = interm_res[-1][-1]


#%% Prepare the variables describing the optimization progress
interm_cond_nums = []
timing = []
x_accepts = []
y_accepts = []

for (v, f, accept, tstamp) in interm_res:
    interm_cond_nums.append(np.log10(f))

    if accept:
        x_accepts.append(len(interm_cond_nums)-1)
        y_accepts.append(np.log10(f))
        
    timing.append(tstamp)

timing = np.diff(np.array(timing))

#%% Plot initial and final configurations
fig1 = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
sens_array.plot(v0, fig=fig1)

fig2 = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
sens_array.plot(v_final, fig=fig2)

mlab.sync_camera(fig1, fig2)


#%% Plot error vs iteration
plt.figure()
plt.plot(interm_cond_nums)
plt.plot(x_accepts, y_accepts, 'ok')
plt.xlabel('iterations')
plt.legend([r'$\log_{10}(R_{cond})$', 'accepted'])
plt.title('L=%i, %i sensors' % (params['L'], params['n_coils']))


#%% Plot distances to the iner helmet surface
if not (sens_array._R_outer is None):
    plt.figure()
    plt.hist(v_final[-sens_array._n_coils:] - sens_array._R_inner, 20)
    plt.xlabel('distance to the inner surface, m')
    plt.ylabel('n of sensors')
    plt.title('L=%i, %i sensors' % (params['L'], params['n_coils']))


#%% Plot the timing
plt.figure()
plt.bar(np.arange(len(timing)), timing)
plt.xlabel('iterations')
plt.ylabel('duration, s')


#%% Print some statistics
print('Initial condition number is 10^%0.3f' % np.log10(sens_array.comp_fitness(v0)))
print('Final condition number is 10^%0.3f' % np.log10(sens_array.comp_fitness(v)))
print('The lowest condition number is 10^%0.3f' % min(interm_cond_nums))
print('Iteration duration: mean %i s, min %i s, max %i s' % (timing.mean(), timing.min(), timing.max()))

