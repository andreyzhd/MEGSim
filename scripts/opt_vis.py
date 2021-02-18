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

from megsimutils.optimize import BarbuteArraySL

INP_PATH = '/home/andrey/scratch/out'

#%% Read the data
# Read the starting timestamp, etc
fl = open('%s/start.pkl' % INP_PATH, 'rb')
params, t_start, v0, sens_array, constraint_penalty = pickle.load(fl)
fl.close()

if constraint_penalty is None:
    func = (lambda v : sens_array.comp_fitness(v))
else:
    func = (lambda v : sens_array.comp_fitness(v) + constraint_penalty.compute(v))
        
# Read the intermediate results
interm_res = []
interm_res.append((v0, func(v0), False, t_start))

for fname in sorted(pathlib.Path(INP_PATH).glob('iter*.pkl')):
    print('Reading %s ...' % fname)
    fl = open (fname, 'rb')
    v, f, accept, tstamp = pickle.load(fl)
    fl.close()
    assert isclose(func(v), f, rel_tol=1e-6)
    interm_res.append((v, f, accept, tstamp))
    
assert len(interm_res) > 1  # should have at least one intermediate result
    
# Try to read the final result
try:
    fl = open('%s/final.pkl' % INP_PATH, 'rb')
    opt_res, tstamp = pickle.load(fl)
    fl.close()
    
    interm_res.append((opt_res.x, func(opt_res.x), True, tstamp))
except:
    print('Could not find the final result, using the last intermediate result instead')
       

#%% Prepare the variables describing the optimization progress
interm_func = []
interm_cond_nums = []
timing = []
x_accepts = []
y_accepts = []

for (v, f, accept, tstamp) in interm_res:
    interm_func.append(f)
    interm_cond_nums.append(sens_array.comp_fitness(v))

    if accept:
        x_accepts.append(len(interm_cond_nums)-1)
        y_accepts.append(np.log10(f))
        
    timing.append(tstamp)

timing = np.diff(np.array(timing))

#%% Plot initial and final configurations
#fig1 = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
#sens_array.plot(v0, fig=fig1)

#fig2 = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
#sens_array.plot(interm_res[-1][0] , fig=fig2)

#mlab.sync_camera(fig1, fig2)


#%% Plot error vs iteration
plt.figure()
plt.plot(np.log10(interm_cond_nums))
if not(constraint_penalty is None):
    plt.plot(np.log10(interm_func))
plt.plot(x_accepts, y_accepts, 'ok')
plt.xlabel('iterations')
if constraint_penalty is None:
    plt.legend([r'$\log_{10}(R_{cond})$', 'accepted'])
else:
    plt.legend([r'$\log_{10}(R_{cond})$', r'$\log_{10}(R_{cond}+C_{penalty})$', 'accepted'])
plt.title('L=%i, %i sensors' % (params['L'], np.sum(params['n_sens'])))


#%% Plot distances to the iner helmet surface
if isinstance(sens_array, BarbuteArraySL) and (not (sens_array._R_outer is None)):
    plt.figure()
    plt.hist(interm_res[-1][0][-sens_array._n_sens:] - sens_array._R_inner, 20)
    plt.xlabel('distance to the inner surface, m')
    plt.ylabel('n of sensors')
    plt.title('L=%i, %i sensors' % (params['L'], params['n_sens']))


#%% Plot the timing
plt.figure()
plt.bar(np.arange(len(timing)), timing)
plt.xlabel('iterations')
plt.ylabel('duration, s')


#%% Print some statistics
print('Initial condition number is 10^%0.3f' % np.log10(interm_cond_nums[0]))
print('Final condition number is 10^%0.3f' % np.log10(interm_cond_nums[-1]))
print('The lowest condition number is 10^%0.3f' % np.min(np.log10(interm_cond_nums)))
print('Iteration duration: mean %i s, min %i s, max %i s' % (timing.mean(), timing.min(), timing.max()))


#%% Interactive. I have no idea how it works - I've just copied it from the
# internet - andrey

from traits.api import HasTraits, Range, on_trait_change
from traitsui.api import View, Group

fig1 = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))

class Slider(HasTraits):
    iteration    = Range(0, len(interm_res)-1, 1, )    

    def __init__(self, figure):
        HasTraits.__init__(self)
        self._figure = figure
        sens_array.plot(interm_res[self.iteration][0], fig=figure)
        mlab.title('iteration %i' % self.iteration, figure=figure, size=0.5)
        
    @on_trait_change('iteration')
    def slider_changed(self):
        sens_array.plot(interm_res[self.iteration][0], fig=self._figure)
        mlab.title('iteration %i' % self.iteration, figure=self._figure, size=0.5)

    view = View(Group("iteration"))

my_model = Slider(fig1)
my_model.configure_traits()