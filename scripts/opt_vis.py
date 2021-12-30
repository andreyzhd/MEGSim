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
from megsimutils.arrays import BarbuteArraySL, BarbuteArraySLGrid, noise_max, noise_mean
from megsimutils import VolumeSlicer
from megsimutils.utils import uniform_sphere_dipoles, comp_inf_capacity


INP_PATH = '/home/andrey/scratch/out'
NBINS = 20


N_DIPOLES_LEADFIELD = 10000
R_LEADFIELD = 0.1 # m
DIPOLE_STR = 5 * 1e-8 # A * m
SQUID_NOISE = 10 * 1e-15 # Tesla

   

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
    assert isclose(func(v), f, rel_tol=1e-4)
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
assert params['R_inner'] > R_LEADFIELD
dlocs, dnorms = uniform_sphere_dipoles(N_DIPOLES_LEADFIELD, R_LEADFIELD, seed=1)

interm_noise_max = []
interm_noise_mean = []
interm_inf = []
timing = []
x_accepts = []
y_accepts = []

d_hist = np.zeros((len(interm_res), NBINS))
angle_hist = np.zeros((len(interm_res), NBINS))

i = 0
for (v, f, accept, tstamp) in interm_res:
    noise = sens_array.comp_interp_noise(v)
    interm_noise_max.append(noise_max(noise))
    interm_noise_mean.append(noise_mean(noise))
    slocs, snorms = sens_array._v2sens_geom(v)
    interm_inf.append(comp_inf_capacity(slocs, snorms, dlocs, dnorms, DIPOLE_STR, SQUID_NOISE))

    if accept:
        x_accepts.append(len(interm_noise_max)-1)
        y_accepts.append(f)
        
    timing.append(tstamp)
    d_hist[i,:], _ = np.histogram(interm_res[i][0][-sens_array._n_sens:], bins=NBINS, range=(sens_array._R_inner, sens_array._R_outer))
    
    # Compute the angle histogram
    rmags, nmags = sens_array._v2sens_geom(v)
    rmags[rmags[:,2] < 0, 2] = 0 # set all negative z values to 0
    assert isclose((np.linalg.norm(nmags, axis=1)).min(), 1, rel_tol=1e-6)
    assert isclose((np.linalg.norm(nmags, axis=1)).max(), 1, rel_tol=1e-6)
      
    prods = np.diagonal(nmags @ (rmags.T / np.linalg.norm(rmags, axis=1))).copy()
    prods[prods>1] = 1  # Not really needed, used to avoid warnings when cos slightly exceeds 1 (due to round-off errors)
    angles = np.arccos(prods)    
    angle_hist[i,:], _ = np.histogram(angles, bins=NBINS, range=(0, np.pi))
    
    i += 1
    

timing = np.diff(np.array(timing))

#%% Plot initial and final configurations
#fig1 = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
#sens_array.plot(v0, fig=fig1)

#fig2 = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
#sens_array.plot(interm_res[-1][0] , fig=fig2)

#mlab.sync_camera(fig1, fig2)


#%% Plot information capacity
plt.figure()
plt.plot(interm_inf)
plt.xlabel('iterations')
plt.ylabel('bits')
plt.legend(['total information per sample'])
plt.title('L=(%i, %i), %i sensors, optimized for %s' % (params['l_int'], params['kwargs']['l_ext'], np.sum(params['n_sens']), params['kwargs']['noise_stat'].__name__))



#%% Plot error vs iteration
plt.figure()
plt.plot(interm_noise_max)
plt.plot(interm_noise_mean)
plt.plot(x_accepts, y_accepts, 'ok')
plt.ylim((0, np.percentile(interm_noise_max, 95)))
plt.xlabel('iterations')
plt.legend(['max noise', 'mean noise', 'accepted'])
plt.title('L=(%i, %i), %i sensors, optimized for %s' % (params['l_int'], params['kwargs']['l_ext'], np.sum(params['n_sens']), params['kwargs']['noise_stat'].__name__))


#%% Plot distances to the inner helmet surface
plt.figure()
if len(interm_res) > NBINS * 4:
    plt.imshow(d_hist[::len(interm_res)//NBINS//4].T)
else:
    plt.imshow(d_hist.T)
    
plt.colorbar()
plt.title('Histogram of sensor depthes, L=(%i, %i), %i sensors' % (params['l_int'], params['kwargs']['l_ext'], np.sum(params['n_sens'])))

#%% Plot distances to the inner helmet surface -- log
plt.figure()
if len(interm_res) > NBINS * 4:
    plt.imshow(np.log(d_hist[::len(interm_res)//NBINS//4].T + 1))
else:
    plt.imshow(np.log(d_hist.T + 1))
plt.colorbar()
plt.title('Histogram of sensor depthes (log), L=(%i, %i), %i sensors' % (params['l_int'], params['kwargs']['l_ext'], np.sum(params['n_sens'])))


#%% Plot angles (wrt surface normal)
plt.figure()
if len(interm_res) > NBINS * 4:
    plt.imshow(angle_hist[::len(interm_res)//NBINS//4].T)
else:
    plt.imshow(angle_hist.T)
    
plt.colorbar()
plt.title('Histogram of sensor angles (w.r.t. surface normals), L=(%i, %i), %i sensors' % (params['l_int'], params['kwargs']['l_ext'], np.sum(params['n_sens'])))


if isinstance(sens_array, BarbuteArraySL) and (not (sens_array._R_outer is None)):
    plt.figure()
    plt.hist(interm_res[-1][0][-sens_array._n_sens:] - sens_array._R_inner, NBINS)
    plt.xlabel('distance to the inner surface, m')
    plt.ylabel('n of sensors')
    plt.title('L=(%i, %i), %i sensors' % (params['l_int'], params['kwargs']['l_ext'], params['n_sens']))
    
    
# Plot the histogram of angles in the last iteration
plt.figure()
plt.hist(angle_hist[-1,:], NBINS)
plt.xlabel('angle to surface normal, rads')
plt.ylabel('n of sensors')
plt.title('L=(%i, %i), %i sensors' % (params['l_int'], params['kwargs']['l_ext'], params['n_sens']))


#%% Plot the timing
plt.figure()
plt.bar(np.arange(len(timing)), timing)
plt.xlabel('iterations')
plt.ylabel('duration, s')

plt.show()

#%% Print some statistics
#print('Initial condition number is 10^%0.3f' % np.log10(interm_cond_nums[0]))
#print('Final condition number is 10^%0.3f' % np.log10(interm_cond_nums[-1]))
#print('The lowest condition number is 10^%0.3f' % np.min(np.log10(interm_cond_nums)))
#print('Iteration duration: mean %i s, min %i s, max %i s' % (timing.mean(), timing.min(), timing.max()))


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

#%% Plot the distribution of the interpolation error
sens_array = BarbuteArraySLGrid(params['n_sens'],
                                params['l_int'],
                                R_inner=params['R_inner'],
                                R_outer=params['R_outer'],
                                n_samp_layers=params['n_samp_layers'],
                                n_samp_per_layer=params['n_samp_per_layer'],
                                **params['kwargs'], grid_sz=100)


noise_grd = sens_array.noise_grid(interm_res[-1][0])
m = VolumeSlicer(data=noise_grd)
m.configure_traits()