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

from mne.preprocessing.maxwell import _sss_basis
from megsimutils.utils import _prep_mf_coils_pointlike

INP_PATH = '/home/andrey/scratch/out.289'
NBINS = 20

class SensorArrayDebugWrapper():
    """ This class is a hack used to compute various functions of the SensorArrays internal state for debuging purposes.
    """
    def __init__(self, sensor_array):
        self.__sensor_array = sensor_array
        sensor_array.comp_fitness(sensor_array.get_init_vector())   # do this to make sure that the forward matrices are initialized.
        
    def comp_stat_debug(self, v):
        """
        Compute various debug values on the enclosed SensorArray object (like alternative version of the fitness function, etc.)
        """            
        v = self.__sensor_array._validate_inp(v)
        rmags, nmags = self.__sensor_array._v2sens_geom(v)
        bins, n_coils, mag_mask, slice_map = _prep_mf_coils_pointlike(rmags, nmags)[2:]
        allcoils = (rmags, nmags, bins, n_coils, mag_mask, slice_map)
        
        # Forward matrices inside the __sensor_array should be initialized. That
        # means comp_fitness be been called at least once on __sensor_array
        # before reaching this line.
        assert self.__sensor_array._SensorArray__forward_matrices != None
        
        all_norms = []
        for exp, S_samp in zip(self.__sensor_array._SensorArray__exp, self.__sensor_array._SensorArray__forward_matrices):
            S = _sss_basis(exp, allcoils)
            Sp = np.linalg.pinv(S)

            all_norms.append(np.linalg.norm(S_samp @ Sp, axis=1))

        noise = np.max(np.column_stack(all_norms), axis=1)
        return noise.max(), noise.mean()
   

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
#    assert isclose(func(v), f, rel_tol=1e-6)
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
wrapper = SensorArrayDebugWrapper(sens_array)

interm_func = []
interm_func_recomp = []
interm_noise_mean = []
timing = []
x_accepts = []
y_accepts = []

hist = np.zeros((len(interm_res), NBINS))

i = 0
for (v, f, accept, tstamp) in interm_res:
    interm_func.append(f)
    noise_max, noise_mean = wrapper.comp_stat_debug(v)
    interm_func_recomp.append(noise_max)
    interm_noise_mean.append(noise_mean)

    if accept:
        x_accepts.append(len(interm_func_recomp)-1)
        y_accepts.append(f)
        
    timing.append(tstamp)
    hist[i,:], _ = np.histogram(interm_res[i][0][-sens_array._n_sens:], bins=NBINS, range=(sens_array._R_inner, sens_array._R_outer))
    i += 1
    

timing = np.diff(np.array(timing))

#%% Plot initial and final configurations
#fig1 = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
#sens_array.plot(v0, fig=fig1)

#fig2 = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
#sens_array.plot(interm_res[-1][0] , fig=fig2)

#mlab.sync_camera(fig1, fig2)


#%% Plot error vs iteration
plt.figure()
plt.plot(interm_func_recomp)
plt.plot(interm_noise_mean)
plt.plot(interm_func)
plt.plot(x_accepts, y_accepts, 'ok')
plt.xlabel('iterations')
plt.legend(['max noise (recomputed)', 'mean noise (recomputed)', 'max noise (loaded)', 'accepted'])
plt.title('L=(%i, %i), %i sensors' % (params['l_int'], params['kwargs']['l_ext'], np.sum(params['n_sens'])))


#%% Plot distances to the iner helmet surface
plt.figure()
plt.imshow(hist[::len(interm_res)//20//4].T)
plt.colorbar()
plt.title('Histogram of sensor depthes, L=(%i, %i), %i sensors' % (params['l_int'], params['kwargs']['l_ext'], np.sum(params['n_sens'])))

#%% Plot distances to the iner helmet surface -- log
plt.figure()
plt.imshow(np.log(hist[::len(interm_res)//20//4].T + 1))
plt.colorbar()
plt.title('Histogram of sensor depthes (log), L=(%i, %i), %i sensors' % (params['l_int'], params['kwargs']['l_ext'], np.sum(params['n_sens'])))

if isinstance(sens_array, BarbuteArraySL) and (not (sens_array._R_outer is None)):
    plt.figure()
    plt.hist(interm_res[-1][0][-sens_array._n_sens:] - sens_array._R_inner, 20)
    plt.xlabel('distance to the inner surface, m')
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