#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize the results of the sensor optimization. Read the results to be visualized
from the folder passed as command-lie parameter to this script.
"""

import math
import sys
from math import isclose
import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
from megsimutils.arrays import BarbuteArraySL, BarbuteArraySLGrid, noise_max, noise_mean
from megsimutils.volume_slicer import VolumeSlicer
from megsimutils.utils import uniform_sphere_dipoles, comp_inf_capacity
from megsimutils import read_opt_res

NBINS = 20
MAX_N_ITER = 100 # math.inf

# Read the input folder
if len(sys.argv) != 2:
    raise RuntimeError('Wrong number of parameters. Specify the input path as a single parameter.')

#%% Read the data
params, sens_array, interm_res, opt_res, iter_indx = read_opt_res(sys.argv[-1], max_n_samp=MAX_N_ITER)
       
#%% Prepare the variables describing the optimization progress
timing = []
angle_hist = np.zeros((len(interm_res), NBINS))

i = 0
for (v, f, accept, tstamp) in interm_res:
    slocs, snorms = sens_array._v2sens_geom(v)
    timing.append(tstamp)

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


#%% Plot angles (wrt surface normal)
plt.figure()
if len(interm_res) > NBINS * 4:
    plt.imshow(angle_hist[::len(interm_res)//NBINS//4].T)
else:
    plt.imshow(angle_hist.T)
    
plt.colorbar()
plt.title('Histogram of sensor angles (w.r.t. surface normals), L=(%i, %i), %i sensors' % (params['l_int'], params['l_ext'], np.sum(params['n_sens'])))


# Plot the histogram of angles in the last iteration
plt.figure()
plt.hist(angle_hist[-1,:], NBINS)
plt.xlabel('angle to surface normal, rads')
plt.ylabel('n of sensors')
plt.title('L=(%i, %i), %i sensors' % (params['l_int'], params['l_ext'], params['n_sens']))


#%% Plot the timing
plt.figure()
plt.bar(np.arange(len(timing)), timing)
plt.xlabel('iterations')
plt.ylabel('duration, s')

plt.show()


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


#%% For 3D arrays only
if isinstance(sens_array, BarbuteArraySL) and (not (sens_array._R_outer is None)):
    plt.figure()
    plt.hist(interm_res[-1][0][-sens_array._n_sens:] - sens_array._R_inner, NBINS)
    plt.xlabel('distance to the inner surface, m')
    plt.ylabel('n of sensors')
    plt.title('L=(%i, %i), %i sensors' % (params['l_int'], params['l_ext'], params['n_sens']))

    # Plot the distribution of the interpolation error
    sens_array = BarbuteArraySLGrid(params['n_sens'],
                                    params['l_int'],
                                    params['l_ext'],
                                    R_inner=params['R_inner'],
                                    R_outer=params['R_outer'],
                                    n_samp_layers=params['n_samp_layers'],
                                    n_samp_per_layer=params['n_samp_per_layer'],
                                    **params['kwargs'], grid_sz=100)

    noise_grd = sens_array.noise_grid(interm_res[-1][0])
    m = VolumeSlicer(data=noise_grd)
    m.configure_traits()
