#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 18:22:47 2021

@author: andrey
"""


from mayavi import mlab
from megsimutils.viz import _plot_brain
from read_opt_res import read_opt_res

INP_PATH = '/home/andrey/storage/Data/MEGSim/2022-03-18_paper_RC_full_run/run_thin/out'
N_ITER = 100 # math.inf # Number of iterations to load

#%% Read the data
params, sens_array, interm_res, opt_res, iter_indx = read_opt_res(INP_PATH, max_n_samp=N_ITER)

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
        _plot_brain(figure)
        mlab.title('iteration %i' % self.iteration, figure=figure, size=0.5)
        
    @on_trait_change('iteration')
    def slider_changed(self):
        sens_array.plot(interm_res[self.iteration][0], fig=self._figure)
        _plot_brain(self._figure)
        mlab.title('iteration %i' % iter_indx[self.iteration], figure=self._figure, size=0.5)

    view = View(Group("iteration"))

my_model = Slider(fig1)
my_model.configure_traits()
