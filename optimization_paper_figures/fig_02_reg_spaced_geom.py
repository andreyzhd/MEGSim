#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 17:14:11 2021

@author: andrey
"""
import pickle
import pathlib
from mayavi import mlab

INP_FLDR_NAME = 'opt_orient_only/303_sens'


#%% Read the data
n_sens_range_opt = []
r_conds_opt = []

fl = open('%s/start.pkl' % INP_FLDR_NAME, 'rb')
params, t_start, v0, sens_array, constraint_penalty = pickle.load(fl)
fl.close()

fl_list = sorted(pathlib.Path(INP_FLDR_NAME).glob('iter*.pkl'))
    
assert len(fl_list) > 0 # should have at least one intermediate result
print('Reading %s ...' % fl_list[-1])
fl = open (fl_list[-1], 'rb')
v, f, accept, tstamp = pickle.load(fl)
fl.close()


#%% Plot
fig1 = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
sens_array.plot(sens_array.evenly_spaced_radial_v(False), fig=fig1)

fig2 = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
sens_array.plot(sens_array.evenly_spaced_radial_v(True), fig=fig2)

fig3 = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
sens_array.plot(sens_array.evenly_spaced_rand_v(), fig=fig3)

fig4 = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
sens_array.plot(v, fig=fig4)

mlab.sync_camera(fig1, fig2)
mlab.sync_camera(fig1, fig3)
mlab.sync_camera(fig1, fig4)
