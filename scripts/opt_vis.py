#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 02:46:46 2020

@author: andrey

Compute condition number vs l and radius
"""
#%% Inits
import pickle
import pathlib
import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt
from megsimutils.optimize import Objective, CondNumber

INP_PATH = '/home/andrey/scratch/out'
FINAL_FNAME = 'final.pkl'
INTERM_PREFIX = 'iter'
START_FNAME = 'start.pkl'

#%% Read the data
# Read the starting timestamp
fl = open('%s/%s' % (INP_PATH, START_FNAME), 'rb')
(params, t_start) = pickle.load(fl)
fl.close()

# Read the intermediate results
interm_res = []
for fname in pathlib.Path(INP_PATH).glob('%s*.pkl' % INTERM_PREFIX):
    fl = open (fname, 'rb')
    opt_vars, f, accept, tstamp = pickle.load(fl)
    fl.close()
    interm_res.append((opt_vars, f, accept, tstamp))
    
# Read the final result
fl = open('%s/%s' % (INP_PATH, FINAL_FNAME), 'rb')
rmags0, cosmags0, x, y, z, x_cosmags, y_cosmags, z_cosmags, cond_num0, cond_num, opt_res, tstamp = pickle.load(fl)
fl.close()

print('Initial condition number is 10^%0.3f' % cond_num0)
print('Final condition number is 10^%0.3f' % cond_num)


#%% Plot

fig1 = mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
mlab.clf(fig1)
mlab.points3d(rmags0[:,0], rmags0[:,1], rmags0[:,2], resolution=32, scale_factor=0.01, color=(0,0,1))
mlab.quiver3d(rmags0[:,0], rmags0[:,1], rmags0[:,2], cosmags0[:,0], cosmags0[:,1], cosmags0[:,2])
mlab.points3d(0, 0, 0, resolution=32, scale_factor=0.01, color=(0,1,0), mode='axes')

fig2 = mlab.figure(2, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
mlab.clf(fig2)
mlab.points3d(x, y, z, resolution=32, scale_factor=0.01, color=(0,0,1))
mlab.quiver3d(x, y, z, x_cosmags, y_cosmags, z_cosmags)
mlab.points3d(0, 0, 0, resolution=32, scale_factor=0.01, color=(0,1,0), mode='axes')

mlab.sync_camera(fig1, fig2)

# Plot the optimization progress
#plt.plot((np.array(list(tstamp for (opt_vars, f, accept, tstamp) in interm_res)) - t_start) / 3600, 'o')

bins = np.arange(params['n_coils'], dtype=np.int64)
mag_mask = np.ones(params['n_coils'], dtype=np.bool)

objective = Objective(params['R'], params['L'], bins, params['n_coils'], mag_mask, params['theta_bound'])
cond_num_comp = CondNumber(params['R'], params['L'], bins, params['n_coils'], mag_mask)

interm_cond_nums = []
interm_objs = []
x_accepts = []
y_accepts = []

for (opt_vars, f, accept, tstamp) in interm_res:
    interm_cond_nums.append(cond_num_comp.compute(opt_vars))
    interm_objs.append(objective.compute(opt_vars))
    if accept:
        x_accepts.append(len(interm_cond_nums)-1)
        y_accepts.append(objective.compute(opt_vars))
    
plt.plot(interm_cond_nums)
#plt.plot(interm_objs)
#plt.plot(x_accepts, y_accepts, 'ok')
plt.xlabel('iterations')
#plt.legend([r'$R_{cond}$', r'$R_{cond}$ + constraint penalty', 'accepted'])
