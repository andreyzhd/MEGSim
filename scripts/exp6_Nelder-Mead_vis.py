#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 02:46:46 2020

@author: andrey

Visualize the results of Nelder-Mead minimization experiment
"""
#%% Inits
import pickle
from mayavi import mlab
import matplotlib.pyplot as plt

INP_PATH = '/tmp/out'
FINAL_FNAME = 'final.pkl'
INTERM_PREFIX = 'iter'
START_FNAME = 'start.pkl'


# Read the starting timestamp, parameters
fl = open('%s/%s' % (INP_PATH, START_FNAME), 'rb')
(params, t_start) = pickle.load(fl)
fl.close()

# Read the final result
fl = open('%s/%s' % (INP_PATH, FINAL_FNAME), 'rb')
rmags0, cosmags0, x, y, z, x_cosmags, y_cosmags, z_cosmags, cond_num0, cond_num, opt_res, tstamp, objective = pickle.load(fl)
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
plt.plot(objective.history)
plt.title('Optimization history')
plt.xlabel('iteration')
plt.ylabel('objective function')
