#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 02:46:46 2020

@author: andrey

Compute condition number vs l and radius
"""
#%% Inits
import pickle
import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt
from megsimutils.optimize import Objective, CondNumber

INP_PATH = '/tmp/out'
FINAL_FNAME = 'final.pkl'
INTERM_PREFIX = 'iter'
TSTAMP_FNAME = 't_start.pkl'

# TODO: replace the parameters below with reading from the file
N_COILS = 100
R = 0.15
L = 9 
THETA_BOUND = np.pi / 2

# Read the starting timestamp
fl = open('%s/%s' % (INP_PATH, TSTAMP_FNAME), 'rb')
t_start = pickle.load(fl)
fl.close()

# Read the final result
fl = open('%s/%s' % (INP_PATH, FINAL_FNAME), 'rb')
rmags0, cosmags0, x, y, z, x_cosmags, y_cosmags, z_cosmags, cond_num0, cond_num, opt_res, tstamp = pickle.load(fl)
fl.close()

# Read the intermediate results
interm_res = []
for i in range(opt_res.nit):
    fl = open ('%s/%s%03i.pkl' % (INP_PATH, INTERM_PREFIX, i), 'rb')
    opt_vars, f, accept, tstamp = pickle.load(fl)
    fl.close()
    interm_res.append((opt_vars, f, accept, tstamp))

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

bins = np.arange(N_COILS, dtype=np.int64)
mag_mask = np.ones(N_COILS, dtype=np.bool)

objective = Objective(R, L, bins, N_COILS, mag_mask, THETA_BOUND)
cond_num_comp = CondNumber(R, L, bins, N_COILS, mag_mask)

interm_cond_nums = []
interm_objs = []

for (opt_vars, f, accept, tstamp) in interm_res:
    interm_cond_nums.append(cond_num_comp.compute(opt_vars))
    interm_objs.append(objective.compute(opt_vars))
    
plt.plot(np.log10(interm_cond_nums))
#plt.plot(np.log10(interm_objs))
