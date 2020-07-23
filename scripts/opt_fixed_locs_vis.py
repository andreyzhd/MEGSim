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
from megsimutils.utils import pol2xyz, xyz2pol

INP_PATH = '/home/andrey/scratch/out'
FINAL_FNAME = 'final.pkl'
INTERM_PREFIX = 'iter'
START_FNAME = 'start.pkl'

#%% Read the data
# Read the starting timestamp, etc
fl = open('%s/%s' % (INP_PATH, START_FNAME), 'rb')
params, t_start, rmags, v0, cond_num_comp = pickle.load(fl)
fl.close()

# Read the intermediate results
interm_res = []
interm_res.append((v0, cond_num_comp.compute(v0), False, t_start))

for fname in pathlib.Path(INP_PATH).glob('%s*.pkl' % INTERM_PREFIX):
    fl = open (fname, 'rb')
    v, f, accept, tstamp = pickle.load(fl)
    fl.close()
    assert cond_num_comp.compute(v) == f
    interm_res.append((v, f, accept, tstamp))
    
assert len(interm_res) > 1  # should have at least one intermediate result
    
# Try to read the final result
try:
    fl = open('%s/%s' % (INP_PATH, FINAL_FNAME), 'rb')
    v, opt_res, tstamp = pickle.load(fl)
    fl.close()
except:
    print('Could not find the final result, using the last intermediate result instead')
       
    v = interm_res[-1][0]   
    tstamp = interm_res[-1][-1]
    
print('Initial condition number is 10^%0.3f' % np.log10(cond_num_comp.compute(v0)))
print('Final condition number is 10^%0.3f' % np.log10(cond_num_comp.compute(v)))


#%% Plot
x_cosmags0, y_cosmags0, z_cosmags0 = pol2xyz(1, v0[:params['n_coils']], v0[params['n_coils']:])
x_cosmags, y_cosmags, z_cosmags = pol2xyz(1, v[:params['n_coils']], v[params['n_coils']:])

fig1 = mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
mlab.clf(fig1)
mlab.points3d(rmags[:,0], rmags[:,1], rmags[:,2], resolution=32, scale_factor=0.01, color=(0,0,1))
mlab.quiver3d(rmags[:,0], rmags[:,1], rmags[:,2], x_cosmags0, y_cosmags0, z_cosmags0)
mlab.points3d(0, 0, 0, resolution=32, scale_factor=0.01, color=(0,1,0), mode='axes')

fig2 = mlab.figure(2, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
mlab.clf(fig2)
mlab.points3d(rmags[:,0], rmags[:,1], rmags[:,2], resolution=32, scale_factor=0.01, color=(0,0,1))
mlab.quiver3d(rmags[:,0], rmags[:,1], rmags[:,2], x_cosmags, y_cosmags, z_cosmags)
mlab.points3d(0, 0, 0, resolution=32, scale_factor=0.01, color=(0,1,0), mode='axes')

mlab.sync_camera(fig1, fig2)

# Plot the optimization progress
#plt.plot((np.array(list(tstamp for (opt_vars, f, accept, tstamp) in interm_res)) - t_start) / 3600, 'o')

bins = np.arange(params['n_coils'], dtype=np.int64)
mag_mask = np.ones(params['n_coils'], dtype=np.bool)

interm_cond_nums = []
interm_objs = []
x_accepts = []
y_accepts = []

for (v, f, accept, tstamp) in interm_res:
    interm_cond_nums.append(np.log10(f))
    if accept:
        x_accepts.append(len(interm_cond_nums)-1)
        y_accepts.append(np.log10(f))
    
plt.plot(interm_cond_nums)
plt.plot(x_accepts, y_accepts, 'ok')
plt.xlabel('iterations')
plt.legend([r'$ļog_{10}(R_{cond})$', 'accepted'])
