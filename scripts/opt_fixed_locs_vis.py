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
from math import isclose
import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt
from megsimutils.utils import pol2xyz, fold_uh

INP_PATH = '/home/andrey/scratch/out_L09'
FINAL_FNAME = 'final.pkl'
INTERM_PREFIX = 'iter'
START_FNAME = 'start.pkl'

RANDOM_SEED = 42
N_PERM = 1000


#%% Read the data
# Read the starting timestamp, etc
fl = open('%s/%s' % (INP_PATH, START_FNAME), 'rb')
params, t_start, rmags, v0, cond_num_comp = pickle.load(fl)
fl.close()

# Read the intermediate results
interm_res = []
interm_res.append((v0, cond_num_comp.compute(v0), False, t_start))

for fname in sorted(pathlib.Path(INP_PATH).glob('%s*.pkl' % INTERM_PREFIX)):
    print('Reading %s ...' % fname)
    fl = open (fname, 'rb')
    v, f, accept, tstamp = pickle.load(fl)
    fl.close()
    assert isclose(cond_num_comp.compute(v), f)
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
    
theta0, phi0 = fold_uh(v0[:params['n_coils']], v0[params['n_coils']:])
theta, phi = fold_uh(v[:params['n_coils']], v[params['n_coils']:])

#%% Plot
x_cosmags0, y_cosmags0, z_cosmags0 = pol2xyz(1, theta0, phi0)
x_cosmags, y_cosmags, z_cosmags = pol2xyz(1, theta, phi)

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
interm_cond_nums = []
interm_objs = []
timing = []
x_accepts = []
y_accepts = []

for (v, f, accept, tstamp) in interm_res:
    interm_cond_nums.append(np.log10(f))
    if accept:
        x_accepts.append(len(interm_cond_nums)-1)
        y_accepts.append(np.log10(f))
        
    timing.append(tstamp)

timing = np.diff(np.array(timing))


#%% Plot error vs iteration
plt.figure()
plt.plot(interm_cond_nums)
plt.plot(x_accepts, y_accepts, 'ok')
plt.xlabel('iterations')
plt.legend([r'$ļog_{10}(R_{cond})$', 'accepted'])
plt.title('L=%i, %i sensors' % (params['L'], params['n_coils']))


#%% Plot the timing
plt.figure()
plt.bar(np.arange(len(timing)), timing)
plt.xlabel('iterations')
plt.ylabel('duration, s')


#%% Plot (v - v0) histogram as image
plt.figure()
plt.hist2d(theta-theta0, phi-phi0, cmap='plasma', bins=10)
plt.xlabel('dtheta')
plt.ylabel('dphi')
plt.colorbar()

#%% Randomely permute dtheta an dphi
np.random.seed(RANDOM_SEED)
cond_num_rand = np.zeros(N_PERM)

for i in range(N_PERM):
    perm = np.random.permutation(params['n_coils'])
    dtheta = theta - theta0
    dphi = phi - phi0

    vp = np.concatenate((theta0 + dtheta[perm], phi0 + dphi[perm]))
    cond_num_rand[i] = np.log10(cond_num_comp.compute(vp))
    
plt.figure()
plt.hist(cond_num_rand, bins=50, label='initial guess randomly perturbed')
maxhist = (np.histogram(cond_num_rand, bins=50)[0]).max()
plt.plot(min(interm_cond_nums)*np.ones(2), (0, maxhist), '--k', label='best solution')
plt.plot(np.log10(cond_num_comp.compute(v0))*np.ones(2), (0, maxhist), '-.k', label='initial guess')
plt.xlabel(r'$ļog_{10}(R_{cond})$')
plt.legend()
plt.title('L=%i, %i sensors' % (params['L'], params['n_coils']))


#%% Print some statistics
print('Initial condition number is 10^%0.3f' % np.log10(cond_num_comp.compute(v0)))
print('Final condition number is 10^%0.3f' % np.log10(cond_num_comp.compute(v)))
print('The lowest condition number is 10^%0.3f' % min(interm_cond_nums))
print('Iteration duration: mean %i s, min %i s, max %i s' % (timing.mean(), timing.min(), timing.max()))
