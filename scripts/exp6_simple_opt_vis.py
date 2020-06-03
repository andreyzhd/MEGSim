#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 02:46:46 2020

@author: andrey

Compute condition number vs l and radius
"""
#%% Inits
import pickle
from mayavi import mlab

DATA_FNAME = '/tmp/opt.pkl'

f = open(DATA_FNAME, 'rb')
rmags0, x, y, z, cond_num0, cond_num = pickle.load(f)
f.close()

print('Initial condition number is 10^%0.3f' % cond_num0)
print('Final condition number is 10^%0.3f' % cond_num)


#%% Plot

fig1 = mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
mlab.clf(fig1)
mlab.points3d(rmags0[:,0], rmags0[:,1], rmags0[:,2], resolution=32, scale_factor=0.01, color=(0,0,1))
mlab.points3d(0, 0, 0, resolution=32, scale_factor=0.01, color=(0,1,0), mode='axes')

fig2 = mlab.figure(2, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
mlab.clf(fig2)
mlab.points3d(x, y, z, resolution=32, scale_factor=0.01, color=(0,0,1))
mlab.points3d(0, 0, 0, resolution=32, scale_factor=0.01, color=(0,1,0), mode='axes')

mlab.sync_camera(fig1, fig2)



