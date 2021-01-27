#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 15:27:41 2021

@author: andrey

3D visualizations of multiple VSH components
"""

import numpy as np
from mayavi import mlab

from mne.preprocessing.maxwell import _sss_basis
from megsimutils.utils import spherepts_golden, _prep_mf_coils_pointlike, _deg_ord_idx
import megsimutils
import matplotlib

N_POINTS = 20
N_POINTS_BG = 10000
DEPTH_BG = 0.2
L = 10

CMAP = 'cool'

rhelm = spherepts_golden(N_POINTS)
nhelm = rhelm.copy()

npoints = rhelm.shape[0]

# Compute VSH's
sss_params = {'origin': np.zeros(3), 'int_order': L, 'ext_order': 0}
allcoils = _prep_mf_coils_pointlike(np.tile(rhelm, (3,1)), np.repeat(np.eye(3), npoints, axis=0))
S = _sss_basis(sss_params, allcoils)
vsh = np.stack((S[:npoints,:], S[npoints:2*npoints,:], S[2*npoints:]), axis=2) # npoints-by-ncomp-by-3

#%% Plot
fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
mlab.clf(fig)

# Plot background
inner_locs = spherepts_golden(N_POINTS_BG) * (1 - DEPTH_BG)
pts = mlab.points3d(inner_locs[:,0], inner_locs[:,1], inner_locs[:,2], opacity=0, figure=fig)
mesh = mlab.pipeline.delaunay3d(pts)
mlab.pipeline.surface(mesh, figure=fig, color=(0.5, 0.5, 0.5), opacity=0.7)

# Plot sensor locations
mlab.points3d(rhelm[:,0], rhelm[:,1], rhelm[:,2], resolution=32, scale_factor=0.01, color=(0,0,1))

# Plot VSH components

# for l in range(1, L+1):
#     for m in range(l+1):
#         mlab.quiver3d(rhelm[:,0], rhelm[:,1], rhelm[:,2],
#                       vsh[:,_deg_ord_idx(l, m),0], vsh[:,_deg_ord_idx(l, m),1], vsh[:,_deg_ord_idx(l, m),2],
#                       figure=fig, color=matplotlib.cm.get_cmap('tab20').colors[_deg_ord_idx(l, m)])

for ind in np.arange(10):
    mlab.quiver3d(rhelm[:,0], rhelm[:,1], rhelm[:,2],
                  vsh[:,ind,0], vsh[:,ind,1], vsh[:,ind,2],
                  figure=fig, color=matplotlib.cm.get_cmap('tab20').colors[ind % 20])
    
    
# %% project some vshs onto spherical local coords
pol = megsimutils.utils.xyz2pol(*rhelm.T)
er, etheta, ephi  = megsimutils.utils.local_axes(pol[1], pol[2])

vsh_proj_er = (vsh[:, :, :] * er[:,None,:]).sum(axis=-1)
#plt.hist(vsh_proj_er[15, :])

vsh_proj_er = (vsh[:, :, :] * er[:,None,:]).sum(axis=-1)
vsh_proj_etheta = (vsh[:, :, :] * etheta[:,None,:]).sum(axis=-1)
vsh_proj_ephi = (vsh[:, :, :] * ephi[:,None,:]).sum(axis=-1)

vsh_r, vsh_theta, vsh_phi = utils.xyz2pol(vsh_proj_er, vsh_proj_etheta, vsh_proj_ephi)










 
    