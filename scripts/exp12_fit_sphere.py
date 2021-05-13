#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 04:54:25 2021

@author: andrey
"""

import pathlib
import numpy as np
import scipy.optimize

import open3d
from mayavi import mlab

import mne
from mne.io.constants import FIFF

from megsimutils.viz import _mlab_trimesh, _plot_sphere
from megsimutils.utils import spherepts_golden

K = 2
MARG = 0.016
N_POINTS = 1000
ALPHA = 1   # Parameter for generating a tri-mesh from point cloud

data_path = pathlib.Path(mne.datasets.sample.data_path())
bem_file = data_path / 'subjects/sample/bem/sample-5120-5120-5120-bem-sol.fif'
bs = mne.read_bem_surfaces(bem_file)

# head
head = [s for s in bs if s['id'] == FIFF.FIFFV_BEM_SURF_ID_HEAD][0]
assert head['coord_frame'] == FIFF.FIFFV_COORD_MRI
head_pts, head_tris = head['rr'], head['tris']

# brain
brain = [s for s in bs if s['id'] == FIFF.FIFFV_BEM_SURF_ID_BRAIN][0]
assert brain['coord_frame'] == FIFF.FIFFV_COORD_MRI
brain_pts, brain_tris = brain['rr'], brain['tris']

# Shift and rotate the head and the brain
center = (brain_pts.min(axis=0) + brain_pts.max(axis=0)) / 2
head_pts -= center
brain_pts -= center

rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

head_pts = head_pts @ rot
brain_pts = brain_pts @ rot


# elliptical approximation of the head
sc = np.abs(brain_pts.max(axis=0) - brain_pts.min(axis=0) + 2*np.ones(3)*MARG) / 2
approx_pts = (spherepts_golden(N_POINTS) * sc)

# Generate a mesh
point_cloud = open3d.geometry.PointCloud()
point_cloud.points = open3d.utility.Vector3dVector(approx_pts)

mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, ALPHA)
approx_tris = np.asarray(mesh.triangles)
approx_pts = np.asarray(mesh.vertices)


#%% Optimization

def k_sphere_fit_xyz(v, brain_pts, head_pts):
    """No restrictions on origins' locations"""
    assert v.shape[0] % 4 == 0
    k = v.shape[0] // 4
    
    spheres = np.reshape(v, (k,4))
    inner_costs = []
    outer_costs = []
    
    for i in range(k):
         inner_costs.append(np.linalg.norm(brain_pts-spheres[i,:3], axis=1) - spheres[i,3])
         outer_costs.append(spheres[i,3] - np.linalg.norm(head_pts-spheres[i,:3], axis=1))
        
    inner_cost = np.max(np.min(np.column_stack(inner_costs), axis=1))
    outer_cost = np.max(np.column_stack(outer_costs))

    return max(inner_cost, outer_cost)
    
def xr_2_xyzr(v):
    assert v.shape[0] % 2 == 0
    k = v.shape[0] // 2
    x, r = *np.reshape(v, (k,2)).T,
    spheres = np.zeros((k,4))
    spheres[:,0] = x
    spheres[:,3] = r
    return spheres.flatten()

def xzr_2_xyzr(v):
    assert v.shape[0] % 3 == 0
    k = v.shape[0] // 3
    x, z, r = *np.reshape(v, (k,3)).T,
    spheres = np.zeros((k,4))
    spheres[:,0] = x
    spheres[:,2] = z
    spheres[:,3] = r
    return spheres.flatten()
       
    
# Preparing starting point, boundaries
xyz0 = brain_pts.mean(axis=0)
xyzr0 = np.append(xyz0, np.max(np.linalg.norm(brain_pts-xyz0, axis=1)))

xyz_min = brain_pts.min(axis=0)
xyz_max = brain_pts.max(axis=0)


r_min = np.min(np.linalg.norm(brain_pts-xyz0, axis=1))
r_max = np.max(np.linalg.norm(head_pts-xyz0, axis=1))

bounds_xyz = np.column_stack((np.append(xyz_min, r_min), np.append(xyz_max, r_max)))
bounds_x = np.column_stack((np.append(xyz_min[0], r_min), np.append(xyz_max[0], r_max)))
bounds_xz = np.column_stack((np.append(xyz_min[(0,2),], r_min), np.append(xyz_max[(0,2),], r_max)))

# Select the function and bounds to optimize
# exp_coords = lambda v : v
# bounds = bounds_xyz

# exp_coords = xr_2_xyzr
# bounds = bounds_x

exp_coords = xzr_2_xyzr
bounds = bounds_xz

k_sphere_fit = lambda v,  brain_pts, head_pts : k_sphere_fit_xyz(exp_coords(v), brain_pts, head_pts)

while True:
    opt_res_head = scipy.optimize.dual_annealing(k_sphere_fit, np.tile(bounds, (K, 1)), args=(brain_pts, head_pts), callback=(lambda x, f, accept : print('f = %f' % f)), maxiter=1000)
    print('Best fit for the real head is %f' % k_sphere_fit(opt_res_head.x, brain_pts, head_pts))
    if k_sphere_fit(opt_res_head.x, brain_pts, head_pts) <= 0:
        break
    else:
        print('Best fit > 0, retrying')

while True:
    opt_res_approx = scipy.optimize.dual_annealing(k_sphere_fit, np.tile(bounds, (K, 1)), args=(brain_pts, approx_pts), callback=(lambda x, f, accept : print('f = %f' % f)), maxiter=1000)
    print('Best fit for the approximation is %f' % k_sphere_fit(opt_res_approx.x, brain_pts, approx_pts))

    if k_sphere_fit(opt_res_approx.x, brain_pts, approx_pts) <= 0:
        break
    else:
        print('Best fit > 0, retrying')
        
print('Best fit for the approximation fits the real head with real head with fitness %f' % k_sphere_fit(opt_res_head.x, brain_pts, head_pts))

#%% Plot the results
fig0 = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))

_mlab_trimesh(brain_pts, brain_tris)
_mlab_trimesh(head_pts, head_tris, color=(0.5, 0.5, 0.5), opacity=0.5)

spheres = np.reshape(exp_coords(opt_res_head.x), (K,4))
for i in range(K):
    _plot_sphere(spheres[i,:3], spheres[i,3], N_POINTS, fig0, color=(0, 0, 0.5), opacity=0.5)

##------------------------
fig1 = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))

_mlab_trimesh(brain_pts, brain_tris)
#_mlab_trimesh(head_pts, head_tris, color=(0.5, 0.5, 0.5), opacity=0.5)
_mlab_trimesh(approx_pts, approx_tris, color=(0.5, 0.5, 0.5), opacity=0.5)

spheres = np.reshape(exp_coords(opt_res_approx.x), (K,4))
for i in range(K):
    _plot_sphere(spheres[i,:3], spheres[i,3], N_POINTS, fig1, color=(0, 0, 0.5), opacity=0.5)


##------------------------
fig2 = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
_mlab_trimesh(brain_pts, brain_tris)
_mlab_trimesh(head_pts, head_tris, color=(0.5, 0.5, 0.5), opacity=0.5)
_mlab_trimesh(approx_pts, approx_tris, color=(0, 0, 0.5), opacity=0.5)

mlab.sync_camera(fig0, fig1)
mlab.sync_camera(fig0, fig2)
