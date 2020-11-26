#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demonstration of mesh functions

author: Jussi (jnu@iki.fi)

"""
# %% init
import numpy as np
from mayavi import mlab
import matplotlib.pylab as plt
import trimesh
from IPython import get_ipython

from megsimutils.mesh import _delaunay_tri, _my_tri, trimesh_equidistant_vertices
from megsimutils.viz import _mlab_points3d, _mlab_quiver3d, _mlab_trimesh
from megsimutils.array_geometry import barbute, spherical


# set up IPython
ip = get_ipython()
if ip is not None:
    ip.magic("matplotlib qt")
    ip.magic("reload_ext autoreload")  # these will enable module autoreloading
    ip.magic("autoreload 2")


# %% make a barbute array and triangulate it
NSENSORS_UPPER = 500
NSENSORS_LOWER = 500
ARRAY_RADIUS = 0.1
HEIGHT_LOWER = 0.12
PHISPAN_LOWER = 1.75 * np.pi

Sc, Sn = barbute(
    NSENSORS_UPPER, NSENSORS_LOWER, ARRAY_RADIUS, HEIGHT_LOWER, PHISPAN_LOWER
)
pts = Sc.copy()

# unfortunately, delaunay doesn't do a great job here, since the projection to
# 2D plane is a bit pathological
tris = _delaunay_tri(pts)
fig = mlab.figure()
_mlab_trimesh(pts, tris, figure=fig, representation='wireframe')
mlab.title('triangulated')

# supersample mesh to 5 mm resolution
fig = mlab.figure()
pts_, tris_ = trimesh.remesh.subdivide_to_size(pts, tris, 5e-3)
_mlab_trimesh(pts_, tris_, figure=fig, representation='wireframe')
mlab.title('supersampled')

# make a trimesh object and clean it a bit
tm = trimesh.Trimesh(vertices=pts_, faces=tris_)
trimesh.smoothing.filter_humphrey(tm)  # not sure if necessary
trimesh.repair.fix_winding(tm)
trimesh.repair.fix_inversion(tm)
trimesh.repair.fill_holes(tm)
# reobtain the points and vertices
pts_, tris_ = tm.vertices, tm.faces

# find the equidistant vertices
sensor_verts = trimesh_equidistant_vertices(tm, 500)

# convert trimesh TrackedArray -> np.array for speed
Sc = np.array(pts_[sensor_verts, :])
Sn = tm.vertex_normals[sensor_verts, :]

# voila (?)
fig = mlab.figure()
_mlab_trimesh(pts_, tris_, figure=fig)
_mlab_points3d(Sc, figure=fig, scale_factor=.006)
mlab.title('resulting array')





