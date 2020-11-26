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

from megsimutils.mesh import _delaunay_tri, _my_tri
from megsimutils.viz import _mlab_points3d, _mlab_trimesh
from megsimutils.array_geometry import barbute, spherical


# set up IPython
ip = get_ipython()
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
tris = _delaunay_tri(pts)
fig = mlab.figure()
_mlab_trimesh(pts, tris, figure=fig, representation='wireframe')

fig = mlab.figure()
pts_, tris_ = trimesh.remesh.subdivide_to_size(pts, tris, 5e-3)
_mlab_trimesh(pts_, tris_, figure=fig, representation='wireframe')





