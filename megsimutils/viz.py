#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization functions for megsim.

author: Jussi (jnu@iki.fi)
"""
from mayavi import mlab


# the following wrappers exist mostly to allow direct passing of (Nx3) points
# matrices as args


def _mlab_trimesh(pts, tris, **kwargs):
    """Plots trimesh specified by pts and tris into given figure.
    pts : (N x 3) array-like
    """
    x, y, z = pts.T
    return mlab.triangular_mesh(x, y, z, tris, **kwargs)


def _mlab_mesh(pts, **kwargs):
    """Plots trimesh specified by pts and tris into given figure.
    pts : (N x 3) array-like
    """
    x, y, z = pts.T
    return mlab.mesh(x, y, z, **kwargs)


def _mlab_quiver3d(rr, nn, **kwargs):
    """Plots vector field as arrows.
    rr : (N x 3) array-like
        The locations of the vectors.
    nn : (N x 3) array-like
        The vectors.
    """
    vx, vy, vz = rr.T
    u, v, w = nn.T
    return mlab.quiver3d(vx, vy, vz, u, v, w, **kwargs)


def _mlab_points3d(rr, *args, **kwargs):
    """Plots points.
    rr : (N x 3) array-like
        The locations of the vectors.
    Note that the api to mayavi points3d is weird, there is no way to specify colors and sizes
    individually. See:
    https://stackoverflow.com/questions/22253298/mayavi-points3d-with-different-size-and-colors
    """
    vx, vy, vz = rr.T
    return mlab.points3d(vx, vy, vz, *args, **kwargs)
