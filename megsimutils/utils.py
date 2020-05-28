#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Util functions for megsim.

"""

import numpy as np


def spherepts_golden(N):
    """Approximate uniformly distributed points on a unit sphere.

    Very fast, but I'm not sure how it works. Translated from old undocumented
    Matlab code.

    Parameters
    ----------
    n : int
        Number of points.

    Returns
    -------
    ndarray
        (N x 3) array of Cartesian point coordinates.
    """
    # create linearly spaced azimuthal coordinate
    dlong = np.pi * (3 - np.sqrt(5))
    longs = np.linspace(0, (N - 1) * dlong, N)
    # create linearly spaced z coordinate
    dz = 2 / N
    z = np.linspace(1 - dz / 2, -dz / 2 - 1 + 2 / N, N)
    r = np.sqrt(1 - z ** 2)
    # this looks like the usual cylindrical -> Cartesian transform?
    return np.column_stack((r * np.cos(longs), r * np.sin(longs), z))


def spherical_shell(npts, rmin, rmax):
    """Generate random points inside a spherical shell"""
    r = np.random.rand(npts) * (rmax - rmin) + rmin
    phi = np.random.randn(npts) * np.pi  # polar angle
    theta = np.random.randn(npts) * 2 * np.pi  # azimuth
    X = r * np.sin(phi) * np.cos(theta)
    Y = r * np.sin(phi) * np.sin(theta)
    Z = r * np.cos(phi)
    return np.column_stack((X, Y, Z))


def _rotate_to(v1, v2):
    """Return a rotation matrix which rotates unit vector v1 to v2.
    
    See https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    NB: may do weird things for some rotations (e.g. identity)
    """
    assert v1.shape == v2.shape == (3,)
    assert np.linalg.norm(v1) == np.linalg.norm(v2) == 1.0
    vs = v1 + v2
    return 2 * np.outer(vs, vs) / np.dot(vs, vs) - np.eye(3)


def _vector_angles(V, W):
    """Angles in degrees between column vectors of V and W. V must be dxM
    and W either dxM (for pairwise angles) or dx1 (for all-to-one angles)"""
    assert V.shape[0] == W.shape[0]
    if V.ndim == 1:
        V = V[:, None]
    if W.ndim == 1:
        W = W[:, None]
    assert V.shape[1] == W.shape[1] or W.shape[1] == 1
    Vn = V / np.linalg.norm(V, axis=0)
    Wn = W / np.linalg.norm(W, axis=0)
    dots = np.sum(Vn * Wn, axis=0)
    dots = np.clip(dots, -1, 1)
    return np.arccos(dots) / np.pi * 180
