#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Magnetic field computations.

"""

import numpy as np



def spherical_shell(npts, rmin, rmax):
    """Generate random points inside a spherical shell"""
    r = np.random.rand(npts) * (rmax-rmin) + rmin
    phi = np.random.randn(npts) * np.pi  # polar angle
    theta = np.random.randn(npts) * 2*np.pi  # azimuth
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
