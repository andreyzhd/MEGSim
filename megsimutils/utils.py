#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Util functions for megsim.

"""

import numpy as np


def spherepts_golden(N, angle=4*np.pi):
    """Approximate uniformly distributed points on a unit sphere.

    This is the "golden ratio algorithm".
    See: http://blog.marmakoide.org/?p=1

    Parameters
    ----------
    n : int
        Number of points.
        
    angle : float
        Solid angle (symmetrical around z axis) covered by the points. By
        default, the whole sphere. Must be between 0 and 4*pi

    Returns
    -------
    ndarray
        (N, 3) array of Cartesian point coordinates.
    """
    # create linearly spaced azimuthal coordinate
    dlong = np.pi * (3 - np.sqrt(5))
    longs = np.linspace(0, (N - 1) * dlong, N)
    # create linearly spaced z coordinate
    z_top = 1
    z_bottom = 1 - 2 * (angle/(4*np.pi))
    dz = (z_top - z_bottom) / N
    
    z = np.linspace(z_top - dz/2, z_bottom + dz/2, N)
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


def local_axes(theta, phi):
    """Compute local radial and tangential directions. theta, phi should be
    arrays of the same dimension. Returns arrays of the dimension d+1, where d
    id the input dimension. The last dimension is of the length 3, and it
    corresponds to x, y, and z coordinates of the tangential/radial unit
    vectors.
    
    Based on Hill 1954 doi:10.1119/1.1933682"""
    e_r = [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]
    e_theta = [np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)]
    e_phi = [-np.sin(phi), np.cos(phi), np.zeros_like(theta)]
    return  np.stack(e_r, axis=-1), np.stack(e_theta, axis=-1), np.stack(e_phi, axis=-1)


def xyz2pol(x, y, z):
    """ Convert from Cartesian to polar coordinates. x, y, z should be arrays
    of the same dimension"""
    r = np.linalg.norm(np.stack((x,y,z)), axis=0)
    phi = np.arctan2(y, x)
    phi[phi<0] += 2*np.pi
    theta = np.arccos(z / r)
    
    return r, theta, phi

def pol2xyz(r, theta, phi):
    """ Convert from polar to Cartesian coordinates. r, theta, phi should be
    arrays of the same dimension (r can also be a scalar)"""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return x, y, z

def fold_uh(theta, phi):
    """Fold theta and phi to the upper hemisphere. The resulting theta is
    between 0 and pi/2, phi -- between 0 and 2*pi. If the vector given by 
    (theta, phi) points down (theta > pi/2), reflect it."""
    
    x, y, z = pol2xyz(1, theta, phi)
    # reflect vectors pointing down
    x[z < 0] *= -1
    y[z < 0] *= -1
    z[z < 0] *= -1
    r, theta, phi = xyz2pol(x, y, z)
    return theta, phi