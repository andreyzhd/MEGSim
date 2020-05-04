#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Magnetic field computations.

"""

import numpy as np


def dipfld_sph(Q, rQ, R, r0):
    """Compute magnetic field of current dipole in a conducting sphere.
    
    Implementation of the Sarvas formula.

    Parameters
    ----------
    Q : ndarray
        (1 x 3) dipole moment (Am)
    rQ : ndarray
        (1 x 3) dipole location (m)
    R : ndarray
        (N x 3) field points (m)
    r0 : ndarray
        (1 x 3) origin of sphere (m)
    
    Returns
    -------
    ndarray
        (N x 3) the magnetic field (T)
    """
    Q = Q.squeeze()  # to allow either (1,3) or (3,) shaped args
    rQ = rQ.squeeze()
    r0 = r0.squeeze()
    rQ = rQ - r0  # source locations rel. to conductor model origin
    R = R - r0  # field points rel. to conductor model origin
    Rl = np.sqrt(np.sum(R ** 2, 1))
    A = R - rQ  # field points rel. to dipole location
    Al = np.sqrt(np.sum(A ** 2, 1))
    F = Al * (Rl * Al + Rl ** 2 - np.dot(R, rQ))
    # einsum used for dot products of rows in matrices
    v1 = Al ** 2 / Rl + np.einsum('ij,ij->i', A, R) / Al + 2 * Al + 2 * Rl
    v2 = Al + 2 * Rl + np.einsum('ij,ij->i', A, R) / Al
    # element-wise products with broadcasting
    gF = R * v1[:, np.newaxis] - v2[:, np.newaxis] * rQ
    QxrQ = np.cross(Q, rQ)
    M1 = F[:, np.newaxis] * QxrQ
    M2 = gF * np.dot(R, QxrQ)[:, np.newaxis]
    B = 1e-7 * (M1 - M2) * (F ** (-2))[:, np.newaxis]
    return B


def magdipfld(m, rm, R):
    """Compute field of a magnetic dipole.

    Parameters
    ----------
    m : ndarray
        (1 x 3) the magnetic moment (Am²)
    rm : ndarray
        (1 x 3) dipole location (m)
    R : ndarray
        (N x 3) field points (m)

    Returns
    -------
    ndarray
        (Nx3) the magnetic field (T)
    """
    m = np.squeeze(m)
    R -= np.squeeze(rm)
    Rl = np.sqrt(np.sum(R ** 2, axis=1))
    B = 1e-7 * (3 * (R.T * np.dot(R, m) * Rl ** (-5)).T - np.outer(Rl ** -3, m))
    return B


def biot_savart(
    P, r, pts_per_edge=1e3, close_loop=True
):
    """Biot-Savart law for a unitary current.

    Parameters
    ----------
    P : ndarray (Np x 3)
        Vertices of polygon describing the current loop. The polygon will be
        closed automatically if close_loop=True.
    r : ndarray (Nf x 3)
        Field points.
    pts_per_edge : int
        Number of integration points per edge.
    close_loop : bool
        Whether to close the current loop (copies the first point into an endpoint).

    Returns
    -------
    ndarray
        (N x 3) the magnetic field (T)
    """
    r = np.atleast_2d(r)  # support (N,3) and (,3) shaped inputs
    P = np.atleast_2d(P)
    if close_loop and not np.array_equal(P[-1, :], P[0, :]):
        P = np.append(P, P[0, None, :], axis=0)
    nverts = P.shape[0] - 1
    nfld = r.shape[0]
    dP = np.diff(P, axis=0)
    fld = np.zeros((nfld, 3))
    # for each edge: discretize, compute the integral and accumulate field
    for v in np.arange(nverts):
        dlvec = dP[v, :] / pts_per_edge
        l = P[v, :] + np.arange(pts_per_edge)[:, None] * dlvec
        r_l = r[:, np.newaxis] - l  # pairwise r-l vectors by broadcasting
        # cross product of dl with all r-l pairs, normalized by (r-l)**3
        cp = np.cross(dlvec, r_l) / np.linalg.norm(r_l, axis=-1, keepdims=True) ** 3
        fld += np.sum(cp, axis=1)
    return 1e-7 * fld


def _biot_savart_vec(P, r, pts_per_edge=1e3, close_loop=True):
    """Biot-Savart law for a unitary current.

    This version avoids looping over edges. However in some cases it's actually
    slower than the above version with the loop, so it's mostly a curiosity.

    Parameters
    ----------
    P : ndarray (Np x 3)
        Vertices of polygon describing the current loop. The polygon will be
        closed automatically if close_loop=True.
    r : ndarray (Nf x 3)
        Field points.
    pts_per_edge : int
        Number of integration points per edge.
    close_loop : bool
        Whether to close the current loop (copies the first point into an endpoint).

    Returns
    -------
    ndarray
        (N x 3) the magnetic field (T)
    """
    r = np.atleast_2d(r)  # support (N,3) and (,3) shaped inputs
    P = np.atleast_2d(P)
    if close_loop and not np.array_equal(P[-1, :], P[0, :]):
        P = np.append(P, P[0, None, :], axis=0)
    nverts = P.shape[0] - 1
    dP = np.diff(P, axis=0)
    dlvecs = dP / pts_per_edge
    # this is a bit messy: form a (M x 3) matrix of all discretized edge points
    l = (
        (P[:-1, :, None] + dlvecs[:, :, None] * np.arange(pts_per_edge))
        .swapaxes(1, 2)
        .reshape(int(nverts * pts_per_edge), 3)
    )
    r_l = r[:, np.newaxis] - l  # pairwise r-l vectors by broadcasting
    dlvecs_ = dlvecs.repeat(pts_per_edge, axis=0)
    # cross product of dl with all r-l pairs, normalized by (r-l)**3
    cp = np.cross(dlvecs_, r_l) / np.linalg.norm(r_l, axis=-1, keepdims=True) ** 3
    return 1e-7 * np.sum(cp, axis=1)
