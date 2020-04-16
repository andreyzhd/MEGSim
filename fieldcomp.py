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
        (3,) dipole moment (Am)
    rQ : ndarray
        (3,) dipole location (m)
    R : ndarray
        (Nx3) field points (m)
    r0 : ndarray
        (3,) origin of sphere (m)
    
    Returns
    -------
    fld : ndarray
        (Nx3) the magnetic field (T)
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
