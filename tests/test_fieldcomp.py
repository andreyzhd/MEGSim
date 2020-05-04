#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 13:35:37 2019

@author: jussi
"""

from numpy.testing import assert_allclose
import numpy as np

from megsimutils import dipfld_sph, biot_savart, magdipfld
from megsimutils.utils import spherical_shell, _rotate_to


def test_bs_vs_wire():
    """Test Biot-Savart law vs. long wire"""
    P = np.array([[-1e3, 0, 0], [1e3, 0, 0]])
    r = np.array([[0, -1, 0]])
    fld = biot_savart(P, r, close_loop=False)
    assert_allclose(fld, [[0.0, 0.0, -2e-7]], rtol=1e-6)


def test_bs_vs_magdip():
    """Test Biot-Savart law vs. magnetic dipole.

    Generate a circular current loop in xy plane and compute field by
    Biot-Savart law. From a distance, the field should look like the equivalent
    magnetic dipole.
    """
    # generate the field points (spherical shell)
    nfld = 1000
    rmin = 3
    rmax = 6
    r = spherical_shell(nfld, rmin, rmax)
    # generate the loop in xy plane
    rcirc = 0.01  # radius in m
    th = np.linspace(0, 2 * np.pi, num=200, endpoint=False)
    x = rcirc * np.cos(th)
    y = rcirc * np.sin(th)
    z = np.zeros(th.shape)
    P = np.column_stack((x, y, z))
    # set up equivalent magnetic dipole
    m = np.array([0, 0, 1]) * np.pi * rcirc ** 2
    rm = np.array([0, 0, 0])
    # compare
    fld_bs = biot_savart(P, r)
    fld_md = magdipfld(m, rm, r)
    # do not use relative tolerance here, since some field components may become
    # arbitrarily small
    assert_allclose(fld_bs, fld_md, rtol=0, atol=1e-15)
