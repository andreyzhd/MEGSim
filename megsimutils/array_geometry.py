#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Util functions for megsim.

"""

import numpy as np
from mne.io.constants import FIFF
from mne.transforms import rotation3d_align_z_axis
import matplotlib.pylab as plt
from mayavi import mlab
from mne.preprocessing.maxwell import _sss_basis
from mne.transforms import _deg_ord_idx, _pol_to_cart, _cart_to_sph
from scipy.spatial import ConvexHull, Delaunay

from megsimutils.utils import spherepts_golden


def hockey_helmet(locs_dens, chin_strap_angle=np.pi / 8, inner_r=0.15, outer_r=None):
    """Create a hockey-helmet-like (a hemisphere and a chin strap) dense mesh
    of possible sensor locations. Locations are distributed approximately
    evenly. locs_dens defines the sensor density -- it's the number of sensors
    per 4*pi steradian. The helmet is optionally double-layered (if outer_r is
    not None).
    
    Returns four arrays -- x, y, and z coordinates of all
    the candidate points on a sphere and a boolean index array indicating
    which locations belong to the helmet.
    """
    pts = spherepts_golden(locs_dens)
    r, theta, phi = xyz2pol(pts[:, 0], pts[:, 1], pts[:, 2])
    helm_indx = ((phi > 0) & (phi < np.pi)) | (
        (phi > (3 / 2) * np.pi - chin_strap_angle) & (phi < (3 / 2) * np.pi)
    )

    x, y, z = pts[:, 0], pts[:, 2], pts[:, 1]  # y and z axis are swapped on purpose
    x_inner, y_inner, z_inner = x * inner_r, y * inner_r, z * inner_r
    if not (outer_r is None):
        assert outer_r > inner_r
        x_outer, y_outer, z_outer = x * outer_r, y * outer_r, z * outer_r
        return (
            np.concatenate((x_inner, x_outer)),
            np.concatenate((y_inner, y_outer)),
            np.concatenate((z_inner, z_outer)),
            np.concatenate((helm_indx, helm_indx)),
        )
    else:
        return x_inner, y_inner, z_inner, helm_indx


def barbute(nsensors_upper, nsensors_lower, array_radius, height_lower, phispan_lower):
    """Create an Italian war helmet.

    The array consists of spherical upper part (positive z)
    and cylindrical lower part (negative z). """

    # make the upper part
    Sc1 = spherepts_golden(nsensors_upper, angle=2 * np.pi)
    Sn1 = Sc1.copy()
    Sc1 *= array_radius

    # add some neg-z sensors on a cylindrical surface
    if nsensors_lower > 0:
        # estimate N of sensors in z and phi directions, so that total number
        # of sensors is approximately correct
        Nh = np.sqrt(nsensors_lower) * np.sqrt(
            height_lower / (phispan_lower * array_radius)
        )
        Nphi = nsensors_lower / Nh
        Nh = int(np.round(Nh))
        Nphi = int(np.round(Nphi))
        phis = np.linspace(0, phispan_lower, Nphi, endpoint=False)  # the phi angles
        zs = np.linspace(-height_lower, 0, Nh, endpoint=False)
        Sc2 = list()
        for phi in phis:
            for z in zs:
                Sc2.append([array_radius * np.cos(phi), array_radius * np.sin(phi), z])
        Sc2 = np.array(Sc2)
        Sn2 = Sc2.copy()
        Sn2[:, 2] = 0  # make normal vectors cylindrical
        Sn2 = (Sn2.T / np.linalg.norm(Sn2, axis=1)).T
        Sc = np.row_stack((Sc1, Sc2))
        Sn = np.row_stack((Sn1, Sn2))
    else:
        Sc = Sc1
        Sn = Sn1

    # optionally, make 90 degree flips for a subset of sensor normals
    FLIP_SENSORS = 0
    if FLIP_SENSORS:
        print(f'*** flipping {FLIP_SENSORS} sensors')
        to_flip = np.random.choice(Sc.shape[0], FLIP_SENSORS, replace=False)
        for k in to_flip:
            flipvec = _random_unit(3)
            Sn[k, :] = np.cross(Sn[k, :], flipvec)

    return Sc, Sn



