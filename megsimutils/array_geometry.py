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

from megsimutils.utils import spherepts_golden, xyz2pol


def hockey_helmet(
    locs_dens,
    chin_strap_angle=np.pi / 8,
    inner_r=0.15,
    outer_r=None,
    symmetric_strap=False,
):
    """Create a hockey-helmet-like (a hemisphere and a chin strap) dense mesh
    of possible sensor locations. Locations are distributed approximately
    evenly. locs_dens defines the sensor density -- it's the number of sensors
    per 4*pi steradian. The helmet is optionally double-layered (if outer_r is
    not None).
    
    Returns two N-by-3 arrays -- locations and orientations
    """
    pts = spherepts_golden(locs_dens)
    r, theta, phi = xyz2pol(pts[:, 0], pts[:, 1], pts[:, 2])
    offset = (0, chin_strap_angle / 2)[symmetric_strap]
    helm_indx = ((phi > 0) & (phi < np.pi)) | (
        (phi > (3 / 2) * np.pi - chin_strap_angle + offset)
        & (phi < (3 / 2) * np.pi + offset)
    )

    helm_pts = pts[helm_indx, :][:, (0, 2, 1)]  # y and z axis are swapped on purpose

    if outer_r is None:
        return helm_pts * inner_r, helm_pts
    else:
        assert outer_r > inner_r
        return (
            np.vstack((helm_pts * inner_r, helm_pts * outer_r)),
            np.vstack((helm_pts, helm_pts)),
        )


def spherical(nsensors, array_radius, angle=4 * np.pi):
    """Make a spherical array"""
    Sc = spherepts_golden(nsensors, angle)
    Sn = Sc.copy()
    Sc *= array_radius
    return Sc, Sn


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
    return Sc, Sn


def double_barbute(
    nsensors_upper, nsensors_lower, inner_r, outer_r, height_lower, phispan_lower
):
    rhelm_in, nhelm_in = barbute(
        nsensors_upper, nsensors_lower, inner_r, height_lower, phispan_lower
    )
    rhelm_out, nhelm_out = barbute(
        nsensors_upper, nsensors_lower, outer_r, height_lower, phispan_lower
    )
    return np.vstack((rhelm_in, rhelm_out)), np.vstack((nhelm_in, nhelm_out))
