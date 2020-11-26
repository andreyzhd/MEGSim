#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate different sensor arrays.

author: Jussi (jnu@iki.fi)

"""
# %% init
import numpy as np
import matplotlib.pylab as plt
from mayavi import mlab
from IPython import get_ipython
from mne.preprocessing.maxwell import _sss_basis
from mne.transforms import _deg_ord_idx, _pol_to_cart, _cart_to_sph

from megsimutils.utils import (
    _normalized_basis,
    _idx_deg_ord,
    _deg_ord_idx,
    _normalized_basis,
    _vector_angles,
    spherepts_golden,
    _sssbasis_cond_pointlike,
    _mlab_points3d,
    _mlab_quiver3d,
    _random_unit,
    _sanity_check_array,
    _flip_normals,
)

from megsimutils.array_geometry import barbute


# set up IPython
ip = get_ipython()
# ip.magic("gui qt5")  # needed for mayavi plots
ip.magic("matplotlib qt")


# %% simple 1-layer barbute
NSENSORS_UPPER = 500
NSENSORS_LOWER = 500
ARRAY_RADIUS = 0.1
HEIGHT_LOWER = 0.12
PHISPAN_LOWER = 1.75 * np.pi
sss_params = {'origin': [0.0, 0.0, 0.0], 'int_order': 16, 'ext_order': 3}

Sc1, Sn1 = barbute(
    NSENSORS_UPPER, NSENSORS_LOWER, ARRAY_RADIUS, HEIGHT_LOWER, PHISPAN_LOWER
)
_sssbasis_cond_pointlike(Sc1, Sn1, sss_params)


# %% 2-layer version of simple barbute
Sc1, Sn1 = barbute(
    NSENSORS_UPPER, NSENSORS_LOWER, ARRAY_RADIUS, HEIGHT_LOWER, PHISPAN_LOWER
)
Sc2, Sn2 = barbute(
    NSENSORS_UPPER, NSENSORS_LOWER, ARRAY_RADIUS, HEIGHT_LOWER, PHISPAN_LOWER
)
Sc2 *= 1.1
Sc = np.row_stack((Sc1, Sc2))
Sn = np.row_stack((Sn1, Sn2))
_sssbasis_cond_pointlike(Sc, Sn, sss_params)


# %% 2-layer version of simple barbute
li = list()
for layer_scaling in np.arange(1.01, 5, 0.05):
    Sc1, Sn1 = barbute(
        NSENSORS_UPPER, NSENSORS_LOWER, ARRAY_RADIUS, HEIGHT_LOWER, PHISPAN_LOWER
    )
    Sc2, Sn2 = barbute(
        NSENSORS_UPPER, NSENSORS_LOWER, ARRAY_RADIUS, HEIGHT_LOWER, PHISPAN_LOWER
    )
    Sc2 *= layer_scaling
    Sc = np.row_stack((Sc1, Sc2))
    Sn = np.row_stack((Sn1, Sn2))
    li.append((layer_scaling, _sssbasis_cond_pointlike(Sc, Sn, sss_params)))
plt.plot(*np.array(li).T)


# %% try varying sensor orientations in 2nd layer
Sc1, Sn1 = barbute(
    NSENSORS_UPPER, NSENSORS_LOWER, ARRAY_RADIUS, HEIGHT_LOWER, PHISPAN_LOWER
)
Sc2, Sn2 = barbute(
    NSENSORS_UPPER, NSENSORS_LOWER, ARRAY_RADIUS, HEIGHT_LOWER, PHISPAN_LOWER
)
Sc2 *= 1.1

# randomly orient 2nd layer sensors
V = np.random.randn(*Sn2.shape)
Sn2 = (V.T / np.linalg.norm(V, axis=1)).T

Sc = np.row_stack((Sc1, Sc2))
Sn = np.row_stack((Sn1, Sn2))

_sssbasis_cond_pointlike(Sc, Sn, sss_params)


# %% plot some basis vecs
Lms = ((3, 1), (8, 1), (15, 1))
inds = [_deg_ord_idx(*Lm) for Lm in Lms]
plt.figure()
for ind in inds:
    plt.semilogy(Sin[:, ind])
legs = [f'{L=}, {m=}' for L, m in Lms]
plt.legend(legs)

