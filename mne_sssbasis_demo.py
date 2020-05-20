#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Demonstrates how to use the fast _sss_basis implementation from mne-python
on both existing arrays (read info structure from raw data) and newly generated arrays.

@author: Jussi (jnu@iki.fi)
"""

# %% init
import numpy as np
from scipy.linalg import subspace_angles
from pathlib import Path

import mne
from mne.preprocessing.maxwell import _sss_basis, _sss_basis_basic, _prep_mf_coils


def _sphere_points(n_pts):
    """Cartesian coords for random points on a unit sphere"""
    phi = np.random.randn(n_pts) * np.pi  # polar angle
    theta = np.random.randn(n_pts) * 2 * np.pi  # azimuth
    X = np.sin(phi) * np.cos(theta)
    Y = np.sin(phi) * np.sin(theta)
    Z = np.cos(phi)
    return np.column_stack((X, Y, Z))


# %% demonstration on Vectorview array (reads array geometry from a file)
#
data_path = Path(mne.datasets.sample.data_path())
raw_file = data_path / 'MEG/sample/sample_audvis_raw.fif'
info = mne.io.read_info(raw_file)
#
# _prep_mf_coils converts coil data into following format, used by the
# the optimized _sss_basis() function:
#
# rmags : integration point locations (Np x 3) (m)
# cosmags : coil normal vectors at integration points,
#           weighted by integration weights (Np x 3)
# bins : sensor index for each integration point (Np x 3)
# n_coils : total number of coils
# mag_mask : (n_coils x 1) binary ndarray, where values are True for magnetometers
#            and False for gradiometers
# slice_map : dict where keys are sensor indices, and values are slices
#             into rmags / cosmag matrices (e.g. entry 0: slice(0, 8, None)
#             would mean that sensor 0 comprises of indices 0..7 in the integration
#             point matrix)
#
rmags, cosmags, bins, n_coils, mag_mask, slice_map = _prep_mf_coils(info)
allcoils = (rmags, cosmags, bins, n_coils, mag_mask, slice_map)
sss_origin = np.array([0.0, 0.0, 0.0])  # origin of device coords
LIN, LOUT = 8, 3
nvecs_in = LIN ** 2 + 2 * LIN
nvecs_out = LOUT ** 2 + 2 * LOUT
exp = {'origin': sss_origin, 'int_order': LIN, 'ext_order': LOUT}
S = _sss_basis(exp, allcoils)
S[2::3, :] *= 100  # scale magnetometers by 100 (slightly affects cond etc.)
S /= np.linalg.norm(S, axis=0)  # normalize basis
# cond is about 190
print(np.linalg.cond(S))
# angle spectrumn is about 4.5 - 23 deg (matches result in Samu's IEEE paper)
print(subspace_angles(S[:, :nvecs_in], S[:, nvecs_in:]) / np.pi * 180)


# %% let's make a pointlike spherical array with radial normals
# this results in a singular SSS basis
r1 = 0.1
n_coils = 400

rmags_one = _sphere_points(n_coils)
cosmags = rmags_one
rmags = r1 * rmags_one
mag_mask = np.ones(n_coils).astype(bool)
slice_map = {k: slice(k, k + 1, None) for k in range(n_coils)}
bins = np.arange(n_coils)
allcoils = (rmags, cosmags, bins, n_coils, mag_mask, slice_map)

sss_origin = np.array([0.0, 0.0, 0.0])  # origin of device coords
LIN, LOUT = 6, 6
nvecs_in = LIN ** 2 + 2 * LIN
nvecs_in = LOUT ** 2 + 2 * LOUT
exp = {'origin': sss_origin, 'int_order': LIN, 'ext_order': LOUT}
S = _sss_basis(exp, allcoils)
S /= np.linalg.norm(S, axis=0)  # normalize basis
# cond is huge, since the basis is singular
print(np.linalg.cond(S))
# angle spectrum is basically all zeros - no difference between bases
print(subspace_angles(S[:, :nvecs_in], S[:, nvecs_in:]) / np.pi * 180)


# %% let's make a pointlike spherical array with two layers
# see Appendix A of Samu's JAP paper
r1 = 0.1
r2 = 0.15
n_coils = 400
n_half = int(n_coils / 2)

rmags_one = _sphere_points(n_coils)
cosmags = rmags_one
rmags = rmags_one.copy()
rmags[:n_half, :] *= r1
rmags[n_half:, :] *= r2
mag_mask = np.ones(n_coils).astype(bool)
slice_map = {k: slice(k, k + 1, None) for k in range(n_coils)}
bins = np.arange(n_coils)
allcoils = (rmags, cosmags, bins, n_coils, mag_mask, slice_map)

sss_origin = np.array([0.0, 0.0, 0.0])  # origin of device coords
LIN, LOUT = 6, 6
exp = {'origin': sss_origin, 'int_order': LIN, 'ext_order': LOUT}
S = _sss_basis(exp, allcoils)
S /= np.linalg.norm(S, axis=0)
# cond is nice now (~10), matching the result in Samu's JAP paper
print(np.linalg.cond(S))
# the largest principal angle goes up to >80 deg (ditto)
print(subspace_angles(S[:, :nvecs_in], S[:, nvecs_in:]) / np.pi * 180)

