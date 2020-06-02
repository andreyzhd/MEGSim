#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:33:35 2020

@author: andrey

Compute condition numbers using mne and my own implementation of VSH and
compare the results.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing.maxwell import _sss_basis, _prep_mf_coils

from megsimutils import sph_v
from megsimutils.utils import xyz2pol

L_MAX = 30

data_path = Path(mne.datasets.sample.data_path())
raw_file = data_path / 'MEG/sample/sample_audvis_raw.fif'
info = mne.io.read_info(raw_file)
rmags, cosmags, bins, n_coils, mag_mask, slice_map = _prep_mf_coils(info)

allcoils = (rmags, cosmags, bins, n_coils, mag_mask, slice_map)
sss_origin = np.array([0.0, 0.0, 0.0])  # origin of device coords

##-----------------------------------------------------------------------------
# Compute the basis with mne
#
cond_mne = np.zeros(L_MAX)
for l in range(1, L_MAX+1):
    print('Computing the MNE basis for l=%i' % l)
    exp = {'origin': sss_origin, 'int_order': l, 'ext_order': 0}
    S = _sss_basis(exp, allcoils)
    S /= np.linalg.norm(S, axis=0)  # normalize basis
    cond_mne[l-1] = np.linalg.cond(S)

##-----------------------------------------------------------------------------
# Compute the basis with sph_v
#
r, theta, phi = xyz2pol(rmags[:,0], rmags[:,1], rmags[:,2])
n_points = len(r)

# Compute the harmonics at all integration points
v = np.zeros((L_MAX+1, 2*(L_MAX+1), n_points, 3), dtype=np.complex128)
for l in range(1, L_MAX+1):
    for m in range(-l, l+1):        
        print('Computing v for l=%i, m=%i onto sensor array' % (l, m))
        for i in range(n_points):
            v[l, m, i,:] = sph_v(l, m, theta[i], phi[i]) / (r[i]**(l+2))
            
# Compute the projections
projs = []

for l in range(1, L_MAX+1):
    for m in range(-l, l+1):
        print('Computing the projections of v for l=%i, m=%i onto sensor array' % (l, m))
        proj = np.zeros(n_coils, dtype=np.complex128)
        for i in range(n_coils):
            proj[i] = (cosmags[slice_map[i]] * v[l,m,slice_map[i],:]).sum()
        projs.append(proj)

# Compute the condition numbers
cond_sph = np.zeros(L_MAX)
for l in range(1, L_MAX+1):
    S_sph = np.stack(projs[:l**2+2*l], axis=1)       
    S_sph /= np.linalg.norm(S_sph, axis=0)  # normalize basis
    cond_sph[l-1] = np.linalg.cond(S_sph)

##-----------------------------------------------------------------------------
# Plot the results
#
width = 0.35
x_loc = np.arange(1, L_MAX+1)
plt.bar(x_loc - width/2, np.log10(cond_mne), width, label='MNE')
plt.bar(x_loc + width/2, np.log10(cond_sph), width, label='SPH')
plt.legend()
plt.xlabel('l')
plt.ylabel(r'$\log_{10}(R_{cond})$')