#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:40:00 2020

@author: andrey
Investigate how much rotation of the coordinate system can affect the 
condition number
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import mne
from mne.preprocessing.maxwell import _sss_basis, _prep_mf_coils

L_MAX = 17

data_path = Path(mne.datasets.sample.data_path())
raw_file = data_path / 'MEG/sample/sample_audvis_raw.fif'
info = mne.io.read_info(raw_file)

rot_x = np.array([[1,0,0],[0,0,1],[0,-1,0]])
rot_z = np.array([[0,1,0],[-1,0,0],[0,0,1]])

rmags, cosmags, bins, n_coils, mag_mask, slice_map = _prep_mf_coils(info)
allcoils = (rmags, cosmags, bins, n_coils, mag_mask, slice_map)
allcoils_rot_x = (np.matmul(rmags, rot_x), np.matmul(cosmags, rot_x), bins, n_coils, mag_mask, slice_map)
allcoils_rot_z = (np.matmul(rmags, rot_z), np.matmul(cosmags, rot_z), bins, n_coils, mag_mask, slice_map)
sss_origin = np.array([0.0, 0.0, 0.0])  # origin of device coords

conds = np.zeros(L_MAX)
conds_rot_x = np.zeros(L_MAX)
conds_rot_z = np.zeros(L_MAX)

for l in range(1, L_MAX+1):
    exp = {'origin': sss_origin, 'int_order': l, 'ext_order': 0}
    S = _sss_basis(exp, allcoils)    
    S /= np.linalg.norm(S, axis=0)  # normalize basis
    conds[l-1] = np.linalg.cond(S)
    
    S_rot_x = _sss_basis(exp, allcoils_rot_x)    
    S_rot_x /= np.linalg.norm(S_rot_x, axis=0)  # normalize basis
    conds_rot_x[l-1] = np.linalg.cond(S_rot_x)
    
    S_rot_z = _sss_basis(exp, allcoils_rot_z)    
    S_rot_z /= np.linalg.norm(S_rot_z, axis=0)  # normalize basis
    conds_rot_z[l-1] = np.linalg.cond(S_rot_z)
    
print('Rotating around x causes the condition number to change by up to %0.5f %%' % (np.max(np.abs(conds_rot_x-conds) / conds) * 100))
print('Rotating around z causes the condition number to change by up to %0.5f %%' % (np.max(np.abs(conds_rot_z-conds) / conds) * 100))


# %% Plot the results

width = 0.35
x_loc = np.arange(1, L_MAX+1)
plt.bar(x_loc - width/2, np.log10(conds), width, label='Original')
plt.bar(x_loc + width/2, np.log10(conds_rot_x), width, label='Rotated around x')
plt.legend()
plt.xlabel('l')
plt.ylabel(r'$\log_{10}(R_{cond})$')