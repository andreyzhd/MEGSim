#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 18:01:11 2021

@author: andrey
Compare MNE and bfieldtools implementation of VSH
"""

import numpy as np
import bfieldtools.sphtools as sh

from mne.preprocessing.maxwell import _sss_basis
from megsimutils.utils import _prep_mf_coils_pointlike, _idx_deg_ord, _deg_ord_idx

LIN = 16
LOUT = 16 # must be leq than LIN

NSENS = 1000

rmags = np.random.rand(NSENS,3)
nmags = (rmags.T / np.linalg.norm(rmags, axis=1)).T

# nmags = np.zeros((NSENS,3))
# nmags[:,2] = 1
      
bins, n_coils, mag_mask, slice_map = _prep_mf_coils_pointlike(rmags, nmags)[2:]
allcoils = (rmags, nmags, bins, n_coils, mag_mask, slice_map)
        
exp = {'origin': np.array([0, 0, 0]), 'int_order': LIN, 'ext_order': LOUT}
S = _sss_basis(exp, allcoils)


Ba, Bb = sh.basis_fields(rmags, LIN, normalization="default", R=1)
ncomp = Ba.shape[-1]

# Project VSH components on the sensor normals
proj = np.repeat(nmags[:,:,np.newaxis], ncomp, axis=2)
Sa = np.sum(Ba*proj, axis=1)
Sb = np.sum(Bb*proj, axis=1)

# Rearrange VSH components to meet the MNE ordering convention
idx = []
for i in range(ncomp):
    l, m = _idx_deg_ord(i)
    idx.append(_deg_ord_idx(l, -m))
    
Sa = Sa[:,idx]
Sb = Sb[:,idx]
    
Sp = np.hstack((Sa,Sb))[:,:S.shape[1]]

# Compare
rat = (S/Sp) * np.sign((S / Sp)[0,:])
print('maximum discrepancy is %e' % np.max(np.abs(rat-1)))