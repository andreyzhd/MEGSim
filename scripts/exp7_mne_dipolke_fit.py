#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 12:45:25 2020

@author: andrey

Try to fit a dipole with arbitrary sensor array using MNE-Python dipole fitting
"""

import numpy as np
import mne
from mne.io.constants import FIFF
from megsimutils.utils import spherepts_golden, sensordata_to_ch_dicts
from megsimutils import dipfld_sph

N_COILS = 200
R = 0.15
ANGLE = 4*np.pi/(1.5)

Q = np.array([-100, 50, 50]) * 1e-9    # Dipole moment
RQ = np.array([0.07, 0.07, 0.07])         # Dipole location

def _create_sim_info(rmags, cosmags):
        assert len(rmags) == len(cosmags)
    
        n_coils = len(rmags)
        ch_names = ['MEG%03i' % i for i in range(n_coils)]
        info = mne.create_info(ch_names, 1, ch_types='mag', verbose=None)

        coil_type = FIFF.FIFFV_COIL_POINT_MAGNETOMETER
        coil_types = n_coils * [coil_type]
    
        # apply no rotations to integration points
        Iprot = np.zeros(n_coils)
        sensors_ = list(sensordata_to_ch_dicts(rmags, cosmags, Iprot, coil_types))
        info['chs'] = sensors_
        info['nchan'] = len(sensors_)
        info['ch_names'] = [ch['ch_name'] for ch in info['chs']]

        return info

class SimEvoked(mne.Evoked):
    """A variation of Evoked class for simulated data and arbitrary sensor geometry."""
    
    def __init__(self, rmags, cosmags, data):
        """
        Parameters
        ----------
        rmags : N-by-3 matrix of sensor locations
        cosmags : N-by-3 matrix of sensor orientations
        data : N-length vector of sensor readings

        Returns
        -------
        None.

        """
        assert len(rmags) == len(data)
    
        self.info = _create_sim_info(rmags, cosmags)
        self.nave = 1   # number of averages
        #self._aspect_kind = 100 # WTF is this?
        self.comment = 'Simulated evoked file'
        self.times = np.zeros(1)
        self.data = data[:, np.newaxis]
        self._update_first_last()
        self.verbose = None
        self.preload = True
        
                
# Create sensor array
rmags = spherepts_golden(N_COILS, angle=ANGLE) * R
cosmags = spherepts_golden(N_COILS, angle=ANGLE) # radial sensor orientation

# Create a dipole
field = dipfld_sph(Q, RQ, rmags, np.zeros(3))
data = (field*cosmags).sum(axis=1)

sim_evoked = SimEvoked(rmags, cosmags, data)
cov = mne.make_ad_hoc_cov(_create_sim_info(rmags, cosmags), std=1)

dip, res = mne.fit_dipole(sim_evoked, cov, mne.make_sphere_model())