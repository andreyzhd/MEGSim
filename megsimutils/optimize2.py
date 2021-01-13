#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:45:49 2021

@author: andrey
"""

from abc import ABC, abstractmethod
import numpy as np

from mne.preprocessing.maxwell import _sss_basis
from megsimutils.utils import spherepts_golden, pol2xyz, xyz2pol
from megsimutils.utils import _prep_mf_coils_pointlike
  
class SensorArray(ABC):
    """
    Abstract class describing MEG sensor array from the point of view of an
    optimization algorithm.
    """
    @abstractmethod
    def get_init_vector(self):
        """
        Return a valid initial parameter vector

        Returns
        -------
        A 1-d vector.

        """
        pass
    
    @abstractmethod
    def comp_fitness(self, v):
        """
        Compute fitness (value of optimization criterion) for a given
        parameter vector

        Parameters
        ----------
        v : 1-d vector
            Contains sensor array parameters being optimized.

        Returns
        -------
        Float, describing the 'quality' of the vector -- lower values
        correspond to better arrays.

        """
        pass


class FixedLocSpherArray(SensorArray):
    """
    Fixed locations, single-layer spherical array.
    """
    def __init__(self, nsens, angle, l, R=0.15):
        
        rmags = spherepts_golden(nsens, angle=angle) * R
        nmags = spherepts_golden(nsens, angle=angle)
        
        self._rmags, cosmags0, self._bins, self._n_coils, self._mag_mask, self._slice_map = _prep_mf_coils_pointlike(rmags, nmags)
              
        # start with radial sensor orientation
        x_cosmags0, y_cosmags0, z_cosmags0 = cosmags0[:,0], cosmags0[:,1], cosmags0[:,2]
        theta0, phi0 = xyz2pol(x_cosmags0, y_cosmags0, z_cosmags0)[1:3]
        self._v0 = np.concatenate((theta0, phi0)) # initial guess
        
        sss_origin = np.array([0.0, 0.0, 0.0])  # origin of device coords
        self._exp = {'origin': sss_origin, 'int_order': l, 'ext_order': 0}
    
    def get_init_vector(self):
        return self._v0
    
    def comp_fitness(self, v):
        theta_cosmags = v[:self._n_coils]
        phi_cosmags = v[self._n_coils:]
        x_cosmags, y_cosmags, z_cosmags = pol2xyz(1, theta_cosmags, phi_cosmags)
        allcoils = (self._rmags, np.stack((x_cosmags,y_cosmags,z_cosmags), axis=1), 
                    self._bins, self._n_coils, self._mag_mask, self._slice_map)
        
        S = _sss_basis(self._exp, allcoils)
        S /= np.linalg.norm(S, axis=0)
        return np.linalg.cond(S)
