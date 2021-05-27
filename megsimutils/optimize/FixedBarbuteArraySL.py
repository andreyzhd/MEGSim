#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 18:53:04 2021

@author: andrey
"""
import numpy as np

from megsimutils.utils import xyz2pol
from megsimutils.optimize import BarbuteArraySL

class FixedBarbuteArraySL(BarbuteArraySL):
    """Barbute array with fixed locations"""
    def __init__(self, n_sens, l_int, l_ext=0, R_inner=0.15, **kwargs):
        super().__init__(n_sens, l_int, l_ext, R_inner=R_inner, R_outer=None, **kwargs)
        
        # Generate evenly spread sensors
        v_locs = self.uniform_locs(n_sens, R_inner)
        self._rmags = self._v2rmags_shell(v_locs, R_inner, is_thick=False)
        
        self._bounds = np.repeat(np.array(((0, 3*np.pi),)), n_sens*2, axis=0)
        theta0, phi0 = xyz2pol(*(self._rmags).T)[1:3]
        self._v0 = np.concatenate((theta0, phi0))
        
    def _v2sens_geom(self, v):
        nmags = self._v2nmags(v)
        
        if self._is_opm:
            return np.vstack((self._rmags, self._rmags)), nmags
        else:
            return self._rmags, nmags
        
        
        
        
    def evenly_spaced_radial_v(self, truly_radial=False):
        """Generate sensor configuration that is evenly spaced with radial orientations"""
        return super().evenly_spaced_radial_v(truly_radial)[:self._n_sens*2]
  
    
    def evenly_spaced_rand_v(self):
        """Generate sensor configuration that is evenly spaced with random orientations"""
        return super().evenly_spaced_rand_v()[:self._n_sens*2]
    