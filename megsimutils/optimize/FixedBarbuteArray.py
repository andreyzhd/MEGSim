#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 18:53:04 2021

@author: andrey
"""
import numpy as np

from megsimutils.utils import spherepts_golden, xyz2pol
from megsimutils.optimize import BarbuteArray

class FixedBarbuteArray(BarbuteArray):
    """Barbute array with fixed locations"""
    def __on_barbute(self, rmags, phispan_lower):
        """
        Check which sensors are on the barbute. Return True for the sensors on
        the barbute and False fro the sensors in the opening. Ignores z
        coordinate of the sensors on the cylindrical part (i.e. does not check
        the depth, only the phi angle.)

        Parameters
        ----------
        rmags : n_sens-by3 matrix
            Sensor coordinates
        phispan_lower : float between 0 and 2*np.pi

        Returns
        -------
        1-d boolean array of the length n_sens
        """
        indx_top = (rmags[:,2] >= 0)
        indx_side = (np.abs(np.arctan2(rmags[:,1], rmags[:,0])) >= (np.pi - (phispan_lower/2)))
        
        return indx_top | indx_side
    
    
    def __init__(self, nsens, l, R_inner=0.15, height_lower=0.15, phispan_lower=1.5*np.pi, opm=False):
        
        super().__init__(nsens, l, R_inner=R_inner, R_outer=None, height_lower=height_lower, phispan_lower=phispan_lower, opm=opm)
        
        # Generate evenly spread sensors
        is_found = False
        for i in range(nsens, 3*nsens):
            rmags = spherepts_golden(i, hcylind=height_lower/R_inner)
            if np.count_nonzero(self.__on_barbute(rmags, phispan_lower)) == nsens:
                is_found = True
                break
            
        assert is_found
        
        self._rmags = rmags[self.__on_barbute(rmags, phispan_lower)] * R_inner
        self._bounds = np.repeat(np.array(((0, 3*np.pi),)), nsens*2, axis=0)
        theta0, phi0 = xyz2pol(*(self._rmags).T)[1:3]
        self._v0 = np.concatenate((theta0, phi0))
        
        
    def _v2sens_geom(self, v):
        nmags = self._v2nmags(v)
        
        if self._is_opm:
            return np.vstack((self._rmags, self._rmags)), nmags
        else:
            return self._rmags, nmags
        
        
        
        
    