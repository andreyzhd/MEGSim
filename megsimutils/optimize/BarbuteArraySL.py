#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:45:49 2021

@author: andrey
"""
import numpy as np

from megsimutils.utils import xyz2pol
from megsimutils.optimize import BarbuteArray


class BarbuteArraySL(BarbuteArray):
    """
    Single layer barbute helmet, flexible locations (optionally incl depth) and orientations.
    """
    def _v2sens_geom(self, v):
        rmags = self._v2rmags_shell(v[2*self._n_sens:], self._R_inner, self._R_outer is not None)
        nmags = self._v2nmags(v[:2*self._n_sens])

        if self._is_opm:
            return np.vstack((rmags, rmags)), nmags
        else:
            return rmags, nmags
        
        
    def evenly_spaced_radial_v(self, truly_radial=False):
        """Generate sensor configuration that is evenly spaced with radial orientations"""
        v_locs = self.uniform_locs(self._n_sens, self._R_inner)
        if self._R_outer is not None:
            v_locs = np.concatenate((v_locs, self._R_inner * np.ones(self._n_sens)))

        # start with radial sensor orientation
        rmags = self._v2rmags_shell(v_locs, self._R_inner, self._R_outer is not None)
        if not truly_radial:
            rmags[rmags[:,2]<0, 2] = 0  
        theta0, phi0 = xyz2pol(*rmags.T)[1:3]
        
        return np.concatenate((theta0, phi0, v_locs))
    

    def __init__(self, nsens, l, l_ext=0, origin=np.array([0,0,0]),
                 R_inner=0.15, R_outer=None, height_lower=0.15, phispan_lower=1.5*np.pi, frac_trans=0.05, opm=False):
        
        super().__init__(l, l_ext, origin=origin, height_lower=height_lower, phispan_lower=phispan_lower, frac_trans=frac_trans)
        
        self._R_inner = R_inner
        self._R_outer = R_outer
        # self._n_sens models the number of physical sensors, whereas
        # self.__n_coils (in the base class) - the number of field
        # measurements. For example, for OPM sensors, one sensor can make two
        # field measurements (in orthogonal directions), thus __n_coils will be
        # twice the _n_sens.
        self._n_sens = nsens
        self._is_opm=opm
              
        self._v0 = self.evenly_spaced_radial_v() # initial guess
        
        # compute parameter bounds
        theta_phi_bounds = np.repeat(np.array(((0, 3*np.pi),)), nsens*2, axis=0)
        geodes, sweep = self._get_shell_params_bounds(R_inner)
        
        geodes_bounds = np.outer(np.ones(nsens), geodes)
        sweep_bounds = np.outer(np.ones(nsens), sweep)
                
        if R_outer is None:
            self._bounds = np.vstack((theta_phi_bounds, geodes_bounds, sweep_bounds))
        else:
            d_bounds = np.repeat(np.array(((R_inner, R_outer),)), nsens, axis=0)
            self._bounds = np.vstack((theta_phi_bounds, geodes_bounds, sweep_bounds, d_bounds))
