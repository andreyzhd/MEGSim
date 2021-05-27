#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:45:49 2021

@author: andrey
"""
import numpy as np

from megsimutils.utils import xyz2pol
from megsimutils.optimize import BarbuteArray


class BarbuteArrayML(BarbuteArray):
    """
    Multilayer barbute helmet, flexible locations and orientations.
    """
    def _v2sens_geom(self, v):
        v_shells = np.array_split(v, np.cumsum(4*np.array(self._n_sens))[:-1])
        
        rmags_all = []
        nmags_all = []
        
        for nsens, R_inner, v_shell in zip(self._n_sens, self._Rs, v_shells):
            assert v_shell.shape == (nsens*4,)
            rmags = self._v2rmags_shell(v_shell[2*nsens:], R_inner, is_thick=False)        
            nmags = self._v2nmags(v_shell[:2*nsens])

            if self._is_opm:
                rmags_all.append(np.vstack((rmags, rmags)))
            else:
                rmags_all.append(rmags)
                
            nmags_all.append(nmags)
            
        return np.vstack(rmags_all), np.vstack(nmags_all)
    

    def __init__(self, n_sens, Rs, l_int, l_ext=0, **kwargs):
        super().__init__(l_int, l_ext, **kwargs)
        assert len(n_sens) == len(Rs)
        assert (l_int * (l_int+2)) + (l_ext * (l_ext+2)) <= np.sum(n_sens) * ((1,2)[self._is_opm])
        
        self._Rs = Rs
        # self._n_sens models the number of physical sensors, whereas
        # self.__n_coils (in the base class) - the number of field
        # measurements. For example, for OPM sensors, one sensor can make two
        # field measurements (in orthogonal directions), thus __n_coils will be
        # twice the _n_sens.
        self._n_sens = n_sens
        
        # Generate inti vector and bounds
        v_all = []
        bounds_all = []
        
        for nsens, R_inner in zip(n_sens, Rs):
            # start with evenly distributed sensors
            v_locs = self.uniform_locs(nsens, R_inner)
                  
            # start with radial sensor orientation
            rmags = self._v2rmags_shell(v_locs, R_inner, is_thick=False)
            rmags[rmags[:,2]<0, 2] = 0  
            theta, phi = xyz2pol(*rmags.T)[1:3]
            v_all.append(np.concatenate((theta, phi, v_locs)))
            
            # compute parameter bounds
            theta_phi_bounds = np.repeat(np.array(((0, 3*np.pi),)), nsens*2, axis=0)
            geodes, sweep = self._get_shell_params_bounds(R_inner)
            geodes_bounds = np.outer(np.ones(nsens), geodes)
            sweep_bounds = np.outer(np.ones(nsens), sweep)
            bounds_all.append(np.vstack((theta_phi_bounds, geodes_bounds, sweep_bounds)))
        
        self._v0 = np.concatenate(v_all) # initial guess
        self._bounds = np.vstack(bounds_all)
        
        




