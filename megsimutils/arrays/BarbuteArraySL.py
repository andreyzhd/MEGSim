#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:45:49 2021

@author: andrey
"""
import numpy as np

from megsimutils.utils import xyz2pol
from megsimutils.arrays import BarbuteArray

from mne.preprocessing.maxwell import _get_n_moments

class BarbuteArraySL(BarbuteArray):
    """
    Single layer barbute helmet, flexible locations (optionally incl depth) and orientations.
    """

    def __init__(self, n_sens, l_int, l_ext, R_inner=0.15, R_outer=None, n_samp_layers=1, n_samp_per_layer=100, **kwargs):
        super().__init__(l_int, l_ext, **kwargs)
        assert _get_n_moments(l_int) + _get_n_moments(l_ext) <= np.sum(n_sens) * ((1, 2)[self._is_opm])
        assert (R_outer is not None) or (n_samp_layers == 1)
        assert (R_outer is None) or (R_outer > R_inner)

        self._R_inner = R_inner
        self._R_outer = R_outer
        # self._n_sens models the number of physical sensors, whereas
        # self.__n_coils (in the base class) - the number of field
        # measurements. For example, for OPM sensors, one sensor can make two
        # field measurements (in orthogonal directions), thus __n_coils will be
        # twice the _n_sens.
        self._n_sens = n_sens
        self._rng = np.random.default_rng()  # Init random number generator

        # Compute the sampling locations
        self._sampling_locs_rmags, self._sampling_locs_nmags = self._create_sampling_locs(R_inner, (R_inner, R_outer)[
            R_outer is not None], n_samp_layers, n_samp_per_layer)

        # Uniformly spaced sensor locations are heavy to compute, so cache them (only the geodes / sweep part)
        self.__uv_geodes_sweep = self.uniform_locs(n_sens, R_inner)
#        if R_outer is not None:
#            self.__uv_geodes_sweep = np.concatenate((self.__uv_geodes_sweep, R_outer * np.ones(n_sens)))

#        self.__v0 = self.evenly_spaced_radial_v()  # initial guess

        # compute parameter bounds
        theta_phi_bounds = np.repeat(np.array(((0, 3 * np.pi),)), n_sens * 2, axis=0)
        geodes, sweep = self._get_shell_params_bounds(R_inner)

        geodes_bounds = np.outer(np.ones(n_sens), geodes)
        sweep_bounds = np.outer(np.ones(n_sens), sweep)

        if R_outer is None:
            self._bounds = np.vstack((theta_phi_bounds, geodes_bounds, sweep_bounds))
        else:
            d_bounds = np.repeat(np.array(((R_inner, R_outer),)), n_sens, axis=0)
            self._bounds = np.vstack((theta_phi_bounds, geodes_bounds, sweep_bounds, d_bounds))


    def _v2sens_geom(self, v):
        rmags = self._v2rmags_shell(v[2*self._n_sens:], self._R_inner, self._R_outer is not None)
        nmags = self._v2nmags(v[:2*self._n_sens])

        if self._is_opm:
            return np.vstack((rmags, rmags)), nmags
        else:
            return rmags, nmags
        
        
    def _get_sampling_locs(self):
        return self._sampling_locs_rmags, self._sampling_locs_nmags
        
        
    def evenly_spaced_radial_v(self, R=None):
        """Generate sensor configuration that is evenly spaced with radial orientations"""
        R_locs = self._R_inner if R is None else R
        uv_locs = self.__uv_geodes_sweep if self._R_outer is None else np.concatenate((self.__uv_geodes_sweep, R_locs * np.ones(self._n_sens)))
        rmags = self._v2rmags_shell(uv_locs, self._R_inner, self._R_outer is not None)
        rmags[rmags[:,2]<0, 2] = 0
        theta0, phi0 = xyz2pol(*rmags.T)[1:3]
        
        return np.concatenate((theta0, phi0, uv_locs))
    
    
    def evenly_spaced_rand_v(self, R=None):
        """Generate sensor configuration that is evenly spaced with random orientations"""
        R = self._R_inner if R is None else R
        uv_locs = self.__uv_geodes_sweep if self._R_outer is None else np.concatenate((self.__uv_geodes_sweep, R * np.ones(self._n_sens)))
        rvecs = self._rng.standard_normal((self._n_sens, 3))
        theta0, phi0 = xyz2pol(*rvecs.T)[1:3]
        
        return np.concatenate((theta0, phi0, uv_locs))


    def get_init_vector(self, depth=0.0):
        """
        Return an initialization vector, sensor locations evenly spread over a surface
        located somewhere between the inner and outer surface. Orientations
        are normal to the surface.

        :param depth: parameter between 0 an 1 specifying where the surface is located.
        depth=0 means sensors are locate on the barbute's outer surface, depth=1 -- on the
        inner surface, everything else -- in between.
        :return: vector representation of the sensor array that can be fed to a
        gradient descent algorithm.
        """
        assert (depth <= 1) and (depth >= 0)

        if self._R_outer is None:
            # 2D barbute
            return self.evenly_spaced_radial_v()
        else:
            # 3D barbute
            return self.evenly_spaced_radial_v((self._R_inner * depth) + (self._R_outer * (1 - depth)))

    def get_bounds(self):
        return self._bounds


    def plot(self, v, fig=None, opacity_inner=0.7, opacity_outer=0.1, R_enclosure=None):
        super().plot(v, fig=fig, opacity_inner=opacity_inner, opacity_outer=opacity_outer, R_inner=self._R_inner, R_outer=self._R_outer, R_enclosure=R_enclosure)
        