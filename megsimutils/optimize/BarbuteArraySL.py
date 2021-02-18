#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:45:49 2021

@author: andrey
"""
import numpy as np

from megsimutils.utils import spherepts_golden, xyz2pol
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
    

    def __init__(self, nsens, l, origin=np.array([0,0,0]),
                 R_inner=0.15, R_outer=None, height_lower=0.15, phispan_lower=1.5*np.pi, frac_trans=0.05, opm=False):
        
        super().__init__(l, origin=origin, height_lower=height_lower, phispan_lower=phispan_lower, frac_trans=frac_trans)
        
        self._R_inner = R_inner
        self._R_outer = R_outer
        # self._n_sens models the number of physical sensors, whereas
        # self.__n_coils (in the base class) - the number of field
        # measurements. For example, for OPM sensors, one sensor can make two
        # field measurements (in orthogonal directions), thus __n_coils will be
        # twice the _n_sens.
        self._n_sens = nsens
        self._is_opm=opm
        
        # start with evenly distributed sensors
        v_locs = self.uniform_locs(nsens, R_inner)
        if R_outer is not None:
            v_locs = np.concatenate((v_locs, ((R_inner + R_outer) / 2) * np.ones(nsens)))

        # start with radial sensor orientation
        rmags = self._v2rmags_shell(v_locs, R_inner, R_outer is not None)
        rmags[rmags[:,2]<0, 2] = 0  
        theta0, phi0 = xyz2pol(*rmags.T)[1:3]
        
        self._v0 = np.concatenate((theta0, phi0, v_locs)) # initial guess
        
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


    def get_init_vector(self):
        return self._v0
    

    def get_bounds(self):
        return self._bounds


    def plot(self, v, fig=None, plot_bg=True, opacity=0.7):
        from mayavi import mlab
        v = self._validate_inp(v)
        rmags, nmags = self._v2sens_geom(v)

        if fig is None:
            fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
        
        mlab.clf(fig)
        mlab.points3d(rmags[:,0], rmags[:,1], rmags[:,2], resolution=32, scale_factor=0.005, color=(0,0,1))
        mlab.quiver3d(rmags[:,0], rmags[:,1], rmags[:,2], nmags[:,0], nmags[:,1], nmags[:,2], scale_factor=0.02)
        
        if plot_bg:
            inner_locs = spherepts_golden(1000, hcylind=self._height_lower/self._R_inner) * self._R_inner * 0.8
            pts = mlab.points3d(inner_locs[:,0], inner_locs[:,1], inner_locs[:,2], opacity=0, figure=fig)
            mesh = mlab.pipeline.delaunay3d(pts)
            mlab.pipeline.surface(mesh, figure=fig, color=(0.5, 0.5, 0.5), opacity=opacity)
        else:
            mlab.points3d(0, 0, 0, resolution=32, scale_factor=0.01, color=(0,1,0), mode='axes')
