#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:45:49 2021

@author: andrey
"""
import numpy as np

from megsimutils.utils import spherepts_golden, pol2xyz, xyz2pol
from megsimutils.optimize import SensorArray


class BarbuteArray(SensorArray):
    """
    Barbute helmet, flexible locations (optionally incl depth) and orientations.
    """
    def __comp_orth(self, v):
        """Compute two sets of vectors orthogonal to v"""
        def __noncolin(vec):
            """Return a vector guaranteed to be non-collinear to a 3-d vector vec (assuming vec != (0, 0, 0))"""
            res = np.zeros(3)
            res[np.argmin(vec)] = 1
            return res
        
        v_nc = np.stack(list(__noncolin(vec) for vec in v), axis=0)
                    
        orth1 = np.cross(v, v_nc)
        orth1 /= np.linalg.norm(orth1, axis=1)[:,None]
        orth2 = np.cross(v, orth1)
        orth2 /= np.linalg.norm(orth2, axis=1)[:,None]
        
        return orth1, orth2
    
    
    def _v2rmags(self, v):
        z = v[:self._n_sens]
        sweep = v[self._n_sens:2*self._n_sens]
        if self._R_outer is None:
            d = self._R_inner * np.ones(self._n_sens)
        else:
            d = v[2*self._n_sens:]
        
        # opening linearly goes from 0 to 1 as we transition from spherical to cylindrical part
        if self._height_lower > 0:
            opening = -z / (self._frac_trans * self._height_lower / self._R_inner)
            opening[opening<0] = 0
            opening[opening>1] = 1
        else:
            opening = np.zeros(self._n_sens)
        
        phi = ((2*np.pi) * (1-opening) + self._phispan_lower * opening) * sweep + \
              (((2*np.pi - self._phispan_lower) / 2) * opening)
        
        xy = np.ones(self._n_sens)
        indx_up = (z > 0)
        xy[indx_up] = np.sqrt(1 - z[indx_up]**2)
        
        zc = z.copy()
        zc[indx_up] = z[indx_up] * d[indx_up]
        zc[~indx_up] = z[~indx_up] * self._R_inner
        
        x = (xy * np.cos(phi)) * d
        y = (xy * np.sin(phi)) * d
        
        rmags = np.stack((x,y,zc), axis=1)
        if self._is_opm:
            return np.vstack((rmags, rmags))
        else:
            return rmags
        
        
    def _v2nmags(self, v):
        """
        Convert sensor orientattion angles in radians (part of the parameter
        vector) to xyz coordinates

        Parameters
        ----------
        v : 1-d array of the length 2*m (m is the number of sensors).
            Contains phi and theta angles.

        Returns
        -------
        m-by-3 matrix of xyz coordinates (or 2m-by-3 bor OPMs)

        """
        theta_cosmags = v[:self._n_sens]
        phi_cosmags = v[self._n_sens:]
        x_cosmags, y_cosmags, z_cosmags = pol2xyz(1, theta_cosmags, phi_cosmags)
        nmags = np.stack((x_cosmags,y_cosmags,z_cosmags), axis=1)
        
        if self._is_opm:
            return np.vstack(self.__comp_orth(nmags))
        else:
            return nmags


    def _v2sens_geom(self, v):
        rmags = self._v2rmags(v[2*self._n_sens:])
        nmags = self._v2nmags(v[:2*self._n_sens])

        return rmags, nmags
    

    def __init__(self, nsens, l, 
                 R_inner=0.15, R_outer=None, height_lower=0.15, phispan_lower=1.5*np.pi, frac_trans=0.05, opm=False):
        
        super().__init__(l)
        
        self._R_inner = R_inner
        self._R_outer = R_outer
        self._height_lower = height_lower
        # self._n_sens models the number of physical sensors, whereas
        # self.__n_coils (in the base class) - the number of field
        # measurements. For example, for OPM sensors, one sensor can make two
        # field measurements (in orthogonal directions), thus __n_coils will be
        # twice the _n_sens.
        self._n_sens = nsens
        self._phispan_lower = phispan_lower
        self._frac_trans = frac_trans
        self._is_opm=opm
        self._call_cnt = 0
        
        # start with sensors at the top of the helmet
        z0 = np.linspace(0.5, 1, num=nsens, endpoint=False)
        sweep0 = np.modf(((3 - np.sqrt(5)) / 2) * np.arange(nsens))[0]
        if R_outer is None:
            d0 = np.array(())
        else:
            d0 = np.mean((R_inner, R_outer)) * np.ones(nsens)
         
        rmags = self._v2rmags(np.concatenate((z0, sweep0, d0)))
        
        # start with radial sensor orientation
        theta0, phi0 = xyz2pol(*rmags[:nsens,:].T)[1:3]
        
        self._v0 = np.concatenate((theta0, phi0, z0, sweep0, d0)) # initial guess
        
        # compute parameter bounds
        theta_phi_bounds = np.repeat(np.array(((0, 3*np.pi),)), nsens*2, axis=0)
        z_bounds = np.repeat(np.array(((-height_lower/R_inner, 1),)), nsens, axis=0)
        sweep_bounds = np.repeat(np.array(((0, 1),)), nsens, axis=0)
        
        if R_outer is None:
            self._bounds = np.vstack((theta_phi_bounds, z_bounds, sweep_bounds))
        else:
            d_bounds = np.repeat(np.array(((R_inner, R_outer),)), nsens, axis=0)
            self._bounds = np.vstack((theta_phi_bounds, z_bounds, sweep_bounds, d_bounds))


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
