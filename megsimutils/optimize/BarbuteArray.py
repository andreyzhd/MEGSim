#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:45:49 2021

@author: andrey
"""
import numpy as np

from megsimutils.utils import spherepts_golden, pol2xyz
from megsimutils.optimize import SensorArray


class BarbuteArray(SensorArray):
    """
    Base class for implementing various barbute arrays.
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
    
    
    def __on_barbute(self, rmags, phispan_lower):
        """
        Check which sensors are on the barbute. Return True for the sensors on
        the barbute and False for the sensors in the opening. Ignores z
        coordinate of the sensors on the cylindrical part (i.e. does not check
        the depth, only the phi angle.)
        Ignore the _fract_trans (that is assume the opening is rectangular)

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

    
    def _get_shell_params_bounds(self, R_inner):
        return np.array([-self._height_lower / R_inner, np.pi/2]), np.array([0, 1.0])
    
    
    def _v2rmags_shell(self, v, R_inner, is_thick=False):
        """Convert part of the parameter vector describing sensor locations
        for a single shell to xyz coordinates
        
        Parameters
        ----------
        v : 1-d vector of floats
            Part of the parameter vector describing sensor location for a
            single shell (z, sweep, and, optionally, d)
        R_inner : float
        is_thick : Boolean
            If True, the shell is "thick" (each sensor has it's own d). Else,
            all the ds are set to R_inner.

        Returns
        -------
        nsens-by-3 array of sensor coordinates

        """        
        if is_thick:
            assert len(v) % 3 == 0
            n_sens = len(v) // 3
            d = v[2*n_sens:]
        else:
            assert len(v) % 2 == 0
            n_sens = len(v) // 2
            d = R_inner * np.ones(n_sens)    

        geodes = v[:n_sens]
        z = geodes.copy()
        indx_up = (geodes > 0)
        z[indx_up] = np.sin(geodes[indx_up])
        sweep = v[n_sens:2*n_sens]

        # opening linearly goes from 0 to 1 as we transition from spherical to cylindrical part
        if self._height_lower > 0:
            opening = -z / (self._frac_trans * self._height_lower / R_inner)
            opening[opening<0] = 0
            opening[opening>1] = 1
        else:
            opening = np.zeros(n_sens)
        
        phi = ((2*np.pi) * (1-opening) + self._phispan_lower * opening) * sweep + \
              (((2*np.pi - self._phispan_lower) / 2) * opening)
        
        xy = np.ones(n_sens)
        xy[indx_up] = np.sqrt(1 - z[indx_up]**2)       
        z[indx_up] = z[indx_up] * d[indx_up]
        z[~indx_up] = z[~indx_up] * R_inner
        
        x = (xy * np.cos(phi)) * d
        y = (xy * np.sin(phi)) * d
        
        return np.stack((x,y,z), axis=1)
              
    
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
        assert len(v) % 2 == 0
        n_sens = len(v) // 2
        theta_cosmags = v[:n_sens]
        phi_cosmags = v[n_sens:]
        x_cosmags, y_cosmags, z_cosmags = pol2xyz(1, theta_cosmags, phi_cosmags)
        nmags = np.stack((x_cosmags,y_cosmags,z_cosmags), axis=1)
        
        if self._is_opm:
            return np.vstack(self.__comp_orth(nmags))
        else:
            return nmags


    def __init__(self, l, origin=np.array([0,0,0]), height_lower=0.15, phispan_lower=1.5*np.pi, frac_trans=0.05):
        super().__init__(l, origin=origin)

        self._height_lower = height_lower
        self._phispan_lower = phispan_lower
        self._frac_trans = frac_trans


    def get_init_vector(self):
        return self._v0
    

    def get_bounds(self):
        return self._bounds
    

    def uniform_locs(self, n_sens, R_inner):
        """Generate evenly spread sensors. """
        is_found = False
        
        for i in range(n_sens, 3*n_sens):
            rmags = spherepts_golden(i, hcylind=self._height_lower/R_inner)
            if np.count_nonzero(self.__on_barbute(rmags, self._phispan_lower)) == n_sens:
                is_found = True
                break

        assert is_found
        
        x, y, z = (rmags[self.__on_barbute(rmags, self._phispan_lower)]).T
        
        # Conver xyz to v
        phi = np.arctan2(y, x)
        phi[phi<0] += 2*np.pi
        
        # opening linearly goes from 0 to 1 as we transition from spherical to cylindrical part
        if self._height_lower > 0:
            opening = -z / (self._frac_trans * self._height_lower / R_inner)
            opening[opening<0] = 0
            opening[opening>1] = 1
        else:
            opening = np.zeros(n_sens)

        sweep = (phi - (np.pi - self._phispan_lower/2) * opening) / \
                ((2*np.pi) * (1-opening) + self._phispan_lower * opening)
        geodes = z.copy()
        geodes[z>0] = np.arcsin(z[z>0])
        
        return np.concatenate((geodes, sweep))
        
    
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
            R_inner = np.min(np.linalg.norm(rmags, axis=1))
            inner_locs = spherepts_golden(1000, hcylind=self._height_lower/R_inner) * R_inner * 0.8
            pts = mlab.points3d(inner_locs[:,0], inner_locs[:,1], inner_locs[:,2], opacity=0, figure=fig)
            mesh = mlab.pipeline.delaunay3d(pts)
            mlab.pipeline.surface(mesh, figure=fig, color=(0.5, 0.5, 0.5), opacity=opacity)
        else:
            mlab.points3d(0, 0, 0, resolution=32, scale_factor=0.01, color=(0,1,0), mode='axes')