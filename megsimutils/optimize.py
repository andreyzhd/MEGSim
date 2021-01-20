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
    def get_bounds(self):
        """
        Return bounds on parameter vector for the optimization algorithm

        Returns
        -------
        N-by-2 vector of (min, max) values

        """

    
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
    
    @abstractmethod
    def plot(self, v, fig):
        """
        Plot array in 3d

        Parameters
        ----------
        v : 1-d parameter vector
        fig : Mayavi figure to use for plot

        Returns
        -------
        None.

        """
        pass


class FixedLocSpherArray(SensorArray):
    """
    Fixed locations, single-layer spherical array.
    """
    def _v2cosmags(self, v):
        """
        Convert vector of parameters (sensor orientattion angles in radians) to
        xyz coordinates

        Parameters
        ----------
        v : 1-d array of the length 2*m (m is the number of sensors).
            Contains phi and theta angles.

        Returns
        -------
        m-by-3 matrix of xyz coordinates

        """
        theta_cosmags = v[:self._n_coils]
        phi_cosmags = v[self._n_coils:]
        x_cosmags, y_cosmags, z_cosmags = pol2xyz(1, theta_cosmags, phi_cosmags)
        return np.stack((x_cosmags,y_cosmags,z_cosmags), axis=1)

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
    
    def get_bounds(self):
        return(np.repeat(np.array(((0, 3*np.pi),)), len(self._v0), axis=0))
    
    def comp_fitness(self, v):
        allcoils = (self._rmags, self._v2cosmags(v), self._bins, self._n_coils, self._mag_mask, self._slice_map)
        
        S = _sss_basis(self._exp, allcoils)
        S /= np.linalg.norm(S, axis=0)
        return np.linalg.cond(S)

    def plot(self, v, fig=None):
        from mayavi import mlab
        cosmags = self._v2cosmags(v)
        
        if fig is None:
            fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
        
        mlab.clf(fig)
        mlab.points3d(self._rmags[:,0], self._rmags[:,1], self._rmags[:,2], resolution=32, scale_factor=0.01, color=(0,0,1))
        mlab.quiver3d(self._rmags[:,0], self._rmags[:,1], self._rmags[:,2], cosmags[:,0], cosmags[:,1], cosmags[:,2])
        mlab.points3d(0, 0, 0, resolution=32, scale_factor=0.01, color=(0,1,0), mode='axes')


class ThickBarbuteArray(SensorArray):
    """
    Barbute helmet, flexible locations (incl depth) and orientations.
    """
    def _v2rmags(self, v):
        z = v[:self._n_coils]
        sweep = v[self._n_coils:2*self._n_coils]
        d = v[2*self._n_coils:]
        
        # opening linearly goes from 0 to 1 as we transition from spherical to cylindrical part
        opening = -z / (self._frac_trans * self._height_lower / self._R_inner)
        opening[opening<0] = 0
        opening[opening>1] = 1
        
        phi = ((2*np.pi) * (1-opening) + self._phispan_lower * opening) * sweep + \
              (((2*np.pi - self._phispan_lower) / 2) * opening)
        
        xy = np.ones(self._n_coils)
        indx_up = (z > 0)
        xy[indx_up] = np.sqrt(1 - z[indx_up]**2)
        
        z[indx_up] = z[indx_up] * d[indx_up]
        z[~indx_up] = z[~indx_up] * self._R_inner
        
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
        m-by-3 matrix of xyz coordinates

        """
        theta_cosmags = v[:self._n_coils]
        phi_cosmags = v[self._n_coils:]
        x_cosmags, y_cosmags, z_cosmags = pol2xyz(1, theta_cosmags, phi_cosmags)
        return np.stack((x_cosmags,y_cosmags,z_cosmags), axis=1)


    def __init__(self, nsens, l, 
                 R_inner=0.15, R_outer=0.25, height_lower=0.15, phispan_lower=1.5*np.pi, frac_trans=0.05):
        
        self._R_inner = R_inner
        self._R_outer = R_outer
        self._height_lower = height_lower
        self._n_coils = nsens
        self._phispan_lower = phispan_lower
        self._frac_trans = frac_trans
        
        # start with sensors at the top of the helmet
        z0 = np.linspace(0.5, 1, num=nsens, endpoint=False)
        sweep0 = np.modf(((3 - np.sqrt(5)) / 2) * np.arange(nsens))[0]
        d0 = np.mean((R_inner, R_outer)) * np.ones(nsens)
         
        rmags = self._v2rmags(np.concatenate((z0, sweep0, d0)))
        
        # start with radial sensor orientation
        theta0, phi0 = xyz2pol(rmags[:,0], rmags[:,1], rmags[:,2])[1:3]
        nmags = self._v2nmags(np.concatenate((theta0, phi0)))
        
        self._v0 = np.concatenate((z0, sweep0, d0, theta0, phi0)) # initial guess

        rmags, nmags, self._bins, self._n_coils, self._mag_mask, self._slice_map = _prep_mf_coils_pointlike(rmags, nmags)
        sss_origin = np.array([0.0, 0.0, 0.0])  # origin of device coords
        self._exp = {'origin': sss_origin, 'int_order': l, 'ext_order': 0}


    def get_init_vector(self):
        return self._v0
    

    def get_bounds(self):
        z_bounds = np.repeat(np.array(((-self._height_lower/self._R_inner, 1),)), self._n_coils, axis=0)
        sweep_bounds = np.repeat(np.array(((0, 1),)), self._n_coils, axis=0)
        d_bounds = np.repeat(np.array(((self._R_inner, self._R_outer),)), self._n_coils, axis=0)
        theta_phi_bounds = np.repeat(np.array(((0, 3*np.pi),)), self._n_coils*2, axis=0)
        
        return(np.vstack((z_bounds, sweep_bounds, d_bounds, theta_phi_bounds)))


    def comp_fitness(self, v):
        allcoils = (self._v2rmags(v[:3*self._n_coils]), self._v2nmags(v[3*self._n_coils:]), self._bins, self._n_coils, self._mag_mask, self._slice_map)
        
        S = _sss_basis(self._exp, allcoils)
        S /= np.linalg.norm(S, axis=0)
        return np.linalg.cond(S)


    def plot(self, v, fig=None):
        from mayavi import mlab
        rmags = self._v2rmags(v[:3*self._n_coils])
        nmags = self._v2nmags(v[3*self._n_coils:])
        
        if fig is None:
            fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
        
        mlab.clf(fig)
        mlab.points3d(rmags[:,0], rmags[:,1], rmags[:,2], resolution=32, scale_factor=0.01, color=(0,0,1))
        mlab.quiver3d(rmags[:,0], rmags[:,1], rmags[:,2], nmags[:,0], nmags[:,1], nmags[:,2])
        mlab.points3d(0, 0, 0, resolution=32, scale_factor=0.01, color=(0,1,0), mode='axes')
