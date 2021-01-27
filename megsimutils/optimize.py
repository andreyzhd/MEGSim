#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:45:49 2021

@author: andrey
"""
import time
from abc import ABC, abstractmethod
import numpy as np

from mne.preprocessing.maxwell import _sss_basis
from megsimutils.utils import spherepts_golden, pol2xyz, xyz2pol
from megsimutils.utils import _prep_mf_coils_pointlike

class ConstraintPenalty():
    def __init__(self, bounds, frac_margin=0.05, penalty=1e15):
        self._bounds = bounds
        self._penalty = penalty
        self._margin = np.diff(bounds, axis=1)[:,0] * frac_margin
        
    def compute(self, v):
        cost_below = (((self._bounds[:,0] - v) / self._margin) + 1) * np.sqrt(self._penalty)
        cost_below[cost_below<0] = 0
    
        cost_above = (((v - self._bounds[:,1]) / self._margin) + 1) * np.sqrt(self._penalty)
        cost_above[cost_above<0] = 0
    
        return np.sum((cost_below + cost_above) ** 2)
    
    
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



class BarbuteArray(SensorArray):
    """
    Barbute helmet, flexible locations (optionally incl depth) and orientations.
    """
    def _validate_inp(self, v):
        """ Check that the input is within bounds and correct if neccessary
        """
        v = v.copy()
        
        indx_below = (v < self._bounds[:,0])
        indx_above = (v > self._bounds[:,1])
        
        deviat_above = 0
        deviat_below = 0
        
        if indx_above.any():
            deviat_above = np.max((v[indx_above] - self._bounds[indx_above,1]) / np.diff(self._bounds[indx_above,:], axis=1))
            v[indx_above] = self._bounds[indx_above,1]
                        
        if indx_below.any():
            deviat_below = np.max((self._bounds[indx_below,0] - v[indx_below]) / np.diff(self._bounds[indx_below,:], axis=1))
            v[indx_below] = self._bounds[indx_below,0]
            
        max_dev_prc = 100 * max(deviat_above, deviat_below)

        if max_dev_prc >= 10:
            print('\Warning: Large out-of-bound parameter deviation -- %0.3f percent of the parameter range\n' % max_dev_prc)
        
        return v     
        
        
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


    def _comp_orth(self, v):
        """Compute two sets of vectors orthogonal to v"""
        def _noncolin(vec):
            """Return a vector guaranteed to be non-collinear to a 3-d vector vec (assuming vec != (0, 0, 0))"""
            res = np.zeros(3)
            res[np.argmin(vec)] = 1
            return res
        
        v_nc = np.stack(list(_noncolin(vec) for vec in v), axis=0)
                    
        orth1 = np.cross(v, v_nc)
        orth1 /= np.linalg.norm(orth1, axis=1)[:,None]
        orth2 = np.cross(v, orth1)
        orth2 /= np.linalg.norm(orth2, axis=1)[:,None]
        
        return orth1, orth2
    
    
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
            return np.vstack(self._comp_orth(nmags))
        else:
            return nmags
    

    def __init__(self, nsens, l, 
                 R_inner=0.15, R_outer=None, height_lower=0.15, phispan_lower=1.5*np.pi, frac_trans=0.05, opm=False):
        
        self._R_inner = R_inner
        self._R_outer = R_outer
        self._height_lower = height_lower
        # self._n_sens models the number of physical sensors, whereas
        # self._n_coils - the number of field measurements. For example, for
        # OPM sensors, one sensor can make two field measurements (in
        # orthogonal directions), thus _n_coils will be twice the _n_sens.
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
        theta0, phi0 = xyz2pol(rmags[:self._n_sens,0], rmags[:self._n_sens,1], rmags[:self._n_sens,2])[1:3]
        nmags = self._v2nmags(np.concatenate((theta0, phi0)))
        
        self._v0 = np.concatenate((theta0, phi0, z0, sweep0, d0)) # initial guess

        self._bins, self._n_coils, self._mag_mask, self._slice_map = _prep_mf_coils_pointlike(rmags, nmags)[2:]
        sss_origin = np.array([0.0, 0.0, 0.0])  # origin of device coords
        self._exp = {'origin': sss_origin, 'int_order': l, 'ext_order': 0}
        
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


    def comp_fitness(self, v):
        
        # Init the time measures on the first call
        if self._call_cnt == 0:
            self._first_time = time.time()
            self._prev_time = self._first_time
        
        # Update call count / timing statistics
        self._call_cnt += 1
        if self._call_cnt % 1000 == 0:
            new_time = time.time()
            print('comp_fitness has been called %i times at the rate of %0.2f / %0.2f calls per second (running / total)' % \
                  (self._call_cnt, 1000/(new_time-self._prev_time), self._call_cnt/(new_time-self._first_time)))
            self._prev_time = new_time
            
        v = self._validate_inp(v)
        allcoils = (self._v2rmags(v[2*self._n_sens:]), self._v2nmags(v[:2*self._n_sens]), self._bins, self._n_coils, self._mag_mask, self._slice_map)
        
        S = _sss_basis(self._exp, allcoils)
        S /= np.linalg.norm(S, axis=0)
        return np.linalg.cond(S)


    def plot(self, v, fig=None, plot_bg=True, opacity=0.7):
        from mayavi import mlab
        v = self._validate_inp(v)
        rmags = self._v2rmags(v[2*self._n_sens:])
        nmags = self._v2nmags(v[:2*self._n_sens])

        if fig is None:
            fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
        
        mlab.clf(fig)
        mlab.points3d(rmags[:,0], rmags[:,1], rmags[:,2], resolution=32, scale_factor=0.01, color=(0,0,1))
        mlab.quiver3d(rmags[:,0], rmags[:,1], rmags[:,2], nmags[:,0], nmags[:,1], nmags[:,2])
        
        if plot_bg:
            inner_locs = spherepts_golden(1000, hcylind=self._height_lower/self._R_inner) * self._R_inner
            pts = mlab.points3d(inner_locs[:,0], inner_locs[:,1], inner_locs[:,2], opacity=0, figure=fig)
            mesh = mlab.pipeline.delaunay3d(pts)
            mlab.pipeline.surface(mesh, figure=fig, color=(0.5, 0.5, 0.5), opacity=opacity)
        else:
            mlab.points3d(0, 0, 0, resolution=32, scale_factor=0.01, color=(0,1,0), mode='axes')
