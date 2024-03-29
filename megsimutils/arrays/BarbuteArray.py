#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:45:49 2021

@author: andrey
"""
import itertools
import numpy as np

from megsimutils.utils import spherepts_golden, pol2xyz
from megsimutils.arrays import SensorArray

MAX_OFFSETS = 10    # integer, maximum number of different offsets to try with
                    # the golden ratio algorithm

class GoldenRatioError(Exception):
    """Thrown if cannot create a uniform array with the golden ratio algorithm"""
    pass


class BarbuteArray(SensorArray):
    """
    Base class for implementing various barbute arrays.
    """
    def __init__(self, l_int, l_ext, height_lower=0.15, phispan_lower=1.5*np.pi, frac_trans=0.05, ellip_sc=np.array([1.,1.,1.]), opm=False, **kwargs):
        super().__init__(l_int, l_ext, **kwargs)

        self._height_lower = height_lower
        self._phispan_lower = phispan_lower
        self._frac_trans = frac_trans
        self._ellip_sc = ellip_sc
        self._is_opm = opm


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
    
    
    def _on_barbute(self, rmags):
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

        Returns
        -------
        1-d boolean array of the length n_sens
        """
        indx_top = (rmags[:,2] >= 0)
        indx_side = (np.abs(np.arctan2(rmags[:,1], rmags[:,0])) >= (np.pi - (self._phispan_lower/2)))
        
        return indx_top | indx_side

    
    def _get_shell_params_bounds(self, R):
        return np.array([-self._height_lower / R, np.pi/2]), np.array([0, 1.0])
    
    
    def _v2rmags_shell(self, v, R, is_thick=False):
        """Convert part of the parameter vector describing sensor locations
        for a single shell to xyz coordinates
        
        Parameters
        ----------
        v : 1-d vector of floats
            Part of the parameter vector describing sensor location for a
            single shell (z, sweep, and, optionally, d)
        R        : float, shell radius. If the shell is thick, inner radius.
        is_thick : Boolean
            If True, the shell is "thick" (each sensor has it's own d). Else,
            all the ds are set to R.

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
            d = R * np.ones(n_sens)    

        geodes = v[:n_sens]
        z = geodes.copy()
        indx_up = (geodes > 0)
        z[indx_up] = np.sin(geodes[indx_up])
        sweep = v[n_sens:2*n_sens]

        # opening linearly goes from 0 to 1 as we transition from spherical to cylindrical part
        if self._height_lower > 0:
            opening = -z / (self._frac_trans * self._height_lower / R)
            opening[opening<0] = 0
            opening[opening>1] = 1
        else:
            opening = np.zeros(n_sens)
        
        phi = ((2*np.pi) * (1-opening) + self._phispan_lower * opening) * sweep + \
              (((2*np.pi - self._phispan_lower) / 2) * opening)
        
        xy = np.ones(n_sens)
        xy[indx_up] = np.sqrt(1 - z[indx_up]**2)       
        z[indx_up] = z[indx_up] * d[indx_up]
        z[~indx_up] = z[~indx_up] * R
        
        x = (xy * np.cos(phi)) * d
        y = (xy * np.sin(phi)) * d
        
        return np.stack((x,y,z), axis=1) * self._ellip_sc
              
    
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
        
        
    def _create_sampling_locs(self, R_inner, R_outer, n_layers, n_samp_per_layer):
        rmags_all = []
        nmags_all = []
        
        for r in np.linspace(R_inner, R_outer, n_layers):
            v = self.uniform_locs(n_samp_per_layer, r)
            # TODO: The following is a bit ugly, consider refactoring
            # x
            rmags_all.append(self._v2rmags_shell(v, r, False))
            nmags_all.append(np.outer(np.ones(n_samp_per_layer), np.array((1,0,0))))
            
            #y
            rmags_all.append(self._v2rmags_shell(v, r, False))
            nmags_all.append(np.outer(np.ones(n_samp_per_layer), np.array((0,1,0))))
            
            #z
            rmags_all.append(self._v2rmags_shell(v, r, False))
            nmags_all.append(np.outer(np.ones(n_samp_per_layer), np.array((0,0,1))))

        return np.vstack(rmags_all), np.vstack(nmags_all)
 

    def uniform_locs(self, n_sens, R):
        """Generate evenly spread sensors. """
        is_found = False

        for offset, i in itertools.product(range(MAX_OFFSETS), range(n_sens, 2*n_sens)):
            rmags = spherepts_golden(i, hcylind=self._height_lower/R, offset=offset)
            if np.count_nonzero(self._on_barbute(rmags)) == n_sens:
                is_found = True
                break

        if not is_found:
            raise GoldenRatioError("Could not create a uniformly-spaced array with %i sensors, sorry" % n_sens)
        
        x, y, z = (rmags[self._on_barbute(rmags)]).T
        
        # Convert xyz to v
        phi = np.arctan2(y, x)
        phi[phi<0] += 2*np.pi
        
        # opening linearly goes from 0 to 1 as we transition from spherical to cylindrical part
        if self._height_lower > 0:
            opening = -z / (self._frac_trans * self._height_lower / R)
            opening[opening<0] = 0
            opening[opening>1] = 1
        else:
            opening = np.zeros(n_sens)

        sweep = (phi - (np.pi - self._phispan_lower/2) * opening) / \
                ((2*np.pi) * (1-opening) + self._phispan_lower * opening)
        geodes = z.copy()
        geodes[z>0] = np.arcsin(z[z>0])
        
        return np.concatenate((geodes, sweep))
        
    
    def plot(self, v, fig=None, R_inner=None, R_outer=None, R_enclosure=None, opacity_inner=0.7, opacity_outer=0.1):
        from mayavi import mlab     # Import from mayavi only when plot is called. This allows using the rest of the
                                    # code in environments where import from mayavi fails (e.g. HPC clusters, etc.)
        v = self._validate_inp(v)
        rmags, nmags = self._v2sens_geom(v)

        if fig is None:
            fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
        
        mlab.clf(fig)
        mlab.points3d(rmags[:,0], rmags[:,1], rmags[:,2], resolution=32, scale_factor=0.005, color=(0,0,1))
        mlab.quiver3d(rmags[:,0], rmags[:,1], rmags[:,2], nmags[:,0], nmags[:,1], nmags[:,2], scale_factor=0.02)
        
        if R_inner is None:
            mlab.points3d(0, 0, 0, resolution=32, scale_factor=0.01, color=(0,1,0), mode='axes')
        else:
            # Draw the background
            inner_locs = spherepts_golden(1000, hcylind=self._height_lower/R_inner) * R_inner * self._ellip_sc
            pts = mlab.points3d(inner_locs[:,0], inner_locs[:,1], inner_locs[:,2], opacity=0, figure=fig)
            mesh = mlab.pipeline.delaunay3d(pts)
            mlab.pipeline.surface(mesh, figure=fig, color=(0.5, 0.5, 0.5), opacity=opacity_inner)
            
            # Draw the outer boundary of the barbute
            if R_outer is not None:              
                outer_locs = spherepts_golden(1000, hcylind=self._height_lower/R_outer) * R_outer * self._ellip_sc
                pts = mlab.points3d(outer_locs[:,0], outer_locs[:,1], outer_locs[:,2], opacity=0, figure=fig)
                mesh = mlab.pipeline.delaunay3d(pts)
                mlab.pipeline.surface(mesh, figure=fig, color=(0.5, 0.5, 0.5), opacity=opacity_outer)            
            
        
            # Draw a fully transparent enclosure around the helmet to force zoom to a certain value
            if R_enclosure is None:
                if R_outer is None:
                    R_enclosure = R_inner * 1.5
                else:
                    R_enclosure = R_outer * 1.1
                
            enclosure_locs = spherepts_golden(1000, hcylind=self._height_lower/R_inner) * R_enclosure * self._ellip_sc
            pts = mlab.points3d(enclosure_locs[:,0], enclosure_locs[:,1], enclosure_locs[:,2], opacity=0, figure=fig)
            mesh = mlab.pipeline.delaunay3d(pts)
            mlab.pipeline.surface(mesh, figure=fig, color=(0.5, 0.5, 0.5), opacity=0)            
            