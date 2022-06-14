#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:45:49 2021

@author: andrey
"""
import time
import pickle
from abc import ABC, abstractmethod
import numpy as np

from mne.preprocessing.maxwell import _sss_basis, _get_n_moments
from megsimutils.utils import _prep_mf_coils_pointlike

MU0 = 1e-7 * 4 * np.pi
BAD_FITNESS = 1e10  # Large value to be used if fitness function computation fails

def noise_max(noise):
    """An implementation of noise_stat parameter that returns maximal noise over the sampling volume."""
    return noise.max()

def noise_mean(noise):
    """An implementation of noise_stat parameter that returns average noise over the sampling volume."""
    return noise.mean()

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
    Base class for implementing various MEG sensor arrays
    """
    def __init__(self, l_int, l_ext, origin=np.array([0., 0., 0.]), noise_stat=noise_max, debug_fldr=None):
        """
        Constructor for SensorArray

        Parameters
        ----------
        l_int : integer
            Order of the VSH expansion
        origin : 3-d vector
            Coordinates of the expansion origin
        noise_stat : a function that summarizes the noise distribution across
                     the sampling volume in a single number. Receives a 1d numpy
                     array of noise values at different locations across the
                     sampling volume and returns a single float.
        Returns
        -------
        None.

        """
        self.__call_cnt = 0
        self.__exp = {'origin': origin, 'int_order': l_int, 'ext_order': l_ext}
        self.__forward_matrix = None
        self.__debug_fldr = debug_fldr
        self.__noise_stat = noise_stat
        

    def _validate_inp(self, v):
        """ Check that the input is within bounds and correct if necessary
        """
        v = v.copy()
        bounds = self.get_bounds()
        indx_below = (v < bounds[:,0])
        indx_above = (v > bounds[:,1])
        
        deviat_above = 0
        deviat_below = 0
        
        if indx_above.any():
            deviat_above = np.max((v[indx_above] - bounds[indx_above,1]) / np.diff(bounds[indx_above,:], axis=1))
            v[indx_above] = bounds[indx_above,1]
                        
        if indx_below.any():
            deviat_below = np.max((bounds[indx_below,0] - v[indx_below]) / np.diff(bounds[indx_below,:], axis=1))
            v[indx_below] = bounds[indx_below,0]
            
        max_dev_prc = 100 * max(deviat_above, deviat_below)

        if max_dev_prc >= 10:
            print('\Warning: Large out-of-bound parameter deviation -- %0.2f percent of the parameter range\n' % max_dev_prc)
        
        return v


    def comp_interp_noise(self, v):
        """
        Compute interpolation noise (at locations specified by
        _get_sampling_locs()) for a given parameter vector

        Parameters
        ----------
        v : 1-d vector
            Contains sensor array parameters being optimized.

        Returns
        -------
        1-d vector of noise values. The length of the vector is the same as
        the number of sampling locations (given by _get_sampling_locs()).
        Each entry of the vector gives the interpolation noise (measured as
        standard deviation) at the corresponding sampling location.
        """
        n_int = _get_n_moments(self.__exp['int_order']) # dim of the internal part of the VSH basis

        # Compute forward matrix if it doesn't exist (needs to be done only once)
        if self.__forward_matrix is None:
            rmags_samp, nmags_samp = self._get_sampling_locs()
            bins_samp, n_coils_samp, mag_mask_samp, slice_map_samp = _prep_mf_coils_pointlike(rmags_samp, nmags_samp)[2:]
            allcoils_samp = (rmags_samp, nmags_samp, bins_samp, n_coils_samp, mag_mask_samp, slice_map_samp)
            full_forward_matrix = _sss_basis(self.__exp, allcoils_samp)

            # Select only the components corresponding to the internal basis
            # (we only use the internal basis for the magnetic field interpolation).
            self.__forward_matrix = full_forward_matrix[:, :n_int]

        # Compute the interpolation noise
        v = self._validate_inp(v)
        rmags, nmags = self._v2sens_geom(v)
        bins, n_coils, mag_mask, slice_map = _prep_mf_coils_pointlike(rmags, nmags)[2:]
        allcoils = (rmags, nmags, bins, n_coils, mag_mask, slice_map)

        S = _sss_basis(self.__exp, allcoils)
        Sp = np.linalg.pinv(S)[:n_int, :] # Select only the internal basis

        return np.linalg.norm(self.__forward_matrix @ Sp, axis=1)


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
        # Init the time measures on the first call
        if self.__call_cnt == 0:
            self.__first_time = time.time()
            self.__prev_time = self.__first_time
        
        # Update call count / timing statistics
        self.__call_cnt += 1
        if self.__call_cnt % 1000 == 0:
            new_time = time.time()
            print('comp_fitness has been called %i times at the rate of %0.2f / %0.2f calls per second (running / total)' % \
                  (self.__call_cnt, 1000/(new_time-self.__prev_time), self.__call_cnt/(new_time-self.__first_time)))
            self.__prev_time = new_time
        
        try:
            noise = self.comp_interp_noise(v)
            res = self.__noise_stat(noise)
        except Exception as excp:
            print('Exception when running comp_fitness for %i-th time.' % self.__call_cnt)
            
            if self.__debug_fldr is None:
                print('No debug folder, not saving anything')
            else:
                fname = '%s/debug_dump_%06i.pkl' % (self.__debug_fldr, self.__call_cnt)
                print('Trying to save debug dump as %s ...' % fname)
                try:
                    fl = open(fname, 'wb')
                    pickle.dump((v, self, excp, time.time()), fl)
                    fl.close()
                    print('\t\tSuccess!')
                except:
                    print('\t\tFailed!')

            res = BAD_FITNESS
            
        return res
    
    
    @abstractmethod
    def plot(self, v, fig=None, **kwargs):
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
    
    
    @abstractmethod
    def _v2sens_geom(self, v):
        """
        Convert parameter vector (as seen by the optimization algorithm) to
        coil locations and orientations in a format accepted by mne-python.

        Parameters
        ----------
        v : TYPE
            1-d parameter vector

        Returns
        -------
        rmags, nmags -- 2 arrays of size n_coils-by-3

        """
        pass

        
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
        pass


    @abstractmethod
    def _get_sampling_locs(self):
        """
        Return sampling locations--a discrete approximatition of all possible
        locations within the sampling volume

        Returns
        -------
        rmags, nmags -- 2 arrays of size n_sampling_locations-by-3

        """
        pass