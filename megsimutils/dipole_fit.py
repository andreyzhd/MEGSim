#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 04:14:29 2020

@author: andrey
"""

from multiprocessing import Pool
import numpy as np
import scipy

from megsimutils.utils import local_axes, pol2xyz
from megsimutils import dipfld_sph

class _DipoleFitter():
    """Auxillary class for dipole fitting. It's job to hold all kinds of
    relevant variables (e.g. array geometry, etc.) to simplify parallel
    execution of the grid search."""
    
    def __init__(self, rmags, cosmags, data, search_params):
        self._rmags = rmags
        self._cosmags = cosmags
        self._data = data
        self._theta_max = search_params['theta_max']
        self._n_theta = search_params['n_theta']
        self._n_phi = search_params['n_phi']
        
    def search_fixed_R(self, r):
        print('Starting dipole search for r = %f ...' % r)

        locs = np.zeros([self._n_theta*self._n_phi, 3])
        qs = np.zeros([self._n_theta*self._n_phi, 3])
        resids = np.zeros(self._n_theta*self._n_phi)
        
        cnt = 0
        for theta in np.linspace(-self._theta_max, self._theta_max, self._n_theta):
            for phi in np.linspace(0, 2*np.pi, self._n_phi, endpoint=False):
                loc = np.array(pol2xyz(r, theta, phi))
                tg_0, tg_1 = local_axes(theta, phi)[1:3]    # locally tangential vectors
                meas_0 = (dipfld_sph(tg_0, loc, self._rmags, np.zeros(3)) * self._cosmags).sum(axis=1)
                meas_1 = (dipfld_sph(tg_1, loc, self._rmags, np.zeros(3)) * self._cosmags).sum(axis=1)
            
                x , resid = scipy.linalg.lstsq(np.stack((meas_0, meas_1), axis=1), self._data)[0:2]
                
                locs[cnt,:] = loc
                qs[cnt,:] = tg_0*x[0] + tg_1*x[1]
                resids[cnt] = resid
                cnt += 1
                
        print('Finished dipole search for r = %f' % r)
        return locs, qs, resids
         

def bf_dipole_fit(rmags, cosmags, data, search_params):
    """
    Fit the dipole by using an extensive search (brute-force)

    Parameters
    ----------
    rmags : M-by-3 vector of sensor locations
    cosmags : M-by-3 vector of sensor orientations (sensors are assumend to be magnetometers)
    data : M-long vector of sensor readings
    search_params : dictionary of parameters controlling the grid search
        rmin : Minimum radius for the dipole search
        rmax : Maximum radius for the dipole search
        theta_max : Maximum theta angle for the dipole search (the minimum is -theta_max).
        n_r : number of steps in R
        n_theta : number of steps in theta
        n_phi : number of steps in phi
    Returns
    -------
    best_loc : estimated dipole location
    best_q : estimated dipole moment
    best_resid : residual error 

    """
    
    df = _DipoleFitter(rmags, cosmags, data, search_params)
    
    p = Pool()
    res = p.map(df.search_fixed_R, np.linspace(search_params['rmin'], search_params['rmax'], search_params['n_r']))
    #res = []
    #for r in np.linspace(search_params['rmin'], search_params['rmax'], search_params['n_r']):
    #    res.append(df.search_fixed_R(r))
      
    llocs, lqs, lresids = list(zip(*res))
    
    locs = np.vstack(llocs)
    qs = np.vstack(lqs)
    resids = np.concatenate(lresids)
    
    best_indx = np.argmin(resids)
    
    return locs[best_indx,:], qs[best_indx,:], resids[best_indx]
    