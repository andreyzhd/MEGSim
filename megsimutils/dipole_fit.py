#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 04:14:29 2020

@author: andrey
"""

import numpy as np
import scipy

from megsimutils.utils import local_axes, pol2xyz
from megsimutils import dipfld_sph

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
    best_resid = np.Inf
    for r in np.linspace(search_params['rmin'], search_params['rmax'], search_params['n_r']):
        print('r = %f' % r)
        for theta in np.linspace(-search_params['theta_max'], search_params['theta_max'], search_params['n_theta']):
            for phi in np.linspace(0, 2*np.pi, search_params['n_phi'], endpoint=False):
                loc = np.array(pol2xyz(r, theta, phi))
                tg_0, tg_1 = local_axes(theta, phi)[1:3]    # locally tangential vectors
                meas_0 = (dipfld_sph(tg_0, loc, rmags, np.zeros(3)) * cosmags).sum(axis=1)
                meas_1 = (dipfld_sph(tg_1, loc, rmags, np.zeros(3)) * cosmags).sum(axis=1)
            
                x , resid = scipy.linalg.lstsq(np.stack((meas_0, meas_1), axis=1), data)[0:2]
                if resid < best_resid:
                    best_loc = loc
                    best_q = tg_0*x[0] + tg_1*x[1]
                    best_resid = resid
                    
    return best_loc, best_q, best_resid
    