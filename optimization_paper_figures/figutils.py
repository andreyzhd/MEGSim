#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some utils used by the scripts generating the paper figures 

"""

import numpy as np
import matplotlib.pyplot as plt
from mne.preprocessing.maxwell import _sss_basis

from megsimutils.utils import uniform_sphere_dipoles, comp_inf_capacity, comp_snr
from megsimutils.arrays import noise_max, noise_mean
from megsimutils.utils import _prep_mf_coils_pointlike

from megsimutils import read_opt_res

def vis_noise_inf_opt(inp_path, max_n_iter, n_dipoles_leadfield, r_leadfield, dipole_str, squid_noise):
    """
    Plot noise amplification factor, channel information capacity as a function of optimization iteration
    """
    
    #%% Read the data
    params, sens_array, interm_res, opt_res, iter_indx = read_opt_res(inp_path, max_n_samp=max_n_iter)
    
    #%% Prepare the variables describing the optimization progress
    assert params['R_inner'] > r_leadfield
    dlocs, dnorms = uniform_sphere_dipoles(n_dipoles_leadfield, r_leadfield, seed=0)
    
    n_iter = len(interm_res)
    interm_noise_max = np.zeros((n_iter,))
    interm_noise_mean = np.zeros((n_iter,))
    interm_inf = np.zeros((n_iter,))
    
    # empty array for storing SNRs
    slocs, snorms = sens_array._v2sens_geom(sens_array.get_init_vector())
    interm_snr = np.zeros((n_iter, slocs.shape[0]))
    
    r_conds = np.zeros((n_iter,))    # DEBUG
    
    for i in range(n_iter):
        print('Computing stuff for iteration %i out of %i ...' % (i, n_iter))
        v, f, accept, tstamp = interm_res[i]
        noise = sens_array.comp_interp_noise(v)
        interm_noise_max[i] = noise_max(noise)
        interm_noise_mean[i] = noise_mean(noise)
        slocs, snorms = sens_array._v2sens_geom(v)
        interm_inf[i] = comp_inf_capacity(slocs, snorms, dlocs, dnorms, dipole_str, squid_noise)
    
        # Compute SNR
        interm_snr[i, :] = comp_snr(slocs, snorms, dlocs, dnorms, dipole_str, squid_noise)
    
        # Ugly hack -- accessing the SensorArray private variable is necessary to
        # compute the VSH matrix
        bins, n_coils, mag_mask, slice_map = _prep_mf_coils_pointlike(slocs, snorms)[2:]
        allcoils = (slocs, snorms, bins, n_coils, mag_mask, slice_map)
        S = _sss_basis(sens_array._SensorArray__exp, allcoils)
    
        S /= np.linalg.norm(S, axis=0)
        r_conds[i] = np.linalg.cond(S)
    
    #%% Plot information capacity
    plt.figure()
    plt.plot(iter_indx, interm_inf)
    plt.xlabel('iterations')
    plt.ylabel('bits')
    plt.legend(['total information per sample'])
    
    #%% Plot error vs iteration
    plt.figure()
    plt.semilogy(iter_indx, interm_noise_max)
    plt.semilogy(iter_indx, interm_noise_mean)
    #plt.ylim((0, np.percentile(interm_noise_max, 99)))
    plt.xlabel('iterations')
    plt.ylabel('noise amplification factor')
    plt.legend(['max', 'avg'])
    
    #%% Plot the condition number
    plt.figure()
    plt.semilogy(iter_indx, r_conds)
    plt.xlabel('iterations')
    plt.ylabel('Condition number (normalized)')
    
    #%% Plot SNR
    plt.figure()
    plt.plot(iter_indx, np.log10(np.median(interm_snr, axis=1)) * 10)
    plt.plot(iter_indx, np.log10(np.mean(interm_snr, axis=1)) * 10)
    plt.plot(iter_indx, np.log10(np.min(interm_snr, axis=1)) * 10, '.')
    plt.plot(iter_indx, np.log10(np.max(interm_snr, axis=1)) * 10, '.')
    plt.legend(['median', 'mean', 'min', 'max'])
    
    plt.xlabel('iterations')
    plt.ylabel('Power SNR, dB')
    
    #%% Scatter plot - condition number vs noise amplification factor
    plt.figure()
    plt.plot(interm_noise_max, r_conds, '*')
    plt.xlabel('noise amplification factor')
    plt.ylabel('Condition number (normalized)')
    
    print('L=(%i, %i), %i sensors, optimized for %s' % (params['l_int'], params['l_ext'], np.sum(params['n_sens']), params['kwargs']['noise_stat'].__name__))
    
