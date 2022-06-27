"""
Plot noise amplification factor, channel information capacity as a function of optimization iteration
"""
import numpy as np
import matplotlib.pyplot as plt
import math

from read_opt_res import read_opt_res
from megsimutils.utils import uniform_sphere_dipoles, comp_inf_capacity
from megsimutils.arrays import noise_max, noise_mean

# DEBUG
from mne.preprocessing.maxwell import _sss_basis
from megsimutils.utils import _prep_mf_coils_pointlike
# ~DEBUG

INP_PATH = '/home/andrey/storage/Data/MEGSim/2022-03-18_paper_RC_full_run/run_thin/out'
N_ITER = 100 # math.inf # Number of iterations to load

N_DIPOLES_LEADFIELD = 1000
R_LEADFIELD = 0.1 # m
DIPOLE_STR = 5 * 1e-8 # A * m
SQUID_NOISE = 10 * 1e-15 # T

#%% Read the data
params, sens_array, interm_res, opt_res, iter_indx = read_opt_res(INP_PATH, max_n_samp=N_ITER)

#%% Prepare the variables describing the optimization progress
assert params['R_inner'] > R_LEADFIELD
dlocs, dnorms = uniform_sphere_dipoles(N_DIPOLES_LEADFIELD, R_LEADFIELD, seed=0)

interm_noise_max = []
interm_noise_mean = []
interm_inf = []

r_conds = []    # DEBUG

for (v, f, accept, tstamp) in interm_res:
    noise = sens_array.comp_interp_noise(v)
    interm_noise_max.append(noise_max(noise))
    interm_noise_mean.append(noise_mean(noise))
    slocs, snorms = sens_array._v2sens_geom(v)
    interm_inf.append(comp_inf_capacity(slocs, snorms, dlocs, dnorms, DIPOLE_STR, SQUID_NOISE))

    # DEBUG
    bins, n_coils, mag_mask, slice_map = _prep_mf_coils_pointlike(slocs, snorms)[2:]
    allcoils = (slocs, snorms, bins, n_coils, mag_mask, slice_map)
    S = _sss_basis(sens_array._SensorArray__exp, allcoils)

    S /= np.linalg.norm(S, axis=0)
    r_conds.append(np.linalg.cond(S))
    # ~DEBUG

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

# DEBUG
#%% Plot the condition number
plt.figure()
plt.semilogy(iter_indx, r_conds)
plt.xlabel('iterations')
plt.ylabel('Condition number (normalized)')

#%% Scatter plot - condition number vs noise amplification factor
plt.figure()
plt.plot(interm_noise_max, r_conds, '*')
plt.xlabel('noise amplification factor')
plt.ylabel('Condition number (normalized)')
# ~DEBUG

print('L=(%i, %i), %i sensors, optimized for %s' % (params['l_int'], params['l_ext'], np.sum(params['n_sens']), params['kwargs']['noise_stat'].__name__))

plt.show()