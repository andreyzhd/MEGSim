"""
Plot noise amplification factor, channel information capacity as a function
of optimization iteration. Read the data from the folder given as a parameter.
"""

import math
import sys

import numpy as np
import matplotlib.pyplot as plt
from mne.preprocessing.maxwell import _sss_basis

from megsimutils.utils import uniform_sphere_dipoles, comp_inf_capacity, comp_snr
from megsimutils.arrays import noise_max, noise_mean
from megsimutils.utils import _prep_mf_coils_pointlike
from megsimutils import read_opt_res

MAX_N_ITER = 100 # math.inf # Number of iterations to load
N_DIPOLES_LEADFIELD = 1000
R_LEADFIELD = 0.1 # m
DIPOLE_STR = 1e-8 # A * m
SQUID_NOISE = 1e-14 # T

# Check the command-line parameters
if len(sys.argv) != 2:
    raise RuntimeError('Wrong number of parameters. Specify the input path as a single parameter.')

# %% Read the data
params, sens_array, interm_res, opt_res, iter_indx = read_opt_res(sys.argv[-1], max_n_samp=MAX_N_ITER)

# %% Prepare the variables describing the optimization progress
assert params['R_inner'] > R_LEADFIELD
dlocs, dnorms = uniform_sphere_dipoles(N_DIPOLES_LEADFIELD, R_LEADFIELD, seed=0)

n_iter = len(interm_res)
interm_noise_max = np.zeros((n_iter,))
interm_noise_mean = np.zeros((n_iter,))
interm_inf = np.zeros((n_iter,))

# empty array for storing SNRs
slocs, snorms = sens_array._v2sens_geom(sens_array.get_init_vector())
interm_snr = np.zeros((n_iter, slocs.shape[0]))

r_conds = np.zeros((n_iter,))  # DEBUG

for i in range(n_iter):
    print('Computing stuff for iteration %i out of %i ...' % (i, n_iter))
    v, f, accept, tstamp = interm_res[i]
    noise = sens_array.comp_interp_noise(v)
    interm_noise_max[i] = noise_max(noise)
    interm_noise_mean[i] = noise_mean(noise)
    slocs, snorms = sens_array._v2sens_geom(v)
    interm_inf[i] = comp_inf_capacity(slocs, snorms, dlocs, dnorms, DIPOLE_STR, SQUID_NOISE)

    # Compute SNR
    interm_snr[i, :] = comp_snr(slocs, snorms, dlocs, dnorms, DIPOLE_STR, SQUID_NOISE)

    # Ugly hack -- accessing the SensorArray private variable is necessary to
    # compute the VSH matrix
    bins, n_coils, mag_mask, slice_map = _prep_mf_coils_pointlike(slocs, snorms)[2:]
    allcoils = (slocs, snorms, bins, n_coils, mag_mask, slice_map)
    S = _sss_basis(sens_array._SensorArray__exp, allcoils)

    S /= np.linalg.norm(S, axis=0)
    r_conds[i] = np.linalg.cond(S)

# %% Plot information capacity
plt.figure()
plt.plot(iter_indx, interm_inf)
plt.xlabel('iterations')
plt.ylabel('bits')
plt.legend(['total information per sample'])

# %% Plot error vs iteration
plt.figure()
plt.semilogy(iter_indx, interm_noise_max)
plt.semilogy(iter_indx, interm_noise_mean)
# plt.ylim((0, np.percentile(interm_noise_max, 99)))
plt.xlabel('iterations')
plt.ylabel('noise amplification factor')
plt.legend(['max', 'avg'])

# %% Plot the condition number
plt.figure()
plt.semilogy(iter_indx, r_conds)
plt.xlabel('iterations')
plt.ylabel('Condition number (normalized)')

# %% Plot SNR
plt.figure()
plt.plot(iter_indx, np.log10(np.median(interm_snr, axis=1)) * 10)
plt.plot(iter_indx, np.log10(np.mean(interm_snr, axis=1)) * 10)
plt.plot(iter_indx, np.log10(np.min(interm_snr, axis=1)) * 10, '.')
plt.plot(iter_indx, np.log10(np.max(interm_snr, axis=1)) * 10, '.')
plt.legend(['median', 'mean', 'min', 'max'])

plt.xlabel('iterations')
plt.ylabel('Power SNR, dB')

# %% Scatter plot - condition number vs noise amplification factor
plt.figure()
plt.plot(interm_noise_max, r_conds, '*')
plt.xlabel('noise amplification factor')
plt.ylabel('Condition number (normalized)')

plt.show()

print('L=(%i, %i), %i sensors, optimized for %s' % (params['l_int'], params['l_ext'], np.sum(params['n_sens']), params['kwargs']['noise_stat'].__name__))

