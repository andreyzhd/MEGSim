"""
Plot noise amplification factor, channel information capacity as a function
of optimization iteration. Read the data from the folder given as a parameter.
"""

import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from megsimutils.utils import uniform_sphere_dipoles, comp_inf_capacity, comp_snr
from megsimutils.arrays import noise_max, noise_mean
from megsimutils.utils import _prep_mf_coils_pointlike
from megsimutils import read_opt_res

MAX_N_ITER = 100 #math.inf # math.inf # Number of iterations to load
N_DIPOLES_LEADFIELD = 1000
R_LEADFIELD = 0.07 # m
DIPOLE_STR = 1e-8 # A * m
SQUID_NOISE = 1e-14 # T

# Check the command-line parameters
if len(sys.argv) != 2:
    raise RuntimeError('Wrong number of parameters. Specify the input path as a single parameter.')

##-------------------------------------------------------------------------
# Read the data
#
opt_res = []
for run_fldr in Path(sys.argv[-1]).iterdir():
    if run_fldr.is_dir():
        opt_re = dict(zip(('params', 'sens_array', 'interm_res', 'opt_res', 'iter_indx'), read_opt_res(str(run_fldr)+'/out', max_n_samp=MAX_N_ITER)))
        opt_res.append(opt_re)
        np.testing.assert_equal(opt_re['params'], opt_res[0]['params'])   # All the runs should have the same parameters

##-------------------------------------------------------------------------
# Prepare the variables describing the optimization progress
#
assert opt_res[0]['params']['R_inner'] > R_LEADFIELD
dlocs, dnorms = uniform_sphere_dipoles(N_DIPOLES_LEADFIELD, R_LEADFIELD, seed=0)
plot_vars = []

for opt_re in opt_res:
    n_iter = len(opt_re['interm_res'])
    interm_noise_max = np.zeros((n_iter,))
    interm_noise_mean = np.zeros((n_iter,))
    interm_inf = np.zeros((n_iter,))

    # empty array for storing SNRs
    sens_array = opt_re['sens_array']
    slocs, snorms = sens_array._v2sens_geom(sens_array.get_init_vector())
    interm_snr = np.zeros((n_iter, slocs.shape[0]))

    for i in range(n_iter):
        print('Computing stuff for iteration %i out of %i ...' % (i, n_iter))
        v, f, accept, tstamp = opt_re['interm_res'][i]
        noise = sens_array.comp_interp_noise(v)
        interm_noise_max[i] = noise_max(noise)
        interm_noise_mean[i] = noise_mean(noise)
        slocs, snorms = sens_array._v2sens_geom(v)
        interm_inf[i] = comp_inf_capacity(slocs, snorms, dlocs, dnorms, DIPOLE_STR, SQUID_NOISE)

    plot_vars.append(dict(zip(('interm_noise_max', 'interm_noise_mean', 'interm_inf', 'interm_snr'), (interm_noise_max, interm_noise_mean, interm_inf, interm_snr))))

##-------------------------------------------------------------------------
# Do the plotting
#

# %% Plot information capacity
plt.figure()
for opt_re, plot_var in zip(opt_res, plot_vars):
    plt.plot(opt_re['iter_indx'], plot_var['interm_inf'])
plt.xlabel('iterations')
plt.ylabel('bits')
plt.legend(['total information per sample'])

# %% Plot error vs iteration
plt.figure()
for opt_re, plot_var in zip(opt_res, plot_vars):
    plt.semilogy(opt_re['iter_indx'], plot_var['interm_noise_max'])
    plt.semilogy(opt_re['iter_indx'], plot_var['interm_noise_mean'])

plt.xlabel('iterations')
plt.ylabel('noise amplification factor')
#plt.legend(['max', 'avg'])

"""
# %% Plot SNR
plt.figure()
plt.plot(iter_indx, np.log10(np.median(interm_snr, axis=1)) * 10)
plt.plot(iter_indx, np.log10(np.mean(interm_snr, axis=1)) * 10)
plt.plot(iter_indx, np.log10(np.min(interm_snr, axis=1)) * 10, '.')
plt.plot(iter_indx, np.log10(np.max(interm_snr, axis=1)) * 10, '.')
plt.legend(['median', 'mean', 'min', 'max'])

plt.xlabel('iterations')
plt.ylabel('Power SNR, dB')

plt.show()

print('L=(%i, %i), %i sensors, optimized for %s' % (params['l_int'], params['l_ext'], np.sum(params['n_sens']), params['kwargs']['noise_stat'].__name__))

"""
plt.show()