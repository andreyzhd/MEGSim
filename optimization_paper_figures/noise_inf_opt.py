"""
Plot noise amplification factor, channel information capacity as a function of optimization iteration
"""
import numpy as np
import matplotlib.pyplot as plt
import math

from read_opt_res import read_opt_res
from megsimutils.utils import uniform_sphere_dipoles, comp_inf_capacity
from megsimutils.arrays import noise_max, noise_mean


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

for (v, f, accept, tstamp) in interm_res:
    noise = sens_array.comp_interp_noise(v)
    interm_noise_max.append(noise_max(noise))
    interm_noise_mean.append(noise_mean(noise))
    slocs, snorms = sens_array._v2sens_geom(v)
    interm_inf.append(comp_inf_capacity(slocs, snorms, dlocs, dnorms, DIPOLE_STR, SQUID_NOISE))

#%% Plot information capacity
plt.figure()
plt.plot(iter_indx, interm_inf)
plt.xlabel('iterations')
plt.ylabel('bits')
plt.legend(['total information per sample'])
print('L=(%i, %i), %i sensors, optimized for %s' % (params['l_int'], params['kwargs']['l_ext'], np.sum(params['n_sens']), params['kwargs']['noise_stat'].__name__))


#%% Plot error vs iteration
plt.figure()
plt.semilogy(iter_indx, interm_noise_max)
plt.semilogy(iter_indx, interm_noise_mean)
#plt.ylim((0, np.percentile(interm_noise_max, 99)))
plt.xlabel('iterations')
plt.ylabel('noise amplification factor')
plt.legend(['max', 'avg'])
print('L=(%i, %i), %i sensors, optimized for %s' % (params['l_int'], params['kwargs']['l_ext'], np.sum(params['n_sens']), params['kwargs']['noise_stat'].__name__))

plt.show()