"""
Study optimization procedure's impact on array's total information capacity
"""
import numpy as np
import matplotlib.pyplot as plt

from read_opt_res import read_opt_res
from megsimutils.utils import uniform_sphere_dipoles, comp_inf_capacity
from megsimutils.arrays import noise_max, noise_mean


INP_PATH = '/home/andrey/scratch/out'

N_DIPOLES_LEADFIELD = 10000
R_LEADFIELD = 0.1 # m
DIPOLE_STR = 5 * 1e-8 # A * m
SQUID_NOISE = 10 * 1e-15 # T

#%% Read the data
params, sens_array, interm_res, opt_res = read_opt_res(INP_PATH)

#%% Prepare the variables describing the optimization progress
assert params['R_inner'] > R_LEADFIELD
dlocs, dnorms = uniform_sphere_dipoles(N_DIPOLES_LEADFIELD, R_LEADFIELD, seed=1)

interm_noise_max = []
interm_noise_mean = []
interm_inf = []
timing = []
x_accepts = []
y_accepts = []

for (v, f, accept, tstamp) in interm_res:
    noise = sens_array.comp_interp_noise(v)
    interm_noise_max.append(noise_max(noise))
    interm_noise_mean.append(noise_mean(noise))
    slocs, snorms = sens_array._v2sens_geom(v)
    interm_inf.append(comp_inf_capacity(slocs, snorms, dlocs, dnorms, DIPOLE_STR, SQUID_NOISE))

    if accept:
        x_accepts.append(len(interm_noise_max) - 1)
        y_accepts.append(f)

    timing.append(tstamp)

timing = np.diff(np.array(timing))


#%% Plot information capacity
plt.figure()
plt.plot(interm_inf)
plt.xlabel('iterations')
plt.ylabel('bits')
plt.legend(['total information per sample'])
plt.title('L=(%i, %i), %i sensors, optimized for %s' % (params['l_int'], params['kwargs']['l_ext'], np.sum(params['n_sens']), params['kwargs']['noise_stat'].__name__))


#%% Plot error vs iteration
plt.figure()
plt.plot(interm_noise_max)
plt.plot(interm_noise_mean)
plt.plot(x_accepts, y_accepts, 'ok')
plt.ylim((0, np.percentile(interm_noise_max, 95)))
plt.xlabel('iterations')
plt.legend(['max noise', 'mean noise', 'accepted'])
plt.title('L=(%i, %i), %i sensors, optimized for %s' % (params['l_int'], params['kwargs']['l_ext'], np.sum(params['n_sens']), params['kwargs']['noise_stat'].__name__))

plt.show()