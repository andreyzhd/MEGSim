"""
Plot noise amplification factor and channel information capacity as a function
of optimization iteration.
"""

# %% Initialize
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from megsimutils.utils import uniform_sphere_dipoles, comp_inf_capacity
from megsimutils.arrays import noise_max, noise_mean
from megsimutils.utils import _sssbasis_cond_pointlike
from megsimutils import read_opt_res
from megsimutils.envutils import _ipython_setup

_ipython_setup()

# set computation parameters
MAX_N_ITER = 100  # number of intermediate iteration results to load
N_DIPOLES_LEADFIELD = 1000
R_LEADFIELD = 0.07  # m
DIPOLE_STR = 2e-8  # A * m
SQUID_NOISE = 1e-14  # T


# %% Read the data
#
# The data directory datadir must contain one or more directories corresponding to optimization runs,
# and each of those must contain an "out" directory with the results.
#
datadir = Path("/path_to_dataset")

opt_res = []
k = 0
for run_fldr in datadir.iterdir():
    if run_fldr.is_dir():
        opt_re = dict(
            zip(
                ('params', 'sens_array', 'interm_res', 'opt_res', 'iter_indx'),
                read_opt_res(str(run_fldr) + '/out', max_n_samp=MAX_N_ITER),
            )
        )
        opt_res.append(opt_re)
        np.testing.assert_equal(
            opt_re['params'], opt_res[0]['params']
        )  # All the runs should have the same parameters
        k += 1
print('done')


# %% Prepare the variables describing the optimization progress
assert opt_res[0]['params']['R_inner'] > R_LEADFIELD
dlocs, dnorms = uniform_sphere_dipoles(N_DIPOLES_LEADFIELD, R_LEADFIELD, seed=0)
plot_vars = []
sss_params = {'origin': [0.0, 0.0, 0.0], 'int_order': 8, 'ext_order': 3}
nres = len(opt_res)

for res_ind, opt_re in enumerate(opt_res, 1):
    print(f'Reading result dataset {res_ind}...')
    n_iter = len(opt_re['interm_res'])
    interm_noise_max = np.zeros((n_iter,))
    interm_noise_mean = np.zeros((n_iter,))
    interm_inf = np.zeros((n_iter,))
    interm_sss_cond = np.zeros((n_iter,))

    # empty array for storing SNRs
    sens_array = opt_re['sens_array']
    slocs, snorms = sens_array._v2sens_geom(sens_array.get_init_vector())
    interm_snr = np.zeros((n_iter, slocs.shape[0]))

    # for every result set, there are N=100 intermediate results stored
    # the intermediate results correspond to different iteration indices in each set
    for i in range(n_iter):
        print(f'Computing stuff for set {res_ind}/{nres}, intermediate {i+1}/{n_iter}')
        v, f, accept, tstamp = opt_re['interm_res'][i]
        noise = sens_array.comp_interp_noise(v)
        interm_noise_max[i] = noise_max(noise)
        interm_noise_mean[i] = noise_mean(noise)
        slocs, snorms = sens_array._v2sens_geom(v)
        interm_inf[i] = comp_inf_capacity(
            slocs, snorms, dlocs, dnorms, DIPOLE_STR, SQUID_NOISE
        )
        interm_sss_cond[i] = _sssbasis_cond_pointlike(slocs, snorms, sss_params)

    plot_vars.append(
        dict(
            zip(
                (
                    'interm_noise_max',
                    'interm_noise_mean',
                    'interm_inf',
                    'interm_snr',
                    'interm_sss_cond',
                ),
                (
                    interm_noise_max,
                    interm_noise_mean,
                    interm_inf,
                    interm_snr,
                    interm_sss_cond,
                ),
            )
        )
    )
print('done')


# %% Plot condition number of SSS matrix
plt.figure()
for opt_re, plot_var in zip(opt_res, plot_vars):
    plt.plot(opt_re['iter_indx'], plot_var['interm_sss_cond'], color='k', linewidth=0.5)
plt.xlabel('N of iterations')
plt.ylabel('Condition number of total SSS basis')


# %% Plot information capacity
plt.figure()
for opt_re, plot_var in zip(opt_res, plot_vars):
    plt.plot(opt_re['iter_indx'], plot_var['interm_inf'])
plt.xlabel('N of iterations')
plt.ylabel('Total information (bits)')


# %% Plot mean and maximum interpolation error
plt.figure()
for opt_re, plot_var in zip(opt_res, plot_vars):
    plt.semilogy(opt_re['iter_indx'], plot_var['interm_noise_max'])
    plt.semilogy(opt_re['iter_indx'], plot_var['interm_noise_mean'])
plt.xlabel('N of iterations')
plt.ylabel('Noise amplification factor')
plt.legend(['Max. NA', 'Mean NA'])
plt.savefig('noise_vs_iter_2D.png')
