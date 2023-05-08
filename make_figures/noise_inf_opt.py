"""
Plot noise amplification factor, channel information capacity as a function
of optimization iteration. Read the data from the folder given as a parameter.
"""

# %%
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from megsimutils.utils import uniform_sphere_dipoles, comp_inf_capacity, comp_snr
from megsimutils.arrays import noise_max, noise_mean
from megsimutils.utils import _prep_mf_coils_pointlike
from megsimutils.utils import _sssbasis_cond_pointlike
from megsimutils import read_opt_res
from IPython import get_ipython
from megsimutils.viz import _mlab_points3d, _mlab_quiver3d

ip = get_ipython()
ip.magic("gui qt5")  # needed for mayavi plots
plt.rcParams['figure.dpi'] = 150

MAX_N_ITER = 100 #math.inf # math.inf # Number of iterations to load
N_DIPOLES_LEADFIELD = 1000
R_LEADFIELD = 0.07 # m
DIPOLE_STR = 1e-8 # A * m
SQUID_NOISE = 1e-14 # T

# Check the command-line parameters
#if len(sys.argv) != 2:
#    raise RuntimeError('Wrong number of parameters. Specify the input path as a single parameter.')

# %% read the data
#datadir = Path(sys.argv[-1])  # as a cmd line argument
# or specify it here
datadir = Path("/home/jussi/andrey_ms_data_v2/2023-04-25_paper_RC4_full_run/3D_init_outer/")

opt_res = []
MAXDIRS = 1000  # DEBUG: limit number of input dirs
k = 0
for run_fldr in datadir.iterdir():
    if k < MAXDIRS and run_fldr.is_dir():
        opt_re = dict(zip(('params', 'sens_array', 'interm_res', 'opt_res', 'iter_indx'), read_opt_res(str(run_fldr) + '/out', max_n_samp=MAX_N_ITER)))
        opt_res.append(opt_re)
        np.testing.assert_equal(opt_re['params'], opt_res[0]['params'])   # All the runs should have the same parameters
        k += 1
print('done')


# %% Prepare the variables describing the optimization progress
assert opt_res[0]['params']['R_inner'] > R_LEADFIELD
dlocs, dnorms = uniform_sphere_dipoles(N_DIPOLES_LEADFIELD, R_LEADFIELD, seed=0)
plot_vars = []
sss_params = {'origin': [0., 0., 0.], 'int_order': 8, 'ext_order': 3}

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
        print('Computing stuff for intermediate %i out of %i ...' % (i, n_iter))
        v, f, accept, tstamp = opt_re['interm_res'][i]
        noise = sens_array.comp_interp_noise(v)
        interm_noise_max[i] = noise_max(noise)
        interm_noise_mean[i] = noise_mean(noise)
        slocs, snorms = sens_array._v2sens_geom(v)
        interm_inf[i] = comp_inf_capacity(slocs, snorms, dlocs, dnorms, DIPOLE_STR, SQUID_NOISE)
        interm_sss_cond[i] = _sssbasis_cond_pointlike(slocs, snorms, sss_params)

    plot_vars.append(dict(zip(('interm_noise_max', 'interm_noise_mean', 'interm_inf', 'interm_snr', 'interm_sss_cond'),
                              (interm_noise_max, interm_noise_mean, interm_inf, interm_snr, interm_sss_cond))))
print('done')


# %% like above, but extract sensor array geometries only
# (at some intermediate result)
locs_all = list()
norms_all = list()
angles_all = list()

for res_ind, opt_re in enumerate(opt_res):
    n_iter = len(opt_re['interm_res'])
    print(f'Result {res_ind}: {n_iter} iterations')
    sens_array = opt_re['sens_array']
    # v, f, accept, tstamp = opt_re['interm_res'][n_iter - 1]  # converged result
    # v, f, accept, tstamp = opt_re['interm_res'][0]  # starting result
    v, f, accept, tstamp = opt_re['interm_res'][99]  # some interm. result
    slocs, snorms = sens_array._v2sens_geom(v)
    locs_all.append(slocs)
    norms_all.append(snorms)
    radials = (slocs.T / np.linalg.norm(slocs, axis=1)).T  # radial unit vectors at sensor locs
    angles = (np.arccos(np.diag(radials @ snorms.T)) / np.pi * 180)  # sensor normals vs. radial dir
    angles_all.append(angles)


# %%
aa = np.array(angles_all).flatten()
aam = aa % 90
plt.hist(aa, 50)
plt.xlabel('Angle (deg)')
plt.ylabel('N')
#plt.savefig('angle_dist.png')



# %% plot distance histogram
slocs = locs_all[0]
negz = np.where(slocs[:, 2] < 0)[0]
slocs[negz, 2] = 0
origin_dist = np.linalg.norm(slocs, axis=1)
plt.hist(origin_dist, 5)




# %% plots for sanity check
_mlab_points3d(slocs, scale_factor=.01)
_mlab_quiver3d(slocs, snorms, scale_factor=.04)






##-------------------------------------------------------------------------
# Do the plotting
#
# %% Plot SSS cond - all data overlaid
plt.figure()
for opt_re, plot_var in zip(opt_res, plot_vars):
    plt.plot(opt_re['iter_indx'], plot_var['interm_sss_cond'], color='k', linewidth=.5)
plt.xlabel('N of iterations')
plt.ylabel('Condition number of total SSS basis')
plt.savefig('sss_cond_25_runs.png')


# %% Plot information capacity - all data overlaid
plt.figure()
for opt_re, plot_var in zip(opt_res, plot_vars):
    plt.plot(opt_re['iter_indx'], plot_var['interm_inf'], color='k', linewidth=.5)
plt.xlabel('N of iterations')
plt.ylabel('Total information (bits)')
plt.savefig('total_information_25_runs.png')


# %% Plot error vs iteration - all data overlaid
plt.figure()
for opt_re, plot_var in zip(opt_res, plot_vars):
    plt.semilogy(opt_re['iter_indx'], plot_var['interm_noise_max'], color='b', linewidth=.5)
    plt.semilogy(opt_re['iter_indx'], plot_var['interm_noise_mean'], color='k', linewidth=.5)

plt.xlabel('N of iterations')
plt.ylabel('Noise amplification factor')
plt.legend(['Maximum amplification factor ($q$)', 'Mean amplification factor'])
plt.savefig('noise_amplification_25_runs.png')






# %% Plot error vs iteration
# resample on common grid
plt.figure()
from math import inf

# find min number of iterations
it_max = inf
for opt_re, plot_var in zip(opt_res, plot_vars):
    if (this_it_max := max(opt_re['iter_indx'])) < it_max:
        it_max = this_it_max

# resample
iter_grid = np.arange(it_max)
noise_max_all = np.zeros((len(opt_res), it_max))
noise_mean_all = np.zeros((len(opt_res), it_max))
ind = 0
for opt_re, plot_var in zip(opt_res, plot_vars):
    noise_max_all[ind, :] = np.interp(iter_grid, opt_re['iter_indx'], plot_var['interm_noise_max'])
    noise_mean_all[ind, :] = np.interp(iter_grid, opt_re['iter_indx'], plot_var['interm_noise_mean'])
    ind += 1


plt.figure()
#plt.semilogy(iter_grid, noise_max_all.mean(axis=0))
mean_curve = noise_max_all.mean(axis=0)
std_lower_curve = mean_curve-noise_max_all.std(axis=0)
std_upper_curve = mean_curve+noise_max_all.std(axis=0)
plt.semilogy(iter_grid, std_lower_curve)
plt.semilogy(iter_grid, std_upper_curve)
plt.semilogy(iter_grid, mean_curve)

    





# %%

    plt.semilogy(opt_re['iter_indx'], plot_var['interm_noise_max'], color='b')
    plt.semilogy(opt_re['iter_indx'], plot_var['interm_noise_mean'], color='k')

plt.xlabel('iterations')
plt.ylabel('noise amplification factor')
#plt.legend(['max', 'avg'])




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


plt.show()