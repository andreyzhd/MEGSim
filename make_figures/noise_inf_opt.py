"""
Plot noise amplification factor, channel information capacity as a function
of optimization iteration. Read the data from the folder given as a parameter.
"""

# %%
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pickle

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
DIPOLE_STR = 2e-8 # A * m
SQUID_NOISE = 1e-14 # T

# Check the command-line parameters
#if len(sys.argv) != 2:
#    raise RuntimeError('Wrong number of parameters. Specify the input path as a single parameter.')

# %% read the data
#datadir = Path(sys.argv[-1])  # as a cmd line argument
# or specify it here
datadir = "/home/jussi/andrey_ms_data_v4/2023-05-22_paper_RC4_full_run/2D"
datadir = Path(datadir)

opt_res = []
MAXDIRS = 8  # DEBUG: limit number of input dirs
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
        interm_inf[i] = comp_inf_capacity(slocs, snorms, dlocs, dnorms, DIPOLE_STR, SQUID_NOISE)
        interm_sss_cond[i] = _sssbasis_cond_pointlike(slocs, snorms, sss_params)

    plot_vars.append(dict(zip(('interm_noise_max', 'interm_noise_mean', 'interm_inf', 'interm_snr', 'interm_sss_cond'),
                              (interm_noise_max, interm_noise_mean, interm_inf, interm_snr, interm_sss_cond))))
print('done')



# %% save a dataset
import pickle
print(datadir)
outfn = 'opt_res_' + datadir.stem + '.pickle'
print(outfn)
pickle.dump(opt_res, open(outfn, 'wb'))
outfn = 'plot_vars_' + datadir.stem + '.pickle'
print(outfn)
pickle.dump(plot_vars, open(outfn, 'wb'))



# %% load the large datasets with different starting locs
import pickle
opt_res_outer = pickle.load(open('opt_res_3D_init_outer.pickle', 'rb'))
plot_vars_outer = pickle.load(open('plot_vars_3D_init_outer.pickle', 'rb'))
opt_res_inner = pickle.load(open('opt_res_3D_init_inner.pickle', 'rb'))
plot_vars_inner = pickle.load(open('plot_vars_3D_init_inner.pickle', 'rb'))
opt_res_middle = pickle.load(open('opt_res_3D_init_middle.pickle', 'rb'))
plot_vars_middle = pickle.load(open('plot_vars_3D_init_middle.pickle', 'rb'))
# select a single set
opt_res = opt_res_outer
plot_vars = plot_vars_outer


# %% load the datasets with varying Lint
opt_res_lint = dict()
plot_vars_lint = dict()
for lint in range(5, 11):
    opt_res_lint[lint] = pickle.load(open(f'opt_res_3D_lint{lint}.pickle', 'rb'))
    plot_vars_lint[lint] = pickle.load(open(f'plot_vars_3D_lint{lint}.pickle', 'rb'))








# %% like above, but extract sensor array geometries only
# for plotting sensor locations/normals during optimization
# (at some intermediate result)
# interm_idx is the intermediate result index 0..99
def _get_intermediate_sensordist(opt_res, interm_idx):
    locs_all = list()
    norms_all = list()
    angles_all = list()

    for res_ind, opt_re in enumerate(opt_res):
        n_iter = len(opt_re['interm_res'])
        sens_array = opt_re['sens_array']
        v, f, accept, tstamp = opt_re['interm_res'][interm_idx]
        slocs, snorms = sens_array._v2sens_geom(v)
        locs_all.append(slocs)
        norms_all.append(snorms)
        radials = (slocs.T / np.linalg.norm(slocs, axis=1)).T  # radial unit vectors at sensor locs
        dots = np.diag(radials @ snorms.T)
        dots = np.clip(dots, -1, 1)  # prevent overflow of 1 for arccos
        angles = np.arccos(dots) / np.pi * 180  # sensor normals vs. radial dir
        angles_all.append(angles)
    return locs_all, angles_all

# % Plot locations/orientations during optimizations
fig, axes = plt.subplots(2, 2);

for interm_idx, ax in zip([0, 10, 49, 99], axes.flat):
    _, angles_all = _get_intermediate_sensordist(opt_res, interm_idx)
    # % plot the angle distribution
    aa = np.array(angles_all).flatten()  # all runs
    # angles 90-180 deg -> fold into 0-90
    aa[np.where(aa > 90)] = 180 - aa[np.where(aa > 90)]
    bins = np.linspace(0, 90, 31)
    
    ax.hist(aa, bins=bins)
    if interm_idx in [49, 99]:
        ax.set_xlabel('Angle from outward radial ($\degree$)')
    if interm_idx in [0, 49]:
       ax.set_ylabel('Total N of sensors')
    if interm_idx == 0:
        ax.set_title('Starting condition')
    else:
        completed = 100 * interm_idx / 99 
        ax.set_title(f'{completed:.0f}% of iterations completed')
   
plt.tight_layout()
plt.savefig(f'angle_distribution_vs_iteration_outer.png')


# %% plot distance histogram - distance from helmet surface
plt.figure()
interm_idx = 99
locs_all, _ = _get_intermediate_sensordist(opt_res, interm_idx)
slocs = np.concatenate(locs_all)
bins = np.linspace(0, 100, 30)
negz = np.where(slocs[:, 2] < 0)[0]
slocs[negz, 2] = 0
surface_dist = np.linalg.norm(slocs, axis=1) - .15
surface_dist *= 1000
plt.hist(surface_dist, bins=bins)
plt.xlabel('Distance from inner surface (mm)')
plt.ylabel('Total N of sensors')
if interm_idx == 0:
    plt.title('Starting condition')
else:
    completed = 100 * interm_idx / 99 
    plt.title(f'{completed:.0f}% of iterations completed')
plt.savefig(f'angle_dist_at_{interm_idx}.png')
np.count_nonzero(surface_dist > 50) / len(surface_dist)


## Plot condition/total info etc.

#i3ht0983qy8ht
590



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
    #plt.plot(opt_re['iter_indx'], plot_var['interm_inf'], color='k', linewidth=.5)
    plt.plot(opt_re['iter_indx'], plot_var['interm_inf'])
# plt.title(f'Total information, {R_LEADFIELD=}, {DIPOLE_STR=}')
#plt.title('Total information')
plt.xlabel('N of iterations')
plt.ylabel('Total information (bits)')
plt.savefig('inf_vs_iter_2d.png')


# %% Plot error vs iteration - all data overlaid
plt.figure()
for opt_re, plot_var in zip(opt_res, plot_vars):
    plt.semilogy(opt_re['iter_indx'], plot_var['interm_noise_max']) # , color='b', linewidth=.5)
    plt.semilogy(opt_re['iter_indx'], plot_var['interm_noise_mean']) #, color='k', linewidth=.5)

plt.xlabel('N of iterations')
plt.ylabel('Noise amplification factor')
plt.legend(['Max. NA', 'Mean NA'])
plt.savefig('noise_vs_iter_2D.png')


# %% Plot NA - all data overlaid - comparison different Lin
fig, ax = plt.subplots()
liwidth = None
# data needs to be loaded first
for lint in opt_res_lint.keys():
    opt_res = opt_res_lint[lint]
    plot_vars = plot_vars_lint[lint]
    for opt_re, plot_var in zip(opt_res, plot_vars):
        label = fr'$L_\alpha={lint}$'
        plt.semilogy(opt_re['iter_indx'], plot_var['interm_noise_max'], '-', linewidth=liwidth, label=label)
        #plt.semilogy(opt_re['iter_indx'], plot_var['interm_noise_mean'], '--', linewidth=liwidth, label=label)
handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles), reversed(labels))
ax.set_xlabel('N of iterations')
ax.set_ylabel('Max. noise amplification factor')
plt.savefig('noise_amplification_compare_lin.png')



# %% Plot NA- all data overlaid - comparison of inner / middle / outer initial locs
plt.rcParams['figure.dpi'] = 150
fig, ax1 = plt.subplots()
NPLOT = 8  # how many runs to plot for each condition

liwidth = .8
ind = 0
for opt_re, plot_var in zip(opt_res_outer[:NPLOT], plot_vars_outer[:NPLOT]):
    label = 'Max. NA, init at outer surface' if ind == 0 else None
    ax1.semilogy(opt_re['iter_indx'], plot_var['interm_noise_max'], 'b', linewidth=liwidth, label=label)
    label = 'Mean NA, init at outer surface' if ind == 0 else None
    ax1.semilogy(opt_re['iter_indx'], plot_var['interm_noise_mean'], 'b--', linewidth=liwidth, label=label)
    ind += 1
ind = 0
for opt_re, plot_var in zip(opt_res_inner[:NPLOT], plot_vars_inner[:NPLOT]):
    label = 'Max. NA, init at inner surface' if ind == 0 else None    
    ax1.semilogy(opt_re['iter_indx'], plot_var['interm_noise_max'], 'r', linewidth=liwidth, label=label)
    label = 'Mean NA, init at inner surface' if ind == 0 else None    
    ax1.semilogy(opt_re['iter_indx'], plot_var['interm_noise_mean'], 'r--', linewidth=liwidth, label=label)
    ind += 1
ind = 0
for opt_re, plot_var in zip(opt_res_middle[:NPLOT], plot_vars_middle[:NPLOT]):
    label = 'Max. NA, init between surfaces' if ind == 0 else None    
    ax1.semilogy(opt_re['iter_indx'], plot_var['interm_noise_max'], 'g', linewidth=liwidth, label=label)
    label = 'Mean NA, init between surfaces' if ind == 0 else None    
    ax1.semilogy(opt_re['iter_indx'], plot_var['interm_noise_mean'], 'g--', linewidth=liwidth, label=label)
    ind += 1
ax1.legend()
ax1.set_xlabel('N of iterations')
ax1.set_ylabel('Noise amplification factor')
plt.savefig('noise_amplification_compare_starting_conds.png')


#%% plot total info- all data overlaid - comparison of inner / middle / outer initial locs
fig, ax2 = plt.subplots()
liwidth = .8
ind = 0
for opt_re, plot_var in zip(opt_res_outer[:NPLOT], plot_vars_outer[:NPLOT]):
    label = 'Total information, init at outer surface' if ind == 0 else None
    ax2.plot(opt_re['iter_indx'], plot_var['interm_inf'], 'b', linewidth=liwidth, label=label)
    ind += 1
ind = 0
for opt_re, plot_var in zip(opt_res_inner[:NPLOT], plot_vars_inner[:NPLOT]):
    label = 'Total information, init at inner surface' if ind == 0 else None
    ax2.plot(opt_re['iter_indx'], plot_var['interm_inf'], 'r', linewidth=liwidth, label=label)
    ind += 1
ind = 0
for opt_re, plot_var in zip(opt_res_middle[:NPLOT], plot_vars_middle[:NPLOT]):
    label = 'Total information, init between surfaces' if ind == 0 else None        
    ax2.plot(opt_re['iter_indx'], plot_var['interm_inf'], 'g', linewidth=liwidth, label=label)
    ind += 1
ax2.legend()
ax2.set_xlabel('N of iterations')
ax2.set_ylabel('Total info')
plt.tight_layout()
plt.savefig('total_info_compare_starting_conds.png')




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