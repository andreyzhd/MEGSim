"""
Iteratively optimize sensor array locations. Sensors are located on a dense grid.
Add one sensor at a time, selecting the best grid location. Grid can be quite
arbitrary (e.g. hockey-helmet-like).
"""
#%% Initializations
import pickle
import time
from multiprocessing import Pool
from itertools import repeat
import numpy as np
from mne.preprocessing.maxwell import _sss_basis

from megsimutils.utils import hockey_helmet
from megsimutils.optimize import _build_slicemap

N_CAND_LOCS = 5000 # Total number of candidate locations (over the whole sphere) before cutting the out the helmet
L = 16
N_SENS = 2 * L * (L+2)
OUT_FNAME = '/home/andrey/scratch/iter_opt.pkl'
THREAD_POOL_SIZE = 2
RANDOM_CONTROL = False

#%% Start measuring time
t_start = time.time()

#%% Build the helmet and compute VSH basis for all the candidate locations
x_helm, y_helm, z_helm = hockey_helmet(N_CAND_LOCS)
helm = np.stack((x_helm, y_helm, z_helm), axis=1)

n_coils = len(x_helm)
bins = np.arange(n_coils, dtype=np.int64)
mag_mask = np.ones(n_coils, dtype=np.bool)

allcoils = (helm, helm, bins, n_coils, mag_mask, _build_slicemap(bins, n_coils))
sss_origin = np.array([0.0, 0.0, 0.0])  # origin of device coords
exp = {'origin': sss_origin, 'int_order': L, 'ext_order': 0}
S = _sss_basis(exp, allcoils)

#%% Iteratively search for the best sensor set
n_locs, n_comps = S.shape
sens_indx = -np.ones(N_SENS, dtype=int)
best_cond_nums = -np.ones(N_SENS)
free_locs = list(range(n_locs))

# place the first sensor at the maximum of the first component
print('Placing sensor 1 out of %i ...' % N_SENS)
if RANDOM_CONTROL:
    best_sens_indx = np.random.choice(free_locs)
else:
    best_sens_indx = np.argmax(np.abs(S[:,0]))
    
sens_indx[0] = best_sens_indx
free_locs.remove(best_sens_indx)
best_cond_nums[0] = 1

def _test_next_sensor(S, sens_indx, i, j):
    Sp = S[np.append(sens_indx[:i], j), :(i+1)]
    Sp /= np.linalg.norm(Sp, axis=0)
    return np.linalg.cond(Sp)

pool = Pool(THREAD_POOL_SIZE)

for i in range(1, N_SENS):
    print('Placing sensor %i out of %i ...' % (i+1, N_SENS))

    if RANDOM_CONTROL:
        best_sens_indx = np.random.choice(free_locs)
        best_cond_nums[i] = _test_next_sensor(S, sens_indx, i, best_sens_indx)
    else:
        cond_ns = pool.starmap(_test_next_sensor, zip(repeat(S), repeat(sens_indx), repeat(i), free_locs), chunksize=len(free_locs)//THREAD_POOL_SIZE)

        res_indx = np.argmin(cond_ns)
        best_cond_nums[i] = cond_ns[res_indx]
        best_sens_indx = free_locs[res_indx]

    sens_indx[i] = best_sens_indx
    free_locs.remove(best_sens_indx)

assert len(np.unique(sens_indx)) == len(sens_indx)  # all the indices should be unique

print('Execution took %i seconds' % (time.time() - t_start))

#%% Save the results
fl = open(OUT_FNAME, 'wb')
pickle.dump((x_helm, y_helm, z_helm, sens_indx, best_cond_nums), fl)
fl.close()

