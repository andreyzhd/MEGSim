"""
Iteratively optimize sensor array locations. Sensors are located on a dense grid.
Add one sensor at a time, selecting the best grid location. Grid can be quite
arbitrary (e.g. hockey-helmet-like).
"""
#%% Initializations
import pickle
import time
import numpy as np
from mne.preprocessing.maxwell import _sss_basis

from megsimutils.utils import spherepts_golden, xyz2pol
from megsimutils.optimize import _build_slicemap

N_CAND_LOCS = 1000 # Total number of candidate locations (over the whole sphere) before cutting the out the helmet
L = 9
N_SENS = 2 * L * (L+2)
OUT_FNAME = '/home/andrey/scratch/iter_opt.pkl'

def helmet(nlocs):
    pts = spherepts_golden(nlocs)
    r, theta, phi = xyz2pol(pts[:,0], pts[:,1], pts[:,2])
    helm_indx = ((phi>0) & (phi<np.pi)) | ((phi>(11/8)*np.pi) & (phi<(12/8)*np.pi))
    return pts[helm_indx, 0], pts[helm_indx, 2], pts[helm_indx, 1]

#%% Start measuring time
t_start = time.time()

#%% Build the helmet and compute VSH basis for all the candidate locations
x_helm, y_helm, z_helm = helmet(N_CAND_LOCS)
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
best_sens_indx = np.argmax(np.abs(S[:,0]))
sens_indx[0] = best_sens_indx
free_locs.remove(best_sens_indx)
best_cond_nums[0] = 1

for i in range(1, N_SENS):
    print('Placing sensor %i out of %i ...' % (i+1, N_SENS))
    cond_ns = np.ones(n_locs) * np.inf
    for j in free_locs:
        Sp = S[np.append(sens_indx[:i], j), :(i+1)]
        Sp /= np.linalg.norm(Sp, axis=0)
        cond_ns[j] = np.linalg.cond(Sp)

    best_sens_indx = np.argmin(cond_ns)
    sens_indx[i] = best_sens_indx
    free_locs.remove(best_sens_indx)
    best_cond_nums[i] = cond_ns[best_sens_indx]

assert len(np.unique(sens_indx)) == len(sens_indx)  # all the indices should be unique

print('Execution took %i seconds' % (time.time() - t_start))

#%% Save the results
fl = open(OUT_FNAME, 'wb')
pickle.dump((x_helm, y_helm, z_helm, sens_indx, best_cond_nums), fl)
fl.close()

