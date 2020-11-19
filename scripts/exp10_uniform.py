"""
Uniformly distribute sensors over a helmet and compute condition number.
"""
#%% Inits
import pickle

import numpy as np
from mne.preprocessing.maxwell import _sss_basis

from megsimutils.utils import hockey_helmet
from megsimutils.optimize import _build_slicemap

L = 16
N_SENS = 2 * L * (L+2)

CHIN_STRAP_ANGLES = (0, np.pi/8, np.pi/4, np.pi/16, 3*np.pi/16)
INNER_R = 0.15

OUT_FNAME_TMPL = '/home/andrey/scratch/uniform_chin_%0.3frad.pkl'

#%% Do the job
min_n_sens = L * (L+2)

for chin_strap_angle in CHIN_STRAP_ANGLES:
    next_n_sens = min_n_sens
    loc_dens = min_n_sens
    best_cond_nums = -np.ones(N_SENS)

    while next_n_sens <= N_SENS:
        x_sphere, y_sphere, z_sphere, helm_indx = hockey_helmet(loc_dens, chin_strap_angle=chin_strap_angle, inner_r=INNER_R)
        loc_dens += 1

        if np.count_nonzero(helm_indx) < next_n_sens:
            continue

        # Found a new helmet configuration
        helm = np.stack((x_sphere[helm_indx], y_sphere[helm_indx], z_sphere[helm_indx]), axis=1)

        bins = np.arange(next_n_sens, dtype=np.int64)
        mag_mask = np.ones(next_n_sens, dtype=np.bool)

        allcoils = (helm, helm, bins, next_n_sens, mag_mask, _build_slicemap(bins, next_n_sens))
        sss_origin = np.array([0.0, 0.0, 0.0])  # origin of device coords
        exp = {'origin': sss_origin, 'int_order': L, 'ext_order': 0}
        S = _sss_basis(exp, allcoils)
        S /= np.linalg.norm(S, axis=0)
        best_cond_nums[next_n_sens-1] = np.linalg.cond(S)

        next_n_sens += 1

    best_cond_nums[:min_n_sens-1] = best_cond_nums[min_n_sens-1] # Pad the condition number values for small numbers of sensors

    # Save the results
    fl = open(OUT_FNAME_TMPL % chin_strap_angle, 'wb')
    pickle.dump(({'x_sphere' : x_sphere,
                'y_sphere' : y_sphere,
                'z_sphere' : z_sphere,
                'helm_indx' : helm_indx,
                'sens_indx' : np.arange(N_SENS),
                'best_cond_nums' : best_cond_nums,
                'L' : L,
                'RANDOM_CONTROL' : False,
                'chin_strap_angle' : chin_strap_angle}), fl)
    fl.close()
