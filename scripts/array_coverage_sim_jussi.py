#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulate the effect of array coverage on the SSS basis.

author: Jussi (jnu@iki.fi)

"""

# %%
import numpy as np
import matplotlib.pylab as plt
from mayavi import mlab
from IPython import get_ipython
from mne.preprocessing.maxwell import _sss_basis
from mne.transforms import _deg_ord_idx

from megsimutils import utils



def _random_unit(N):
    """Return random unit vector in N-dimensional space"""
    v = np.random.randn(N)
    return v / np.linalg.norm(v)


def _prep_mf_coils_pointlike(rmags, nmags):
    """Prepare the coil data for pointlike magnetometers.
    
    rmags, nmags are sensor locations and normals respectively, with shape (N,3)
    """
    n_coils = rmags.shape[0]
    mag_mask = np.ones(n_coils).astype(bool)
    slice_map = {k: slice(k, k + 1, None) for k in range(n_coils)}
    bins = np.arange(n_coils)
    return rmags, nmags, bins, n_coils, mag_mask, slice_map


def _sssbasis_cond_pointlike(rmags, nmags, sss_params, cond_type='int'):
    """Calculate basis matrix condition for a pointlike basis.

    cond : str
        Which condition number to return. 'total' for whole basis, 'int' for
        internal basis, 'l_split' for each L order separately, 'l_cumul' for
        cumulative L orders.
    """
    allcoils = _prep_mf_coils_pointlike(rmags, nmags)
    S = _sss_basis(sss_params, allcoils)
    S /= np.linalg.norm(S, axis=0)  # normalize basis
    Lin = sss_params['int_order']
    nvecs_in = Lin ** 2 + 2 * Lin
    Sin = S[:, :nvecs_in]
    if cond_type == 'total':
        cond = np.linalg.cond(S)
    elif cond_type == 'int':
        cond = np.linalg.cond(Sin)
    elif cond_type == 'l_split' or cond_type == 'l_cumul':
        cond = list()
        for L in np.arange(1, Lin + 1):
            ind0 = _deg_ord_idx(L, -L) if cond_type == 'l_split' else 0
            ind1 = _deg_ord_idx(L, L)
            cond.append(np.linalg.cond(Sin[:, ind0 : ind1 + 1]))
    else:
        raise ValueError('invalid cond argument')
    return cond


# set up IPython
ip = get_ipython()
# ip.magic("gui qt5")  # needed for mayavi plots
ip.magic("matplotlib qt")


# %% run the simulation

# adjustable params
LIN, LOUT = 8, 3
sss_origin = np.array([0.0, 0.0, 0.0])  # origin of device coords
Nsensors = 300  # how many sensors in total
Nflip = 150  # how many tangential sensors out of total
Narrays = 100  # how many different arrays to create (=solid angle spacing)
Nnegs = 0  # how many 'low-lying reference sensors' to include
# which condition number to calculate; see _sssbasis_cond_pointlike above
cond_type = 'l_cumul'


to_flip = np.random.choice(Nsensors - 5, Nflip, replace=False)
nvecs_in = LIN ** 2 + 2 * LIN
sss_params = {'origin': sss_origin, 'int_order': LIN, 'ext_order': LOUT}
polars = np.linspace(np.pi / 4, np.pi, Narrays)  # the polar angle limits
solids = 2 * np.pi * (1 - np.cos(polars))  # corresponding solid angles
conds = list()

# create spherical caps of varying polar angle, compute the condition numbers
for polar_lim, solid in zip(polars, solids):
    # make the spherical cap
    npts = int(Nsensors * 4 * np.pi / solid)
    pts = utils.spherepts_golden(npts)
    polar = np.arccos(pts[:, 2])  # polar angle for each sensor
    pts_in_cap = np.where(polar <= polar_lim)[0]  # sensors included in the cap
    # find some 'low-lying reference' sensors and add them (rather, make sure they
    # are included in the array)
    zneg_inds = np.where(pts[:, 2] < -0.5)[0]
    pts_zneg = zneg_inds[np.random.choice(len(zneg_inds), Nnegs, replace=False)]
    pts_to_use = np.union1d(pts_zneg, pts_in_cap)
    rmags = pts[pts_to_use, :]
    nmags = rmags.copy()
    # flip Nflip sensors into a random tangential orientation
    for k in to_flip:
        nmags[k, :] = np.cross(nmags[k, :], _random_unit(3))
    cond = _sssbasis_cond_pointlike(rmags, nmags, sss_params, cond_type=cond_type)
    conds.append(cond)


# %% plot
fig, ax = plt.subplots()
LOGMODE = False if cond_type == 'l_split' else True
conds = np.array(conds)
if LOGMODE:
    ax.plot(solids / np.pi, np.log10(conds))
    ylabel = 'log10 of condition number'
else:
    ax.plot(solids / np.pi, conds)
    ylabel = 'condition number'
ax.set_ylabel(ylabel)
ax.set_xlabel('solid angle / pi')
if conds.ndim > 1:
    fig.legend(np.arange(1, LIN + 1))
if cond_type == 'l_split':
    title = 'Condition for individual Lin orders'
elif cond_type == 'l_cumul':
    title = 'Condition for cumulative Lin orders'
elif cond_type == 'total':
    title = 'Condition for total basis'
elif cond_type == 'int':
    title = 'Condition for internal basis'
fig.suptitle(title)

# %%
