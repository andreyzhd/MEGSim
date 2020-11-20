#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulate the effect of array coverage on the SSS basis.

author: Jussi (jnu@iki.fi)

"""
# %% init
import numpy as np
import matplotlib.pylab as plt
from mayavi import mlab
from IPython import get_ipython
from mne.preprocessing.maxwell import _sss_basis
from mne.transforms import _deg_ord_idx, _pol_to_cart, _cart_to_sph

from megsimutils.utils import (
    _normalized_basis,
    _idx_deg_ord,
    _deg_ord_idx,
    _normalized_basis,
    _vector_angles,
    spherepts_golden,
    _sssbasis_cond_pointlike,
    _mlab_points3d,
    _mlab_quiver3d,
    _random_unit,
)

from megsimutils.array_geometry import barbute


# set up IPython
ip = get_ipython()
# ip.magic("gui qt5")  # needed for mayavi plots
ip.magic("matplotlib qt")



# %%
Sc1, Sn1 = barbute(500, 500, .1, .1, 1.5 * np.pi)

Sc2, Sn2 = barbute(500, 500, .1, .1, 1.5 * np.pi)

Sc2 *= 10


Sc = np.row_stack((Sc1, Sc2))
Sn = np.row_stack((Sn1, Sn2))


_mlab_points3d(Sc, scale_factor=.01)

sss_params = {'origin': [0., 0., 0.], 'int_order': 16, 'ext_order': 3}
_sssbasis_cond_pointlike(Sc, Sn, sss_params)


sss_params = {'origin': [0., 0., 0.], 'int_order': 16, 'ext_order': 3}
_sssbasis_cond_pointlike(Sc1, Sn1, sss_params)


S, Sin, Sout = _normalized_basis(Sc, Sn, sss_params)


# %% plot some basis vecs
Lms = ((3, 1), (8, 1), (15, 1))
inds = [_deg_ord_idx(*Lm) for Lm in Lms]
plt.figure()
for ind in inds:
    plt.semilogy(Sin[:, ind])
legs = [f'{L=}, {m=}' for L, m in Lms]
plt.legend(legs)


