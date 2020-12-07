#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization of VSHs.

author: Jussi (jnu@iki.fi)

"""
# %% init
import numpy as np
from mayavi import mlab
from scipy.spatial import ConvexHull
from IPython import get_ipython

from megsimutils.viz import _mlab_trimesh
from megsimutils.array_geometry import spherical
from megsimutils.utils import _normalized_basis, _deg_ord_idx
from megsimutils.fileutils import _named_tempfile, _montage_figs


# %% make a dense radial array
NSENSORS = 10_000
rmags, nmags = spherical(NSENSORS, 1)


# %% evaluate some VSH functions
sss_params = {'origin': [0.0, 0.0, 0.0], 'int_order': 16, 'ext_order': 3}
S, Sin, Sout = _normalized_basis(rmags, nmags, sss_params)
tris = ConvexHull(rmags).simplices

# %% plot VSHs into a montage
# this is complicated by the fact that mayavi does not provide subplot functionality
# the workaround is to output each plot into a temporary .png file, then montage
# them using imagemagick

# fig parameters for .png files
FIG_BG_COLOR = (0.3, 0.3, 0.3)
FIGSIZE = (400, 300)
NCOLS_MAX = 11
assert NCOLS_MAX % 2 == 1  # we want it odd for symmetry
FONT_SIZE = FIGSIZE[0] / 500  # heuristic


degrees = [2, 4, 8, 10, 12, 14, 16]
# degrees = range(2, 17, 2)
# neglect some orders (make step larger than 1)
order_steps = np.ceil([(2 * d + 1) / (NCOLS_MAX - 1) for d in degrees]).astype(int)


def _append_empty_fig(fignames):
    """Create and append an empty .png fig to fignames"""
    fig = mlab.figure(bgcolor=FIG_BG_COLOR)
    fname = _named_tempfile(suffix='.png')
    mlab.savefig(fname, size=FIGSIZE, figure=fig)
    fignames.append(fname)


fignames = list()
mlab.options.offscreen = True

for deg, order_step in zip(degrees, order_steps):
    # this is to make sure l=0 is included
    orders_neg = list(range(-deg, 0, order_step))
    orders_pos = [-o for o in reversed(orders_neg)]
    orders_to_plot = orders_neg + [0] + orders_pos
    nempty = NCOLS_MAX - len(orders_to_plot)
    for x in range(int(nempty / 2)):
        _append_empty_fig(fignames)
    for ord in orders_to_plot:
        fig = mlab.figure(bgcolor=FIG_BG_COLOR)
        idx = _deg_ord_idx(deg, ord)
        _mlab_trimesh(rmags, tris, figure=fig, scalars=Sin[:, idx])
        mlab.title(f'l={deg}, m={ord}', size=FONT_SIZE)
        fname = _named_tempfile(suffix='.png')
        mlab.savefig(fname, size=FIGSIZE, figure=fig)
        fignames.append(fname)
    for x in range(int(nempty / 2)):
        _append_empty_fig(fignames)


mlab.options.offscreen = False

montage_fn = 'vsh_radial.png'
_montage_figs(fignames, montage_fn, ncols_max=NCOLS_MAX)

