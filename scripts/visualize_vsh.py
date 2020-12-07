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

from megsimutils.viz import _mlab_trimesh
from megsimutils.array_geometry import spherical
from megsimutils.utils import _normalized_basis, _deg_ord_idx
from megsimutils.fileutils import _named_tempfile, _montage_figs
from megsimutils.envutils import _ipython_setup


_ipython_setup()



# %% make a dense radial array and a corresponding mesh
NSENSORS = 10_000
rmags, nmags = spherical(NSENSORS, 1)
tris = ConvexHull(rmags).simplices  # for a closed surface, ConvexHull should work


# %% evaluate some VSH functions
sss_params = {'origin': [0.0, 0.0, 0.0], 'int_order': 16, 'ext_order': 3}
S, Sin, Sout = _normalized_basis(rmags, nmags, sss_params)

# %% single plot
fig = mlab.figure()
_mlab_trimesh(rmags, tris, figure=fig, scalars=Sin[:, 2])


# %% plot VSHs into a montage
# this is complicated by the fact that mayavi does not provide subplot functionality
# the workaround is to output each plot into a temporary .png file, then montage
# them using imagemagick

# fig parameters for .png files
FIG_BG_COLOR = (0.3, 0.3, 0.3)  # RGB values for background color
FIGSIZE = (400, 300)  # size of each individual figure
NCOLS = 11  # how many columns in the montaged plot
assert NCOLS % 2 == 1  # we want it odd for symmetry
FONT_SIZE = FIGSIZE[0] / 500  # heuristic for font size
MONTAGE_FILENAME = 'vsh_radial.png'  # filename for the resulting montage

# specify the VSH degrees to plot
degrees = [2, 4, 8, 10, 12, 14, 16]
# neglect some orders (make step larger than 1)
order_steps = np.ceil([(2 * d + 1) / (NCOLS - 1) for d in degrees]).astype(int)


def _append_empty_fig(fignames):
    """Create and append an empty .png fig to fignames"""
    fig = mlab.figure(bgcolor=FIG_BG_COLOR)
    fname = _named_tempfile(suffix='.png')
    mlab.savefig(fname, size=FIGSIZE, figure=fig)
    fignames.append(fname)


# prevent mlab from opening on-screen plots
mlab.options.offscreen = True

fignames = list()
# create the individual VSH plots
for deg, order_step in zip(degrees, order_steps):
    # this is to make sure l=0 is included
    orders_neg = list(range(-deg, 0, order_step))
    orders_pos = [-o for o in reversed(orders_neg)]
    orders_to_plot = orders_neg + [0] + orders_pos
    nempty = NCOLS - len(orders_to_plot)
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

# create a large montage of all the figures
_montage_figs(fignames, MONTAGE_FILENAME, ncols_max=NCOLS)

# restore on-screen plotting
mlab.options.offscreen = False


# %% try visualizations of the vector-valued field

# let's make a grid of orthogonal unit vectors
