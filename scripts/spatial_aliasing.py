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
import matplotlib.pyplot as plt

from megsimutils.viz import _mlab_trimesh, _mlab_points3d
from megsimutils.array_geometry import spherical
from megsimutils.utils import _normalized_basis, _deg_ord_idx
from megsimutils.fileutils import _named_tempfile, _montage_figs
from megsimutils.envutils import _ipython_setup


_ipython_setup()



def _stupid_sphere(N):
    """Make a simple sphere"""
    Sc = list()
    for phi in np.linspace(0, 2*np.pi, N):
        for th in np.linspace(1/N, np.pi, N, endpoint=False):
            Sc.append([np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)])
    Sc = np.array(Sc)
    return Sc, Sc


sss_params = {'origin': [0.0, 0.0, 0.0], 'int_order': 32, 'ext_order': 3}


# %% make a simple spherical array
NSENSORS = 11000
rmags, nmags = _stupid_sphere(int(np.sqrt(NSENSORS)))
tris = ConvexHull(rmags).simplices  # for a closed surface, ConvexHull should work
S, Sin, Sout = _normalized_basis(rmags, nmags, sss_params)



# %% make "golden ratio" array
NSENSORS = 1000
rmags, nmags = spherical(NSENSORS, 1, angle=4 * np.pi)
tris = ConvexHull(rmags).simplices  # for a closed surface, ConvexHull should work
S, Sin, Sout = _normalized_basis(rmags, nmags, sss_params)


# %% test
L = 24
m = L
fig = mlab.figure()
idx = _deg_ord_idx(L, m)
mesh = _mlab_trimesh(rmags, tris, figure=fig, scalars=Sin[:, idx])
_mlab_points3d(rmags, scale_factor=.01, color=(0., 0., 0.))



# %% plot aliasing
L = 4
Ls = range(4, 28, 4)
VIEW = [45, 85, 4.7, np.array([0., 0., 0.14])]  # mlab camera view
TRUEN = rmags.shape[0]
# prevent mlab from opening on-screen plots
mlab.options.offscreen = True

for L in Ls:
    m = L
    fig = mlab.figure()
    idx = _deg_ord_idx(L, m)
    mesh = _mlab_trimesh(rmags, tris, figure=fig, scalars=Sin[:, idx])

    _mlab_points3d(rmags, scale_factor=.01, color=(0., 0., 0.))
    vi = [45, 55, 4.45, np.array([0., 0., 0.14])]
    mlab.view(*vi)
    mlab.title(f'{L=}, {m=}, {TRUEN} sensors', size=.75, height=.9)
    mlab.view(*VIEW)

    mlab.savefig(f'aliasing_{L:03}_{m}_{TRUEN}.png')


# restore on-screen plotting
mlab.options.offscreen = False


# %% condition numbers, irregular array

plt.figure()

NSENSORS = 1000
ARRAY_DESC = 'irregular'

rmags, nmags = spherical(NSENSORS, 1, angle=4 * np.pi)
tris = ConvexHull(rmags).simplices  # for a closed surface, ConvexHull should work
S, Sin, Sout = _normalized_basis(rmags, nmags, sss_params)

conds = list()
for L in Ls:
    sss_params = {'origin': [0.0, 0.0, 0.0], 'int_order': L, 'ext_order': 0}
    S, Sin, Sout = _normalized_basis(rmags, nmags, sss_params)
    conds.append(np.linalg.cond(Sin))

plt.plot(Ls, np.log10(conds))

NSENSORS = 10000
rmags, nmags = spherical(NSENSORS, 1, angle=4 * np.pi)
tris = ConvexHull(rmags).simplices  # for a closed surface, ConvexHull should work
S, Sin, Sout = _normalized_basis(rmags, nmags, sss_params)

conds = list()
for L in Ls:
    sss_params = {'origin': [0.0, 0.0, 0.0], 'int_order': L, 'ext_order': 0}
    S, Sin, Sout = _normalized_basis(rmags, nmags, sss_params)
    conds.append(np.linalg.cond(Sin))

plt.xticks(Ls)
plt.plot(Ls, np.log10(conds))
plt.xlabel('Lin')
plt.ylabel('log10 of cond(Sin)')
plt.title(f'Condition number vs. Lin, {ARRAY_DESC} array')
plt.legend(['undersampling', 'proper sampling'])
plt.savefig(f'basis_cond_{ARRAY_DESC}.png', dpi=200, transparent=False, facecolor='w')



# %% condition numbers, regular array

plt.figure()

ARRAY_DESC = 'regular'


NSENSORS = 1100
rmags, nmags = _stupid_sphere(int(np.sqrt(NSENSORS)))
tris = ConvexHull(rmags).simplices  # for a closed surface, ConvexHull should work
S, Sin, Sout = _normalized_basis(rmags, nmags, sss_params)

conds = list()
for L in Ls:
    sss_params = {'origin': [0.0, 0.0, 0.0], 'int_order': L, 'ext_order': 0}
    S, Sin, Sout = _normalized_basis(rmags, nmags, sss_params)
    conds.append(np.linalg.cond(Sin))

plt.plot(Ls, np.log10(conds))

NSENSORS = 11000
rmags, nmags = _stupid_sphere(int(np.sqrt(NSENSORS)))
tris = ConvexHull(rmags).simplices  # for a closed surface, ConvexHull should work
S, Sin, Sout = _normalized_basis(rmags, nmags, sss_params)

conds = list()
for L in Ls:
    sss_params = {'origin': [0.0, 0.0, 0.0], 'int_order': L, 'ext_order': 0}
    S, Sin, Sout = _normalized_basis(rmags, nmags, sss_params)
    conds.append(np.linalg.cond(Sin))

plt.xticks(Ls)
plt.plot(Ls, np.log10(conds))
plt.xlabel('Lin')
plt.ylabel('log10 of cond(Sin)')
plt.title(f'Condition number vs. Lin, {ARRAY_DESC} array')
plt.legend(['undersampling', 'proper sampling'])
plt.savefig(f'basis_cond_{ARRAY_DESC}.png', dpi=200, transparent=False, facecolor='w')



# %% plot VSHs into a montage
# this is complicated by the fact that mayavi does not provide subplot functionality
# the workaround is to output each plot into a temporary .png file, then montage
# them using imagemagick

# fig parameters for .png files
FIG_BG_COLOR = (0.3, 0.3, 0.3)  # RGB values for background color
FIGSIZE = (400, 300)  # size of each individual figure
NCOLS = 9  # how many columns in the montaged plot
assert NCOLS % 2 == 1  # we want it odd for symmetry
FONT_SIZE = 1.75  # heuristic for font size
TITLE_HEIGHT = .85
MONTAGE_FILENAME = 'vsh_radial.png'  # filename for the resulting montage
VIEW = [45, 55, 4.7, np.array([0., 0., 0.14])]  # mlab camera view

# specify the VSH degrees to plot
degrees = [2, 4, 8, 13, 16]
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
        mlab.view(*VIEW)
        mlab.title(f'l={deg}, m={ord}', size=FONT_SIZE, height=TITLE_HEIGHT)
        # hack for spacing
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
