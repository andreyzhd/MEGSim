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
from scipy.spatial import ConvexHull, Delaunay


from megsimutils import utils

def _random_unit(N):
    """Return random unit vector in N-dimensional space"""
    v = np.random.randn(N)
    return v / np.linalg.norm(v)


def _idx_deg_ord(idx):
    """Returns (degree, order) tuple for a given multipole index."""
    # this is just an ugly inverse of _deg_ord_idx, do not expect speed
    for deg in range(1, 20):
        for ord in range(-deg, deg + 1):
            if _deg_ord_idx(deg, ord) == idx:
                return deg, ord
    return None


def _prep_mf_coils_pointlike(rmags, nmags):
    """Prepare the coil data for pointlike magnetometers.
    
    rmags, nmags are sensor locations and normals respectively, with shape (N,3)
    """
    n_coils = rmags.shape[0]
    mag_mask = np.ones(n_coils).astype(bool)
    slice_map = {k: slice(k, k + 1, None) for k in range(n_coils)}
    bins = np.arange(n_coils)
    return rmags, nmags, bins, n_coils, mag_mask, slice_map


def _normalized_basis(rmags, nmags, sss_params):
    """Compute normalized SSS basis matrices for a pointlike array."""
    allcoils = _prep_mf_coils_pointlike(rmags, nmags)
    S = _sss_basis(sss_params, allcoils)
    S /= np.linalg.norm(S, axis=0)  # normalize basis
    nvecs_in = sss_params['int_order'] ** 2 + 2 * sss_params['int_order']
    Sin, Sout = S[:, :nvecs_in], S[:, nvecs_in:]
    return S, Sin, Sout


def _sssbasis_cond_pointlike(rmags, nmags, sss_params, cond_type='int'):
    """Calculate basis matrix condition for a pointlike array.

    cond : str
        Which condition number to return. 'total' for whole basis, 'int' for
        internal basis, 'l_split' for each L order separately, 'l_cumul' for
        cumulative L orders, 'single' for individual basis vectors.
    """
    Lin = sss_params['int_order']
    S, Sin, _ = _normalized_basis(rmags, nmags, sss_params)
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
    elif cond_type == 'single':
        cond = list()
        for v in np.arange(nvecs_in):
            cond.append(np.linalg.cond(Sin[:, 0 : v + 1]))
    else:
        raise ValueError('invalid cond argument')
    return cond


def _mlab_points3d(rr, *args, **kwargs):
    """Plots points.
    rr : (N x 3) array-like
        The locations of the vectors.
    Note that the api to mayavi points3d is weird, there is no way to specify colors and sizes
    individually. See:
    https://stackoverflow.com/questions/22253298/mayavi-points3d-with-different-size-and-colors
    """
    vx, vy, vz = rr[:, 0], rr[:, 1], rr[:, 2]
    return mlab.points3d(vx, vy, vz, *args, **kwargs)


def _mlab_quiver3d(rr, nn, **kwargs):
    """Plots vector field as arrows.
    rr : (N x 3) array-like
        The locations of the vectors.
    nn : (N x 3) array-like
        The vectors.
    """
    vx, vy, vz = rr[:, 0], rr[:, 1], rr[:, 2]
    u, v, w = nn[:, 0], nn[:, 1], nn[:, 2]
    return mlab.quiver3d(vx, vy, vz, u, v, w, **kwargs)


def _mlab_trimesh(pts, tris, **kwargs):
    """Plots trimesh specified by pts and tris into given figure.
    pts : (N x 3) array-like
    """
    x, y, z = pts.T
    return mlab.triangular_mesh(x, y, z, tris, **kwargs)


def _delaunay_tri(rr):
    """Surface triangularization based on 2D proj and Delaunay"""
    # this is a straightforward projection to xy plane
    com = rr.mean(axis=0)
    rr = rr - com
    xy = _pol_to_cart(_cart_to_sph(rr)[:, 1:][:, ::-1])
    # do Delaunay for the projection and hope for the best
    return Delaunay(xy).simplices


# set up IPython
ip = get_ipython()
# ip.magic("gui qt5")  # needed for mayavi plots
ip.magic("matplotlib qt")


# %% confusion matrix
S, Sin, Sout = _normalized_basis(rmags, nmags, sss_params)
# _mlab_quiver3d(rmags, nmags)
nvecs = Sin.shape[1]
anmat = np.zeros((nvecs, nvecs))
for k in range(nvecs):
    for l in range(nvecs):
        anmat[k, l] = utils._vector_angles(Sin[:, k], Sin[:, l])
        if k != l and anmat[k, l] < 80 and not anmat[l, k]:
            print('%s and %s are confused' % (_idx_deg_ord(k), _idx_deg_ord(l)))
plt.figure()
plt.imshow(anmat)
plt.colorbar()
plt.title('Confusion matrix for coverage of 4*pi')

# l, m = 1, 1
# ind = _deg_ord_idx(l, m)
# sig = Sin[:, ind]
# tri = _delaunay_tri(rmags)
# _mlab_trimesh(rmags, tri, scalars=sig)



# %% make 'strip array' to investigate polar sampling of VSHs
polars = np.linspace(0, np.pi, 100)
phi = 90 / 180 * np.pi  # azimuth
zs = np.cos(polars)
xs = np.sin(polars) * np.cos(phi)
ys = np.sin(polars) * np.sin(phi)
rmags_strip = np.column_stack((xs, ys, zs))
nmags_strip = rmags_strip.copy()
nmags_strip = (nmags_strip.T / np.linalg.norm(nmags_strip, axis=1)).T
#_mlab_quiver3d(rmags, nmags)
S, Sin, Sout = _normalized_basis(rmags_strip, nmags_strip, sss_params)
inds = [k for k in range(20) if _idx_deg_ord(k)[1] == 0]
tups = ['l=%d, m=%d' % _idx_deg_ord(k) for k in inds]
plt.plot(polars / np.pi, Sin[:, inds])
plt.legend(tups)
plt.title('Polar dependency of some VSHs')
plt.xlabel('Polar angle / pi')
plt.ylabel('VSH radial projection')


# %% effect of array coverage (solid angle) on condition number

# adjustable params
LIN, LOUT = 16, 3
sss_origin = np.array([0.0, 0.0, 0.0])  # origin of device coords
Nsensors = 1000  # how many sensors in total
Nflip = 0  # how many tangential sensors out of total
Narrays = 100  # how many different arrays to create (=solid angle spacing)
Nnegs = 0  # how many 'low-lying reference sensors' to include
# which condition number to calculate; see _sssbasis_cond_pointlike above
cond_type = 'int'

to_flip = np.random.choice(Nsensors - 5, Nflip, replace=False)
nvecs_in = LIN ** 2 + 2 * LIN
sss_params = {'origin': sss_origin, 'int_order': LIN, 'ext_order': LOUT}
polars = np.linspace(np.pi / 4, np.pi, Narrays)  # the polar angle limits
solids = 2 * np.pi * (1 - np.cos(polars))  # corresponding solid angles
conds = list()

# create spherical caps of varying polar angle, compute the condition numbers
for polar_lim, solid in zip(polars, solids):
    rmags = utils.spherepts_golden(Nsensors, solid)
    nmags = rmags.copy()
    # flip Nflip sensors into a random tangential orientation
    for k in to_flip:
        nmags[k, :] = np.cross(nmags[k, :], _random_unit(3))
        nmags[k, :] /= np.linalg.norm(nmags[k, :])
    cond = _sssbasis_cond_pointlike(rmags, nmags, sss_params, cond_type=cond_type)
    conds.append(cond)


# %% plot ips
fig = mlab.figure()
_mlab_points3d(rmags, figure=fig, scale_factor=0.06)


# %% plot the condition numbers
fig, ax = plt.subplots()
LOGMODE = False if cond_type == 'l_split' else True
conds = np.array(conds)
if LOGMODE:
    ax.plot(solids / np.pi, np.log10(conds))
    ylabel = 'Condition number (log10)'
else:
    ax.plot(solids / np.pi, conds)
    ylabel = 'Condition number'
ax.set_ylabel(ylabel)
ax.set_xlabel('Array coverage, solid angle / pi')
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
elif cond_type == 'single':
    title = 'Condition for single cumulative'
title += f', Nsensors={Nsensors}'
title += f', Lint={LIN}'
title += '\n(spherical array, pointlike radial sensors)'
fig.suptitle(title)

