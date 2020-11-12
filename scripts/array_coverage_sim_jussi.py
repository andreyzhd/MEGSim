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
    _vector_angles,
    spherepts_golden,
    _sssbasis_cond_pointlike,
    _mlab_points3d,
    _mlab_quiver3d,
    _random_unit,
)


# set up IPython
ip = get_ipython()
# ip.magic("gui qt5")  # needed for mayavi plots
ip.magic("matplotlib qt")


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
polars = np.linspace(np.pi / 2, np.pi, Narrays)  # the polar angle limits
solids = 2 * np.pi * (1 - np.cos(polars))  # corresponding solid angles
conds = list()

# create spherical caps of varying polar angle, compute the condition numbers
for polar_lim, solid in zip(polars, solids):
    rmags = spherepts_golden(Nsensors, solid)
    nmags = rmags.copy()
    # optionally, flip Nflip sensors into a random tangential orientation
    for k in to_flip:
        nmags[k, :] = np.cross(nmags[k, :], _random_unit(3))
        nmags[k, :] /= np.linalg.norm(nmags[k, :])
    cond = _sssbasis_cond_pointlike(rmags, nmags, sss_params, cond_type=cond_type)
    conds.append(cond)


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


# %% cylindrical array


def barbute(nsensors_upper, nsensors_lower, array_radius, height_lower, phispan_lower):
    """Create an Italian war helmet.

    The array consists of spherical upper part (positive z)
    and cylindrical lower part (negative z). """

    # make the upper part
    Sc1 = spherepts_golden(nsensors_upper, angle=2 * np.pi)
    Sn1 = Sc1.copy()
    Sc1 *= array_radius

    # add some neg-z sensors on a cylindrical surface
    if nsensors_lower > 0:
        # estimate N of sensors in z and phi directions, so that total number
        # of sensors is approximately correct
        Nh = np.sqrt(nsensors_lower) * np.sqrt(
            height_lower / (phispan_lower * array_radius)
        )
        Nphi = nsensors_lower / Nh
        Nh = int(np.round(Nh))
        Nphi = int(np.round(Nphi))
        phis = np.linspace(0, phispan_lower, Nphi, endpoint=False)  # the phi angles
        zs = np.linspace(-height_lower, 0, Nh, endpoint=False)
        Sc2 = list()
        for phi in phis:
            for z in zs:
                Sc2.append([array_radius * np.cos(phi), array_radius * np.sin(phi), z])
        Sc2 = np.array(Sc2)
        Sn2 = Sc2.copy()
        Sn2[:, 2] = 0  # make normal vectors cylindrical
        Sn2 = (Sn2.T / np.linalg.norm(Sn2, axis=1)).T
        Sc = np.row_stack((Sc1, Sc2))
        Sn = np.row_stack((Sn1, Sn2))
    else:
        Sc = Sc1
        Sn = Sn1

    # optionally, make 90 degree flips for a subset of sensor normals
    FLIP_SENSORS = 0
    if FLIP_SENSORS:
        print(f'*** flipping {FLIP_SENSORS} sensors')
        to_flip = np.random.choice(Sc.shape[0], FLIP_SENSORS, replace=False)
        for k in to_flip:
            flipvec = _random_unit(3)
            Sn[k, :] = np.cross(Sn[k, :], flipvec)

    return Sc, Sn


# %% evaluate several barbute helmets
nsensors_upper = 500
nsensors_lower = 500
height_min = 0.05
height_max = 0.15
nheights = 8
nphis = 30
array_radius = 0.1

LIN, LOUT = 16, 3
sss_origin = np.array([0.0, 0.0, 0.0])  # origin of device coords
sss_params = {'origin': sss_origin, 'int_order': LIN, 'ext_order': LOUT}

heights_lower = np.linspace(height_min, height_max, nheights)
phispans_lower = np.linspace(2 * np.pi / 10, 2 * np.pi, nphis)

conds = np.zeros((nheights, nphis))
for i, height_lower in enumerate(heights_lower):
    for j, phispan_lower in enumerate(phispans_lower):
        rmags, nmags = barbute(
            nsensors_upper, nsensors_lower, array_radius, height_lower, phispan_lower
        )
        cond = _sssbasis_cond_pointlike(rmags, nmags, sss_params, cond_type='int')
        conds[i, j] = cond


# %% plot conds for the barbutes
fig, ax = plt.subplots()
ax.plot(phispans_lower / np.pi, np.log10(conds.T))
ax.set_ylabel('Condition number (log10)')
ax.set_xlabel('Azimuthal coverage of lower part / pi')
heights_legend = ['%.2f m' % h for h in heights_lower]
ax.legend(heights_legend, title='Height')
title = 'Effect of cylindrical part height & coverage for barbute helmets'
title += f'\n(Nsensors={nsensors_upper} + {nsensors_lower}, Lint = {LIN})'
ax.set_title(title)


# %% plot an exemplary barbute
nsensors_upper = 500
nsensors_lower = 500
array_radius = 0.1
height_lower = 0.1
phispan_lower = 1.8 * np.pi
rmags, nmags = barbute(
    nsensors_upper, nsensors_lower, array_radius, height_lower, phispan_lower
)
fig = mlab.figure()
_mlab_points3d(rmags, figure=fig, scale_factor=0.006)
_mlab_quiver3d(rmags, nmags, figure=fig)


# %% confusion matrix
S, Sin, Sout = _normalized_basis(rmags, nmags, sss_params)
# _mlab_quiver3d(rmags, nmags)
nvecs = Sin.shape[1]
anmat = np.zeros((nvecs, nvecs))
for k in range(nvecs):
    for l in range(nvecs):
        anmat[k, l] = _vector_angles(Sin[:, k], Sin[:, l])
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
# _mlab_quiver3d(rmags, nmags)
S, Sin, Sout = _normalized_basis(rmags_strip, nmags_strip, sss_params)
inds = [k for k in range(20) if _idx_deg_ord(k)[1] == 0]
tups = ['l=%d, m=%d' % _idx_deg_ord(k) for k in inds]
plt.plot(polars / np.pi, Sin[:, inds])
plt.legend(tups)
plt.title('Polar dependency of some VSHs')
plt.xlabel('Polar angle / pi')
plt.ylabel('VSH radial projection')

