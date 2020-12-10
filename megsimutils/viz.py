#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization functions for megsim.

author: Jussi (jnu@iki.fi)
"""

import mne
import numpy as np
import trimesh
from mayavi import mlab
from mne.transforms import _cart_to_sph, _pol_to_cart, apply_trans
from scipy.spatial import ConvexHull, Delaunay

# the following wrappers exist mostly to allow direct passing of (Nx3) points
# matrices as args


def _mlab_trimesh(pts, tris, **kwargs):
    """Plots trimesh specified by pts and tris into given figure.
    pts : (N x 3) array-like
    """
    x, y, z = pts.T
    return mlab.triangular_mesh(x, y, z, tris, **kwargs)


def _mlab_mesh(pts, **kwargs):
    """Plots trimesh specified by pts and tris into given figure.
    pts : (N x 3) array-like
    """
    x, y, z = pts.T
    return mlab.mesh(x, y, z, **kwargs)


def _mlab_quiver3d(rr, nn, **kwargs):
    """Plots vector field as arrows.
    rr : (N x 3) array-like
        The locations of the vectors.
    nn : (N x 3) array-like
        The vectors.
    """
    vx, vy, vz = rr.T
    u, v, w = nn.T
    return mlab.quiver3d(vx, vy, vz, u, v, w, **kwargs)


def _mlab_points3d(rr, *args, **kwargs):
    """Plots points.
    rr : (N x 3) array-like
        The locations of the vectors.
    Note that the api to mayavi points3d is weird, there is no way to specify colors and sizes
    individually. See:
    https://stackoverflow.com/questions/22253298/mayavi-points3d-with-different-size-and-colors
    """
    vx, vy, vz = rr.T
    return mlab.points3d(vx, vy, vz, *args, **kwargs)


def _reorder_ccw(rrs, tris):
    """Reorder tris of a convex hull to be wound counter-clockwise."""
    # This ensures that rendering with front-/back-face culling works properly
    com = np.mean(rrs, axis=0)
    rr_tris = rrs[tris]
    dirs = np.sign(
        (
            np.cross(rr_tris[:, 1] - rr_tris[:, 0], rr_tris[:, 2] - rr_tris[:, 0])
            * (rr_tris[:, 0] - com)
        ).sum(-1)
    ).astype(int)
    return np.array([t[::d] for d, t in zip(dirs, tris)])


def _mne_tri(rr):
    """Surface triangularization, stolen from mne-python.
    XXX: this is somehow broken, see _delaunay_tri which works"""
    rr = rr[np.unique(ConvexHull(rr).simplices)]
    com = rr.mean(axis=0)
    xy = _pol_to_cart(_cart_to_sph(rr - com)[:, 1:][:, ::-1])
    return _reorder_ccw(rr, Delaunay(xy).simplices)


def _delaunay_tri(rr):
    """Surface triangularization based on 2D proj and Delaunay"""
    # this is a straightforward projection to xy plane
    com = rr.mean(axis=0)
    rr = rr - com
    xy = _pol_to_cart(_cart_to_sph(rr)[:, 1:][:, ::-1])
    # do Delaunay for the projection and hope for the best
    return Delaunay(xy).simplices


def _my_tri(rr):
    """Another attempt at sensor array triangulation.
    Idea is to use a convex hull and strip the unnecessary 'bottom'
    surface by point manipulation.
    Currently assumes that z is the 'up' direction.
    XXX: Delaunay gives nicer results than ConvexHull
    """
    # we want to place an extra point below the array bottom
    npts = rr.shape[0]
    ctr = rr.mean(axis=0)
    l = np.linalg.norm(rr - ctr, axis=1).mean()
    extrap = ctr - np.array([0.0, 0.0, l])
    # do a convex hull, including the extra point
    # this means that the bottom will be closed via the extra point
    rr_ = np.concatenate((rr, extrap[None, :]))
    # now delete all simplices that contain the extra point
    # this should get rid of the 'bottom' surface
    sps = ConvexHull(rr_).simplices
    inds_extra = np.where(sps == npts)[0]
    sps = np.delete(sps, inds_extra, axis=0)
    # filter & clean up mesh - currently not used
    tm = trimesh.Trimesh(vertices=rr, faces=sps)
    # trimesh.repair.fix_winding(tm)
    # trimesh.repair.fix_inversion(tm)
    # trimesh.repair.fill_holes(tm)
    pts_, tris_ = tm.vertices, tm.faces
    # return tm.faces
    return sps


def _get_plotting_inds(info):
    """Get magnetometer inds only, if they exist. Otherwise, get gradiometer inds."""
    mag_inds = mne.pick_types(info, meg='mag')
    grad_inds = mne.pick_types(info, meg='grad')
    # if mags exist, use only them
    if mag_inds.size:
        inds = mag_inds
    elif grad_inds.size:
        inds = grad_inds
    else:
        raise RuntimeError('no MEG sensors found')
    return inds


def _info_meg_locs(info):
    """Return sensor locations for MEG sensors"""
    return np.array(
        [
            info['chs'][k]['loc'][:3]
            for k in range(info['nchan'])
            if info['chs'][k]['kind'] == FIFF.FIFFV_MEG_CH
        ]
    )


def _make_array_tri(info, to_headcoords=True):
    """Make triangulation of array for topography views.
    If array has both mags and grads, triangulation will be based on mags
    only. Corresponding sensor indices are returned as inds.
    If to_headcoords, returns the sensor locations in head coordinates."""
    inds = _get_plotting_inds(info)
    locs = _info_meg_locs(info)[inds, :]
    if to_headcoords:
        locs = apply_trans(info['dev_head_t'], locs)
    locs_tri = _delaunay_tri(locs)
    return inds, locs, locs_tri


def _sigvec_topoplot(sigvec, info, **kwargs):
    """Make a topoplot of sigvec.
    
    For mixed magnetometer/gradiometer arrays, will use magnetometers only.
    """
    inds, locs, tri = _make_array_tri(info)
    sigvec = np.squeeze(sigvec)
    sigvec_ = sigvec[inds]
    _mlab_trimesh(locs, tri, scalars=sigvec_, **kwargs)
    mlab.view(azimuth=0, elevation=0, distance=0.5)

