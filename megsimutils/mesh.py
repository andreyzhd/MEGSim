#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mesh handling functions for megsim.

author: Jussi (jnu@iki.fi)
"""

import numpy as np
from mne.transforms import _pol_to_cart, _cart_to_sph
from scipy.spatial import Delaunay, ConvexHull
import trimesh


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
    The idea is to use a convex hull and strip out the unnecessary 'bottom'
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


def _vertex_nhood(v, tris, n, include_self=True):
    """N-neighborhood (all n-connected vertices) of vertex index v, given tris"""
    nhood = np.array([v])
    for k in range(n):
        # find all triangles that include this vertex
        tri_inds = np.unique(np.where(np.isin(tris, nhood))[0])
        # collect all unique vertices for those triangles
        nbs = np.unique(tris[tri_inds, :])
        nbs_new = np.setdiff1d(nbs, nhood, assume_unique=True)
        # form new neighborhood by including the new vertices, continue search
        nhood = np.concatenate((nhood, nbs_new))
    return nhood if include_self else np.delete(nhood, np.where(nhood == v))


def _mesh_delete(pts, tris, inds):
    """Delete mesh points given by inds and remap tris array"""
    mask = np.ones(pts.shape[0], dtype=bool)
    mask[inds] = False
    pts_ = pts[mask, :]  # reduced pts array
    # row indices of tris that refer to deleted pts
    tris_bad = np.where(~np.all(~np.isin(tris, inds), axis=1))[0]
    tris_ = np.delete(tris, tris_bad, axis=0)
    # we need to remap the tris array since indices changed
    oldinds = np.where(mask)[0]
    tris_ = np.digitize(tris_.ravel(), oldinds, right=True).reshape(tris_.shape)
    return pts_, tris_


def trimesh_equidistant_vertices(tm, nverts):
    """Find N approximately equidistant vertices in a trimesh.
    
    Uses a point repulsion algorithm (kind of gradient descent).
    The neighborhood of each point P is considered in turn. P is moved into
    the neighbor that yields the greatest reduction in global energy. This is
    continued until NIT_MAX is reached or energy cannot be reduced any more.
    Energy is computed as a sum of inverse squared distances (analogous to
    electrostatic point repulsion). Should converge for convex surfaces but may
    hit serious local minima for non-convex ones.

    Parameters
    ----------

    tm : trimesh.Trimesh
        The mesh.
    nverts : int
        Number of equidistant vertices to find.

    Returns
    -------
    locs : np.array (nverts,)
        The equidistant vertices (hopefully)

    """

    def _1toall_energy(points, ind):
        """Compute 1-to-all energy of Nx3 point set (point ind to all).
        This will be minimized by the iteration"""
        P = points[ind, :]
        points = np.delete(points, ind, axis=0)
        dists_squared = np.sum((points - P) ** 2, 1)
        # return minimum intersensor distance
        # (change sign for minimizer so it gets maximized)
        return -dists_squared.min()
        # return sum of inverse squared distances, analoguous to energy
        # of point cloud of charges; may return inf if identical points
        # exist
        # with np.errstate(divide='ignore'):
        #    return np.sum(dists_squared**-1)

    NIT_MAX = int(1e3)
    pts = np.array(tm.vertices)  # get rid of trimesh TrackedArray for speed
    # assign random initial vertices
    locs = np.random.choice(pts.shape[0], nverts, replace=False)
    print('iterating for %d vertices, please wait...' % nverts)
    for k in range(NIT_MAX):
        improved = False
        # move each vertex in turn
        for locind, loc in enumerate(locs):
            e0 = _1toall_energy(pts[locs, :], locind)  # current energy
            locs_ = locs.copy()
            for ne in tm.vertex_neighbors[loc]:
                locs_[locind] = ne  # move vertex to its neighbor
                e = _1toall_energy(pts[locs_, :], locind)
                if e < e0:
                    improved = True
                    locs[locind] = ne  # save best move so far
                    e0 = e
        if not improved:
            print('iteration converged in %d rounds' % k)
            break
    else:
        print('maximum n of iterations reached')
    return locs
