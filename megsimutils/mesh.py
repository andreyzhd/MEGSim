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
