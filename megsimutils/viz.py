#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization functions for megsim.

author: Jussi (jnu@iki.fi)
"""
import pathlib
import mne
import numpy as np
import trimesh
from mayavi import mlab
from mne.transforms import _cart_to_sph, _pol_to_cart, apply_trans
from mne.io.constants import FIFF
from scipy.spatial import ConvexHull, Delaunay
from megsimutils.utils import spherepts_golden

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


def _mlab_colorblobs(
    src_coords, color_data, pt_scale=None, normalize_data=True, **kwargs
):
    """Visualize scalar functions on a 'blob style' source space."""
    if pt_scale is None:
        pt_scale = 0.002  # seems reasonable for a source space
    if normalize_data:
        color_data /= color_data.max()
    nodes = _mlab_points3d(src_coords, scale_factor=pt_scale, **kwargs)
    nodes.glyph.scale_mode = 'scale_by_vector'
    nodes.mlab_source.dataset.point_data.scalars = color_data


def viz_field(locs, field, norms, \
              show_arrows=False, proj_field=True, \
              cmap_range=None, colormap='seismic', \
              opacity=1, inner_surf=None, figure=None):
    """
    Visualize amagnetic field over a surface

    Parameters
    ----------
    locs : 3-by-M float
        location of the points that define the surface
    field : 3-by-M float
        vectors of magnetic fields
    norms : 3-by-M float
        Normal vectors for the surface
    show_arrows : bool, optional
        Show/hide arrows. The default is False.
    proj_field : bool, optional
        If True, color depicts the magnitude of the field component normal to
        the surcace. If False - absolute value of the field. The default is True.
    cmap_range : (float, float), optional
        Minimal and maximal values for the colormap. If not specified, minimum
        amd maximum of the data is used.
    colormap : colormap, optional
        The default is 'seismic'.
    opacity : float, optional
        Surface opacity, between 0 and 1. The default is 1.
    inner_surf : float, optional
        If specified, plot an inner surface at a distance inner_surf below the
        main surface. Serves as a background to help better see the main
        surface/arrows. If not specified, do not plot the inner surface.
    figure : mayavi figure to use, optional
        If not specified, create a new figure.

    """
    if figure is None:
        figure = mlab.figure()
    if cmap_range is None:
        vmax = np.max(np.abs(np.sum(field*norms, axis=1)))
        cmap_range = (-vmax, vmax)
    if proj_field:
        vals = np.sum(field*norms, axis=1)
    else:
        vals = np.sign(np.sum(field*norms, axis=1)) * np.linalg.norm(field, axis=1)
    pts = mlab.points3d(locs[:,0], locs[:,1], locs[:,2], vals, opacity=0, figure=figure)
    mesh = mlab.pipeline.delaunay3d(pts)
    mlab.pipeline.surface(mesh, colormap=colormap, vmin=cmap_range[0], vmax=cmap_range[1], opacity=opacity, figure=figure)
    
    if show_arrows:
        mlab.quiver3d(locs[:,0], locs[:,1], locs[:,2], field[:,0], field[:,1], field[:,2], color=(0,0,0), figure=figure)
    
    if inner_surf is not None:
        inner_locs = locs - norms*inner_surf
        pts = mlab.points3d(inner_locs[:,0], inner_locs[:,1], inner_locs[:,2], opacity=0, figure=figure)
        mesh = mlab.pipeline.delaunay3d(pts)
        mlab.pipeline.surface(mesh, figure=figure)

def _plot_sphere(center, r, npoints, fig, **kwargs):
    """Plot a sphere"""
    x, y, z = *(((spherepts_golden(npoints) * r) + center).T),
    pts = mlab.points3d(x, y, z, opacity=0, figure=fig)
    mesh = mlab.pipeline.delaunay3d(pts)
    mlab.pipeline.surface(mesh, figure=fig, **kwargs)


def _plot_brain(fig=None):
    """Plot the brain"""
    # source spacing; normally 'oct6', 'oct4' for sparse source space
    SRC_SPACING = 'oct6'

    # Rotation matrix to convert from MNE to barbute coordinates
    ROT_MAT = np.array(((0,-1,0),(1,0,0),(0,0,1)))

    if fig is None:
        fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))

    data_path = pathlib.Path(mne.datasets.sample.data_path())
    subjects_dir = data_path / 'subjects'
    subject = 'sample'

    # create the volume source space
    src_cort = mne.setup_source_space(
        subject, spacing=SRC_SPACING, subjects_dir=subjects_dir, add_dist=False
    )

    # src_cort is indexed by hemisphere (0=left, 1=right)
    # separate meshes for left & right hemi
    _mlab_trimesh(src_cort[0]['rr']@ROT_MAT, src_cort[0]['tris'], figure=fig, color=(85/255,170/255,255/255))
    _mlab_trimesh(src_cort[1]['rr']@ROT_MAT, src_cort[1]['tris'], figure=fig, color=(85/255,170/255,255/255))
    
    