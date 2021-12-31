#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Util functions for megsim.

"""
from deprecated import deprecated
from math import modf
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from mne.io.constants import FIFF
from mne.transforms import rotation3d_align_z_axis
import matplotlib.pylab as plt
from mne.preprocessing.maxwell import _sss_basis
from mne.transforms import _deg_ord_idx, _pol_to_cart, _cart_to_sph
from scipy.spatial import ConvexHull, Delaunay
from pathlib import Path
import subprocess
import platform
from megsimutils.fieldcomp import leadfield_sph


def _random_unit(N):
    """Return random unit vector in N-dimensional space"""
    v = np.random.randn(N)
    return v / np.linalg.norm(v)


def _idx_deg_ord(idx):
    """Returns (degree, order) tuple for a given multipole index."""
    # this is just an ugly inverse of _deg_ord_idx, do not expect speed
    for deg in range(1, 100):
        for ord in range(-deg, deg + 1):
            if _deg_ord_idx(deg, ord) == idx:
                return deg, ord
    return None


def _flip_normals(Sn, Nflip):
    """Flip randomly chosen sensor normals 90 degrees"""
    to_flip = np.random.choice(Sn.shape[0], Nflip, replace=False)
    for k in to_flip:
        flipvec = _random_unit(3)
        Sn[k, :] = np.cross(Sn[k, :], flipvec)


def _sanity_check_array(Sc, Sn):
    """Sanity check for a simple array description.
    Sc is (Nx3) sensor locations, Sn is (Nx3) sensor normals.
    """
    assert Sc.shape[0] == Sn.shape[0]
    assert Sc.shape[1] == 3
    assert Sn.shape[1] == 3
    # check that all normal vectors have length close to 1
    assert_array_almost_equal(np.linalg.norm(Sn, axis=1), 1)


def _prep_mf_coils_pointlike(rmags, nmags):
    """Prepare the coil data for pointlike magnetometers.

    rmags, nmags are sensor locations and normals respectively, with shape (N,3)
    """
    _sanity_check_array(rmags, nmags)
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


def spherepts_golden(N, angle=4 * np.pi, hcylind=None, offset=0):
    """Approximate uniformly distributed points on a unit sphere or
    cylinder-sphere.

    This is the "golden ratio algorithm".
    See: http://blog.marmakoide.org/?p=1

    Parameters
    ----------
    n : int
        Number of points.

    angle : float
        Solid angle (symmetrical around z axis) covered by the points. By
        default, the whole sphere. Must be between 0 and 4*pi

    hcylind : float
        Height of the cylindrical part. If not None, angle parameter is ignored.
        
    offset : float. The result is rotated by offset radians around z axis

    Returns
    -------
    ndarray
        (N, 3) array of Cartesian point coordinates.
    """
    # create linearly spaced azimuthal coordinate
    dlong = np.pi * (3 - np.sqrt(5))
    longs = np.linspace(0, (N - 1) * dlong, N) + offset
    # create linearly spaced z coordinate
    z_top = 1
    if hcylind is None:
        z_bottom = 1 - 2 * (angle / (4 * np.pi))
    else:
        z_bottom = -hcylind
    dz = (z_top - z_bottom) / N

    z = np.linspace(z_top - dz / 2, z_bottom + dz / 2, N)
    r = np.sqrt(np.abs(1 - z ** 2)) # Use abs to avoid warning, if z > 1, the r value computed here is discarded anyway.
    if hcylind is not None:
        r[z<0] = 1

    # this looks like the usual cylindrical -> Cartesian transform?
    return np.column_stack((r * np.cos(longs), r * np.sin(longs), z))


def spherical_shell(npts, rmin, rmax):
    """Generate random points inside a spherical shell"""
    r = np.random.rand(npts) * (rmax - rmin) + rmin
    phi = np.random.randn(npts) * np.pi  # polar angle
    theta = np.random.randn(npts) * 2 * np.pi  # azimuth
    X = r * np.sin(phi) * np.cos(theta)
    Y = r * np.sin(phi) * np.sin(theta)
    Z = r * np.cos(phi)
    return np.column_stack((X, Y, Z))


def _rotate_to(v1, v2):
    """Return a rotation matrix which rotates unit vector v1 to v2.

    See https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    NB: may do weird things for some rotations (e.g. identity)
    """
    assert v1.shape == v2.shape == (3,)
    assert np.linalg.norm(v1) == np.linalg.norm(v2) == 1.0
    vs = v1 + v2
    return 2 * np.outer(vs, vs) / np.dot(vs, vs) - np.eye(3)


def _vector_angles(V, W):
    """Angles in degrees between column vectors of V and W. V must be dxM
    and W either dxM (for pairwise angles) or dx1 (for all-to-one angles)"""
    assert V.shape[0] == W.shape[0]
    if V.ndim == 1:
        V = V[:, None]
    if W.ndim == 1:
        W = W[:, None]
    assert V.shape[1] == W.shape[1] or W.shape[1] == 1
    Vn = V / np.linalg.norm(V, axis=0)
    Wn = W / np.linalg.norm(W, axis=0)
    dots = np.sum(Vn * Wn, axis=0)
    dots = np.clip(dots, -1, 1)
    return np.arccos(dots) / np.pi * 180


def local_axes(theta, phi):
    """Compute local radial and tangential directions. theta, phi should be
    arrays of the same dimension. Returns arrays of the dimension d+1, where d
    is the input dimension. The last dimension is of the length 3, and it
    corresponds to x, y, and z coordinates of the tangential/radial unit
    vectors.

    Based on Hill 1954 doi:10.1119/1.1933682"""
    e_r = [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    e_theta = [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)]
    e_phi = [-np.sin(phi), np.cos(phi), np.zeros_like(theta)]
    return np.stack(e_r, axis=-1), np.stack(e_theta, axis=-1), np.stack(e_phi, axis=-1)


def xyz2pol(x, y, z):
    """Convert from Cartesian to polar coordinates. x, y, z should be arrays
    of the same dimension"""
    r = np.linalg.norm(np.stack((x, y, z)), axis=0)
    phi = np.arctan2(y, x)
    phi[phi < 0] += 2 * np.pi
    theta = np.arccos(z / r)

    return r, theta, phi


def pol2xyz(r, theta, phi):
    """Convert from polar to Cartesian coordinates. r, theta, phi should be
    arrays of the same dimension (r can also be a scalar)"""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return x, y, z


def fold_uh(theta, phi):
    """Fold theta and phi to the upper hemisphere. The resulting theta is
    between 0 and pi/2, phi -- between 0 and 2*pi. If the vector given by
    (theta, phi) points down (theta > pi/2), reflect it."""

    x, y, z = pol2xyz(1, theta, phi)
    # reflect vectors pointing down
    x[z < 0] *= -1
    y[z < 0] *= -1
    z[z < 0] *= -1
    r, theta, phi = xyz2pol(x, y, z)
    return theta, phi


##-----------------------------------------------------------------------------
# Functions for creating MNE-Python's Info object from given sensor locations
# and orientations (Copied from another project by Dr. J. Nurminen)
#
def _rot_around_axis_mat(a, theta):
    """Matrix for rotating right-handedly around vector a by theta degrees"""
    theta = theta / 180 * np.pi
    x, y, z = a[0], a[1], a[2]
    ctheta = np.cos(theta)
    stheta = np.sin(theta)
    return np.array(
        [
            [
                ctheta + (1 - ctheta) * x ** 2,
                (1 - ctheta) * x * y - stheta * z,
                (1 - ctheta) * x * z + stheta * y,
            ],
            [
                (1 - ctheta) * y * x + stheta * z,
                ctheta + (1 - ctheta) * y ** 2,
                (1 - ctheta) * y * z - stheta * x,
            ],
            [
                (1 - ctheta) * z * x - stheta * y,
                (1 - ctheta) * z * y + stheta * x,
                ctheta + (1 - ctheta) * z ** 2,
            ],
        ]
    )


def _sensordata_to_loc(Sc, Sn, Iprot):
    """Convert sensor data from Sc (Mx3 locations) and Sn (Mx3 normals) into
    mne loc matrices, used in e.g. info['chs'][k]['loc']. Integration data is
    handled separately via the coil definitions.

    Sn is the desired sensor normal, used to align the xy-plane integration
    points. Iprot (Mx1, degrees) can optionally be applied to first rotate the
    integration point in the xy plane. Rotation is CCW around z-axis.
    """
    assert Sn.shape[0] == Sc.shape[0]
    assert Sn.shape[1] == Sc.shape[1] == 3
    for k in range(Sc.shape[0]):
        # get rotation matrix corresponding to desired sensor orientation
        R2 = rotation3d_align_z_axis(Sn[k, :])
        # orient integration points in their xy plane
        R1 = _rot_around_axis_mat([0, 0, 1], Iprot[k])
        rot = R2 @ R1
        loc = np.zeros(12)
        loc[:3] = Sc[k, :]  #  first 3 elements are the loc
        loc[3:] = rot.T.flat  # next 9 elements are the flattened rot matrix
        yield loc


def sensordata_to_ch_dicts(Sc, Sn, Iprot, coiltypes):
    """Convert sensor data from Sc (Mx3 locations) and Sn (Mx3 normals) into
    mne channel dicts (e.g. info['chs'][k]"""
    locs = _sensordata_to_loc(Sc, Sn, Iprot)
    for k, (loc, coiltype) in enumerate(zip(locs, coiltypes)):
        ch = dict()
        number = k + 1
        ch['loc'] = loc
        ch['ch_name'] = 'MYMEG %d' % number
        ch['coil_type'] = coiltype
        ch['coord_frame'] = FIFF.FIFFV_COORD_DEVICE
        ch['kind'] = FIFF.FIFFV_MEG_CH
        ch['logno'] = number
        ch['range'] = 1
        ch['scanno'] = number
        ch['unit'] = FIFF.FIFF_UNIT_T
        ch['unit_mul'] = FIFF.FIFF_UNITM_NONE
        yield ch


@deprecated(reason="You should use megsimutils.array_geometry.hockey.helmet instead")
def hockey_helmet(locs_dens, chin_strap_angle=np.pi / 8, inner_r=0.15, outer_r=None):
    """Create a hockey-helmet-like (a hemisphere and a chin strap) dense mesh
    of possible sensor locations. Locations are distributed approximately
    evenly. locs_dens defines the sensor density -- it's the number of sensors
    per 4*pi steradian. The helmet is optionally double-layered (if outer_r is
    not None).

    Returns four arrays -- x, y, and z coordinates of all
    the candidate points on a sphere and a boolean index array indicating
    which locations belong to the helmet.
    """
    pts = spherepts_golden(locs_dens)
    r, theta, phi = xyz2pol(pts[:, 0], pts[:, 1], pts[:, 2])
    helm_indx = ((phi > 0) & (phi < np.pi)) | (
        (phi > (3 / 2) * np.pi - chin_strap_angle) & (phi < (3 / 2) * np.pi)
    )

    x, y, z = pts[:, 0], pts[:, 2], pts[:, 1]  # y and z axis are swapped on purpose
    x_inner, y_inner, z_inner = x * inner_r, y * inner_r, z * inner_r
    if not (outer_r is None):
        assert outer_r > inner_r
        x_outer, y_outer, z_outer = x * outer_r, y * outer_r, z * outer_r
        return (
            np.concatenate((x_inner, x_outer)),
            np.concatenate((y_inner, y_outer)),
            np.concatenate((z_inner, z_outer)),
            np.concatenate((helm_indx, helm_indx)),
        )
    else:
        return x_inner, y_inner, z_inner, helm_indx


def uniform_sphere_dipoles(n, r, seed=0):
    """Generate n dipoles with random locations and orientations uniformly spread throughout a sphere centered at the
    origin. If the seed is None, entropy from the OS will be used for the random number generator, making the set of
    dipoles irreproducible.

    Parameters
    ----------
    n : int
        number of dipoles
    r : float
        sphere radius (m)
    seed : None or int
        random number generator seed.

    Returns
    -------
    rmags: ndarray
        (n x 3) dipole locations (m)
    nmags: ndarray
        (n x 3) dipole orientations (unit vectors)
    """
    rng = np.random.default_rng(seed=seed)
    locs = rng.normal(size=(3, n))
    locs /= np.linalg.norm(locs, axis=0)
    locs *= r * np.cbrt(rng.uniform(size=n))

    norms = rng.normal(size=(3, n))
    norms /= np.linalg.norm(norms, axis=0)

    return locs.T, norms.T


def comp_inf_capacity(slocs, snorms, dlocs, dnorms, signal, noise):
    """ Compute a total information capacity of a pointlike magnetometer sensor
    array (see Kemppainen and Ilmoniemi 1989, https://doi.org/10.1007/978-1-4613-0581-1_141).
    Assume gaussian IID source dipoles and gaussian IID additive sensor noise.

    :param slocs: Sensor locations (m)
    :param snorms: Sensor orientaions (unit vectors)
    :param dlocs: Dipole locations (m)
    :param dnorms: Dipole orientations
    :param signal: Total dipole moment (root-mean-square over all the dipoles, A * m)
    :param noise: Single sensor noise, T
    :return: Number of bits per sample
    """
    n_dipoles = dlocs.shape[0]
    lf = leadfield_sph(slocs, snorms, dlocs, dnorms)
    eigs = np.linalg.svd((lf.T @ lf) / n_dipoles, compute_uv=False, hermitian=True)
    return np.sum(np.log2((signal**2) * eigs / (noise**2) + 1)) / 2


def subsample(l: int, k: int, rng: np.random.Generator = None):
    """
    Return a sequence of at most k (k may be math.inf) indices into a sequence
    of length l, approximately evenly spread. The result always starts with 0,
    if k > 1, it ends with l-1. Indices are unique and ordered from smallest to
    largest. If k >= l, return list(range(l)).

    Note: This is a simple recursive implementation. The recursion depth is
    approximately k. If you get 'maximum recursion depth exceeded' error,
    increase the maximum recursion depth with sys.setrecursionlimit().
    """
    if rng is None:
        rng = np.random.default_rng(seed=0)

    if k == 1 or l == 1:
        return [0]
    else:
        frac, base = modf(l - max(l / k, 1))
        if rng.uniform() < frac:
            base += 1
        res = subsample(int(base), k-1, rng)
        res.append(l-1)
        return res