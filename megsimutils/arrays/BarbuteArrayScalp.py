import numpy as np
import trimesh
import trimesh.sample

from megsimutils.arrays import BarbuteArraySL
from megsimutils.viz import _load_anatomy

# Downsample the head mesh to this number of vertices to speed up 
# computations
_NVERT_HEAD_HEAD = 700

def _map_barbute_to_scalp_point(point, R_inner, R_outer, head_pts, d=0.007):
    """ Map a single 3D point from a 3D-barbute space to a scalp-array-3D-barbute space
    """
    point = point.copy()

    if point[2] >= 0:
        transl = np.array([0., 0., 0.])
    else:
        transl = np.array([0., 0., -point[2]])
        point[2] = 0

    r = np.linalg.norm(point)
    assert r >= R_inner * 0.99
    assert r <= R_outer * 1.01

    point_norm = point / r

    projs = (head_pts + transl) @ point_norm
    dists = np.linalg.norm(np.outer(projs, point_norm) - (head_pts + transl), axis=1)
    dists[projs<=0] = np.max(dists)

    R_inner_local = projs[np.argmin(dists)] + d
    assert R_inner_local <= R_inner
    frac = (r - R_inner) / (R_outer - R_inner)

    r_new = R_inner_local + frac * (R_outer - R_inner_local)

    point_new = point_norm * r_new - transl

    return point_new


def _map_barbute_to_scalp(pts, R_inner, R_outer, head_pts, d=0.007):
    """ Map a set of points 3D point from a 3D-barbute space to a scalp-array-3D-barbute space
    """
    return np.vstack(list(_map_barbute_to_scalp_point(point, R_inner, R_outer, head_pts, d=d) for point in pts))


class BarbuteArrayScalp(BarbuteArraySL):
    def __init__(self, n_sens, l_int, l_ext, R_inner=0.15, R_outer=None, n_samp_layers=1, n_samp_per_layer=100, dist_sens_to_scalp=0.007, **kwargs):
        assert R_outer is not None
        super().__init__(n_sens,
                         l_int,
                         l_ext,
                         R_inner=R_inner,
                         R_outer=R_outer,
                         n_samp_layers=n_samp_layers,
                         n_samp_per_layer=n_samp_per_layer,
                         **kwargs)
        
        head_mesh = _load_anatomy()['head']
        # Downsample the head mesh to speed up computation
        mesh = trimesh.Trimesh(head_mesh['rr'], head_mesh['tris'])
        self.__head_pts = trimesh.sample.sample_surface_even(mesh, _NVERT_HEAD_HEAD)[0]

        self.__dist_sens_to_scalp = dist_sens_to_scalp
        self._sampling_locs_rmags = _map_barbute_to_scalp(self._sampling_locs_rmags, self._R_inner, self._R_outer, self.__head_pts, d=self.__dist_sens_to_scalp)

    def _v2sens_geom(self, v):
        rmags, nmags = super()._v2sens_geom(v)

        return _map_barbute_to_scalp(rmags, self._R_inner, self._R_outer, self.__head_pts, d=self.__dist_sens_to_scalp), nmags
  