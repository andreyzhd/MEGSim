import numpy as np
from megsimutils.arrays import BarbuteArraySL
from megsimutils.utils import spherepts_golden

class BarbuteArraySLGrid(BarbuteArraySL):

    def __in_imag_vol(self, rmags):
        """
        Check which sensors are in the barbute. Return True for the sensors in
        the barbute and False for the sensors outside of the barbute.
        Ignore the _fract_trans (that is assume the opening is rectangular). This function is similar to
        BarbuteArray._on_barbute, but in addition checks the "depth" of each sensor's location.

        Parameters
        ----------
        rmags : n_sens-by3 matrix
            Sensor coordinates

        Returns
        -------
        1-d boolean array of the length n_sens
        """
        r = np.linalg.norm(rmags, axis=1)
        r[rmags[:,2]<0] = np.linalg.norm(rmags[rmags[:,2]<0, :-1], axis=1)
        return (r > self._R_inner) & (r < self._R_outer) & self._on_barbute(rmags)


    def __init__(self, n_sens, l_int, grid_sz=100, **kwargs):
        super().__init__(n_sens, l_int, **kwargs)

        assert self._R_outer is not None
        d = max(self._R_outer, self._height_lower)

        x, y, z = np.mgrid[-d:d:grid_sz*1j, -d:d:grid_sz*1j, -d:d:grid_sz*1j]
        all_locs = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
        self.__imag_locs_indx = self.__in_imag_vol(all_locs)

        self.__sampling_locs_rmags = np.tile(all_locs[self.__imag_locs_indx,:], (3,1))
        self.__sampling_locs_nmags = np.vstack((np.outer(np.ones(np.count_nonzero(self.__imag_locs_indx)), np.array((1, 0, 0))),
                                                np.outer(np.ones(np.count_nonzero(self.__imag_locs_indx)), np.array((0, 1, 0))),
                                                np.outer(np.ones(np.count_nonzero(self.__imag_locs_indx)), np.array((0, 0, 1)))))
        self.__grid_sz = grid_sz


    def _get_sampling_locs(self):
        return self.__sampling_locs_rmags, self.__sampling_locs_nmags


    def noise_grid(self, v):
        noise = self.comp_interp_noise(v)
        noise = noise.reshape(len(noise)//3, 3, order='F')
        noise = noise.max(axis=1)

        all_noise = np.zeros_like(self.__imag_locs_indx, dtype='float64')
        all_noise[self.__imag_locs_indx] = noise
        return all_noise.reshape(self.__grid_sz, self.__grid_sz, self.__grid_sz)


    def plot_grid(self, fig=None):
        from mayavi import mlab

        if fig is None:
            fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))

        mlab.clf(fig)
        mlab.points3d(self.__all_locs[:, 0], self.__all_locs[:, 1], self.__all_locs[:, 2], resolution=4, scale_factor=0.0025, color=(0, 0, 1), opacity=0.0)
        mlab.points3d(self.__all_locs[self.__imag_locs_indx, 0], self.__all_locs[self.__imag_locs_indx, 1], self.__all_locs[self.__imag_locs_indx, 2], resolution=32, scale_factor=0.005, color=(0, 0, 1))

        # Draw the background
        inner_locs = spherepts_golden(1000, hcylind=self._height_lower / self._R_inner) * self._R_inner * self._ellip_sc
        pts = mlab.points3d(inner_locs[:, 0], inner_locs[:, 1], inner_locs[:, 2], opacity=0, figure=fig)
        mesh = mlab.pipeline.delaunay3d(pts)
        mlab.pipeline.surface(mesh, figure=fig, color=(0.5, 0.5, 0.5), opacity=0.5)