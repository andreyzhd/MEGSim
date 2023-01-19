"""
Visualize the sampling volumes, 2D or 3D. Read the data from the folder given as a parameter.
"""
import sys

import numpy as np
from mayavi import mlab
from megsimutils import read_opt_res
from megsimutils.viz import _plot_anatomy, _plot_sphere

R_ENCL = 0.3 # m, radius of an invisible sphere to plot to force the plot to a given scale.

# Check the command-line parameters
if len(sys.argv) != 2:
    raise RuntimeError('Wrong number of parameters. Specify the input path as a single parameter.')

sens_array = read_opt_res(sys.argv[-1], max_n_samp=3)[1]

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
rmags = sens_array._sampling_locs_rmags
mlab.clf(fig)
mlab.points3d(rmags[:, 0], rmags[:, 1], rmags[:, 2], resolution=32, scale_factor=0.005, color=(0, 0, 1))
fig.scene.parallel_projection = True
_plot_anatomy(head_opacity=1, brain_opacity=0, cortex_opacity=0, figure=fig)
_plot_sphere(np.array((0, 0, 0)), R_ENCL, 100, fig, opacity=0)
mlab.view(azimuth=40, elevation=75, figure=fig)

mlab.show()

