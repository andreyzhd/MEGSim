"""
Plot an example of a scalp array. If the output folder given as
a parameter, also save the image there.
"""
import sys
from mayavi import mlab
from megsimutils.arrays import BarbuteArrayScalp, BarbuteArraySL
from megsimutils.viz import _plot_anatomy


R_ENCL = 0.3 # m, radius of an invisible sphere to plot to force the plot to a given scale.

# Scalp array
sens_array = BarbuteArrayScalp(240, 10, 3, 0.15, 0.25, 5, 500, dist_sens_to_scalp=0.007)
v0 = sens_array.get_init_vector(1.)

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
fig.scene.parallel_projection = True
sens_array.plot(v0, fig=fig, R_enclosure=R_ENCL, opacity_inner=0., opacity_outer=0.1)

_plot_anatomy(figure=fig)
mlab.view(azimuth=40, elevation=75, figure=fig)
if len(sys.argv) == 2:
    mlab.savefig(f'{sys.argv[1]}/scalp_array_example.png', figure=fig, magnification=8)


# 3D array for comparison
sens_array = BarbuteArraySL(240, 10, 3, 0.15, 0.25, 5, 500)
v0 = sens_array.get_init_vector(1.)

fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
fig.scene.parallel_projection = True
sens_array.plot(v0, fig=fig, R_enclosure=R_ENCL, opacity_inner=0.3, opacity_outer=0.1)

_plot_anatomy(figure=fig)
mlab.view(azimuth=40, elevation=75, figure=fig)
if len(sys.argv) == 2:
    mlab.savefig(f'{sys.argv[1]}/scalp_array_example_ref.png', figure=fig, magnification=8)
