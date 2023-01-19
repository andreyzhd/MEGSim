#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot some examples of sensor configurations from an optimization run. Read the
run data from the folder given as a parameter.
"""

import sys
from mayavi import mlab
from megsimutils.viz import _plot_anatomy
from megsimutils import read_opt_res

R_ENCL = 0.3 # m, radius of an invisible sphere to plot to force the plot to a given scale.
N_ITER = 4 # Number of iterations to load

# Check the command-line parameters
if len(sys.argv) != 2:
    raise RuntimeError('Wrong number of parameters. Specify the input path as a single parameter.')

# Read the data
params, sens_array, interm_res, opt_res, iter_indx = read_opt_res(sys.argv[-1], max_n_samp=N_ITER)

for ir in interm_res:
    fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
    fig.scene.parallel_projection = True
    sens_array.plot(ir[0], fig=fig, R_enclosure=R_ENCL, opacity_inner=0.3, opacity_outer=0.1)
    _plot_anatomy(figure=fig)
    mlab.view(azimuth=40, elevation=75, figure=fig)

print('Iterations:')
print(iter_indx)

mlab.show()
