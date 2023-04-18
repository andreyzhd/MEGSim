"""
Show final array configurations for multiple runs for side-dy-side comparison.
"""

import sys
from pathlib import Path

import numpy as np
import sys
from mayavi import mlab
from megsimutils.viz import _plot_anatomy

from megsimutils import read_opt_res

R_ENCL = 0.3 # m, radius of an invisible sphere to plot to force the plot to a given scale.
MAX_N_ITER = 5


# Check the command-line parameters
if len(sys.argv) != 2:
    raise RuntimeError('Wrong number of parameters. Specify the input path as a single parameter.')

##-------------------------------------------------------------------------
# Read the data
#
opt_res = []
for run_fldr in Path(sys.argv[-1]).iterdir():
    if run_fldr.is_dir():
        opt_re = dict(zip(('params', 'sens_array', 'interm_res', 'opt_res', 'iter_indx'), read_opt_res(str(run_fldr)+'/out', max_n_samp=MAX_N_ITER)))
        opt_res.append(opt_re)
        np.testing.assert_equal(opt_re['params'], opt_res[0]['params'])   # All the runs should have the same parameters

##-------------------------------------------------------------------------
# Plot
#
for i, opt_re in enumerate(opt_res):
    ir = opt_re['interm_res'][-1]

    fig = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
    fig.scene.parallel_projection = True
    opt_re['sens_array'].plot(ir[0], fig=fig, R_enclosure=R_ENCL, opacity_inner=0.3, opacity_outer=0.1)
    _plot_anatomy(figure=fig)
    #lab.view(azimuth=40, elevation=75, figure=fig)

    if i == 0:
        ref_fig = fig
    else:
        mlab.sync_camera(ref_fig, fig)
mlab.show()