#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run sensor array optimization for the 3D case. Save the results of the optimization
to the folder passed as command-lie parameter to this script. The folder should
exist and be empty.
"""

#%% Imports
import sys
import numpy as np
from megsimutils.arrays import noise_max
from opt_run import opt_run

#%% Parameter definitions
PARAMS = {'n_sens': 240,
          'R_inner': 0.15,
          'R_outer': 0.25,
          'l_int': 10,
          'l_ext': 3,
          'n_samp_layers': 5,
          'n_samp_per_layer': 500,
          'init_depth': 1.0,
          'kwargs': {'height_lower': 0.15,
                     'opm': False,
                     'origin': np.array([[0., 0., 0.],]),
                     'ellip_sc': np.array([1., 1., 1.]),
                     'noise_stat': noise_max}}
NITER = 1000

# Check the command-line parameters
if len(sys.argv) != 2:
    raise RuntimeError('Wrong number of parameters. Specify the output path as a single parameter.')

opt_run(PARAMS, NITER, sys.argv[-1])