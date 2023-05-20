#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run sensor array optimization for the 2D case. Save the results of the optimization
to the folder passed as command-lie parameter to this script. The folder should
exist and be empty.
"""

#%% Imports
import sys
import numpy as np
from megsimutils.arrays import noise_max, BarbuteArraySL
from opt_run import opt_run

#%% Parameter definitions
PARAMS = {'n_sens': 240,
          'R_inner': 0.15,
          'R_outer': None,
          'l_int': 10,
          'l_ext': 3,
          'n_samp_layers': 1,
          'n_samp_per_layer': 1000,
          'init_depth': 0,
          'kwargs': {'height_lower': 0.15,
                     'opm': False,
                     'origin': np.array([[0., 0., 0.],]),
                     'ellip_sc': np.array([1., 1., 1.]),
                     'noise_stat': noise_max}}
NITER = 1000

# Check the command-line parameters
if len(sys.argv) != 2:
    raise RuntimeError('Wrong number of parameters. Specify the output path as a single parameter.')

opt_run(BarbuteArraySL, PARAMS, NITER, sys.argv[-1])