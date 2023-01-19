"""
Generate a picture of a uniformly-spaced radial array
"""

import numpy as np
from mayavi import mlab
from megsimutils.arrays import BarbuteArraySL, noise_max

#%% Parameter definitions
PARAMS = {'n_sens' : 240,
          'R_inner' : 0.15,
          'R_outer' : None,
          'l_int' : 10,
          'n_samp_layers' : 1,
          'n_samp_per_layer' : 1000,
          'kwargs' : {#'Re' : 0.2,               # Radius for energy-based normalization
                      'height_lower' : 0.15,
                      'l_ext' : 0,
                      'opm' : False,
                      'origin' : np.array([[0., 0., 0.],]),
                      #'ellip_sc' : np.array([1.2, 1., 1.1])
                      'ellip_sc' : np.array([1., 1., 1.]),
                      'noise_stat' : noise_max
                      }
          }

sens_array = BarbuteArraySL(PARAMS['n_sens'], PARAMS['l_int'], R_inner=PARAMS['R_inner'], R_outer=PARAMS['R_outer'], n_samp_layers=PARAMS['n_samp_layers'], n_samp_per_layer=PARAMS['n_samp_per_layer'], debug_fldr='/tmp/', **PARAMS['kwargs'])
sens_array.plot(sens_array.get_init_vector())
mlab.show()