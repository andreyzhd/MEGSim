"""
Generate a plot of noise vs R and number of sensors for regularly-spaced barbutes
"""

import numpy as np
from mayavi import mlab
from megsimutils.arrays import BarbuteArraySL, noise_max

N_SENS_RANGE = list(range(120, 241, 10))
R_RANGE = np.linspace(0.15, 0.25, 10)

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

noise = np.zeros((len(N_SENS_RANGE), len(R_RANGE)))

for i in range(len(N_SENS_RANGE)):
    for j in range(len(R_RANGE)):
        sens_array = BarbuteArraySL(N_SENS_RANGE[i], PARAMS['l_int'], R_inner=R_RANGE[j], R_outer=PARAMS['R_outer'], n_samp_layers=PARAMS['n_samp_layers'], n_samp_per_layer=PARAMS['n_samp_per_layer'], debug_fldr='/tmp/', **PARAMS['kwargs'])
        noise[i,j] = sens_array.comp_fitness(sens_array.get_init_vector())

x, y = np.meshgrid(N_SENS_RANGE, R_RANGE)
mlab.surf(noise)
mlab.show()