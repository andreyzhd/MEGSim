"""
Generate a plot of noise vs R and number of sensors for regularly-spaced barbutes
"""

import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt
from megsimutils.arrays import BarbuteArraySL, noise_max

N_SENS_RANGE = np.arange(120, 241, 5)
R_RANGE = np.linspace(0.15, 0.25, 20)

#N_SENS_RANGE = np.arange(120, 241, 20)
#R_RANGE = np.linspace(0.15, 0.25, 10)

N_SENS_SUBSAMP = np.arange(4, len(N_SENS_RANGE), 4)
R_SUBSAMP = np.arange(0, len(R_RANGE), 4)


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


#%% Plot
#x, y = np.meshgrid(N_SENS_RANGE, R_RANGE)
#mlab.surf(np.log10(noise))
#mlab.show()


plt.figure()
plt.semilogy(N_SENS_RANGE, noise[:, R_SUBSAMP])
plt.legend(list(('R = %0.2f' % R) for R in R_RANGE[R_SUBSAMP]))
plt.xlabel('Number of sensors')
plt.ylabel('Noise amplification factor')

plt.figure()
plt.semilogy(R_RANGE, (noise[N_SENS_SUBSAMP, :]).T)
plt.legend(list(('%i sensors' % n_sens) for n_sens in N_SENS_RANGE[N_SENS_SUBSAMP]))
plt.xlabel('R, m')
plt.ylabel('Noise amplification factor')
plt.show()