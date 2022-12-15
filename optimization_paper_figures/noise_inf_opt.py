"""
Plot noise amplification factor, channel information capacity as a function
of optimization iteration. Read the data from the folder given as a parameter.
"""

import math
import sys
import matplotlib.pyplot as plt
from figutils import vis_noise_inf_opt

MAX_N_ITER = 100 # math.inf # Number of iterations to load

# Read the input folder
if len(sys.argv) != 2:
    raise RuntimeError('Wrong number of parameters. Specify the input path as a single parameter.')

N_DIPOLES_LEADFIELD = 1000
R_LEADFIELD = 0.1 # m
DIPOLE_STR = 1e-8 # A * m
SQUID_NOISE = 1e-14 # T

vis_noise_inf_opt(sys.argv[-1], MAX_N_ITER, N_DIPOLES_LEADFIELD, R_LEADFIELD, DIPOLE_STR, SQUID_NOISE)

plt.show()