"""
Compute condition number vs layer separation in dual-layer helmets
"""
import numpy as np

import matplotlib.pyplot as plt

from megsimutils.array_geometry import double_barbute, hockey_helmet
from megsimutils.utils import _sssbasis_cond_pointlike

LIN = 16
LOUT = 0
N_PTS = 1000

INNER_R = 0.15
OUT_RANGE = np.linspace(0.16, 1, 100)

sss_params = {'origin': np.zeros(3), 'int_order': LIN, 'ext_order': LOUT}

cs_h = []
cs_b = []

for outer_r in OUT_RANGE:
    rhelm, nhelm = hockey_helmet(4*N_PTS, chin_strap_angle=np.pi/8, inner_r=INNER_R, outer_r=outer_r)
    cs_h.append(_sssbasis_cond_pointlike(rhelm, nhelm, sss_params))
    nsens_h = rhelm.shape[0]

    rhelm, nhelm = double_barbute(N_PTS, N_PTS, INNER_R, outer_r, INNER_R, 1.5*np.pi)
    cs_b.append(_sssbasis_cond_pointlike(rhelm, nhelm, sss_params))
    nsens_b = rhelm.shape[0]

plt.plot(OUT_RANGE-INNER_R, np.log10(cs_h))
plt.plot(OUT_RANGE-INNER_R, np.log10(cs_b))
plt.legend(('Hockey helmet pi/8, %i sensors' % nsens_h, 'Barbute, %i sensors' % nsens_b))
plt.xlabel('Layer separation, m')
plt.ylabel('Log10 of Rcond')

plt.show()

