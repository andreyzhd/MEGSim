"""
Compute condition number vs chin strap width for hockey helmet
"""
import numpy as np

import matplotlib.pyplot as plt

from megsimutils.array_geometry import barbute, hockey_helmet
from megsimutils.utils import _sssbasis_cond_pointlike

LIN = 16
LOUT = 0
N_PTS = 1000
RANGE = np.linspace(np.pi/16, np.pi)

sss_params = {'origin': np.zeros(3), 'int_order': LIN, 'ext_order': LOUT}

cs_a = []
cs_s = []

for chin_strap_angle in RANGE:
    rhelm, nhelm = hockey_helmet(N_PTS, chin_strap_angle=chin_strap_angle, symmetric_strap=False)
    cs_a.append(_sssbasis_cond_pointlike(rhelm, nhelm, sss_params))
    rhelm, nhelm = hockey_helmet(N_PTS, chin_strap_angle=chin_strap_angle, symmetric_strap=True)
    cs_s.append(_sssbasis_cond_pointlike(rhelm, nhelm, sss_params))


plt.plot(RANGE, np.log10(cs_a))
plt.plot(RANGE, np.log10(cs_s))
plt.legend(('Asymmetric chin strap', 'Symmetric chin strap'))
plt.xlabel('Chin strap angle, rads')
plt.ylabel('Log10 of Rcond')

plt.show()

"""
from mayavi import mlab
mlab.points3d(rhelm[:,0], rhelm[:,1], rhelm[:,2])
mlab.show()
"""