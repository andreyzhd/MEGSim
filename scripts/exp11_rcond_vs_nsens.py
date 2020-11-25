"""
Compute condition number vs number of sensors for different geometries (barbute, hockey)
"""
import numpy as np

import matplotlib.pyplot as plt

from megsimutils.array_geometry import barbute, double_barbute, hockey_helmet
from megsimutils.utils import _sssbasis_cond_pointlike

LIN = 16
LOUT = 0

def _eval_helmet(h_func):
    ns = []
    cs = []
    for n in range(130, 500, 5):
        rhelm, nhelm = h_func(n)
        cs.append(_sssbasis_cond_pointlike(rhelm, nhelm, sss_params))
        ns.append(rhelm.shape[0])

    return cs, ns

sss_params = {'origin': np.zeros(3), 'int_order': LIN, 'ext_order': LOUT}

cs, ns = _eval_helmet(lambda n : barbute(n, n, .15, .15, 1.5 * np.pi))
plt.plot(ns, np.log10(cs), 'o')

cs, ns = _eval_helmet(lambda n : hockey_helmet(4*n, chin_strap_angle=np.pi/4))
plt.plot(ns, np.log10(cs), 'o')

cs, ns = _eval_helmet(lambda n : hockey_helmet(4*n, chin_strap_angle=np.pi/8))
plt.plot(ns, np.log10(cs), 'o')

cs, ns = _eval_helmet(lambda n : double_barbute(n//2, n//2, 0.15, 0.2, 0.15, 1.5 * np.pi))
plt.plot(ns, np.log10(cs), 'o')

cs, ns = _eval_helmet(lambda n : hockey_helmet(2*n, chin_strap_angle=np.pi/4, inner_r=0.15, outer_r=0.2))
plt.plot(ns, np.log10(cs), 'o')

cs, ns = _eval_helmet(lambda n : hockey_helmet(2*n, chin_strap_angle=np.pi/8, inner_r=0.15, outer_r=0.2))
plt.plot(ns, np.log10(cs), 'o')


plt.legend(('Barbute', 'Hockey, pi/4', 'Hockey, pi/8', 'Double barbute', 'Double hockey pi/4', 'Double hockey pi/8'))
plt.xlim((LIN*(LIN+2), 1000))
plt.xlabel('Number of sensors')
plt.ylabel('Log10 of Rcond')

plt.show()
