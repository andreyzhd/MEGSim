#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare Biot-Savart to Sarvas formula.
"""

import numpy as np
import matplotlib.pyplot as plt

from megsimutils import biot_savart, dipfld_sph
from megsimutils.utils import spherical_shell, _vector_angles

# set up triangle geometry
tl_min = 1e-3
tl_max = 1000e-3
tls = np.linspace(tl_min, tl_max, num=100)  # sidelength of short (tangential) side

# Sarvas parameters
r0 = np.array([0., 0., 0.])
rQ = np.array([0., 1., 0.])

# the random field points (should be outside of sphere)
nfld = 100
sphere_rad = 1.2
r = spherical_shell(nfld, sphere_rad, sphere_rad+.5)

# compute the angles
angs = list()
for tl in tls:
    Q = np.array([-tl, 0., 0.])
    P = np.array([r0, [tl/2, 1, 0], [-tl/2, 1, 0]])
    if tl**2/4+1 > sphere_rad**2:
        raise RuntimeError('triangle points outside of sphere')
    fld_bs = biot_savart(P, r, pts_per_edge=1000)
    fld_sarvas = dipfld_sph(Q, rQ, r, r0)
    ang = _vector_angles(fld_bs.flatten(), fld_sarvas.flatten())[0]
    angs.append(ang)

plt.figure()
plt.plot(tls, angs)
plt.title('Field from Sarvas vs. Biot-Savart')
plt.xlabel('Length of tangential edge (m)')
plt.ylabel('Angle between magnetic field vectors (deg)')












