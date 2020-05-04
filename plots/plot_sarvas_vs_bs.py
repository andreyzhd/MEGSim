#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare Biot-Savart to Sarvas formula.

This sets up a symmetric (isosceles) Ilmoniemi triangle with different
tangential sidelenghts and compares the resulting magnetic field with current
dipole approximation (the Sarvas formula).
"""

import numpy as np
import matplotlib.pyplot as plt

from megsimutils import biot_savart, dipfld_sph
from megsimutils.utils import spherical_shell, _vector_angles

# set up triangle geometry
tri_y = 1.0  # triangle in xy plane
# min and max sidelength of tangential triangle edge
tl_min = 1e-3
tl_max = 500e-3
tls = np.linspace(tl_min, tl_max, num=100)  # sidelength of short (tangential) side

# Sarvas parameters
r0 = np.array([0.0, 0.0, 0.0])
rQ = np.array([0.0, tri_y, 0.0])

# the field points (should lie outside of sphere)
nfld = 100
sphere_rad = 1.2
sphere_thickness = 0.5
# use random points; set seed for exactly reproducible results
r = spherical_shell(nfld, sphere_rad, sphere_rad + sphere_thickness)

# vary tangential sidelength and compare fields
angs = list()
max_errors = list()
for tl in tls:
    Q = np.array([-tl, 0.0, 0.0])  # strength of equivalent dipole depends on sidelength
    # the Ilmoniemi triangle
    P = np.array([r0, [tl / 2, tri_y, 0], [-tl / 2, tri_y, 0]])
    if tl ** 2 / 4 + 1 > sphere_rad ** 2:
        raise RuntimeError('triangle points are outside of sphere')
    fld_bs = biot_savart(P, r, pts_per_edge=1000)
    fld_sarvas = dipfld_sph(Q, rQ, r, r0)
    max_error = np.max(np.abs(fld_bs - fld_sarvas)) / np.max(np.abs(fld_bs))
    max_errors.append(max_error)
    ang = _vector_angles(fld_bs.flatten(), fld_sarvas.flatten())[0]
    angs.append(ang)

# plot
plt.figure()
# the angle of the Ilmoniemi triangle
tri_angle = 2 * np.arcsin(tls / 2) / np.pi * 180
plt.plot(tri_angle, np.array(angs))
plt.title('Field from Sarvas vs. Biot-Savart')
plt.xlabel('Angle of current triangle (deg)')
plt.ylabel('Subspace angle between magnetic field vectors (deg)')
plt.figure()
plt.plot(tri_angle, 100 * np.array(max_errors))
plt.title('Field from Sarvas vs. Biot-Savart')
plt.xlabel('Angle of current triangle (deg)')
plt.ylabel('Maximum relative error (%)')
plt.ylim([0, 100])