#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script validates dipfld_sph against an independent implementation of the
same algorithm
"""


import numpy as np
import matplotlib.pylab as plt

from megsimutils import dipfld_sph

def dipole_field(rQ, Q, r):
    """Compute field for a dipole is a spherical conductor using the formula from
    Sarvas, 1987 (as described by Nurminen, 2014).
    rQ - 3d vector of the dipole location (in m)
    Q - 3d vector of dipole moment (in A*m)
    r - 3d vector givingt the point at which the field is to be computed (in m)
    The origin is at the center of the spherical conductor.
    Return the 3d vector of the field (in T). """

    a = r - rQ
    na = np.linalg.norm(a)
    nr = np.linalg.norm(r)
    
    # define some handy intermediate expressions
    a_times_r = na * nr
    a_plus_r = na + nr
    r_dot_rQ = r.dot(rQ)
    
    F = na*(a_times_r + nr**2 - r_dot_rQ)
    
    # Compute gradient of F
    wr = (a_plus_r**3 - a_times_r*a_plus_r - nr*r_dot_rQ) / a_times_r
    wrQ = (r_dot_rQ - a_plus_r**2) / na
    
    grad_F = wr*r + wrQ*rQ
    
    B = 1e-7 * (np.cross(F*Q,rQ) - (np.cross(Q,rQ)).dot(r) * grad_F) / F**2
    
    return(B)

rQ = np.array([0, 0.07, 0])
Q = np.array([-50, 50, -100]) * 1e-9

span = np.arange(-0.2, 0.2, 0.01)
x, y, z = np.meshgrid(span, span, span)

R = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=1)
R = R[np.linalg.norm(R, axis=1) > 0.1, :]   # exclude points aroubd origin

# Compute the field using dipfld_sph
B_orig = dipfld_sph(Q, rQ, R, np.array([0, 0, 0]))

# Compute the field using independent implementation
B_valid = np.zeros_like(B_orig)

for i in range(R.shape[0]):
    B_valid[i,:] = dipole_field(rQ, Q, R[i,:])
    
# Compare the results
B_orig = B_orig.flatten()
B_valid = B_valid.flatten()

d = np.abs(B_orig - B_valid) / np.abs(B_valid)
indx = np.argsort(d)

plt.plot(np.log(d) / np.log(10) * 10)
plt.plot(np.log(np.abs(B_valid)) / np.log(10) * 10)
plt.xlabel('field components')
plt.ylabel('log scale, dB')
plt.legend(('Relative error between the implementations (0 dB means the error of 100%)', 'Magnetic field strength (0 dB means 1 T)'))


