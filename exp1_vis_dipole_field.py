#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualise a field of a dipole or Ilmoniemi triangle on a sphere
"""

from mayavi import mlab
import numpy as np

from megsimutils import *

PHI_NSAMP = 100
THETA_NSAMP = 200
R = 0.1 # radius of the sensors sphere

Q = np.array([0, 100, 0]) * 1e-9

Z_SOURCE = 0.7*R # Dipole is located at x=0, y=0, z=Z_SOURCE

# Create a sphere
pi = np.pi

phi, theta = np.meshgrid(np.linspace(0, pi, PHI_NSAMP), np.linspace(0, 2*pi, THETA_NSAMP))

x = R * np.sin(phi) * np.cos(theta)
y = R * np.sin(phi) * np.sin(theta)
z = R * np.cos(phi)

coord = np.stack((x.reshape((PHI_NSAMP*THETA_NSAMP,)), y.reshape((PHI_NSAMP*THETA_NSAMP,)), z.reshape((PHI_NSAMP*THETA_NSAMP,))), axis=1)

# Compute the maximum allowed shift of the Sarvas formula's origin
d = (R**2 + Z_SOURCE**2) / (2*R)

##-----------------------------------------------------------------------------
# Start plotting
#

# don't shift the origin
f_orig = dipfld_sph(Q, np.array([0, 0, Z_SOURCE]), coord, np.array([0, 0, 0]))
s = np.linalg.norm(f_orig, axis=1)

fig1 = mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
mlab.clf(fig1)
mlab.mesh(x, y, z, scalars=s.reshape((THETA_NSAMP, PHI_NSAMP)), colormap='viridis', figure=fig1)
fig1.name = 'Conductor and sensor spheres concentrical'

# shift the origin by -d along the x axis
f_minusd = dipfld_sph(Q, np.array([0, 0, Z_SOURCE]), coord, np.array([-d, 0, 0]))
s = np.linalg.norm(f_minusd, axis=1)

fig2 = mlab.figure(2, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
mlab.clf(fig2)
mlab.mesh(x, y, z, scalars=s.reshape((THETA_NSAMP, PHI_NSAMP)), colormap='viridis', figure=fig2)
fig2.name = 'Conductor sphere shifted by -%0.2f M along x axis' % d

# shift the origin by -´d along the x axis
f_plusd = dipfld_sph(Q, np.array([0, 0, Z_SOURCE]), coord, np.array([d, 0, 0]))
s = np.linalg.norm(f_plusd, axis=1)

fig3 = mlab.figure(3, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
mlab.clf(fig3)
mlab.mesh(x, y, z, scalars=s.reshape((THETA_NSAMP, PHI_NSAMP)), colormap='viridis', figure=fig3)
fig3.name = 'Conductor sphere shifted by %0.2f M along x axis' % d

mlab.sync_camera(fig1, fig2)
mlab.sync_camera(fig1, fig3)


##-----------------------------------------------------------------------------
# Compute and visualize field due to the Ilmoniemi triangle
#

# The tangential side of the triangle extends by EXT in each direction along
# the dipole. Should not extens outside the sphere.
EXT = 0.01
TR_SIDE_NSAMP = 1000

c1 = np.array([0, 0, Z_SOURCE]) - EXT * Q / np.linalg.norm(Q)   # triangle corner
c2 = np.array([0, 0, Z_SOURCE]) + EXT * Q / np.linalg.norm(Q)   # triangle corner

# Make sure the corners are inside the sensor sphere
assert(np.linalg.norm(c1) < R)
assert(np.linalg.norm(c1) < R)

smp = np.linspace(0, 1, TR_SIDE_NSAMP, endpoint=False)
L = np.vstack((np.outer(smp, c1), np.outer(smp, c2-c1) + c1, np.outer(smp, -c2) + c2))

def triangle_field(r, L):
    """Compute field ar location r due to unit current in the closed loop L
    using Biot-Savart law"""
    
    rp = r - L
    dL = np.diff(L, axis=0, append=L[np.newaxis,0,:])

    scl = np.outer(np.linalg.norm(rp, axis=1)**3, np.ones(3))
    B = 1e-7 * (np.cross(dL, rp) / scl).sum(axis=0)
    return B

B = np.apply_along_axis(triangle_field, 1, coord, L)
    
# Plot
s = np.linalg.norm(B, axis=1)

fig4 = mlab.figure(4, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
mlab.clf(fig4)
mlab.mesh(x, y, z, scalars=s.reshape((THETA_NSAMP, PHI_NSAMP)), colormap='viridis', figure=fig4)
fig4.name='Ilmoniemi triangle'
mlab.sync_camera(fig1, fig4)

# Compute the difference between Ilmoniemi triangle and Sarvas. Ilmoniemi
# triangle's current is set to such a value that dipole moment along the
# tangential side alone is equal to Q

compare_fields(B * (np.linalg.norm(Q) / (2*EXT)), f_orig)

