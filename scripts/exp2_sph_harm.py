#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 12:50:29 2020

@author: andrey

Compare spherical harmonics expansion to direct field computation with
Biot-Savart. Compute magnetic field due to Ilmoniemi triangle (strictly
speaking, due to a current loop that is very close but, but not exactly the
Ilmoniemi triangle to avoid some numerical problems).

Store the results in a file to be visualized with another script.
"""
import time
import pickle
import numpy as np
from megsimutils.utils import pol2xyz, xyz2pol
from megsimutils import sph_X_fixed, sph_v

def main():
    THETA_NSAMP = 50
    PHI_NSAMP = 100
    R = 0.1 # radius of the sensors sphere
    Q = np.array([100, 0, 0]) * 1e-9
    Y_SOURCE = 0.7*R # Dipole is located at x=0, y=Y_SOURCE, z=0
    
    EXT = 0.06
    TR_SIDE_NSAMP = 1000
    L_MAX = 10 # Maximum degree for spherical harmonics
    
    DATA_FNAME = '/tmp/vsh.pkl'
    
    t_start = time.time()
    
    # Create a sphere
    theta, phi = np.meshgrid(np.linspace(0.05, np.pi-0.05, THETA_NSAMP), np.linspace(0, 2*np.pi, PHI_NSAMP))
    x, y, z = pol2xyz(R, theta, phi)
    
    coord = np.stack((x.reshape((PHI_NSAMP*THETA_NSAMP,)), y.reshape((PHI_NSAMP*THETA_NSAMP,)), z.reshape((PHI_NSAMP*THETA_NSAMP,))), axis=1)
    
    ##-----------------------------------------------------------------------------
    # Create an Ilmoniemi triangle
    #
    
    # The tangential side of the triangle extends by EXT in each direction along
    # the dipole. Should not extens outside the sphere.
    
    c1 = np.array([0, Y_SOURCE, 0]) - EXT * Q / np.linalg.norm(Q)   # triangle corner
    c2 = np.array([0, Y_SOURCE, 0]) + EXT * Q / np.linalg.norm(Q)   # triangle corner
    cO = np.array([0, 0.0001, 0])
    
    # Make sure the corners are inside the sensor sphere
    assert(np.linalg.norm(c1) < R)
    assert(np.linalg.norm(c2) < R)
    
    
    smp = np.linspace(0, 1, TR_SIDE_NSAMP, endpoint=False)
    L = np.vstack((np.outer(smp, c1-cO) + cO, np.outer(smp, c2-c1) + c1, np.outer(smp, cO-c2) + c2))
    dL = np.diff(L, axis=0, append=L[np.newaxis,0,:])
    
    ##-----------------------------------------------------------------------------
    # Compute the field using Biot-Savart
    #
    def triangle_field(r, L):
        """Compute field ar location r due to unit current in the closed loop L
        using Biot-Savart law"""
        
        rp = r - L
    
        scl = np.outer(np.linalg.norm(rp, axis=1)**3, np.ones(3))
        B = 1e-7 * (np.cross(dL, rp) / scl).sum(axis=0)
        return B
    
    B_bs = np.apply_along_axis(triangle_field, 1, coord, L)
       
    
    ##-----------------------------------------------------------------------------
    # Compute the field using spherical harmonics
    #
    
    # Compute the expansion coefficients
    alphas = np.zeros((L_MAX+1, 2*(L_MAX+1)), dtype=np.complex128)
    for l in range(L_MAX+1):
        for m in range(-l, l+1):
            print('Computing alpha for l = %i, m = %i ...' % (l, m))
            # integrate over the triangle
            r, theta, phi = xyz2pol(L[:,0], L[:,1], L[:,2])
            alpha = 0
            for i in range(len(r)):
                alpha += (r[i]**l) * np.dot(np.conj(sph_X_fixed(l, m, theta[i], phi[i])), dL[i,:])
            
            alphas[l,m] = complex(0,1) * np.sqrt(1/(l+1)) / (2*l+1) * alpha
    
    # Reconstruct the field
    B_vsh = np.zeros_like(B_bs, dtype=np.complex128)
    r, theta, phi = xyz2pol(coord[:,0], coord[:,1], coord[:,2])
    
    for i in range(PHI_NSAMP*THETA_NSAMP):
        print('Computing B_vsh for location no %i out of %i' % (i, PHI_NSAMP*THETA_NSAMP))
        B = np.zeros(3, dtype=np.complex128)
        for l in range(L_MAX+1):
            for m in range(-l, l+1):
                B += alphas[l,m] * sph_v(l, m, theta[i], phi[i]) / (r[i]**(l+2))
                
        B_vsh[i,:] = -4 * np.pi * 1e-7 * B
            
    energy_imag = np.linalg.norm(np.imag(B_vsh), axis=1) ** 2
    energy_total = energy_imag + (np.linalg.norm(np.real(B_vsh), axis=1) ** 2)
    
    print('The highest proportion of energy contained in the imaginary part is %f' % (np.max(energy_imag/energy_total)))
    
    
    ##-----------------------------------------------------------------------------
    # Save the fields
    #
    f = open(DATA_FNAME, 'wb')
    pickle.dump((B_bs, B_vsh, x, y, z, alphas), f)
    f.close()
    
    print('Execution took %i seconds' % (time.time()-t_start))
    
    
if __name__ == '__main__':
  main()