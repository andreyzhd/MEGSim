#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 02:24:48 2020

@author: andrey
Code for computing spherical harmonic expansion of a magnetic field. Based on
Taulu and Kajola 2005 doi:10.1063/1.1935742, Hill 1954 doi:10.1119/1.1933682

This is a quick-and-dirty proof-of-concept. It is (a) very computationaly
ineficient, and (b) cannot correctly handle anything on the z axis (theta=0 or 
theta=pi).
"""

import numpy as np
from scipy.special import sph_harm, lpmv, poch


def sph_Y(l, m, theta, phi):
    """Translate scipy convention to that used in Taulu and Kajola 2005,
    Hill 1954, etc.: swap phi and theta"""
    return sph_harm(m, l, phi, theta)


def _c(l, m):
    assert (4*np.pi*poch(l-m+1, 2*m)) != 0
    return np.sqrt((2*l+1) / (4*np.pi*poch(l-m+1, 2*m)))


def sph_Yp(l, m, theta, phi):
    """Compute derivative of Y wrt theta"""
    
    f1 = -_c(l, m) * np.sin(theta) * np.exp(complex(0,1)*m*phi)
    f2 = (m-l-1) * lpmv(m,l+1,np.cos(theta)) / (np.sin(theta)**2)
    f3 = (l+1) * np.cos(theta) * lpmv(m,l,np.cos(theta)) / (np.sin(theta)**2)
    
    return f1 * (f2 + f3)


def local_axes(theta, phi):
    """Compute local radial and tangential directions. Based on Hill 1954"""
    e_r = [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]
    e_theta = [np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)]
    e_phi = [-np.sin(phi), np.cos(phi), 0]
    return  np.array(e_r), np.array(e_theta), np.array(e_phi)


def sph_X_fixed(l, m, theta, phi):
    """Almost identical to X(l,m,theta,phi) defined in Taulu and kajola 2005, with
    the exception of sqrt(l) factor. That is, the fixed version is multiplied
    by sqrt(l) X_fixed = X * sqrt(l). This avoids division by zero for l=0."""
    e_r, e_theta, e_phi = local_axes(theta, phi)
    
    x_theta = -m * sph_Y(l, m, theta, phi) / np.sqrt(l+1) / np.sin(theta)
    x_phi = -complex(0,1) * sph_Yp(l, m, theta, phi) / np.sqrt(l+1)
    
    return x_theta*e_theta + x_phi*e_phi


def xyz2pol(x, y, z):
    """ Convert from Cartesian to polar coordinates. x, y, z should be arrays
    of the same dimension"""
    r = np.linalg.norm(np.stack((x,y,z)), axis=0)
    phi = np.arctan2(y, x)
    phi[phi<0] += 2*np.pi
    theta = np.arccos(z / r)
    
    return r, theta, phi

def pol2xyz(r, theta, phi):
    """ Convert from polar to Cartesian coordinates. r, theta, phi should be
    arrays of the same dimension (r can also be a scalar)"""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return x, y, z

def sph_v(l, m, theta, phi):
    e_r, e_theta, e_phi = local_axes(theta, phi)
    
    v_r = -(l+1) * sph_Y(l,m,theta,phi)
    v_theta = sph_Yp(l,m,theta,phi)
    v_phi = complex(0,1) * m * sph_Y(l,m,theta,phi) / np.sin(theta)
    
    return v_r*e_r + v_theta*e_theta + v_phi*e_phi