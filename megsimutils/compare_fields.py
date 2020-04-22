#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:02:10 2020

@author: user
"""

import numpy as np

def compare_fields(B1, B2):
    """ B1, B2 are N-by-3 matrices that contain magnetic fields at the same
    points"""
    nB1 = np.linalg.norm(B1, axis=1)
    nB2 = np.linalg.norm(B2, axis=1)
    
    angle = np.arccos(np.sum(B1*B2, axis=1) / (nB1 * nB2))
    d = np.abs(nB1 - nB2) / nB1
    
    print('Maximum difference, magnitude: %0.3f %%, direction: %0.3f degree' % (d.max()*100, angle.max()*180/np.pi))
    