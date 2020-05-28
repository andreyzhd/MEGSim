#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 13:35:37 2019

@author: jussi
"""

from numpy.testing import assert_allclose
import numpy as np

from megsimutils.utils import spherepts_golden


def test_spherepts_golden():
    """Test spherepts_golden()"""
    P = spherepts_golden(5000)
    norms = np.linalg.norm(P, axis=1)
    assert P.shape == (5000, 3)
    assert_allclose(norms, 1)

