# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 13:01:17 2020

@author: Claudia Haindl
"""

import numpy as np

def find_closest(nr,a):
    i0 = np.nonzero(a <= nr)[-1][-1]
    i1 = np.nonzero(nr < a)[0][0]

    if ((a[i0]-nr)**2 <= (a[i1]-nr)**2):
        return i0
    else:
        return i1