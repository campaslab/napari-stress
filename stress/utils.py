# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 10:31:29 2021

@author: johan
"""

import numpy as np
import pandas

def get_IQs(array):
    """Calculate quantiles and interquartile distance for a given array"""
    
    array = np.asarray(array).flatten()
    
    median = np.median(array)
    I25 = np.quantile(array, 0.25)
    I75 = np.quantile(array, 0.75)
    IQ = I75 - I25
    
    return I25, median, I75, IQ

def cart2sph(x,y,z):
    
    radius = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/radius)
    phi = np.zeros_like(theta)
    
    for i in range(len(x)):
        if x[i] >= 0:
            phi[i] = np.arctan(y[i]/x[i])
        else:
            phi[i] = np.arctan(y[i]/x[i]) + np.pi
            
    return phi, theta, radius


def df2XYZ(df):
    "Convert the XYZ columns of a dataframe to an Nx3 array"
    

    return np.vstack([df.X, df.Y, df.Z]).transpose()