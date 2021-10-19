# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 17:34:45 2021

@author: johan
"""

import numpy as np
# import pandas as pd
from scipy.interpolate import RBFInterpolator, interp2d, griddata
import matplotlib.pyplot as plt
import napari


def surf_fit(points, Xq, **kwargs):
    
    """
    First interpolates a set of points around a query point with a set of radial 
    basis functions in a given sample density. The inteprolated points are then approximated 
    with a 2D polynomial
    """
    
    sample_length = kwargs.get('sample_length', 0.1)
    int_method = kwargs.get('int_method', 'rbf')
    
    # get x, y and z coordinates
    x = np.asarray(points[:, 0])
    y = np.asarray(points[:, 1])
    z = np.asarray(points[:, 2])
    
    # # create interpolation grid
    # xi = np.linspace(np.min(x), np.max(x), ((x.max() - x.min()) // sample_length).astype(int))
    # yi = np.linspace(np.min(x), np.max(x), ((y.max() - y.min()) // sample_length).astype(int))
    
    # create interpolation/evaluation grid
    sL = sample_length
    
    # add 1 sL to grid range ro ensure interpolation grid of sufficient size to calculate gradients
    xgrid = np.mgrid[x.min() - sL : x.max() + sL : sL,
                     y.min() - sL : y.max() + sL : sL]
    
    shape_x = xgrid.shape[1]
    shape_y = xgrid.shape[2]
    xgrid = np.asarray(xgrid.reshape(2,-1).T)
    
    # Create polynomial approximation of provided data on a regular grid with set sample length
    if int_method == 'rbf':
        rbf = RBFInterpolator(np.vstack([x,y]).transpose(), z, epsilon=2)
        _x = xgrid[:,0]
        _y = xgrid[:,1]
        _z = rbf(xgrid)
        
    # elif int_method =='grid':
    #     grid = griddata(np.vstack([x,y]).transpose(), z, xgrid.transpose(), method='linear')
        
    elif int_method == 'Poly2d':
        # Fit custom 2D Polynomial function. Some of the moments are missing in matlab - intented?
        z_poly = poly2d(_x, _y)
        coeff, r, rank, s = np.linalg.lstsq(z_poly, _z, rcond=None)
        _z = poly2d(_x, _y, coeff=coeff).sum(axis=1)
    
    # Make data 2D (i.e., grid) again
    _x = _x.reshape(shape_x, -1)
    _y = _y.reshape(shape_x, -1)
    _z = _z.reshape(shape_x, -1)
    
    if _z.shape[0] == 1:
        print('Here')
        pass
    
    # Calculate the mean curvature of the interpolated surface 
    H = mean_curvature(_z, sample_length)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(_x, _y, _z)
    
    # Interpolate mean curvature at query point
    f = interp2d(_x, _y, H, kind='linear')
    H_mean = f(Xq[0], Xq[1])
    
    return H_mean

def mean_curvature(z, spacing):
    """
    Calculates mean curvature based on the partial derivatives of z based on
    the formula from https://en.wikipedia.org/wiki/Mean_curvature
    """
    
    try:
        Zy, Zx = np.gradient(z, spacing)  # First-order partial derivatives
        Zxy, Zxx = np.gradient(Zx, spacing) # Second-order partial derivatives (I)
        Zyy, _ = np.gradient(Zy, spacing) # (II)  (note that Zyx = Zxy)
    except:
        print('Here')
    
    H = (1/2.0) * ((1 + Zxx**2) * Zyy - 2.0 * Zx * Zy * Zxy + (1 + Zyy**2) * Zxx)/ \
        (1 + Zxx**2 + Zyy**2)**(1.5)
    
    return H
    
    
def poly2d(x, y, coeff=np.ones(9)):
    
    assert len(coeff) == 9
    
    return np.array([coeff[0] * np.ones(len(x)),
                     coeff[1] * x,
                     coeff[2] * y,
                     coeff[3] * x*y,
                     coeff[4] * x**2,
                     coeff[5] * x**2 * y,
                     coeff[6] * x*y**2,
                     coeff[7] * y**2,
                     coeff[8] * x**2 * y**2]).T