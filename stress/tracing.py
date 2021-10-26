# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 10:20:17 2021

@author: johan
"""

import numpy as np
import tqdm
from scipy import interpolate
import pandas as pd

def get_traces(image, start_pts, target_pts, **kwargs):
    """
    Generates intensity profiles along traces defined by start points and target points.
    The profiles are interpolated from the input image with linear interpolation

    Parameters
    ----------
    image : array 
        3D-image containing the raw intesity values
    start_pts : Nx3 array
        list of length-3 coordinate vectors that refer to the starting location of each trace.
    target_pts : Mx3 array
        list of length-3 coordinate vectors that refer to the target direction of each trace.

    Returns
    -------
    list: List of intensity profiles
    """
    
    sample_distance = kwargs.get('sample_distance', 1)
    detection = kwargs.get('detection', 'quick_edge')
    fluorescence = kwargs.get('fluorescence', 'interior')
    
    # Do type conversion if points come from pandas array
    if type(target_pts) == pd.core.series.Series:
        target_pts = np.vstack(target_pts)
    if type(start_pts) == pd.core.series.Series:
        start_pts = np.vstack(start_pts)
    
    # If only a single start point is provided
    if target_pts.shape[0] == 3:
        target_pts = target_pts.transpose()
    
    # allocate vector for origin points, repeat N times if only one coordinate is provided
    if len(start_pts.shape) == 1:
        start_pts = np.asarray([start_pts]).repeat(target_pts.shape[0], axis=0)
        
    # Create coords for interpolator
    Z = np.arange(0, image.shape[0], 1)
    Y = np.arange(0, image.shape[1], 1)
    X = np.arange(0, image.shape[2], 1)
    coords = (Z, Y, X)
    
    profiles = []
    surface_points = []
    
    # Iterate over all provided target points
    for idx in tqdm.tqdm(range(target_pts.shape[0]), desc='Shooting rays: '):
        
        profile = []
        r = 0
        
        # calculate normalized vector from center to target
        v = target_pts[idx, :] - start_pts[idx, :]
        v  = v/np.linalg.norm(v)
        
        while True:

            # walk from center outwards
            p = start_pts[idx] + r * v
            try:
                profile.append(interpolate.interpn(coords, image, p)[0])
            except Exception:
                break
                    
            r += sample_distance
            
        if detection == 'quick_edge':
            if fluorescence == 'interior':
                # Find point where gradient is maximal
                idx_border = np.argmax(np.abs(np.diff(profile)))
            
            elif fluorescence == 'surface':
                # Find point where intensity is maximal
                idx_border = np.argmax(profile)
        
        # get coordinate of surface point
        surf_point = start_pts[idx] + (idx_border * sample_distance) * v
        
        # Append to list
        surface_points.append(surf_point)
        profiles.append(profile)

    surface_points = np.asarray(surface_points)
    if surface_points.shape[0] == 3:
        surface_points= surface_points.transpose()
        
    if detection == 'quick_edge':
        errors = np.zeros(np.max(target_pts.shape))
        FitParams = np.zeros(np.max(target_pts.shape))
        
    # df = pd.DataFrame(columns=['XYZ', 'X', 'Y', 'Z', 'FitErrors', 'FitParams'])
    
    # points['XYZ'] = surface_points
    # surface_points = np.asarray(surface_points)
    # # points['X'] = surface_points[:, 0]
    # # points['Y'] = surface_points[:, 1]
    # # points['Z'] = surface_points[:, 2]
    # points['FitErrors'] = errors
    # points['FitParams'] = FitParams
        
    return surface_points, errors, FitParams