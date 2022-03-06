# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 10:20:17 2021

@author: johan
"""

import numpy as np
import tqdm
from scipy import interpolate
import pandas as pd
import vedo
from joblib import Parallel, delayed
import typing

from ._utils import pointcloud_to_vertices4D

def get_traces_4d(image: np.ndarray,
                  start_pts: list,
                  target_pts: list,
                  sample_distance: float = 1.0,
                  detection = 'quick_edge',
                  fluorescence = 'interior',
                  num_threads: int = -1) -> list:
    
    # Parse and check inputs
    assert len(image.shape) == 4
        
    if isinstance(target_pts[0], vedo.pointcloud.Points):
        target_pts = [pt.points() for pt in target_pts]
        
    if isinstance(start_pts[0], vedo.pointcloud.Points):
        start_pts = [pt.points() for pt in start_pts]
    
    # define task to be parallelized
    def task(image, start_pts, target_pts, sample_distace, detection, fluorescence):
        return get_traces(image, start_pts, target_pts, 
                          sample_distance=sample_distance,
                          detection=detection,
                          fluorescence=fluorescence)   

    n_frames = image.shape[0]
    args = list(zip([image[t] for t in range(n_frames)],
                    start_pts, target_pts,
                    [sample_distance] * n_frames,
                    [detection]* n_frames,
                    [fluorescence] * n_frames))
    
    # Execute parallel calculation
    results = Parallel(n_jobs=num_threads)(delayed(task)(*arg) for arg in args)
    
    # Parse outputs
    pts = []
    for res in results:
        pt = vedo.pointcloud.Points(res['points'])
        for key in res.keys():
            pt.pointdata[key] = res[key]
        pts.append(pt)
        
    return pts
        

def get_traces(image: np.ndarray,
               start_pts: np.ndarray,
               target_pts: np.ndarray,
               sample_distance: float =  1.0,
               detection: str = 'quick_edge',
               fluorescence: str = 'interior') -> [np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate intensity profiles along traces.

    The profiles are interpolated from the input image with linear interpolation
    Parameters
    ----------
    image : np.ndarray
        Input intensity image from which traces will be calculated
    start_pts : np.ndarray
        Nx3 or 1x3 array of points that should be used as base points for trace
        vectors. If vector is 1x3-sized, the same coordinate will be used as
        base point for all trace vectors.
    target_pts : np.ndarray
        Nx3-sized array of points that are used as direction vectors for each
        trace. Traces will be calculated along the line from the base points
        `start_pts` towards `target_pts`
    sample_distance : float, optional
        Distance between sampled intensity along a trace. Default is 1.0
    detection : str, optional
        Detection method, can be `quick_edge` or `advanced`. The default is'quick_edge'.
    fluorescence : str, optional
        Fluorescence type of water droppled, can be `interior` or `surface`.
        The default is 'interior'.

    Returns
    -------
    surface_points : np.ndarray
        Nx3 array of points on the surface of the structure in the image.
    errors : np.ndarray
        Nx1 array of error values for the points on the surface.
    FitParams : np.ndarray
        Nx3 array with determined fit parameters if detection was set to `advanced`

    """
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
    for idx in range(target_pts.shape[0]):

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


    return {'points': surface_points, 'errors': errors, 'fitparams': FitParams}
