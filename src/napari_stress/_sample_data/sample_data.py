# -*- coding: utf-8 -*-

import pandas as pd
from napari.types import LayerDataTuple
import os
from pathlib import Path

DATA_ROOT  = os.path.join(Path(__file__).parent)

def get_droplet_point_cloud() -> LayerDataTuple:
    """Generates a sample point cloud of a droplet surface"""

    df = pd.read_csv(os.path.join(DATA_ROOT, 'dropplet_point_cloud.csv'), sep=',')
    coordinates = df[['axis-0', 'axis-1', 'axis-2', 'axis-3']].to_numpy()

    return [(coordinates, {'size': 0.5, 'face_color': 'orange'}, 'points')]

def get_droplet_point_cloud_4d() -> LayerDataTuple:
    """Generates a sample 4d point cloud of a droplet surface"""

    df = pd.read_csv(os.path.join(DATA_ROOT, 'dropplet_point_cloud_4d.csv'), sep=',')
    coordinates = df[['axis-0', 'axis-1', 'axis-2', 'axis-3']].to_numpy()

    return [(coordinates, {'size': 0.5, 'face_color': 'orange'}, 'points')]

def get_droplet_4d() -> LayerDataTuple:
    """
    Loads a sample 4d point cloud of a droplet surface.

    Source:https://github.com/campaslab/STRESS
    """
    from skimage import io

    image = io.imread(os.path.join(DATA_ROOT, 'ExampleTifSequence.tif'))

    return [(image, {}, 'image')]

def make_binary_ellipsoid(major_axis_length: float = 10.0,
                   medial_axis_length: float = 7.0,
                   minor_axis_length: float = 5.0,
                   sampling: float = 0.5,
                   edge_padding: int = 5) -> LayerDataTuple:
    """
    Creates a 3D ellipsoid with the given dimensions and sampling.

    Parameters
    ----------
    major_axis_length : float
        Length of the major axis of the ellipsoid.
    medial_axis_length : float
        Length of the medial axis of the ellipsoid.
    minor_axis_length : float
        Length of the minor axis of the ellipsoid.
    sampling : float
        Sampling of the ellipsoid.
    edge_padding : int
        Number of voxels to pad the ellipsoid with.

    Returns
    -------
    LabelsData
        The ellipsoid as a 3D binary image.
    """
    import vedo
    import numpy as np

    ellipsoid = vedo.Ellipsoid(
        pos=(0, 0, 0),
        axis1=(major_axis_length, 0, 0),
        axis2=(0, medial_axis_length, 0),
        axis3=(0, 0, minor_axis_length)).binarize(spacing=[sampling]*3).tonumpy()
    
    properties = {
        'name': 'Ellipsoid'
    }
    return (np.pad(ellipsoid, edge_padding).astype(int), properties, 'labels')
