# -*- coding: utf-8 -*-

import pandas as pd
from napari.types import LayerDataTuple
import os
from pathlib import Path

DATA_ROOT  = os.path.join(Path(__file__).parent, 'sample_data')

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
