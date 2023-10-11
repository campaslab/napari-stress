# -*- coding: utf-8 -*-

import pandas as pd
from napari.types import LayerDataTuple
import os
from pathlib import Path

DATA_ROOT = os.path.join(Path(__file__).parent)


def get_droplet_point_cloud() -> LayerDataTuple:
    """Generates a sample point cloud of a droplet surface"""

    df = pd.read_csv(os.path.join(DATA_ROOT, "dropplet_point_cloud.csv"), sep=",")
    coordinates = df[["axis-0", "axis-1", "axis-2", "axis-3"]].to_numpy()

    return [(coordinates, {"size": 0.5, "face_color": "orange"}, "points")]


def get_droplet_point_cloud_4d() -> LayerDataTuple:
    """Generates a sample 4d point cloud of a droplet surface"""

    df = pd.read_csv(os.path.join(DATA_ROOT, "dropplet_point_cloud_4d.csv"), sep=",")
    coordinates = df[["axis-0", "axis-1", "axis-2", "axis-3"]].to_numpy()

    return [(coordinates, {"size": 0.5, "face_color": "orange"}, "points")]


def get_droplet_4d() -> LayerDataTuple:
    """
    Loads a sample 4d point cloud of a droplet surface.

    Source:https://github.com/campaslab/STRESS
    """
    from skimage import io

    image = io.imread(os.path.join(DATA_ROOT, "ExampleTifSequence.tif"))

    return [(image, {}, "image")]


def make_blurry_ellipsoid(
    axis_length_a: float = 0.7,
    axis_length_b: float = 0.3,
    axis_length_c: float = 0.3,
    size: int = 64,
    definition_width: int = 5,
) -> LayerDataTuple:
    """Generates a blurry ellipsoid.

    Parameters
    ----------
    axis_length_a : float
        Length of the major axis of the ellipsoid.
        Must be greater than 0 and less than 1.
    axis_length_b : float
        Length of the medial axis of the ellipsoid.
        Must be greater than 0 and less than 1.
    axis_length_c : float
        Length of the minor axis of the ellipsoid.
        Must be greater than 0 and less than 1.
    size : int
        Size of the image.
    definition_width : int
        Steepness of the intensity gradient on the edge
        of the ellipsoid.

    Returns
    -------
    LayerDataTuple
        A blurry ellipsoid.
    """
    import numpy as np

    def sigmoid(x, a):
        return 1 / (1 + np.exp(-a * x))

    x, y, z = np.meshgrid(
        np.linspace(-1, 1, size),
        np.linspace(-1, 1, size),
        np.linspace(-1, 1, size),
        indexing="ij",
    )

    ellipsoid = (
        (x / axis_length_a) ** 2 + (y / axis_length_b) ** 2 + (z / axis_length_c) ** 2
    )

    blurry_image = 1 - sigmoid(ellipsoid - 1, definition_width)
    return (blurry_image, {"name": "blurry_ellipsoid"}, "image")
