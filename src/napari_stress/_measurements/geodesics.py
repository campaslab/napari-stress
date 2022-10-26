import numpy as np
from pygeodesic import geodesic

from napari.types import SurfaceData

import tqdm


def geodesic_distance_matrix(surface: SurfaceData) -> np.ndarray:
    """
    Calculate a pairwise distance matrix for vertices of a surface.

    This calculates the geodesic distances between any two vertices on a given
    surface.

    Parameters
    ----------
    surface : SurfaceData

    Returns
    -------
    distance_matrix : np.ndarray
        Triangular matrix with differences between vertex i and j located
        at `distance_matrix[i, j]`

    """
    geoalg = geodesic.PyGeodesicAlgorithmExact(surface[0], surface[1])

    n_points = len(surface[0])
    distance_matrix = np.zeros((n_points, n_points))
    points = surface[0]

    for idx, pt in tqdm.tqdm(enumerate(points), desc='Calculating geodesic distances'):
        distances, _ = geoalg.geodesicDistances([idx], None)
        distance_matrix[idx, :] = distances

    return distance_matrix
