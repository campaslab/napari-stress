import numpy as np
from pygeodesic import geodesic

from .._utils.frame_by_frame import frame_by_frame
from napari_tools_menu import register_function

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

@register_function(menu="Surfaces > Extract Geodesic path between vertices (pygeodesics, n-STRESS)")
@frame_by_frame
def geodesic_path(surface: SurfaceData, index_1: int, index_2: int
                  ) -> VectorsData:
    """
    Calculate the geodesic path between two index-defined surface vertices .

    Parameters
    ----------
    surface : SurfaceData
    index_1 : int
        Index of start vertex
    index_2 : int
        Index of destination vertex

    Returns
    -------
    VectorsData

    """
    geoalg = geodesic.PyGeodesicAlgorithmExact(surface[0], surface[1])
    distances, path = geoalg.geodesicDistance(index_1, index_2)

    # convert points to vectors from point to point
    vectors = []
    for i in range(len(path)-1):
        vectors.append(path[i+1] - path[i])
    vectors = np.asarray(vectors)
    napari_vectors = np.stack([path[:-1], vectors]).transpose((1,0,2))

    return napari_vectors