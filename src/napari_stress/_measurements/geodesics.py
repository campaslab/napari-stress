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

def correlation_on_surface(surface1: SurfaceData,
                           surface2: SurfaceData,
                           distance_matrix: np.ndarray = None,
                           maximal_distance: float = None) -> dict:
    """
    Calculate (auto-) correlation of features on surface.

    Parameters
    ----------
    surface1 : SurfaceData
    surface2 : SurfaceData

    Returns
    -------
    dict
        Dictionary with information about correlation between `feature1` and
        `feature2` on a surface. The keys of the dictionary are:#
            * `auto_corrs_microns_dists`: Distances for which correlations were binned
            * `auto_correlations_average`: Correlations between averaged feature values
            * `auto_correlations_normalized`: Normalized, un-averaged correlations
            * `auto_correlations_averaged_normalized`: Normalized, averaged correlations

    """
    if distance_matrix is None:
        distance_matrix = geodesic_distance_matrix(surface1)

    n_points = len(surface1[0])
    dists_lbdv_non0 = distance_matrix[np.triu_indices(n_points)]

    if maximal_distance is None:
        maximal_distance = int(np.floor(max(dists_lbdv_non0)))

    # get features from surfaces
    feature1 = surface1[-1]
    feature2 = surface2[-1]

    # Calculate outer product of input features
    Corr_outer_prod_mat_pts = np.dot(feature1.reshape(n_points, 1), feature2.reshape(n_points, 1).T )
    Corr_non0_pts = Corr_outer_prod_mat_pts[np.triu_indices(n_points)]

    # calculate autocorrelation for all points
    # norm, so auto-corrs are 1 at \delta (= |x - x'|) = 0
    auto_corr_norm = np.average(
        np.diag(Corr_outer_prod_mat_pts).flatten(), axis=0
        )

    avg_mean_curv_auto_corrs = []
    dists_used = []

    # calculate autocorrelation with spatially averaged feature values
    for dist_i in range(0, maximal_distance+1):

        sum_mean_curv_corr_d_i, num_mean_curv_corr_d_i = _avg_around_pt(dist_i, dists_lbdv_non0, Corr_non0_pts, maximal_distance)

        if(num_mean_curv_corr_d_i > 0):
            mean_curv_corr_d_i = sum_mean_curv_corr_d_i/num_mean_curv_corr_d_i # average in bin
            if(abs(mean_curv_corr_d_i)/auto_corr_norm <= 1.): # include if corr <= 1
                avg_mean_curv_auto_corrs.append(mean_curv_corr_d_i)
                dists_used.append(dist_i)

    num_dists_used = len(dists_used)

    # We get (geodesic) distances we calculate correlations on, using bump fn averages around these distances:
    auto_corrs_microns_dists = np.array( dists_used, dtype=np.dtype('d')).reshape(num_dists_used, 1)
    auto_corrs_avg = np.array( avg_mean_curv_auto_corrs, dtype=np.dtype('d')).reshape(num_dists_used, 1)
    auto_corrs_avg_normed = auto_corrs_avg / auto_corr_norm

    result = {'auto_correlations_distances': auto_corrs_microns_dists,
              'auto_correlations_average': auto_corrs_avg,
              'auto_correlations_normalized': auto_corr_norm,
              'auto_correlations_averaged_normalized': auto_corrs_avg_normed
              }

    return result

def _avg_around_pt(dist_x_c, dists_pts, vals_at_pts, max_dist_used):
    """Bump function weights on pts centered on dist_x_c."""
    dist_max = max_dist_used/20. #10. #5. #10.
    pts_within_1 = np.where(abs(dists_pts - dist_x_c)<=dist_max, 1., 0.) # only look at points within 1
    #sum_pts_within_1 = np.sum(pts_within_1*vals_at_pts) # we can average over all pts within 1

    num_pts_within_1 = np.sum(pts_within_1) # number of points in bin

    pts_vals_within_1 = np.where(abs(dists_pts - dist_x_c)<=dist_max, vals_at_pts, 0.) # only look at points within 1
    #dists_within_1 = np.where(abs(dists_pts - dist_x_c)<=1., dists_pts, 0.) # only look at points within 1
    weights = np.where( abs(dists_pts - dist_x_c)<=1., np.exp(1.- 1./(dist_max**2 - (dists_pts - dist_x_c)**2 )), 0. )
    sum_weights = np.sum(weights.flatten())
    sum_pts_within_1 = np.sum( np.multiply(pts_vals_within_1, weights).flatten() )

    return sum_pts_within_1, sum_weights #num_pts_within_1