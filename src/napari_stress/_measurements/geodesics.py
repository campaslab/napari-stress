# -*- coding: utf-8 -*-

# from .._stress import plots_SPB as plts


import numpy as np
from scipy import stats
from pygeodesic import geodesic
import pandas as pd

from napari.types import SurfaceData, PointsData, LayerDataTuple

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
    distance_matrix = np.triu(np.ones((n_points, n_points)), k=1)

    # get list of pair indices without permutations (only (1,0), not (0,1)).
    pairs = np.argwhere(distance_matrix == 1)

    for pair in tqdm.tqdm(pairs, desc='Calculating geodesic distances'):
        try:
            distances, _ = geoalg.geodesicDistances([pair[0]], None)
            distance_matrix[pair[0], :] = distances
        except Exception:
            print('hello')

    return distance_matrix


def geodesic_analysis(anisotropic_stress: SurfaceData,
                      anisotropic_stress_tissue: SurfaceData,
                      anisotropic_stress_cell: SurfaceData,
                      maximal_distance: float = None):
    """
    Analyze geodesic distances.

    Parameters
    ----------
    anisotropic_stress: np.ndarray
        total anisotropic stress at every point of droplet surface
    anisotropic_stress_cell: np.ndarray
        cell-scale anisotropic stress on droplet surface
    anisotropic_stress_tissue: np.ndarray
        tissue-scale anisotropic stress on surface of LSQ ellipsoid
    maximal_distance: float
        distance within which surface points should be included in the analysis.

    Returns
    -------
    None.

    """
    GDM = None
    if maximal_distance is None:
        # calculate geodesic distances
        GDM = geodesic_distance_matrix(anisotropic_stress)
        maximal_distance = int(np.floor(GDM.max()))

    # Compute Overall Stress spatial correlations
    autocorrelations = correlation_on_surface(anisotropic_stress,
                                              anisotropic_stress,
                                              distance_matrix=GDM,
                                              maximal_distance=maximal_distance)

    # Compute Cellular Stress spatial correlations
    autocorrelations_cell = correlation_on_surface(anisotropic_stress_cell,
                                                   anisotropic_stress_cell,
                                                   distance_matrix=GDM,
                                                   maximal_distance=maximal_distance)

    # Compute Tissue Stress spatial correlations
    autocorrelations_tissue = correlation_on_surface(anisotropic_stress_tissue,
                                                     anisotropic_stress_tissue,
                                                     distance_matrix=GDM,
                                                     maximal_distance=maximal_distance)

    #########################################################################
    # Do Local Max/Min analysis on 2\gamma*(H - H0) and 2\gamma*(H - H_ellps) data:
    min_max_distances = local_min_max_and_dists(anisotropic_stress, GDM)
    min_max_distances_cell = local_min_max_and_dists(anisotropic_stress_cell, GDM)

    results = {'autocorrelations': autocorrelations,
               'autocorrelations_cell': autocorrelations_cell,
               'autocorrelations_tissue': autocorrelations_tissue,
               'min_max_distances': min_max_distances,
               'min_max_distances_cell': min_max_distances_cell}

    return results


# calculate correlations from input vecs (at same pts), and dists mats:
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
    Corr_outer_prod_mat_pts = np.dot(feature1.reshape(n_points, 1), feature2.reshape(n_points, 1).T)
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
            mean_curv_corr_d_i = sum_mean_curv_corr_d_i/num_mean_curv_corr_d_i  # average in bin
            if(abs(mean_curv_corr_d_i)/auto_corr_norm <= 1.):  # include if corr <= 1
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


def local_extrema_on_surface(surface: SurfaceData) -> np.ndarray:
    """
    Find local extrema of a feature on a surface.

    Parameters
    ----------
    surface : SurfaceData

    Returns
    -------
    local_extrema : np.ndarray
        Array of length `n_vertices` with 1 = maximum and -1 = minimum

    """
    import vedo
    surface_vedo = vedo.mesh.Mesh([surface[0], surface[1]])

    feature = surface[-1]

    local_extrema = np.zeros_like(feature, dtype=int)

    # iterate over all points, find neighbors and check the feature values
    # at the neighboring points
    for idx in range(surface_vedo.N()):
        ids = surface_vedo.connectedVertices(idx)
        feature_at_neighbor = feature[ids]

        if (feature[idx] > feature_at_neighbor).all():
            local_extrema[idx] = 1

        if (feature[idx] < feature_at_neighbor).all():
            local_extrema[idx] = -1

    return local_extrema



def local_min_max_and_dists(surface: SurfaceData,
                            distance_matrix: np.ndarray = None):
    """
    Get local maximum and minimum.

    Parameters
    ----------
    feature : np.ndarray
        Feature on surface
    distance_matrix : np.ndarray
        DESCRIPTION.
    surface : tuple
        DESCRIPTION.

    Returns
    -------
    min_max_distances : pd.DataFrame
        DataFrame with length = n_vertices on surface. The columns denote:
            * `local_maxima_and_minima`: np.ndarray with value +1 if point i is
            a maximum, -1 if it's a minimum or 0 if it's neither.
            * `nearest_min_max_distances`: Distance to nearest minimum or
            maximum
            * `delta_feature_nearest_min_max`: Difference in passed feature
            between nearest minimum and maximum

    pair_distances: pd.DataFrame
        DataFrame with length n_minima/maxima. The columns denote:
            * `min_max_pair_distances`: Distances between pairs of neighboring
            maximum/minimum on surface
            * `min_max_pair_anisotropies`: Anisotropy of passed feature between
            pairs of neighboring maximum/minimum on surface
    """
    # triangles = surface[1]
    feature = surface[2]

    # quad_fit = len(feature)

    # local_max_and_min = np.zeros_like(feature) # should be -1 at local min, +1 at local max, 0 otherwise, 2 if both
    # nearest_min_max_dists = np.zeros_like(feature) # 0 if not local max or min; but for local max is distance to nearest local min, vice versa

    # # Matrix to store distances between maxima and minima
    # max_min_distance_matrix = np.zeros_like(distance_matrix)

    # delta_feature_nearest_min_max = np.zeros_like(feature)  # 2*(H_in_max - H_in_min), from extrema to nearest partner

    # for pt in range(quad_fit):

    #     H_pt = feature[pt]

    #     # set to True, change if False:
    #     pt_is_local_max = True
    #     pt_is_local_min = True

    #     tris_containing_pt = np.where( triangles == pt )
    #     rows_pt = tris_containing_pt[0]

    #     for row_pt in range(len(rows_pt)):
    #         row_num_pt = rows_pt[row_pt]

    #         # compare to other pts in triangle
    #         for other_tri_pt_num in range(3):
    #             other_tri_pt =     int(triangles[row_num_pt, other_tri_pt_num])
    #             H_other_tri_pt = feature[other_tri_pt]

    #             if(H_other_tri_pt < H_pt):
    #                 pt_is_local_min = False
    #             if(H_other_tri_pt > H_pt):
    #                 pt_is_local_max = False

    #     if(pt_is_local_max == True):
    #         if(pt_is_local_min == True):
    #             local_max_and_min[pt] = 2 # local max AND min
    #         else:
    #             local_max_and_min[pt] = 1 # local max (only)

    #     elif(pt_is_local_min == True):
    #         local_max_and_min[pt] = -1 # local min (only)
    #     else:
    #         local_max_and_min[pt] = 0

    local_extrema = local_extrema_on_surface(surface)

    # STEP 1
    # find distances between all maxima/minima
    inter_extremal_distances = np.zeros((len(local_extrema),
                                         len(local_extrema)))

    geoalg = geodesic.PyGeodesicAlgorithmExact(surface[0], surface[1])
    for idx in np.argwhere(local_extrema != 0):
        distances, _ = geoalg.geodesicDistances(idx)
        inter_extremal_distances[idx] = distances * (local_extrema != 0)

    # # STEP 2
    # get indices of extrema and the feature anisotropies between all extrema:
    # I.e., for 5 maxima and five minima, the result should have length 25,
    # for there are 5 neighboring minima for each maximum
    distances_max_to_min = []
    feature_anisotropy_max_to_min = []
    local_min = np.argwhere(local_extrema == -1).squeeze()
    for idx, pt in enumerate(np.argwhere(local_extrema == 1).squeeze()):

        delta_distances = inter_extremal_distances[pt][local_min]
        delta_feature = feature[pt] - feature[local_min]
        distances_max_to_min.append(delta_distances)
        feature_anisotropy_max_to_min.append(delta_feature)

    distances_max_to_min = np.concatenate(distances_max_to_min)
    feature_anisotropy_max_to_min = np.concatenate(
        feature_anisotropy_max_to_min)

    is_minimum = local_extrema == -1
    is_maximum = local_extrema == 1
    distance_to_nearest_extremum = np.zeros_like(local_extrema)

    # STEP 3a:
    # find pairs of maxima and nearest minima
    max_min_pairs = []
    max_min_pair_anisotropy = []

    for idx in np.argwhere(is_maximum).squeeze():

        # get the inter-extremal distances only between point idx and minima
        distances_to_minima = inter_extremal_distances[idx] * is_minimum

        # find index of nearest minimum
        distances_to_minima[distances_to_minima == 0] = np.nan
        idx_nearest_miniumum = np.nanargmin(distances_to_minima)
        delta_feature = feature[idx] - feature[idx_nearest_miniumum]

        distance_to_nearest_extremum
        max_min_pairs.append((idx, idx_nearest_miniumum))
        max_min_pair_anisotropy.append(delta_feature)

    # STEP 3b:
    # find pairs of minimum and nearest maximum
    min_max_pairs = []
    min_max_pair_anisotropy = []

    for idx in np.argwhere(is_minimum).squeeze():
        # get the inter-extremal distances only between point idx and minima
        distances_to_maxima = inter_extremal_distances[idx] * is_maximum

        # find index of nearest maximum
        distances_to_maxima[distances_to_maxima == 0] = np.nan
        idx_nearest_maximum = np.nanargmin(distances_to_maxima)
        delta_feature = feature[idx] - feature[idx_nearest_maximum]

        min_max_pairs.append((idx, idx_nearest_maximum))
        min_max_pair_anisotropy.append(delta_feature)

    # Step 4: Format results as dataframe
    df = pd.DataFrame(np.argwhere(local_extrema != 0).squeeze(),
                      columns=['local_extrema'])
    df['is_maximum'] = local_extrema[df['local_extrema']] == 1
    df['is_minimum'] = local_extrema[df['local_extrema']] == -1

    # num_local_max = np.count_nonzero(local_extrema == 1)
    # num_local_min = np.count_nonzero(local_extrema == -1)

    # local_max_inds = np.where( local_max_and_min == 1 )[0]
    # local_min_inds = np.where( local_max_and_min == -1 )[0]

    # # list of ALL local min/max pairs' distances and difference in input fields:
    # min_max_pair_anisotropies = []
    # min_max_pair_distances = []

    # for pt_max in range(num_local_max):
    #     pt_max_num = local_max_inds[pt_max]
    #     local_max_field = feature[pt_max_num]

    #     for pt_min in range(num_local_min):
    #         pt_min_num = local_min_inds[pt_min]
    #         local_min_field = feature[pt_min_num]

    #         dist_max_min_pt = distance_matrix[pt_min_num, pt_max_num]
    #         Anisotropy_max_min_pts = local_max_field - local_min_field

    #         min_max_pair_anisotropies.append(Anisotropy_max_min_pts)
    #         min_max_pair_distances.append(dist_max_min_pt)

    #         max_min_distance_matrix[pt_min_num, pt_max_num] = dist_max_min_pt
    #         max_min_distance_matrix[pt_max_num, pt_min_num] = dist_max_min_pt

    # min_max_pair_anisotropies = np.array(min_max_pair_anisotropies, dtype=np.dtype('d'))
    # min_max_pair_distances = np.array(min_max_pair_distances, dtype=np.dtype('d'))

    # for pt_max in range(num_local_max):
    #     pt_max_num = local_max_inds[pt_max]

    #     local_min_dists_to_pt = distance_matrix[pt_max_num, local_min_inds]
    #     min_dist_to_local_min = min(local_min_dists_to_pt)
    #     nearest_min_max_dists[pt_max_num] = min_dist_to_local_min

    #     # Calculate 2*(H_max - H_nearest_min):
    #     ind_in_list_of_nearest_min = np.argwhere(local_min_dists_to_pt == min_dist_to_local_min)
    #     pt_num_of_nearest_min = local_min_inds[ind_in_list_of_nearest_min][0,0]

    #     delta_feature_nearest_min_max[pt_max_num] = ( feature[pt_max_num] - feature[pt_num_of_nearest_min] )

    # for pt_min in range(num_local_min):
    #     pt_min_num = local_min_inds[pt_min]

    #     local_max_dists_to_pt = distance_matrix[pt_min_num, local_max_inds]

    #     max_dist_to_local_min = min(local_max_dists_to_pt)
    #     nearest_min_max_dists[pt_min_num] = max_dist_to_local_min

    #     # Calculate 2*(H_min - H_nearest_max):
    #     ind_in_list_of_nearest_max = np.argwhere(local_max_dists_to_pt == max_dist_to_local_min)
    #     pt_num_of_nearest_max = local_max_inds[ind_in_list_of_nearest_max][0,0]
    #     delta_feature_nearest_min_max[pt_min_num] = ( feature[pt_num_of_nearest_max] - feature[pt_min_num] )

    result1 = {'local_max_and_min': local_extrema,
               'nearest_min_max_dists': nearest_min_max_dists,
               'delta_feature_nearest_min_max': delta_feature_nearest_min_max,
               'min_max_distance_matrix': max_min_distance_matrix}

    result2 = {'min_max_pair_distances': distances_max_to_min,
                'min_max_pair_anisotropies': feature_anisotropy_max_to_min}

    return result1, result2
