# -*- coding: utf-8 -*-

# from .._stress import plots_SPB as plts


import numpy as np
from pygeodesic import geodesic

from napari.types import SurfaceData, LayerDataTuple
import tqdm

from .._utils.frame_by_frame import frame_by_frame

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


    # # FIND CDF for 2\gamma*(H - H_0) and exclude points with curvature within
    # # alpha tail of CDF function:
    # cdf_info = _CDF_Analysis_of_Data(anisotropic_stress, alpha_total)
    # num_curvs_above_max_AnisStress = np.count_nonzero(cdf_info['excluded_values']== 1)
    # num_curvs_below_min_AnisStress = np.count_nonzero(cdf_info['excluded_values']== -1)

    # # FIND CDF for 2\gamma*(H - H_ellps):
    # cdf_info_cell = _CDF_Analysis_of_Data(anisotropic_stress_cell, alpha_cell)
    # num_curvs_above_max_AnisCellStress = np.count_nonzero(cdf_info_cell['excluded_values']== 1)
    # num_curvs_below_min_AnisCellStress = np.count_nonzero(cdf_info_cell['excluded_values']== -1)

    # # Find CDF for 2\gamma*(H_Local_Max - H_Local_Min) distribution, using ALL pairs:
    # cdf_info_autocorrelations = _CDF_Analysis_of_Data(autocorrelations['min_max_pair_anisotropies'],
    #                                                   alpha_min_max_exclusion)

    # # # FIND CDF for 2\gamma*(H_Input - H_0_Input):
    # # self.min_val_excl_AnisStress_Input_UV, self.max_val_excl_AnisStress_Input_UV, self.curv_pts_excluded_AnisStress_Input_UV, self.hist_dist_AnisStress_Input_UV = _CDF_Analysis_of_Data(self.Anis_Stress_pts_UV_input, Droplet_Input_Dict['alpha_percentile_excl_AnisStress']) # same \alpha as corresponding lbdv field

    # # # FIND CDF for 2\gamma*(H_Input - H_ellps_{INPUT}):
    # # self.min_val_excl_AnisCellStress_Input_UV, self.max_val_excl_AnisCellStress_Input_UV, self.curv_pts_excluded_AnisCellStress_Input_UV, self.hist_dist_AnisCellStress_Input_UV = _CDF_Analysis_of_Data(self.Anis_Cell_Stress_pts_UV_input, Droplet_Input_Dict['alpha_percentile_excl_AnisCellStress']) # same \alpha as corresponding lbdv field


    # # look at which points exluded from 2\gamma*HmHe are also local min/max of 2\gamma*H:
    # # should be 1 where these align, 0 otherwise
    # HmHe_exlc_vs_Local_H_min_max_pts = np.multiply(cdf_info_cell['excluded_values'].flatten(),
    #                                                     min_max_distances['local_maxima_and_minima'].flatten() )

    # num_local_extrema_curvs_in_cutoff_AnisCellStress_align = np.count_nonzero(HmHe_exlc_vs_Local_H_min_max_pts == 1)
    # num_local_extrema_curvs_in_cutoff_AnisCellStress_anti_align = np.count_nonzero(HmHe_exlc_vs_Local_H_min_max_pts == -1)

    # return metadata


    return autocorrelations

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

# # CDF Analysis for distribution:
# def _CDF_Analysis_of_Data(data: np.ndarray,
#                          delta_prob_extrema_exlc: float = 0.05) -> dict:
#     """Create and analyze cumulative density function of data."""

#     sorted_Data_Field = np.sort(data.flatten())

#     # from: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_histogram.html
#     hist_Data_Field = np.histogram(sorted_Data_Field, bins='auto', density=True)
#     hist_dist = stats.rv_histogram(hist_Data_Field)

#     min_val_excl_Data_Field = hist_dist.ppf(delta_prob_extrema_exlc)
#     max_val_excl_Data_Field = hist_dist.ppf(1. - delta_prob_extrema_exlc)

#     curv_pts_excluded_Data_Field = np.zeros_like(sorted_Data_Field) # 0: where within .5- \delta of median, -1, where CDF<\delta, 1 where CDF>1-\delta
#     curv_pts_excluded_Data_Field = np.where(sorted_Data_Field < min_val_excl_Data_Field, -1, curv_pts_excluded_Data_Field)
#     curv_pts_excluded_Data_Field = np.where(sorted_Data_Field > max_val_excl_Data_Field, 1, curv_pts_excluded_Data_Field)

#     results = {
#         'lower_quantile': min_val_excl_Data_Field,
#         'upper_quantile': max_val_excl_Data_Field,
#         'excluded_values': curv_pts_excluded_Data_Field,
#         'distance_histogram': hist_dist
#         }

    # return results

def local_extrema_analysis(surface: SurfaceData,
                           distance_matrix: np.ndarray = None
                           ) -> LayerDataTuple:
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
            * `nearest_min_max_distances`: Distance to nearest minimum or maximum
            * `delta_feature_nearest_min_max`: Difference in passed feature
            between nearest minimum and maximum

    pair_distances: pd.DataFrame
        DataFrame with length n_minima/maxima. The columns denote:
            * `min_max_pair_distances`: Distances between pairs of neighboring
            maximum/minimum on surface
            * `min_max_pair_anisotropies`: Anisotropy of passed feature between
            pairs of neighboring maximum/minimum on surface
    """
    triangles = surface[1]
    feature = surface[2]

    quad_fit = len(feature)

    local_max_and_min = np.zeros_like(feature) # should be -1 at local min, +1 at local max, 0 otherwise, 2 if both
    nearest_min_max_dists = np.zeros_like(feature) # 0 if not local max or min; but for local max is distance to nearest local min, vice versa

    delta_feature_nearest_min_max = np.zeros_like(feature)  # 2*(H_in_max - H_in_min), from extrema to nearest partner

    for pt in range(quad_fit):

        H_pt = feature[pt]

        # set to True, change if False:
        pt_is_local_max = True
        pt_is_local_min = True

        tris_containing_pt = np.where( triangles == pt )
        rows_pt = tris_containing_pt[0]

        for row_pt in range(len(rows_pt)):
            row_num_pt = rows_pt[row_pt]

            # compare to other pts in triangle
            for other_tri_pt_num in range(3):
                other_tri_pt =     int(triangles[row_num_pt, other_tri_pt_num])
                H_other_tri_pt = feature[other_tri_pt]

                if(H_other_tri_pt < H_pt):
                    pt_is_local_min = False
                if(H_other_tri_pt > H_pt):
                    pt_is_local_max = False

        if(pt_is_local_max == True):
            if(pt_is_local_min == True):
                local_max_and_min[pt] = 2 # local max AND min
            else:
                local_max_and_min[pt] = 1 # local max (only)

        elif(pt_is_local_min == True):
            local_max_and_min[pt] = -1 # local min (only)
        else:
            local_max_and_min[pt] = 0

    num_local_max = np.count_nonzero(local_max_and_min == 1)
    num_local_min = np.count_nonzero(local_max_and_min == -1)

    local_max_inds = np.where( local_max_and_min == 1 )[0]
    local_min_inds = np.where( local_max_and_min == -1 )[0]

    # list of ALL local min/max pairs' distances and difference in input fields:
    min_max_pair_anisotropies = []
    min_max_pair_distances = []

    for pt_max in range(num_local_max):
        pt_max_num = local_max_inds[pt_max]
        local_max_field = feature[pt_max_num]

        for pt_min in range(num_local_min):
            pt_min_num = local_min_inds[pt_min]
            local_min_field = feature[pt_min_num]

            dist_max_min_pt = distance_matrix[pt_min_num, pt_max_num]
            Anisotropy_max_min_pts = local_max_field - local_min_field

            min_max_pair_anisotropies.append(Anisotropy_max_min_pts)
            min_max_pair_distances.append(dist_max_min_pt)

    min_max_pair_anisotropies = np.array(min_max_pair_anisotropies, dtype=np.dtype('d'))
    min_max_pair_distances = np.array(min_max_pair_distances, dtype=np.dtype('d'))

    for pt_max in range(num_local_max):
        pt_max_num = local_max_inds[pt_max]

        local_min_dists_to_pt = distance_matrix[pt_max_num, local_min_inds]
        min_dist_to_local_min = min(local_min_dists_to_pt)
        nearest_min_max_dists[pt_max_num] = min_dist_to_local_min

        # Calculate 2*(H_max - H_nearest_min):
        ind_in_list_of_nearest_min = np.argwhere(local_min_dists_to_pt == min_dist_to_local_min)
        pt_num_of_nearest_min = local_min_inds[ind_in_list_of_nearest_min][0,0]

        delta_feature_nearest_min_max[pt_max_num] = ( feature[pt_max_num] - feature[pt_num_of_nearest_min] )

    for pt_min in range(num_local_min):
        pt_min_num = local_min_inds[pt_min]

        local_max_dists_to_pt = distance_matrix[pt_min_num, local_max_inds]

        max_dist_to_local_min = min(local_max_dists_to_pt)
        nearest_min_max_dists[pt_min_num] = max_dist_to_local_min

        # Calculate 2*(H_min - H_nearest_max):
        ind_in_list_of_nearest_max = np.argwhere(local_max_dists_to_pt == max_dist_to_local_min)
        pt_num_of_nearest_max = local_max_inds[ind_in_list_of_nearest_max][0,0]
        delta_feature_nearest_min_max[pt_min_num] = ( feature[pt_num_of_nearest_max] - feature[pt_min_num] )

    features = {'local_max_and_min': local_max_and_min}
    metadata = {'nearest_min_max_dists': nearest_min_max_dists,
               'delta_feature_nearest_min_max': delta_feature_nearest_min_max,
               'min_max_pair_distances': min_max_pair_distances,
               'min_max_pair_anisotropies': min_max_pair_anisotropies}
    properties = {'features': features,
                  'metadata': metadata,
                  'name': 'Maxima and minima',
                  'face_color': 'local_max_and_min'}
    output_points = (surface[0], properties, 'points')

    return output_points
