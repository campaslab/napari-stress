from typing import List

import numpy as np
from napari.types import LayerDataTuple, SurfaceData, VectorsData
from napari_tools_menu import register_function

from .._utils.frame_by_frame import frame_by_frame


def geodesic_distance_matrix(surface: SurfaceData) -> np.ndarray:
    """
    Calculate a pairwise distance matrix for vertices of a surface.

    This calculates the geodesic distances between any two vertices on a given
    surface. If the surface comprises more than 500 vertices, the computation will
    be parallelized using dask.

    Parameters
    ----------
    surface : SurfaceData

    Returns
    -------
    distance_matrix : np.ndarray
        Triangular matrix with differences between vertex i and j located
        at `distance_matrix[i, j]`

    """
    from pygeodesic import geodesic

    geoalg = geodesic.PyGeodesicAlgorithmExact(surface[0], surface[1])

    n_points = len(surface[0])
    distance_matrix = np.zeros((n_points, n_points))
    points = surface[0]

    if n_points > 500:
        from dask.distributed import Client, get_client, secede, rejoin, worker_client
        from dask import compute

        with worker_client() as client:
            # get the indices of the upper triangle, get pairs and split into chunks
            indices = np.triu_indices(n_points, k=1)
            pairs = np.stack(indices).T
            chunks = np.array_split(pairs, len(pairs) // 5000)

            # calculate distances in parallel
            futures = []
            for chunk in chunks:
                futures.append(client.submit(_geodesic_distances, surface, chunk))

            results = client.gather(futures)
            # Gather results and fill distance matrix
            for chunk, result in zip(chunks, results):
                distance_matrix[chunk[:, 0], chunk[:, 1]] = result

    for idx, pt in enumerate(points):
        distances, _ = geoalg.geodesicDistances([idx], np.arange(idx + 1, n_points))
        distance_matrix[idx, idx + 1 :] = distances
        distance_matrix[idx + 1 :, idx] = distances

    return distance_matrix


def _geodesic_distances(surface, chunk):
    from pygeodesic import geodesic

    geoalg = geodesic.PyGeodesicAlgorithmExact(surface[0], surface[1])
    distances = []

    # check for unique source indices
    unique_source_indices = np.unique(chunk[:, 0])
    for source_index in unique_source_indices:
        target_indices = chunk[chunk[:, 0] == source_index, 1]
        distances += list(geoalg.geodesicDistances([source_index], target_indices)[0])
    return distances


@register_function(menu="Surfaces > Geodesic path (pygeodesics, n-STRESS)")
@frame_by_frame
def geodesic_path(surface: SurfaceData, index_1: int, index_2: int) -> VectorsData:
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
    from pygeodesic import geodesic

    geoalg = geodesic.PyGeodesicAlgorithmExact(surface[0], surface[1])
    distances, path = geoalg.geodesicDistance(index_1, index_2)

    # convert points to vectors from point to point
    vectors = []
    for i in range(len(path) - 1):
        vectors.append(path[i + 1] - path[i])
    vectors = np.asarray(vectors)
    napari_vectors = np.stack([path[:-1], vectors]).transpose((1, 0, 2))

    return napari_vectors


def correlation_on_surface(
    surface1: SurfaceData,
    surface2: SurfaceData,
    distance_matrix: np.ndarray = None,
    maximal_distance: float = None,
) -> dict:
    """
    Calculate (auto-) correlation of features on surface.

    This calculates the correlation of features on a surface with itself
    (auto-correlation) or with another surface (cross-correlation).
    If the two input surfaces are identical, this is the auto-correlation.

    Parameters
    ----------
    surface1 : SurfaceData
    surface2 : SurfaceData

    Returns
    -------
    dict
        Dictionary with keys:
            - auto_correlations_distances
            - auto_correlations_average
            - auto_correlations_normalized
            - auto_correlations_averaged_normalized
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
    Corr_outer_prod_mat_pts = np.dot(
        feature1.reshape(n_points, 1), feature2.reshape(n_points, 1).T
    )
    Corr_non0_pts = Corr_outer_prod_mat_pts[np.triu_indices(n_points)]

    # calculate autocorrelation for all points
    # norm, so auto-corrs are 1 at \delta (= |x - x'|) = 0
    auto_corr_norm = np.average(np.diag(Corr_outer_prod_mat_pts).flatten(), axis=0)

    avg_mean_curv_auto_corrs = []
    dists_used = []

    # calculate autocorrelation with spatially averaged feature values
    for dist_i in range(0, maximal_distance + 1):
        sum_mean_curv_corr_d_i, n_points = _avg_around_pt(
            dist_i, dists_lbdv_non0, Corr_non0_pts, maximal_distance
        )

        if n_points > 0:
            # average in bin
            mean_curv_corr_d_i = sum_mean_curv_corr_d_i / n_points
            # include if corr <= 1
            if abs(mean_curv_corr_d_i) / auto_corr_norm <= 1.0:
                avg_mean_curv_auto_corrs.append(mean_curv_corr_d_i)
                dists_used.append(dist_i)

    num_dists_used = len(dists_used)

    # We get (geodesic) distances we calculate correlations on,
    # using bump fn averages around these distances:
    auto_corrs_microns_dists = np.array(dists_used, dtype=np.dtype("d")).reshape(
        num_dists_used, 1
    )
    auto_corrs_avg = np.array(avg_mean_curv_auto_corrs, dtype=np.dtype("d")).reshape(
        num_dists_used, 1
    )
    auto_corrs_avg_normed = auto_corrs_avg / auto_corr_norm

    result = {
        "auto_correlations_distances": auto_corrs_microns_dists,
        "auto_correlations_average": auto_corrs_avg,
        "auto_correlations_normalized": auto_corr_norm,
        "auto_correlations_averaged_normalized": auto_corrs_avg_normed,
    }

    return result


def _avg_around_pt(dist_x_c, dists_pts, vals_at_pts, max_dist_used):
    """
    Bump function weights on pts centered on dist_x_c.
    """
    # 10. #5. #10.
    dist_max = max_dist_used / 20.0

    # only look at points within 1
    pts_vals_within_1 = np.where(
        abs(dists_pts - dist_x_c) <= dist_max, vals_at_pts, 0.0
    )
    weights = np.where(
        abs(dists_pts - dist_x_c) <= 1.0,
        np.exp(1.0 - 1.0 / (dist_max**2 - (dists_pts - dist_x_c) ** 2)),
        0.0,
    )

    sum_weights = np.sum(weights.flatten())
    sum_pts_within_1 = np.sum(np.multiply(pts_vals_within_1, weights).flatten())

    return sum_pts_within_1, sum_weights


@register_function(menu="Measurement > Local maxima on surface (pygeodesics, n-STRESS)")
def local_extrema_analysis(
    surface: SurfaceData, distance_matrix: np.ndarray = None
) -> List[LayerDataTuple]:
    """
    Get local maximum and minimum and analyze their mutual distances.

    Parameters
    ----------
    surface : SurfaceData
    distance_matrix : np.ndarray, optional
        geodesic distance matrix. The default is None.

    Returns
    -------
    List[LayerDataTuple]
        List of layer data tuples with features and metadata:
            - local_max_and_min (features)
            - nearest_min_max_dists (metadata)
            - nearest_min_max_anisotropies (metadata)
            - min_max_pair_distances (metadata)
            - min_max_pair_anisotropies (metadata)
    """
    triangles = surface[1]
    feature = surface[2]

    quad_fit = len(feature)

    # should be -1 at local min, +1 at local max, 0 otherwise, 2 if both
    local_max_and_min = np.zeros_like(feature)

    # 0 if not local max or min; but for local max is distance to nearest
    # local min, vice versa
    nearest_min_max_dists = np.zeros_like(feature)

    # 2*(H_in_max - H_in_min), from extrema to nearest partner
    delta_feature_nearest_min_max = np.zeros_like(feature)

    for pt in range(quad_fit):
        H_pt = feature[pt]

        # set to True, change if False:
        pt_is_local_max = True
        pt_is_local_min = True

        tris_containing_pt = np.where(triangles == pt)
        rows_pt = tris_containing_pt[0]

        for row_pt in range(len(rows_pt)):
            row_num_pt = rows_pt[row_pt]

            # compare to other pts in triangle
            for other_tri_pt_num in range(3):
                other_tri_pt = int(triangles[row_num_pt, other_tri_pt_num])
                H_other_tri_pt = feature[other_tri_pt]

                if H_other_tri_pt < H_pt:
                    pt_is_local_min = False
                if H_other_tri_pt > H_pt:
                    pt_is_local_max = False

        if pt_is_local_max:
            if pt_is_local_min:
                # local max AND min
                local_max_and_min[pt] = 2
            else:
                # local max (only)
                local_max_and_min[pt] = 1

        elif pt_is_local_min:
            # local min (only)
            local_max_and_min[pt] = -1
        else:
            local_max_and_min[pt] = 0

    num_local_max = np.count_nonzero(local_max_and_min == 1)
    num_local_min = np.count_nonzero(local_max_and_min == -1)

    local_max_inds = np.where(local_max_and_min == 1)[0]
    local_min_inds = np.where(local_max_and_min == -1)[0]

    # list of ALL local min/max pairs'
    # distances and difference in input fields:
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

    min_max_pair_anisotropies = np.array(min_max_pair_anisotropies, dtype=np.dtype("d"))
    min_max_pair_distances = np.array(min_max_pair_distances, dtype=np.dtype("d"))

    pt_num_of_nearest_min = []
    geodesic_paths_nearest_max_min = []
    geodesic_paths_nearest_min_max = []

    for pt_max in range(num_local_max):
        pt_max_num = local_max_inds[pt_max]

        local_min_dists_to_pt = distance_matrix[pt_max_num, local_min_inds]
        min_dist_to_local_min = min(local_min_dists_to_pt)
        nearest_min_max_dists[pt_max_num] = min_dist_to_local_min

        # Calculate 2*(H_max - H_nearest_min):
        ind_in_list_of_nearest_min = np.argwhere(
            local_min_dists_to_pt == min_dist_to_local_min
        )
        pt_num_of_nearest_min = local_min_inds[ind_in_list_of_nearest_min][0, 0]

        delta_feature_nearest_min_max[pt_max_num] = (
            feature[pt_max_num] - feature[pt_num_of_nearest_min]
        )
        geodesic_paths_nearest_max_min.append(
            geodesic_path(surface, pt_max_num, pt_num_of_nearest_min)
        )

    for pt_min in range(num_local_min):
        pt_min_num = local_min_inds[pt_min]

        local_max_dists_to_pt = distance_matrix[pt_min_num, local_max_inds]

        max_dist_to_local_min = min(local_max_dists_to_pt)
        nearest_min_max_dists[pt_min_num] = max_dist_to_local_min

        # Calculate 2*(H_min - H_nearest_max):
        ind_in_list_of_nearest_max = np.argwhere(
            local_max_dists_to_pt == max_dist_to_local_min
        )
        pt_num_of_nearest_max = local_max_inds[ind_in_list_of_nearest_max][0, 0]
        delta_feature_nearest_min_max[pt_min_num] = (
            feature[pt_num_of_nearest_max] - feature[pt_min_num]
        )
        geodesic_paths_nearest_min_max.append(
            geodesic_path(surface, pt_min_num, pt_num_of_nearest_max)
        )

    features = {"local_max_and_min": local_max_and_min}
    metadata = {
        "nearest_pair_distance": nearest_min_max_dists,
        "nearest_pair_anisotropy": delta_feature_nearest_min_max,
        "all_pair_distance": min_max_pair_distances,
        "all_pair_anisotropy": min_max_pair_anisotropies,
    }
    properties = {
        "features": features,
        "metadata": metadata,
        "name": "Maxima and minima",
        "size": 0.5,
        "face_color": "local_max_and_min",
    }
    output_points = (surface[0], properties, "points")

    output_geodesics_min_max = (
        np.concatenate(geodesic_paths_nearest_min_max),
        {
            "name": "Geodesics minima -> nearest maxima",
            "edge_width": 0.2,
            "edge_color": "orange",
        },
        "vectors",
    )
    output_geodesics_max_min = (
        np.concatenate(geodesic_paths_nearest_max_min),
        {
            "name": "Geodesics maxima -> nearest minima",
            "edge_width": 0.2,
            "edge_color": "blue",
        },
        "vectors",
    )

    return [output_points, output_geodesics_max_min, output_geodesics_min_max]
