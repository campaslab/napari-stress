import numpy as np
from .._utils import frame_by_frame
from typing import Tuple


def _fit_and_create_pointcloud(
    pointcloud: "napari.tyes.PointsData",
) -> "napari.tyes.PointsData":
    """
    Fits a quadratic surface to a pointcloud and returns a new pointcloud
    with fitted Z coordinates.

    Parameters:
    pointcloud: a numpy array with shape (n_points, 3), where each row is [Z, Y, X]

    Returns:
    fitted_pointcloud: a numpy array with the fitted Z coordinates
    """
    # Fit the quadratic surface to the Z coordinates
    fitting_params = _fit_quadratic_surface(pointcloud)

    # Apply the fitting parameters to get the fitted ZYX pointcloud
    fitted_pointcloud = _create_fitted_coordinates(pointcloud, fitting_params)

    return fitted_pointcloud


def _fit_quadratic_surface(points: "napari.types.PointsData") -> np.ndarray:
    """
    Fits a quadratic surface to 3D data points using a second-order polynomial.

    Parameters:
    x_coords, y_coords, z_coords: arrays of coordinates of the points

    Returns:
    fitting_params: coefficients of the fitted surface
    """
    num_points = len(points)

    # Design matrix for the second-order polynomial surface
    z = points[:, 0]
    y = points[:, 1]
    x = points[:, 2]
    ones_vec = np.ones(num_points)

    # Assemble the design matrix
    design_matrix = np.column_stack((ones_vec, x, y, x * y, x**2, y**2))
    z_matrix = z.reshape(-1, 1)

    # Linear least squares fitting
    normal_matrix = design_matrix.T @ design_matrix
    fitting_params = np.linalg.pinv(normal_matrix) @ design_matrix.T @ z_matrix

    return fitting_params.flatten()


def _create_fitted_coordinates(points, fitting_params):
    """
    Creates the fitted ZYX pointcloud from the fitting parameters and X, Y coordinates.

    Parameters:
    x_coords, y_coords: arrays of coordinates of the points
    fitting_params: coefficients of the fitted surface

    Returns:
    zyx_pointcloud: new pointcloud with fitted z-coordinates
    """
    num_points = len(points)
    ones_vec = np.ones(num_points)
    x = points[:, 2]
    y = points[:, 1]

    # Assemble the design matrix with the known coordinates
    design_matrix = np.column_stack((ones_vec, x, y, x * y, x**2, y**2))

    # Calculate the fitted z-coordinates
    z_fitted = design_matrix @ fitting_params

    # Create the new pointcloud as a ZYX array
    zyx_pointcloud = np.column_stack((z_fitted, y, x))
    return zyx_pointcloud


def _find_neighbor_indices(
    pointcloud: "napari.types.PointsData", patch_radius: Tuple[float, np.ndarray]
):
    """
    For each point in the pointcloud, find the indices and distances of all points
    within a given radius.

    Parameters:
    -----------
    pointcloud: 'napari.types.PointsData'
        A numpy array with shape (n_points, 3), where each row represents a point with
        coordinates [Z, Y, X].
    patch_radius: float or np.ndarray
        The radius around each point to search for neighbors. Can be a single value or
        a numpy array with the same length as pointcloud.

    Returns:
    --------
    indices: a list where each element is a list of indices of neighbors for the
    corresponding point
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(pointcloud)
    indices = []

    for i, point in enumerate(pointcloud):
        if isinstance(patch_radius, np.ndarray) and len(patch_radius) == len(
            pointcloud
        ):
            idx = tree.query_ball_point(point, r=patch_radius[i])
        else:
            idx = tree.query_ball_point(point, r=patch_radius)
        indices.append(idx)

    return indices


def compute_orientation_matrix(patch_points: "napari.types.PointsData") -> np.ndarray:
    """
    Compute the orientation matrix for a patch of points.

    This function calculates the covariance matrix of the patch points and
    performs an eigen decomposition to find the orientation matrix that
    aligns the patch's normal vector with the z-axis.

    Parameters:
    - patch_points (np.ndarray): An N x 3 array of centered patch points.

    Returns:
    - orient_matrix (np.ndarray): The orientation matrix used to align the patch points with the z-axis.
    """
    from scipy.linalg import eigh

    # Compute the covariance matrix and its eigen decomposition
    n = len(patch_points)
    S = (1 / (n - 1)) * (patch_points.T @ patch_points)
    eigvals, eigvecs = eigh(S)  # 'eigh' is for symmetric matrices like covariance

    # # Sort the eigenvectors by eigenvalues in ascending order
    # # Assuming the normal corresponds to the smallest eigenvalue
    # sorted_indices = np.argsort(eigvals)
    # orient_matrix = eigvecs[:, sorted_indices]

    return eigvecs


def _orient_patch(
    patch_points: "napari.types.PointsData",
    patch_query_point: "napari.types.PointsData",
    reference_point: "napari.types.PointsData" = None,
) -> tuple:
    """
    Reorient a patch of points so that the normal vector points along the z-axis.

    This function takes a patch of points in 3D space, the center point of the patch,
    and a reference center point, and aligns the patch's normal vector with the z-axis.

    Parameters
    ----------
    patch_points : np.ndarray
        An N x 3 array of points representing a patch in 3D space.
    patch_query_point : np.ndarray
        A 1 x 3 array representing the center point of the patch.
    reference_point : np.ndarray
        A 1 x 3 array representing the overall center point for orientation.

    Returns
    -------
    Xn_out : np.ndarray
        The reoriented patch points as an N x 3 array.
    Xq_out : np.ndarray
        The reoriented center point of the patch as a 1 x 3 array.
    eigvals : np.ndarray
        Eigenvalues from the eigen decomposition of the covariance matrix.
    computed_patch_center : np.ndarray
        The mean center of the patch points as a 1 x 3 array.
    orient_matrix : np.ndarray
        The orientation matrix used to align the patch points with the z-axis.

    Raises
    ------
    ValueError
        If the `patch_center_point` contains NaN or infinite values.
    """
    # Check for NaN or infinite values in the center point of the patch
    if np.isnan(np.sum(patch_query_point)) or not np.isfinite(
        np.sum(patch_query_point)
    ):
        raise ValueError("Center point of the patch must be a finite 1x3 array")

    if reference_point is None:
        reference_point = np.asarray([0, 0, 0])

    # Mean center of the patch points
    computed_patch_center = patch_points.mean(axis=0)

    # Center the patch points and the given center point
    points_centered = patch_points - computed_patch_center
    query_point = patch_query_point - computed_patch_center
    Xct = reference_point - computed_patch_center

    # Calculate the orientation matrix
    orient_matrix = compute_orientation_matrix(points_centered)

    # Reorient the center point of the patch
    query_point_transformed = query_point @ orient_matrix
    YCenter = Xct @ orient_matrix

    # Determine if the patch needs to be flipped
    if query_point_transformed[0] - YCenter[0] > 0:
        flip_upside_down = np.diag([-1, -1, 1])
        orient_matrix = orient_matrix @ flip_upside_down
        query_point_transformed = query_point @ orient_matrix
        YCenter = Xct @ orient_matrix

    # Reorient all points in the patch
    patch_transformed = points_centered @ orient_matrix

    # # Calculate eigenvalues for the covariance matrix
    # _, eigvals = np.linalg.eigh(np.cov(points_centered.T))

    return (patch_transformed, query_point_transformed, orient_matrix)


def _calculate_mean_curvature_on_patch(
    query_point: "napari.types.PointsData", fitting_params: np.ndarray
) -> tuple:
    """
    Calculate the mean curvature on a patch of points.

    This function calculates the mean curvature on a patch of points by
    fitting a quadratic surface to the patch and calculating the mean
    curvature from the fitted surface.

    Parameters
    ----------
    patch_cloud : np.ndarray
        An N x 3 array of points representing a patch in 3D space.

    Returns
    -------
    mean_curvatures : np.ndarray
        An N x 1 array of mean curvature values for each point in the patch.
    principal_curvatures : np.ndarray
        An N x 2 array of principal curvature values for each point in the patch.
    """
    # Unpack the parameters
    p10 = fitting_params[1]
    p01 = fitting_params[2]
    p11 = fitting_params[3]
    p20 = fitting_params[4]
    p02 = fitting_params[5]

    # Compute mean curvature and principal curvatures for each point in patch_cloud
    mean_curvatures = []
    principal_curvatures = []

    x = query_point.squeeze()[2]
    y = query_point.squeeze()[1]
    X = np.array([1, x, y])

    tht_u = np.array([p10, 2 * p20, p11 * y])
    hu = X @ tht_u
    tht_v = np.array([p01, p11 * x, 2 * p02])
    hv = X @ tht_v
    tht_uu = np.array([2 * p20, 0, 0])
    huu = X @ tht_uu
    tht_vv = np.array([2 * p02, 0, 0])
    hvv = X @ tht_vv
    tht_uv = np.array([p11, 0, 0])
    huv = X @ tht_uv

    # Compute mean curvature
    H = ((1 + hu**2) * hvv - 2 * hu * hv * huv + (1 + hv**2) * huu) / (
        2 * (1 + hu**2 + hv**2) ** (3 / 2)
    )
    mean_curvatures.append(H)

    # Compute principal curvatures
    k1 = 2 * max(p02, p20)
    k2 = 2 * min(p02, p20)
    principal_curvatures.append((k1, k2))

    return mean_curvatures, principal_curvatures


def _estimate_patch_radii(
    pointcloud: "napari.types.PointsData",
    k1: np.ndarray = None,
    minimum_permitted_range: float = 2.5,
    min_num_per_patch: int = 18,
):
    """
    Calculate patch radii for a point cloud based on principal curvatures.

    This function calculates the patch radii for a point cloud based on the
    principal curvatures of each point. The patch radii are calculated as the
    geometric mean of the minimum radius and the radius calculated from the
    principal curvatures. The minimum radius is calculated as the minimum
    distance between a point and its nearest neighbor. The patch radii are
    adjusted based on the minimum number of neighbors required for each patch.

    Parameters
    ----------
    pointcloud : np.ndarray
        An N x 3 array of points representing a point cloud in 3D space.
    k1 : np.ndarray
        An N x 1 array of principal curvatures for each point in the point cloud.
    minimum_permitted_range : float
        The minimum permitted range for the patch radii. This value is used to
        ensure that the patch radii are not too small.
    min_num_per_patch : int
        The minimum number of neighbors required for each patch.

    Returns
    -------
    patch_radii : np.ndarray
    """
    from .._approximation import least_squares_ellipsoid
    from .._measurements import curvature_on_ellipsoid
    from ..types import (
        _METADATAKEY_PRINCIPAL_CURVATURES1,
        _METADATAKEY_PRINCIPAL_CURVATURES2,
    )

    if k1 is None:
        # measure curvature on fitted patches first: Approximate by ellipsoid

        ellipsoid = least_squares_ellipsoid(pointcloud)
        curvatures = curvature_on_ellipsoid(ellipsoid, pointcloud)[1]["features"]
        principal_curvatures = [
            curvatures[key]
            for key in [
                _METADATAKEY_PRINCIPAL_CURVATURES1,
                _METADATAKEY_PRINCIPAL_CURVATURES2,
            ]
        ]

        # Assume this is 1 for every point as per the requirement
        k1 = np.stack(principal_curvatures).max(axis=0)
    len_scale2 = 2.0 * k1 ** (-1)

    # Calculate the geometric mean of the length scales to get the patch radius
    small_patch_length_scale = 1
    patch_radii = np.sqrt(small_patch_length_scale * len_scale2)

    # Ensure the patch radius is not smaller than the minimum patch radius
    patch_radii[patch_radii < minimum_permitted_range] = minimum_permitted_range

    # List to store the updated radii
    updated_patch_radii = np.copy(patch_radii)

    # Retrieve the list of lists containing neighbor indices for each point
    neighbor_indices_list = _find_neighbor_indices(pointcloud, patch_radii)

    for i, curv in enumerate(k1):
        # Get the current point's neighbors
        neighbors = neighbor_indices_list[i]

        if neighbors:  # Check if there are any neighbors
            k_neighbors = k1[neighbors]
            hot_spots = np.abs(k_neighbors / curv) > 2

            if np.any(hot_spots):
                distances = np.linalg.norm(
                    pointcloud[neighbors][hot_spots] - pointcloud[i], axis=1
                )
                if distances.size > 0:
                    rp = np.min(distances) - 0.1
                    updated_patch_radii[i] = max(rp, minimum_permitted_range)

        # Ensure there are at least min_num_per_patch neighbors
        if len(neighbors) < min_num_per_patch:
            all_distances = np.linalg.norm(pointcloud - pointcloud[i], axis=1)
            nearest_indices = np.argsort(all_distances)[:min_num_per_patch]
            updated_patch_radii[i] = np.max(all_distances[nearest_indices])

    return updated_patch_radii


@frame_by_frame
def fit_patches(
    point_cloud: "napari.types.PointsData", search_radius: float = 1
) -> "napari.types.PointsData":
    """
    Fit a quadratic surface to each point's neighborhood in a point cloud and
    adjust the point positions to the fitted surface.

    Parameters
    ----------
    point_cloud : np.ndarray
        A numpy array with shape (n_points, 3), where each row represents a point
        with coordinates [Z, Y, X].
    search_radius : float or np.ndarray
        The radius around each point to search for neighbors. Can be a single value or
        a numpy array with the same length as point_cloud.

    Returns
    -------
    fitted_point_cloud : np.ndarray
        The point cloud with points adjusted to the fitted quadratic surface.
    """
    num_points = len(point_cloud)  # Number of points in the point cloud
    fitted_point_cloud = np.copy(point_cloud)  # Initialize fitted point cloud
    min_neighbors = 6  # Minimum number of neighbors required to perform fitting

    # Compute neighbors for each point in the point cloud
    neighbor_indices = _find_neighbor_indices(point_cloud, search_radius)

    for idx in range(num_points):
        current_point = point_cloud[idx, :]
        neighbors_idx = neighbor_indices[idx]
        patch = point_cloud[neighbors_idx, :]

        # Proceed only if there are enough points in the neighborhood
        if len(patch) < min_neighbors:
            continue  # Not enough neighbors, skip to the next point

        # Orient the patch for the current point
        oriented_patch, oriented_query_point, orient_matrix = _orient_patch(
            patch, current_point, np.mean(point_cloud, axis=0)
        )
        patch_center = patch.mean(axis=0)

        # Perform the quadratic surface fitting
        fitting_params = _fit_quadratic_surface(oriented_patch)

        # Calculate the new fitted point
        fitted_query_point = _create_fitted_coordinates(
            oriented_query_point[None, :], fitting_params
        )
        # fitted_patch = _create_fitted_coordinates(oriented_patch, fitting_params)

        # fitted_patch_reoriented = fitted_patch @ orient_matrix.T + patch_center
        fitted_point_cloud[idx, :] = (
            fitted_query_point[None, :] @ orient_matrix.T + patch_center
        )

    return fitted_point_cloud


@frame_by_frame
def iterative_curvature_adaptive_patch_fitting(
    point_cloud: "napari.types.PointsData",
    n_iterations: int = 3,
    minimum_neighbors: int = 6,
    minimum_search_radius: int = 1,
) -> "napari.types.PointsData":
    # Initialize fitted point cloud and find neighbors
    fitted_point_cloud = np.copy(point_cloud)
    search_radii = _estimate_patch_radii(
        point_cloud, minimum_permitted_range=minimum_search_radius
    )

    for it in range(n_iterations):
        neighbor_indices = _find_neighbor_indices(point_cloud, search_radii)
        mean_curvatures = [np.nan] * len(point_cloud)
        principal_curvatures = [np.nan] * len(point_cloud)

        # Compute neighbors for each point in the point cloud
        for idx in range(len(point_cloud)):
            current_point = point_cloud[idx, :]
            neighbors_idx = neighbor_indices[idx]
            patch = point_cloud[neighbors_idx, :]

            # Proceed only if there are enough points in the neighborhood
            if len(patch) < minimum_neighbors:
                principal_curvatures[idx] = np.array([np.nan, np.nan])
                continue  # Not enough neighbors, skip to the next point

            # Orient the patch for the current point
            oriented_patch, oriented_query_point, orient_matrix = _orient_patch(
                patch, current_point, np.mean(point_cloud, axis=0)
            )
            patch_center = patch.mean(axis=0)

            # Perform the quadratic surface fitting
            fitting_params = _fit_quadratic_surface(oriented_patch)

            # Calculate the new fitted point
            fitted_query_point = _create_fitted_coordinates(
                oriented_query_point[None, :], fitting_params
            )
            # fitted_patch = _create_fitted_coordinates(oriented_patch, fitting_params)

            fitted_point_cloud[idx, :] = (
                fitted_query_point[None, :] @ orient_matrix.T + patch_center
            )

            mean_curv, principal_curv = _calculate_mean_curvature_on_patch(
                fitted_query_point, fitting_params
            )

            mean_curvatures[idx] = mean_curv
            principal_curvatures[idx] = principal_curv[0]

        # Update the search radii
        mean_curvatures = np.array(mean_curvatures)
        principal_curvatures = np.array(principal_curvatures).squeeze()
        point_cloud = fitted_point_cloud

        search_radii = _estimate_patch_radii(
            point_cloud, principal_curvatures.max(axis=1)
        )

    return fitted_point_cloud
