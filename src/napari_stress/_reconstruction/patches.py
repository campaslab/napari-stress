import numpy as np

def fit_and_create_pointcloud(pointcloud: 'napari.tyes.PointsData'
                              ) -> 'napari.tyes.PointsData':
    """
    Fits a quadratic surface to a pointcloud and returns a new pointcloud
    with fitted Z coordinates.

    Parameters:
    pointcloud: a numpy array with shape (n_points, 3), where each row is [Z, Y, X]
    
    Returns:
    fitted_pointcloud: a numpy array with the fitted Z coordinates
    """
    # Extract X, Y, Z coordinates from the pointcloud
    z_coords = pointcloud[:, 0]
    y_coords = pointcloud[:, 1]
    x_coords = pointcloud[:, 2]
    
    # Fit the quadratic surface to the Z coordinates
    fitting_params = fit_quadratic_surface(x_coords, y_coords, z_coords)
    
    # Apply the fitting parameters to get the fitted ZYX pointcloud
    fitted_pointcloud = create_fitted_coordinates(x_coords, y_coords, fitting_params)
    
    return fitted_pointcloud

def fit_quadratic_surface(x_coords, y_coords, z_coords):
    """
    Fits a quadratic surface to 3D data points using a second-order polynomial.
    
    Parameters:
    x_coords, y_coords, z_coords: arrays of coordinates of the points
    
    Returns:
    fitting_params: coefficients of the fitted surface
    """
    num_points = len(x_coords)

    # Design matrix for the second-order polynomial surface
    ones_vec = np.ones(num_points)
    x_lin = x_coords
    y_lin = y_coords
    x_y_cross = x_coords * y_coords
    x_quad = x_coords ** 2
    y_quad = y_coords ** 2

    # Assemble the design matrix
    design_matrix = np.column_stack((ones_vec, x_lin, y_lin, x_y_cross, x_quad, y_quad))
    z_matrix = z_coords.reshape(-1, 1)

    # Linear least squares fitting
    normal_matrix = design_matrix.T @ design_matrix
    fitting_params = np.linalg.pinv(normal_matrix) @ design_matrix.T @ z_matrix

    return fitting_params.flatten()

def create_fitted_coordinates(x_coords, y_coords, fitting_params):
    """
    Creates the fitted ZYX pointcloud from the fitting parameters and X, Y coordinates.
    
    Parameters:
    x_coords, y_coords: arrays of coordinates of the points
    fitting_params: coefficients of the fitted surface
    
    Returns:
    zyx_pointcloud: new pointcloud with fitted z-coordinates
    """
    num_points = len(x_coords)
    ones_vec = np.ones(num_points)
    x_lin = x_coords
    y_lin = y_coords
    x_y_cross = x_coords * y_coords
    x_quad = x_coords ** 2
    y_quad = y_coords ** 2

    # Assemble the design matrix with the known coordinates
    design_matrix = np.column_stack((ones_vec, x_lin, y_lin, x_y_cross, x_quad, y_quad))

    # Calculate the fitted z-coordinates
    z_fitted = design_matrix @ fitting_params

    # Create the new pointcloud as a ZYX array
    zyx_pointcloud = np.column_stack((z_fitted, y_coords, x_coords))
    return zyx_pointcloud


def find_neighbor_indices(pointcloud, patch_radii):
    """
    For each point in the pointcloud, find the indices and distances of all points within a given radius.
    
    Parameters:
    pointcloud: a numpy array with shape (n_points, 3), where each row is [Z, Y, X]
    patch_radii: a single value or a numpy array with the same length as pointcloud indicating the radius around each point to search for neighbors
    
    Returns:
    indices: a list where each element is a list of indices of neighbors for the corresponding point
    distances: a list where each element is a list of distances to the corresponding neighbors
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(pointcloud)
    indices = []
    distances = []
    
    for i, (point, radius) in enumerate(zip(pointcloud, patch_radii)):
        idx, dist = tree.query(point, k=None, distance_upper_bound=radius)
        # Exclude points at a distance of infinity (outside the search radius)
        valid_indices = idx < len(pointcloud)
        indices.append(idx[valid_indices].tolist())
        distances.append(dist[valid_indices].tolist())
        
    return indices, distances


def compute_orientation_matrix(patch_points: 'napari.types.PointsData') -> np.ndarray:
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
    _, eigvecs = eigh(S)  # 'eigh' is for symmetric matrices like covariance
    
    # Ensure the normal vector points along the z-axis by reversing the order of eigenvectors
    orient_matrix = eigvecs[:, [2, 1, 0]]

    return orient_matrix

def orient_patch(patch_points: 'napari.types.PointsData',
                 patch_center_point: 'napari.types.PointsData',
                 center_point: 'napari.types.PointsData') -> tuple:
    """
    Reorient a patch of points so that the normal vector points along the z-axis.

    This function takes a patch of points in 3D space, the center point of the patch,
    and a reference center point, and aligns the patch's normal vector with the z-axis.

    Parameters
    ----------
    patch_points : np.ndarray
        An N x 3 array of points representing a patch in 3D space.
    patch_center_point : np.ndarray
        A 1 x 3 array representing the center point of the patch.
    center_point : np.ndarray
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
    if np.isnan(np.sum(patch_center_point)) or not np.isfinite(np.sum(patch_center_point)):
        raise ValueError('Center point of the patch must be a finite 1x3 array')
    
    # Filter out NaN values in patch points
    valid_indices = ~np.isnan(patch_points[:, 0])
    x = patch_points[valid_indices, 0]
    y = patch_points[valid_indices, 1]
    z = patch_points[valid_indices, 2]
    
    # Calculate the mean of the patch
    xo, yo, zo = np.mean(x), np.mean(y), np.mean(z)
    computed_patch_center = np.array([xo, yo, zo])
    
    # Center the patch points and the given center point
    X = np.column_stack((x - xo, y - yo, z - zo))
    Xq = patch_center_point - computed_patch_center
    Xct = center_point - computed_patch_center
    
    # Calculate the orientation matrix
    orient_matrix = compute_orientation_matrix(X)

    # Reorient the center point of the patch
    Yq = Xq @ orient_matrix
    YCenter = Xct @ orient_matrix
    
    # Determine if the patch needs to be flipped
    if Yq[2] - YCenter[2] > 0:
        flip_upside_down = np.diag([1, -1, -1])
        orient_matrix = orient_matrix @ flip_upside_down
        Yq = Xq @ orient_matrix
        YCenter = Xct @ orient_matrix
    
    # Reorient all points in the patch
    Xn_out = X @ orient_matrix
    
    # Calculate eigenvalues for the covariance matrix
    _, eigvals = np.linalg.eigh(np.cov(X.T))

    # Output
    Xq_out = Yq
    Xn_out = Xn_out  # Reoriented patch points

    return Xn_out, Xq_out, eigvals, computed_patch_center

def fit_patches(point_cloud: np.ndarray,
                search_radius) -> 'napari.types.PointsData':
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
    neighbor_indices = find_neighbor_indices(point_cloud, search_radius)

    for idx in range(num_points):
        current_point = point_cloud[idx, :]
        neighbors_idx = neighbor_indices[idx]
        neighbors = point_cloud[neighbors_idx, :]

        # Proceed only if there are enough points in the neighborhood
        if len(neighbors) < min_neighbors:
            continue  # Not enough neighbors, skip to the next point

        # Orient the patch for the current point
        oriented_patch, oriented_query_point, _, patch_center = orient_patch(
            neighbors, current_point, np.mean(point_cloud, axis=0))

        # Perform the quadratic surface fitting
        x_coords, y_coords, z_coords = oriented_patch.T
        fitting_params = fit_quadratic_surface(x_coords, y_coords, z_coords)

        # Calculate the new fitted point
        fitted_point = create_fitted_coordinates(
            oriented_query_point[0], oriented_query_point[1], fitting_params)

        # Adjust and store the fitted point based on the original patch center
        fitted_point_cloud[idx, :] = fitted_point + patch_center

    return fitted_point_cloud

