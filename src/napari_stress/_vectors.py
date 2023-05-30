import numpy as np
import pandas as pd
from napari_tools_menu import register_function
from ._utils.frame_by_frame import frame_by_frame


@register_function(
        menu="Points > Move point along vectors by absolute value (n-STRESS)")
@frame_by_frame
def absolute_move_points_along_vector(points: "napari.types.PointsData",
                                      vectors: "napari.types.VectorsData",
                                      position: float = 1,
                                      pointwise_position: np.ndarray = None
                             ) -> "napari.types.PointsData":
    """
    Move points along vectors by an absolute value.

    Args:
        points (napari.types.PointsData): Points
        vectors (napari.types.VectorsData): Vectors
        position (float, optional): Position along vector. Defaults to 1.0.
        pointwise_position (np.ndarray, optional): Position for each point.
            Defaults to None. If given, this overrides the `position` argument.

    Returns:
        napari.types.PointsData: Moved points
    """
    vector_length = np.linalg.norm(vectors[:, 1], axis=1)[:, np.newaxis]
    unit_vector = vectors[:, 1] / vector_length
    return points + position * unit_vector


@register_function(
        menu="Points > Move point along vectors by relative factor (n-STRESS)")
@frame_by_frame
def relative_move_points_along_vector(points: "napari.types.PointsData",
                                      vectors: "napari.types.VectorsData",
                                      position: float = 0.5,
                                      pointwise_position: np.ndarray = None
                             ) -> "napari.types.PointsData":
    """
    Move points along vectors by a relative factor.

    This function moves points along vectors by a relative factor.
    The position argument determines the relative position along the
    vector. For example, a position of 0.5 moves the points to the
    middle of the vector.

    Args:
        points (napari.types.PointsData): Points
        vectors (napari.types.VectorsData): Vectors
        position (float, optional): Position along vector. Defaults to 0.5.
        pointwise_position (np.ndarray, optional): Position for each point.
            Defaults to None. If given, this overrides the `position` argument.

    Returns:
        napari.types.PointsData: Moved points
    """
    return points + position * vectors[:, 1]


@register_function(
        menu="Points > Calculate pairwise distance vectors (n-STRESS)")
@frame_by_frame
def pairwise_point_distances(points: 'napari.types.PointsData',
                             fitted_points: 'napari.types.PointsData'
                             ) -> 'napari.types.VectorsData':
    """
    Calculate pairwise distance vectors between pointclouds.

    For this to work, the two pointclouds must have the same
    number of points. This can be the case, for example, when
    one pointcloud is a fitted ellipsoid and the other is the
    original pointcloud.

    Parameters
    ----------
    points : PointsData
    fitted_points : PointsData

    Raises
    ------
    ValueError
        Both pointclouds must have the same number of points.

    Returns
    -------
    VectorsData

    """
    if not len(points) == len(fitted_points):
        raise ValueError('Both pointclouds must have same length, but had'
                         f'{len(points)} and {len(fitted_points)}.')

    delta = points - fitted_points
    return np.stack([fitted_points, delta]).transpose((1, 0, 2))


@register_function(
        menu="Points > Calculate normal vectors on pointcloud (n-STRESS, vedo)")
@frame_by_frame
def normal_vectors_on_pointcloud(
    points: "napari.types.PointsData"
    ) -> "napari.types.VectorsData":
    """
    Calculate normal vectors on pointcloud.

    Args:
        pointcloud (napari.types.PointsData): Pointcloud

    Returns:
        napari.types.VectorsData: Normal vectors
    """
    import vedo

    pointcloud = vedo.pointcloud.Points(points)
    center_of_mass = pointcloud.centerOfMass()
    pointcloud.compute_normals_with_pca(orientation_point=center_of_mass)

    normals = pointcloud.pointdata["Normals"]
    normals = np.stack([points, normals]).transpose((1, 0, 2))

    return normals


@register_function(
        menu="Surfaces > Calculate normal vectors on surface (n-STRESS, vedo)")
@frame_by_frame
def normal_vectors_on_surface(
    surface: "napari.types.SurfaceData",
    ) -> "napari.types.VectorsData":
    """
    Calculate normal vectors on surface.

    Args:
        surface (napari.types.SurfaceData): Surface mesh

    Returns:
        napari.types.VectorsData: Normal vectors
    """
    import vedo

    surface = vedo.mesh.Mesh((surface[0], surface[1]))
    surface.compute_normals()
    normals = surface.pointdata["Normals"]

    normals = np.stack([surface.points(), normals]).transpose((1, 0, 2))

    return normals


def sample_intensity_along_vector(
    sample_vectors: "napari.types.sample_vectorsData",
    image: "napari.types.ImageData",
    sampling_distance: float = 1.0,
    interpolation_method: str = "linear",
    ) -> pd.DataFrame:
    """
    Sample intensity along sample_vectors of equal length.

    Args:
        sample_vectors (napari.types.sample_vectorsData): sample_vectors
            along which to measure intensity
        image (napari.types.ImageData): Image to sample
        sampling_distance (float, optional): Distance between samples.
            Defaults to 1.0.
        interpolation_method (str, optional): Interpolation method.
            Defaults to 'linear'.
    """
    from scipy.interpolate import RegularGridInterpolator

    # Create coords for interpolator
    X1 = np.arange(0, image.shape[0], 1)
    X2 = np.arange(0, image.shape[1], 1)
    X3 = np.arange(0, image.shape[2], 1)
    interpolator = RegularGridInterpolator(
        (X1, X2, X3),
        image,
        bounds_error=False,
        fill_value=np.nan,
        method=interpolation_method,
    )

    # get start point and unit normal vector
    start_points = sample_vectors[:, 0]
    lengths = np.linalg.norm(sample_vectors[:, 1], axis=1)[:, np.newaxis]
    unit_vector = sample_vectors[:, 1] / lengths

    # calculate number of steps
    steps = np.round(lengths / sampling_distance).mean().astype(int)

    # sample intensity
    intensity = np.zeros((len(start_points), steps))

    coordinates_along_sample_vectors = []
    for idx in range(len(start_points)):
        _directions = [
            sampling_distance * unit_vector[idx] * i for i in np.arange(steps)
        ]
        _directions = np.stack(_directions)

        _sample_vectors = start_points[idx] + _directions
        coordinates_along_sample_vectors.append(_sample_vectors)

    for idx in range(len(start_points)):
        intensity[idx, :] = interpolator(coordinates_along_sample_vectors[idx])

    intensity = pd.DataFrame(np.stack(intensity))
    intensity.columns = [f"step_{idx}" for idx in range(steps)]

    return intensity
