import numpy as np
from napari_tools_menu import register_function
from ._utils.frame_by_frame import frame_by_frame

from typing import Annotated


@register_function(
    menu="Points > Move point along vectors by absolute value (n-STRESS)"
)
@frame_by_frame
def absolute_move_points_along_vector(
    points: "napari.types.PointsData",
    vectors: "napari.types.VectorsData",
    position: float = 1,
    pointwise_position: np.ndarray = None,
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
    menu="Points > Move point along vectors by relative factor (n-STRESS)"
)
@frame_by_frame
def relative_move_points_along_vector(
    points: "napari.types.PointsData",
    vectors: "napari.types.VectorsData",
    position: float = 0.5,
    pointwise_position: np.ndarray = None,
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


@register_function(menu="Points > Calculate pairwise distance vectors (n-STRESS)")
@frame_by_frame
def pairwise_point_distances(
    points: "napari.types.PointsData", fitted_points: "napari.types.PointsData"
) -> "napari.types.VectorsData":
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
        raise ValueError(
            "Both pointclouds must have same length, but had"
            f"{len(points)} and {len(fitted_points)}."
        )

    delta = points - fitted_points
    return np.stack([fitted_points, delta]).transpose((1, 0, 2))


@register_function(
    menu="Points > Calculate normal vectors on pointcloud (n-STRESS, vedo)",
)
@frame_by_frame
def normal_vectors_on_pointcloud(
    points: "napari.types.PointsData",
    length_multiplier: Annotated[float, {"max": 100, "min": -100}] = 1.0,
    center: bool = True,
) -> "napari.types.VectorsData":
    """
    Calculate normal vectors on pointcloud.

    Parameters
    ----------
    points : PointsData
    length_multiplier : float, optional
        Length multiplier for normal vectors, by default 1.0
    center : bool, optional
        Center normal vectors on surface, by default True

    Returns
    -------
    VectorsData
        Normal vectors
    """
    import vedo

    pointcloud = vedo.pointcloud.Points(points)
    center_of_mass = pointcloud.center_of_mass()
    pointcloud.compute_normals_with_pca(orientation_point=center_of_mass)
    normals = length_multiplier * pointcloud.pointdata["Normals"]

    if center:
        points = points - normals / 2

    normals = np.stack([points, normals]).transpose((1, 0, 2))

    return normals


@register_function(
    menu="Surfaces > Calculate normal vectors on surface (n-STRESS, vedo)"
)
@frame_by_frame
def normal_vectors_on_surface(
    surface: "napari.types.SurfaceData",
    length_multiplier: Annotated[float, {"max": 100, "min": -100}] = 1.0,
    center: bool = True,
) -> "napari.types.VectorsData":
    """
    Calculate normal vectors on surface.

    Parameters
    ----------
    surface : SurfaceData
    length_multiplier : float, optional
        Length multiplier for normal vectors, by default 1.0
    center : bool, optional
        Center normal vectors on surface, by default True

    Returns
    -------
    VectorsData
        Normal vectors
    """
    import vedo

    surface = vedo.mesh.Mesh((surface[0], surface[1]))
    surface.compute_normals()
    normals = length_multiplier * surface.pointdata["Normals"]
    base = surface.vertices

    if center:
        base = base - normals / 2

    normals = np.stack([base, normals]).transpose((1, 0, 2))

    return normals
