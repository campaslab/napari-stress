from typing import TYPE_CHECKING

from napari_timelapse_processor import frame_by_frame

if TYPE_CHECKING:
    import napari

from .expansion import EllipsoidExpander, SphericalHarmonicsExpander


@frame_by_frame
def fit_ellipsoid_to_pointcloud(
    points: "napari.types.PointsData",
) -> "napari.types.VectorsData":
    """
    Fit an ellipsoid to a set of points.

    Parameters
    ----------
    points : napari.types.PointsData
        The points to fit an ellipsoid to.

    Returns
    -------
    ellipsoid : napari.types.VectorsData
        The fitted ellipsoid.
    """
    expander = EllipsoidExpander()
    expander.fit(points)

    ellipsoid = expander.coefficients_

    return ellipsoid


@frame_by_frame
def expand_points_on_fitted_ellipsoid(
    points: "napari.types.PointsData",
) -> "napari.types.PointsData":
    """
    Project a set of points on a fitted ellipsoid.

    Parameters
    ----------
    points : napari.types.PointsData
        The points to project.

    Returns
    -------
    projected_points : napari.types.PointsData
        The projected points.
    """
    expander = EllipsoidExpander()
    expanded_points = expander.fit_expand(points)

    return expanded_points


@frame_by_frame
def normals_on_fitted_ellipsoid(
    points: "napari.types.PointsData",
) -> "napari.types.VectorsData":
    """
    Calculate normals on the ellipsoid fitted to the pointcloud.

    Parameters
    ----------
    points : napari.types.PointsData
        The points to calculate normals for. Should be points on the ellipsoid.
        To project points onto the ellipsoid, use
        `expand_points_on_fitted_ellipsoid`.

    Returns
    -------
    normals : napari.types.VectorsData
        The normals on the ellipsoid.
    """
    expander = EllipsoidExpander()
    expander.fit(points)

    return expander.properties["normals"]


@frame_by_frame
def expand_spherical_harmonics(
    points: "napari.types.PointsData",
    max_degree: int = 5,
) -> "napari.types.LayerDataTuple":
    """
    Expand points using spherical harmonics.

    Parameters
    ----------
    points : np.ndarray
        Points to be expanded.
    max_degree : int
        Maximum degree of spherical harmonics expansion.

    Returns
    -------
    expanded_points : np.ndarray
        Expanded points.
    """
    expander = SphericalHarmonicsExpander(max_degree=max_degree)
    expanded_points = expander.fit_expand(points)

    # convert dataframe expander.properties to dict so that one column is one key
    # and the values are the entries in the columns
    properties = {
        "features": {"residuals": expander.properties["residuals"]},
        "metadata": {
            "coefficients": expander.coefficients_,
            "max_degree": max_degree,
            "power_spectrum": expander.properties["power_spectrum"],
        },
        "name": "Spherical Harmonics Expansion",
        "face_color": "residuals",
        "size": 0.5,
        "face_colormap": "inferno",
    }
    points_layer = (expanded_points, properties, "points")
    return points_layer
