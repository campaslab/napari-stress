from typing import TYPE_CHECKING

from napari_tools_menu import register_function

if TYPE_CHECKING:
    import napari

from .._utils import frame_by_frame
from .expansion_ellipsoid import (
    EllipsoidExpander,
    EllipsoidImageExpander,
)
from .expansion_spherical_harmonics import (
    SphericalHarmonicsExpander,
)


@register_function(menu="Points > Fit ellipsoid to pointcloud (n-STRESS)")
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


@register_function(
    menu="Points > Expand point locations on ellipsoid (n-STRESS)"
)
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


@register_function(menu="Points > Fit spherical harmonics (n-STRESS")
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


@frame_by_frame
def expand_ellipsoid_on_image(
    image: "napari.types.ImageData",
    n_points: int = 512,
) -> "napari.types.PointsData":
    """
    Fit and expand an ellipsoid on an image.

    Parameters
    ----------
    image : napari.types.ImageData
        The image to expand the ellipsoid on.
    n_points : int
        Number of points to use for the expansion.

    Returns
    -------
    points : napari.types.PointsData
        The expanded points.
    """
    expander = EllipsoidImageExpander()
    points = expander.fit_expand(image, n_points=n_points)

    # assuming points are Nx3 and scale is length 3,
    # multiply all points with scale

    return points
