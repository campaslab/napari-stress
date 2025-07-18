from typing import TYPE_CHECKING

from napari_timelapse_processor import frame_by_frame

if TYPE_CHECKING:
    import napari

from .expansion_ellipsoid import (
    EllipsoidExpander,
    EllipsoidImageExpander,
)
from .expansion_spherical_harmonics import (
    SphericalHarmonicsExpander,
)


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
def expand_spherical_harmonics_on_lebedev_grid(
    points: "napari.types.PointsData",
    max_degree: int = 5,
    n_quadrature_points: int = 434,
    use_minimal_point_set: bool = False,
    expansion_type: str = "cartesian",
    normalize_spectrum: bool = True,
) -> "napari.types.LayerDataTuple":
    """
    Expand points using Lebedev quadrature and spherical harmonics.

    Parameters
    ----------
    points : np.ndarray
        Points to be expanded.
    max_degree : int
        Maximum degree of spherical harmonics expansion.
    n_quadrature_points : int
        Number of quadrature points to use for Lebedev grid.
    use_minimal_point_set : bool
        Whether to use the minimal pointset for Lebedev quadrature.
    expansion_type : str
        Type of expansion to use ('cartesian', 'radial', etc.).
    normalize_spectrum : bool
        Whether to normalize the spectrum of the expansion.
    Returns
    -------
    expanded_points : np.ndarray
        Expanded points.

    """
    from ..types import (
        _METADATAKEY_H0_ARITHMETIC,
        _METADATAKEY_H0_RADIAL_SURFACE,
        _METADATAKEY_H0_SURFACE_INTEGRAL,
        _METADATAKEY_H0_VOLUME_INTEGRAL,
        _METADATAKEY_MEAN_CURVATURE,
        _METADATAKEY_S2_VOLUME_INTEGRAL,
    )
    from .expansion_spherical_harmonics import (
        LebedevExpander,
    )

    expander = LebedevExpander(
        max_degree=max_degree,
        n_quadrature_points=n_quadrature_points,
        use_minimal_point_set=use_minimal_point_set,
        expansion_type=expansion_type,
        normalize_spectrum=normalize_spectrum,
    )

    expansion_surface = expander.fit_expand(
        points,
    )

    return (
        expansion_surface,
        {
            "features": {
                "mean_curvature": expander.properties["mean_curvature"],
            },
            "metadata": {
                _METADATAKEY_MEAN_CURVATURE: expander.properties[
                    _METADATAKEY_MEAN_CURVATURE
                ],
                _METADATAKEY_H0_ARITHMETIC: expander.properties[
                    _METADATAKEY_H0_ARITHMETIC
                ],
                _METADATAKEY_H0_SURFACE_INTEGRAL: expander.properties[
                    _METADATAKEY_H0_SURFACE_INTEGRAL
                ],
                _METADATAKEY_H0_VOLUME_INTEGRAL: expander.properties[
                    _METADATAKEY_H0_VOLUME_INTEGRAL
                ],
                _METADATAKEY_S2_VOLUME_INTEGRAL: expander.properties[
                    _METADATAKEY_S2_VOLUME_INTEGRAL
                ],
                _METADATAKEY_H0_RADIAL_SURFACE: expander.properties[
                    _METADATAKEY_H0_RADIAL_SURFACE
                ],
            },
        },
        "surface",
    )


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
