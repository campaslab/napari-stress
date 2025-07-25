import napari
from napari.types import LayerDataTuple, PointsData

from .._utils.frame_by_frame import frame_by_frame


@frame_by_frame
def deviation_from_ellipsoidal_mode(
    points: PointsData, max_degree: int = 5, viewer: napari.Viewer = None
) -> LayerDataTuple:
    import numpy as np

    from napari_stress import approximation, vectors

    from .._spherical_harmonics.fit_utils import Least_Squares_Harmonic_Fit
    from .._stress import sph_func_SPB as sph_f
    from .._utils.coordinate_conversion import cartesian_to_elliptical
    from ..types import _METADATAKEY_ELIPSOID_DEVIATION_CONTRIB

    # calculate errors
    expander = approximation.EllipsoidExpander()
    expander.fit(points)
    ellipsoid = expander.coefficients_
    errors = vectors.pairwise_point_distances(points, expander.expand(points))[
        :, 1
    ]
    normals = expander.properties["normals"][:, 1]
    signed_errors = -1.0 * np.multiply(normals, errors).sum(axis=1)

    # least squares harmonic fit
    longitude, latitude = cartesian_to_elliptical(ellipsoid, points)
    coefficients = Least_Squares_Harmonic_Fit(
        fit_degree=max_degree,
        sample_locations=(longitude, latitude),
        values=signed_errors,
    )
    coefficients = np.abs(coefficients)

    coefficients = sph_f.Un_Flatten_Coef_Vec(coefficients, max_degree)

    features = {"signed_elipsoid_deviation": signed_errors}
    metadata = {_METADATAKEY_ELIPSOID_DEVIATION_CONTRIB: coefficients}
    properties = {
        "features": features,
        "metadata": metadata,
        "size": 0.5,
        "face_color": "signed_elipsoid_deviation",
        "face_colormap": "viridis",
    }

    return (points, properties, "points")
