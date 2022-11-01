from napari.types import PointsData, LayerDataTuple
from napari_tools_menu import register_function

from .._utils.frame_by_frame import frame_by_frame
import napari

@register_function(menu="Measurement > Measure deviation from ellipsoid (n-STRESS)")
@frame_by_frame
def deviation_from_ellipsoidal_mode(points: PointsData,
                                    max_degree:int=5,
                                    viewer: napari.Viewer = None) -> LayerDataTuple:
    from napari_stress import approximation
    from .._utils.fit_utils import Least_Squares_Harmonic_Fit
    from .._utils.coordinate_conversion import cartesian_to_elliptical
    from .._stress import sph_func_SPB as sph_f
    import numpy as np

    from ..types import _METADATAKEY_ELIPSOID_DEVIATION_CONTRIB
    
    # calculate errors
    ellipsoid = approximation.least_squares_ellipsoid(points)
    ellipsoid_points = approximation.expand_points_on_ellipse(ellipsoid, points)
    errors = approximation.pairwise_point_distances(points, ellipsoid_points)[:, 1]
    normals = approximation.normals_on_ellipsoid(ellipsoid_points)[:, 1]
    signed_errors = -1.*np.multiply(normals, errors).sum(axis=1)

    # least squares harmonic fit
    longitude, latitude = cartesian_to_elliptical(ellipsoid, points)
    coefficients = Least_Squares_Harmonic_Fit(
                    fit_degree=max_degree,
                    sample_locations = (longitude, latitude),
                    values = signed_errors)
    coefficients = np.abs(coefficients)

    coefficients = sph_f.Un_Flatten_Coef_Vec(coefficients, max_degree)

    features = {'signed_elipsoid_deviation': signed_errors}
    metadata = {_METADATAKEY_ELIPSOID_DEVIATION_CONTRIB: coefficients}
    properties = {'features': features,
                  'metadata': metadata,
                  'size': 0.5,
                  'face_color': 'signed_elipsoid_deviation',
                  'face_colormap': 'viridis'}

    return (points, properties, 'points')

    

