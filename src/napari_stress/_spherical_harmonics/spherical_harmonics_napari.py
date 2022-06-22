# -*- coding: utf-8 -*-
from napari.types import LayerDataTuple, PointsData
import numpy as np
from enum import Enum

from .._utils.frame_by_frame import frame_by_frame
from .spherical_harmonics import shtools_spherical_harmonics_expansion,\
    stress_spherical_harmonics_expansion,\
    integrate_on_manifold
from . import lebedev_info_SPB as lebedev_info
from . import sph_func_SPB as sph_f
from . import euclidian_k_form_SPB as euc_kf

import napari
import warnings
from napari_tools_menu import register_function


class spherical_harmonics_methods(Enum):
    """Available methods for spherical harmonics expansion."""

    shtools = {'function': shtools_spherical_harmonics_expansion}
    stress = {'function': stress_spherical_harmonics_expansion}

@register_function(menu="Points > Fit spherical harmonics (n-STRESS")
@frame_by_frame
def fit_spherical_harmonics(points: PointsData,
                            max_degree: int = 5,
                            implementation: spherical_harmonics_methods = spherical_harmonics_methods.stress
                            ) -> LayerDataTuple:
    """
    Approximate a surface by spherical harmonics expansion.

    Parameters
    ----------
    points : PointsData
    max_degree : int
        Order up to which spherical harmonics should be included for the approximation.

    Returns
    -------
    PointsData
        Pointcloud on surface of a spherical harmonics expansion at the same
        latitude/longitude as the input points.

    See Also
    --------
    [1] https://en.wikipedia.org/wiki/Spherical_harmonics#/media/File:Spherical_Harmonics.png

    """
    # Parse inputs
    if isinstance(implementation, str):
        fit_function = spherical_harmonics_methods.__members__[implementation].value['function']
    else:
        fit_function = implementation.value['function']
    fitted_points, coefficients = fit_function(points, max_degree=max_degree)

    properties, features = {}, {}
    features['error'] = np.linalg.norm(fitted_points - points, axis=1)
    properties['features'] = features
    properties['face_color'] = 'error'
    properties['size'] = 0.5

    return (fitted_points, properties, 'points')

@register_function(menu="Measurement > Surface curvature from points (n-STRESS)",
                   number_of_quadrature_points={'min': 6, 'max': 5180})
@frame_by_frame
def measure_curvature(points: PointsData,
                      max_degree: int = 5,
                      implementation: spherical_harmonics_methods = spherical_harmonics_methods.stress,
                      number_of_quadrature_points: int = 500,
                      use_minimal_point_set: bool = False,
                      viewer: napari.Viewer = None
                      ) -> PointsData:
    """
    Measure curvature on pointcloud surface.

    Parameters
    ----------
    points : PointsData
        DESCRIPTION.
    max_degree : int, optional
        DESCRIPTION. The default is 5.
    implementation : spherical_harmonics_methods, optional
        DESCRIPTION. The default is spherical_harmonics_methods.stress.
    number_of_quadrature_points : int, optional
        DESCRIPTION. The default is 1000.

    Returns
    -------
    PointsData
        DESCRIPTION.

    """
    if isinstance(implementation, str):
        fit_function = spherical_harmonics_methods.__members__[implementation].value['function']
    else:
        fit_function = implementation.value['function']
    fitted_points, coefficients = fit_function(points, max_degree=max_degree)

    # Clip number of quadrature points
    if number_of_quadrature_points > 5810:
        number_of_quadrature_points = 5810

    possible_n_points = np.asarray(list(lebedev_info.pts_of_lbdv_lookup.values()))
    index_correct_n_points = np.argmin(abs(possible_n_points - number_of_quadrature_points))
    number_of_quadrature_points = possible_n_points[index_correct_n_points]

    if use_minimal_point_set:
        number_of_quadrature_points = lebedev_info.look_up_lbdv_pts(max_degree + 1)

    # Only a specific amount of points are needed for spherical harmonics of a
    # given order. Using more points will not give more precise results.
    if number_of_quadrature_points > lebedev_info.look_up_lbdv_pts(max_degree + 1):
        warnings.warn(r'Note: Only {necessary_n_points} are required for exact results.')

    # Create spherical harmonics functions to represent z/y/x
    fit_functions = [
        sph_f.spherical_harmonics_function(x, max_degree) for x in coefficients
        ]

    # Get {Z/Y/X} Coordinates at lebedev points, so we can leverage our code more efficiently (and uniformly) on surface:
    LBDV_Fit = lebedev_info.lbdv_info(max_degree, number_of_quadrature_points)
    lebedev_points  = [
        euc_kf.get_quadrature_points_from_sh_function(f, LBDV_Fit, 'A') for f in fit_functions
        ]
    lebedev_points = np.stack(lebedev_points).squeeze().transpose()

    properties, features = {}, {}
    features['curvature'] = integrate_on_manifold(lebedev_points, LBDV_Fit, max_degree).squeeze()

    properties['features'] = features
    properties['face_color'] = 'curvature'
    properties['size'] = 0.5
    properties['name'] = 'Result of measure curvature'

    if viewer is not None:
        if properties['name'] not in viewer.layers:
            viewer.add_points(lebedev_points, **properties)
        else:
            layer = viewer.layers[properties['name']]
            layer.features = features
            layer.data = lebedev_points
    return lebedev_points
