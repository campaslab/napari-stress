# -*- coding: utf-8 -*-
from cmath import exp
from napari.types import LayerDataTuple, PointsData
from napari.layers import Points
import numpy as np
from enum import Enum

from .._utils.frame_by_frame import frame_by_frame
from .spherical_harmonics import shtools_spherical_harmonics_expansion,\
    stress_spherical_harmonics_expansion,\
    lebedev_quadrature,\
    create_manifold

import napari
from napari_tools_menu import register_function


class spherical_harmonics_methods(Enum):
    """Available methods for spherical harmonics expansion."""

    shtools = {'function': shtools_spherical_harmonics_expansion}
    stress = {'function': stress_spherical_harmonics_expansion}

class expansion_types(Enum):
    """Available coordinate systems for spherical harmonics expansion."""
    cartesian = 'cartesian'
    radial = 'radial'


@register_function(menu="Points > Fit spherical harmonics (n-STRESS")
@frame_by_frame
def fit_spherical_harmonics(points: PointsData,
                            max_degree: int = 5,
                            implementation: spherical_harmonics_methods = spherical_harmonics_methods.stress,
                            expansion_type: expansion_types = expansion_types.cartesian
                            ) -> LayerDataTuple:
    """
    Approximate a surface by spherical harmonics expansion.

    Parameters
    ----------
    points : PointsData
    max_degree : int
        Order up to which spherical harmonics should be included for the approximation.
    implementation: spherical_harmonics_methods
        Which implementation to use for spherical harmonics fit (stress or pyshtools). 
        Default is `spherical_harmonics_methods.stress`
    expansion_type: expansion_type
        Which coordinate to use for expansion. Can be `cartesian` or `radial`. For cartesian
        expansion, x/y/z will be approximated separately with a spherical harmonics expansion
        or radial for radial approximation.


    Returns
    -------
    LayerDataTuple
        Pointcloud on surface of a spherical harmonics expansion at the same
        latitude/longitude as the input points.

    See Also
    --------
    [1] https://en.wikipedia.org/wiki/Spherical_harmonics

    """
    # Parse inputs
    if isinstance(implementation, str):
        fit_function = spherical_harmonics_methods.__members__[implementation].value['function']
    else:
        fit_function = implementation.value['function']
    fitted_points, coefficients = fit_function(points,
                                               max_degree=max_degree,
                                               expansion_type=expansion_type.value)

    properties, features, metadata = {}, {}, {}

    features['error'] = np.linalg.norm(fitted_points - points, axis=1)
    metadata['spherical_harmonics_coefficients'] = coefficients
    metadata['spherical_harmonics_implementation'] = implementation

    properties['features'] = features
    properties['metadata'] = metadata
    properties['face_color'] = 'error'
    properties['size'] = 0.5

    return (fitted_points, properties, 'points')

@register_function(menu="Points > Perform lebedev quadrature (n-STRESS)",
                   number_of_quadrature_points={'min': 6, 'max': 5180})
@frame_by_frame
def perform_lebedev_quadrature(points: Points,
                               number_of_quadrature_points: int = 500,
                               use_minimal_point_set: bool = False,
                               viewer: napari.Viewer = None
                               ) -> LayerDataTuple:
    """
    Perform lebedev quadrature and manifold creaiton on spherical-harmonics expansion.

    Parameters
    ----------
    points : Points
    number_of_quadrature_points : int, optional
        Number of quadrature points to represent the surface. The default is 500.
    use_minimal_point_set : bool, optional
        Whether or not to use the minimally required number of quadrature
        points instead of the number given by `number_of_quadrature_points`.
        Depends on the chosen `max_degree` of the previous spherical harmonics
        expansion.The default is False.
    viewer : napari.Viewer, optional


    Returns
    -------
    LayerDataTuple

    """
    metadata = points.metadata

    if not 'spherical_harmonics_coefficients' in metadata.keys():
        raise ValueError('Missing spherical harmonics coefficients. Use spherical harmonics expansion first')

    max_degree = metadata['spherical_harmonics_coefficients'].shape[-1] - 1

    lebedev_points, LBDV_Fit = lebedev_quadrature(metadata['spherical_harmonics_coefficients'],
                                                  number_of_quadrature_points,
                                                  use_minimal_point_set=use_minimal_point_set)

    # create manifold object for this quadrature
    manifold = create_manifold(lebedev_points, lebedev_fit=LBDV_Fit, max_degree=max_degree)
    metadata['manifold'] = manifold

    properties, features = {}, {}
    properties['features'] = features
    properties['metadata'] = metadata
    properties['size'] = 0.5
    properties['name'] = 'Result of lebedev quadrature'

    lebedev_points = manifold.get_coordinates()

    # if viewer is not None:
    #     if properties['name'] not in viewer.layers:
    #         viewer.add_points(lebedev_points, **properties)
    #     else:
    #         layer = viewer.layers[properties['name']]
    #         layer.features = features
    #         layer.metadata = metadata
    #         layer.data = lebedev_points
    return (lebedev_points, properties, 'points')
