# -*- coding: utf-8 -*-

from napari.types import LayerDataTuple, PointsData

from .._utils.frame_by_frame import frame_by_frame

from . import sph_func_SPB as sph_f
from . import euclidian_k_form_SPB as euc_kf
from . import lebedev_info_SPB as lebedev_info
from ._expansion import spherical_harmonics_methods

import numpy as np

@frame_by_frame
def measure_curvature(points: PointsData,
                      max_degree: int = 5,
                      implementation: spherical_harmonics_methods = spherical_harmonics_methods.stress,
                      number_of_quadrature_points: int = 3000,
                      ) -> PointsData:

    # Parse inputs: Spherical harmonics implementation
    if isinstance(implementation, str):
        fit_function = spherical_harmonics_methods.__members__[implementation].value['function']
    else:
        fit_function = implementation.value['function']
    fitted_points, coefficients = fit_function(points, max_degree=max_degree)
    
    # Get possible number of quadrature points
    if number_of_quadrature_points > 5810:
        number_of_quadrature_points = 5810
    else:
        number_of_quadrature_points = lebedev_info.look_up_lbdv_pts(max_degree + 1)
    
    
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
    
    return lebedev_points