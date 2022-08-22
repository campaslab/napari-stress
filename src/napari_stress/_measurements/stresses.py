# -*- coding: utf-8 -*-
from napari.types import VectorsData, PointsData
from napari.layers import Points

from .curvature import curvature_on_ellipsoid
from ..types import (_METADATAKEY_H0_E123_ELLIPSOID,
                     _METADATAKEY_MEAN_CURVATURE)

def tissue_and_cell_scale_stress(ellipsoid_points: PointsData,
                                 ellipsoid: VectorsData,
                                 quadrature_points_ellipsoid: Points,
                                 gamma: float = 0.0):

    # collect features: H_E123 (stress along ellipsoid axis)
    _, _, metadata = curvature_on_ellipsoid(ellipsoid, ellipsoid_points)
    H_ellipsoid_major_medial_minor = metadata[_METADATAKEY_H0_E123_ELLIPSOID]

    # collect features: H0_ellipsoid
    if isinstance(quadrature_points_ellipsoid, Points):
        H0_ellipsoid = quadrature_points_ellipsoid.metadata[]

    # use H0_Ellpsoid to calculate tissue stress projections:
    sigma_11_e = 2 * gamma * (H_ellipsoid_major_medial_minor[0] - H0_surface_integral_ellipsoid)
    sigma_22_e = 2 * gamma * (H_ellipsoid_major_medial_minor[1] - H0_surface_integral_ellipsoid)
    sigma_33_e = 2 * gamma * (H_ellipsoid_major_medial_minor[2] - H0_surface_integral_ellipsoid)
