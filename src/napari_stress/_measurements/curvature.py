# -*- coding: utf-8 -*-

from .._stress import manifold_SPB as mnfd
from .._stress import euclidian_k_form_SPB as euc_kf
from .._spherical_harmonics.spherical_harmonics import get_normals_on_manifold

from .utils import naparify_measurement

from ..types import manifold
from ..types import (
    _METADATAKEY_MEAN_CURVATURE,
    _METADATAKEY_H0_SURFACE_INTEGRAL,
    _METADATAKEY_H0_ARITHMETIC_AVERAGE
    )

from napari_tools_menu import register_function

import numpy as np

@register_function(menu="Measurement > Measure Gauss-Bonnet error on manifold (n-STRESS")
@naparify_measurement
def gauss_bonnet_test(manifold: manifold) -> (np.ndarray, dict, dict):
    """
    Use Gauss-Bonnet theorem to measure resolution on manifold.

    Parameters
    ----------
    manifold: mnfd.manifold


    See Also
    --------
    https://en.wikipedia.org/wiki/Gauss%E2%80%93Bonnet_theorem
    """
    K_lbdv_pts = euc_kf.Combine_Chart_Quad_Vals(manifold.K_A_pts,
                                                manifold.K_B_pts,
                                                manifold.lebedev_info)
    Gauss_Bonnet_Err = euc_kf.Integral_on_Manny(K_lbdv_pts,
                                                manifold,
                                                manifold.lebedev_info) - 4*np.pi
    Gauss_Bonnet_Rel_Err = abs(Gauss_Bonnet_Err)/(4*np.pi)

    metadata = {'Gauss_Bonnet_error': Gauss_Bonnet_Err,
                'Gauss_Bonnet_relative_error': Gauss_Bonnet_Rel_Err}
    return None, None, metadata

@register_function(menu="Measurement > Measure mean curvature on manifold (n-STRESS")
@naparify_measurement
def calculate_mean_curvature_on_manifold(manifold: mnfd.manifold) -> (np.ndarray, dict, dict):
    """
    Calculate mean curvatures for a given manifold.

    Parameters
    ----------
    manifold: mnfd.manifold

    Returns
    -------
    np.ndarray
        Mean curvature value for every lebedev point.

    """
    normals = get_normals_on_manifold(manifold)

    # Test orientation:
    points = manifold.get_coordinates()
    centered_lbdv_pts = points - points.mean(axis=0)[None, :]

    # Makre sure orientation is inward, so H is positive (for Ellipsoid, and small deviations):
    Orientations = [np.dot(x, y) for x, y in zip(centered_lbdv_pts,  normals)]
    num_pos_orr = np.sum(np.asarray(Orientations).flatten() > 0)

    Orientation = 1. # unchanged (we want INWARD)
    if(num_pos_orr > .5 * len(centered_lbdv_pts)):
        Orientation = -1.

    mean_curvatures = Orientation*euc_kf.Combine_Chart_Quad_Vals(manifold.H_A_pts, manifold.H_B_pts, manifold.lebedev_info).squeeze()
    H0_arithmetic = averaged_mean_curvature(mean_curvatures)
    H0_surface_integral = surface_integrated_mean_curvature(mean_curvatures,
                                                            manifold)

    # aggregate results in dictionary
    features = {'Mean_curvature_at_lebedev_points': mean_curvatures}
    metadata = {'H0_arithmetic_average': H0_arithmetic,
               'H0_surface_integral': H0_surface_integral}

    return None, features, metadata

def averaged_mean_curvature(curvatures: np.ndarray) -> float:
    """Calculate arithmetic average of mean curvature."""
    return curvatures.flatten().mean()

def surface_integrated_mean_curvature(mean_curvatures: np.ndarray,
                                      manifold: mnfd.manifold):
    """Calculate mean curvature by integrating surface area."""
    Integral_on_surface = euc_kf.Integral_on_Manny(mean_curvatures,
                                                   manifold,
                                                   manifold.lebedev_info)
    Integral_on_sphere = euc_kf.Integral_on_Manny(np.ones_like(mean_curvatures).astype(float),
                                                  manifold,
                                                  manifold.lebedev_info)

    return Integral_on_surface/Integral_on_sphere
