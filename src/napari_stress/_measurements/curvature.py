# -*- coding: utf-8 -*-

from .._stress import manifold_SPB as mnfd
from .._stress import euclidian_k_form_SPB as euc_kf
from .._spherical_harmonics.spherical_harmonics import get_normals_on_manifold

from .utils import naparify_measurement

from napari_tools_menu import register_function
import numpy as np

from napari.types import PointsData, VectorsData
import napari

from .._utils import coordinate_conversion as conversion
from ..types import (_METADATAKEY_MEAN_CURVATURE,
                     _METADATAKEY_H0_ELLIPSOID,
                     _METADATAKEY_H_E123_ELLIPSOID,
                     _METADATAKEY_H0_SURFACE_INTEGRAL,
                     _METADATAKEY_H0_ARITHMETIC,
                     _METADATAKEY_GAUSS_BONNET_REL,
                     _METADATAKEY_GAUSS_BONNET_ABS)

@register_function(menu="Measurement > Measure mean curvature on ellipsoid (n-STRESS)")
def curvature_on_ellipsoid(ellipsoid: VectorsData,
                           sample_points: PointsData,
                           viewer: napari.Viewer = None) -> (np.ndarray, dict, dict):
    """
    Calculate curvature at sample points on the surface of an ellipse.

    Parameters
    ----------
    ellipsoid : VectorsData
    sample_points : PointsData
        points on the ellipse surface. Can be generated from a pointcloud using
        the `approximation.expand_points_on_ellipse` function

    Returns
    -------
    sample_points : np.ndarray
        Points at which curvature is measured
    features: dict
        dictionary containing mean curvature values at sample points
    metadata: dict
        dictionary containing global mean curvature H0 of ellipsoid

    See Also
    --------
    https://mathworld.wolfram.com/Ellipsoid.html

    """
    lengths = conversion._axes_lengths_from_ellipsoid(ellipsoid)
    a0 = lengths[0]
    a1 = lengths[1]
    a2 = lengths[2]
    U, V = conversion.cartesian_to_elliptical(ellipsoid, sample_points)

    # calculate point-wise mean curvature H_i
    num_H_ellps = a0*a1*a2* ( 3.*(a0**2 + a1**2) + 2.*(a2**2) + (a0**2 + a1**2 -2.*a2**2)*np.cos(2.*V) - 2.*(a0**2 - a1**2)*np.cos(2.*U)*np.sin(V)**2 )
    den_H_ellps = 8.*( (a0*a1*np.cos(V))**2 + ( a2*np.sin(V) )**2 * ( (a1*np.cos(U))**2 + (a0*np.sin(U))**2 ) )**1.5
    H_ellps_pts = (num_H_ellps/den_H_ellps).squeeze()

    # calculate averaged curvatures H_0: 1st method of H0 computation, for Ellipsoid in UV points
    H0_ellps_avg_ellps_UV_curvs = H_ellps_pts.mean(axis=0)

    # calculate maximum/minimum mean curvatures from largest to shortest axis
    order = np.argsort(lengths)[::-1]
    lengths_sorted = lengths[order]
    a0 = lengths_sorted[0]
    a1 = lengths_sorted[1]
    a2 = lengths_sorted[2]

    H_ellps_e_1 = a0/(2.*a1**2) +  a0/(2.*a2**2)
    H_ellps_e_2 = a1/(2.*a0**2) +  a1/(2.*a2**2)
    H_ellps_e_3 = a2/(2.*a0**2) +  a2/(2.*a1**2)

    H0_ellipsoid_major_minor = [H_ellps_e_1, H_ellps_e_2, H_ellps_e_3]

    # add to viewer if it doesn't exist.
    properties, features, metadata = {}, {}, {}
    features[_METADATAKEY_MEAN_CURVATURE] = H_ellps_pts
    metadata[_METADATAKEY_H0_ELLIPSOID] = H0_ellps_avg_ellps_UV_curvs
    metadata[_METADATAKEY_H_E123_ELLIPSOID] = H0_ellipsoid_major_minor

    properties['features'] = features
    properties['metadata'] = metadata
    properties['face_color'] = _METADATAKEY_MEAN_CURVATURE
    properties['size'] = 0.5
    properties['name'] = 'Result of mean curvature on ellipsoid'

    if viewer is not None:
        if properties['name'] not in viewer.layers:
            viewer.add_points(sample_points, **properties)
        else:
            layer = viewer.layers[properties['name']]
            layer.features[_METADATAKEY_MEAN_CURVATURE] = H_ellps_pts
            layer.metadata[_METADATAKEY_H0_ELLIPSOID] = H0_ellps_avg_ellps_UV_curvs
            layer.metadata[_METADATAKEY_H_E123_ELLIPSOID] = H0_ellipsoid_major_minor

    return sample_points, features, metadata

@register_function(menu="Measurement > Measure Gauss-Bonnet error on manifold (n-STRESS")
@naparify_measurement
def gauss_bonnet_test(manifold: mnfd.manifold) -> (np.ndarray, dict, dict):
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

    metadata = {_METADATAKEY_GAUSS_BONNET_ABS: Gauss_Bonnet_Err,
                _METADATAKEY_GAUSS_BONNET_REL: Gauss_Bonnet_Rel_Err}
    return None, None, metadata

@register_function(menu="Measurement > Measure mean curvature on manifold (n-STRESS")
def calculate_mean_curvature_on_manifold(manifold: mnfd.manifold) -> (dict, dict):
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
    features = {_METADATAKEY_MEAN_CURVATURE: mean_curvatures}
    metadata = {_METADATAKEY_H0_ARITHMETIC: H0_arithmetic,
               _METADATAKEY_H0_SURFACE_INTEGRAL: H0_surface_integral}

    return features, metadata

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
