# -*- coding: utf-8 -*-

from statistics import mean
from .._stress import euclidian_k_form_SPB as euc_kf
from .._spherical_harmonics.spherical_harmonics import get_normals_on_manifold

from napari_tools_menu import register_function
import numpy as np

from napari.types import PointsData, VectorsData, LayerDataTuple
from napari.layers import Layer
from napari import Viewer

from .._utils import coordinate_conversion as conversion
from ..types import (_METADATAKEY_MEAN_CURVATURE,
                     _METADATAKEY_H0_ELLIPSOID,
                     _METADATAKEY_H_E123_ELLIPSOID,
                     _METADATAKEY_H0_SURFACE_INTEGRAL,
                     _METADATAKEY_S2_VOLUME_INTEGRAL,
                     _METADATAKEY_H0_VOLUME_INTEGRAL,
                     _METADATAKEY_H0_ARITHMETIC,
                     _METADATAKEY_H0_RADIAL_SURFACE,
                     _METADATAKEY_GAUSS_BONNET_REL,
                     _METADATAKEY_GAUSS_BONNET_ABS,
                     manifold)

from typing import Tuple

@register_function(menu="Measurement > Measure mean curvature on ellipsoid (n-STRESS)")
def curvature_on_ellipsoid(ellipsoid: VectorsData,
                           sample_points: PointsData) -> LayerDataTuple:
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
    LayerDataTuple
        LayerDataTuple of structured `(data, properties, 'points')`. The results
        of this function are stored in:
            * LayerDataTuple['features']['mean_curvature']
            * LayerDataTuple['metadata']['H_ellipsoid_major_medial_minor']
            * LayerDataTuple['metadata']['H0_ellipsoid']

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

    H0_ellipsoid_major_minor = mean_curvature_on_ellipse_cardinal_points(
        ellipsoid)

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

    return (sample_points, properties, 'points')

def mean_curvature_on_ellipse_cardinal_points(ellipsoid: VectorsData
                                              ) -> list:
    """
    Calculate mean points on major axes tip points of ellipsoid.

    Parameters
    ----------
    ellipsoid : VectorsData

    Returns
    -------
    list
        List with mean curvature at major/medial/minor axis of ellipsoid.

    """
    lengths = conversion._axes_lengths_from_ellipsoid(ellipsoid)
    a0 = lengths[0]
    a1 = lengths[1]
    a2 = lengths[2]

    # calculate maximum/minimum mean curvatures from largest to shortest axis
    order = np.argsort(lengths)[::-1]
    lengths_sorted = lengths[order]
    a0 = lengths_sorted[0]
    a1 = lengths_sorted[1]
    a2 = lengths_sorted[2]

    H_ellps_e_1 = a0/(2.*a1**2) +  a0/(2.*a2**2)
    H_ellps_e_2 = a1/(2.*a0**2) +  a1/(2.*a2**2)
    H_ellps_e_3 = a2/(2.*a0**2) +  a2/(2.*a1**2)

    return [H_ellps_e_1, H_ellps_e_2, H_ellps_e_3]

@register_function(menu="Measurement > Measure Gauss-Bonnet error on manifold (n-STRESS")
def gauss_bonnet_test(input_manifold: manifold, viewer: Viewer = None) -> Tuple[float, float]:
    """
    Use Gauss-Bonnet theorem to measure resolution on manifold.

    Parameters
    ----------
    input_manifold: manifold

    Returns
    -------
    Gauss_Bonnet_Err: float
        Absolute error of Gauss-Bonnet-test
    Gauss_Bonnet_Rel_Err: float
        Relative error of Gauss-Bonnet test (absolute error divided by 4*pi)

    See Also
    --------
    https://en.wikipedia.org/wiki/Gauss%E2%80%93Bonnet_theorem
    """
    layer = None
    if isinstance(input_manifold, Layer):
        layer = input_manifold
        input_manifold = input_manifold.metadata[manifold.__name__]

    K_lbdv_pts = euc_kf.Combine_Chart_Quad_Vals(input_manifold.K_A_pts,
                                                input_manifold.K_B_pts,
                                                input_manifold.lebedev_info)
    Gauss_Bonnet_Err = euc_kf.Integral_on_Manny(K_lbdv_pts,
                                                input_manifold,
                                                input_manifold.lebedev_info) - 4*np.pi
    Gauss_Bonnet_Rel_Err = abs(Gauss_Bonnet_Err)/(4*np.pi)

    if layer is not None:
        layer.metadata[_METADATAKEY_GAUSS_BONNET_ABS] = Gauss_Bonnet_Err
        layer.metadata[_METADATAKEY_GAUSS_BONNET_REL] = Gauss_Bonnet_Rel_Err

    return Gauss_Bonnet_Err, Gauss_Bonnet_Rel_Err

@register_function(menu="Measurement > Measure mean curvature on manifold (n-STRESS)")
def calculate_mean_curvature_on_manifold(input_manifold: manifold
                                         ) -> Tuple[np.ndarray, float, float]:
    """
    Calculate mean curvatures for a given manifold.

    Parameters
    ----------
    manifold: mnfd.manifold

    Returns
    -------
    mean_curvatures: np.ndarray
        Mean curvature value for every quadrature point

    """
    layer = None
    if isinstance(input_manifold, Layer):
        layer = input_manifold
        input_manifold = input_manifold.metadata[manifold.__name__]

    normals = get_normals_on_manifold(input_manifold)

    # Test orientation:
    points = input_manifold.get_coordinates()
    centered_lbdv_pts = points - points.mean(axis=0)[None, :]

    # Makre sure orientation is inward, so H is positive (for Ellipsoid, and small deviations):
    Orientations = [np.dot(x, y) for x, y in zip(centered_lbdv_pts,  normals)]
    num_pos_orr = np.sum(np.asarray(Orientations).flatten() > 0)

    Orientation = 1. # unchanged (we want INWARD)
    if(num_pos_orr > .5 * len(centered_lbdv_pts)):
        Orientation = -1.

    mean_curvatures = Orientation*euc_kf.Combine_Chart_Quad_Vals(
        input_manifold.H_A_pts,
        input_manifold.H_B_pts,
        input_manifold.lebedev_info).squeeze()

    # Arithmetic average of curvature
    H0_arithmetic = numerical_averaged_mean_curvature(mean_curvatures)

    # Integrating surface area - this is the one used for downstream analysis
    H0_surface_integral = surface_integrated_mean_curvature(mean_curvatures,
                                                            input_manifold)

    if layer is not None:
        layer.features[_METADATAKEY_MEAN_CURVATURE] = mean_curvatures
        layer.metadata[_METADATAKEY_H0_ARITHMETIC] = H0_arithmetic
        layer.metadata[_METADATAKEY_H0_SURFACE_INTEGRAL] = H0_surface_integral

    return mean_curvatures, H0_arithmetic, H0_surface_integral

@register_function(menu="Measurement > Measure average mean curvature on manifold (n-STRESS)")
def average_mean_curvatures_on_manifold(input_manifold: manifold) -> Tuple[float, float]:
    """
    Calculate averaged mean curvatures on manifold.

    Args:
    ----
        input_manifold (manifold): Manifold object. Can be of type `napari.Layer` 
            if a `manifold` object is detected in `layer.metadata`

    Returns:
    -------
        H0_arithmetic: float
            arithmetic average of mean curvature
        H0_surface_integral: float
            surface-integrated averaged mean curvature
        H0_volume_integral: float
            volume-integrated averaged mean curvature. Only calculated if
            `manifold.manifold_type` is `radial`.
        S2_volume: float
            volume-integral of manifold. Only calculated if
            `manifold.manifold_type` is `radial`. 
    """

    mean_curvature, _, _ = calculate_mean_curvature_on_manifold(input_manifold)
    
    # If a layer was passed instead of manifold
    layer = None
    if isinstance(input_manifold, Layer):
        layer = input_manifold
        input_manifold = input_manifold.metadata[manifold.__name__]

    # Arithmetic average of curvature
    H0_arithmetic = numerical_averaged_mean_curvature(mean_curvature)

    # Integrating surface area - this is the one used for downstream analysis
    H0_surface_integral = surface_integrated_mean_curvature(mean_curvature,
                                                            input_manifold)
    
    S2_volume = None
    H0_volume_integral = None
    H0_radial_surface = None
    if input_manifold.manifold_type == 'radial':
        S2_volume, H0_volume_integral = volume_integrated_mean_curvature(input_manifold) 
        H0_radial_surface = radial_surface_averaged_mean_curvature(input_manifold)    

    # if a layer was passed instead of manifold - put result in metadata
    if layer is not None:
        layer.metadata[_METADATAKEY_H0_ARITHMETIC] = H0_arithmetic
        layer.metadata[_METADATAKEY_H0_SURFACE_INTEGRAL] = H0_surface_integral
        layer.metadata[_METADATAKEY_H0_VOLUME_INTEGRAL] = H0_volume_integral
        layer.metadata[_METADATAKEY_S2_VOLUME_INTEGRAL] = S2_volume
        layer.metadata[_METADATAKEY_H0_RADIAL_SURFACE] = H0_radial_surface

    return H0_arithmetic, H0_surface_integral, H0_volume_integral, S2_volume, H0_radial_surface


def numerical_averaged_mean_curvature(curvatures: np.ndarray) -> float:
    """Calculate arithmetic average of mean curvature."""
    return curvatures.flatten().mean()

def surface_integrated_mean_curvature(mean_curvatures: np.ndarray,
                                      manifold: manifold):
    """Calculate mean curvature by integrating surface area."""
    Integral_on_surface = euc_kf.Integral_on_Manny(mean_curvatures,
                                                   manifold,
                                                   manifold.lebedev_info)
    Integral_on_sphere = euc_kf.Integral_on_Manny(np.ones_like(mean_curvatures).astype(float),
                                                  manifold,
                                                  manifold.lebedev_info)

    return Integral_on_surface/Integral_on_sphere

def volume_integrated_mean_curvature(input_manifold: manifold) -> float:
    """Determine volume and averaged mean curvature by integrating over radii."""
    from .._stress.sph_func_SPB import S2_Integral
    
    assert input_manifold.manifold_type == 'radial'
    radii = input_manifold.raw_coordinates
    lbdv_info = input_manifold.lebedev_info

    volume = S2_Integral(radii[:, None]**3/3, lbdv_info)
    H0_from_Vol_Int = ((4.*np.pi)/(3.*volume))**(1./3.) # approx ~1/R, for V ~ (4/3)*pi*R^3
    return volume, H0_from_Vol_Int

def radial_surface_averaged_mean_curvature(input_manifold: manifold) -> float:
    """
     Eestimate H0 on Radial Manifold.
     
     Integratal of H_Rad (on surface) divided by Rad surface area

    Args:
        input_manifold (manifold): radial manifold

    Returns:
        float: averaged mean curvature
    """
    mean_curvature, _, _ = calculate_mean_curvature_on_manifold(input_manifold)
    sphere_radii = np.ones_like(mean_curvature)

    area_radial_manifold = euc_kf.Integral_on_Manny(mean_curvature, input_manifold, input_manifold.lebedev_info)
    area_unit_sphere = euc_kf.Integral_on_Manny(sphere_radii, input_manifold, input_manifold.lebedev_info)
    H0 = area_radial_manifold/area_unit_sphere
    return H0


def mean_curvature_on_radial_manifold(input_manifold: manifold) -> np.ndarray:
    """
    Calculate mean curvature on radial manifold

    Args:
        input_manifold (manifold): radial manifold

    Returns:
        np.ndarray: mean curvature
    """
    mean_curvature, _, _ = calculate_mean_curvature_on_manifold(input_manifold)
    H0 = radial_surface_averaged_mean_curvature(input_manifold)
    
    mean_curvature_radial = mean_curvature * abs(H0) / H0
    return mean_curvature_radial

def mean_curvature_differences_radial_cartesian_manifolds(manifold_cartesian: manifold,
                                                          manifold_radial: manifold) -> np.ndarray:
    """
    Calculate difference of radial and cartesian mean curvature calculation

    Args:
        manifold_cartesian (manifold): Cartesian manifold
        manifold_radial (manifold): Radial manifold

    Returns:
        np.ndarray: difference of mean curvatures.
    """
    mean_curvature_cartesian, _, _ = calculate_mean_curvature_on_manifold(manifold_cartesian)
    mean_curvature_radial, _, _ = calculate_mean_curvature_on_manifold(manifold_radial)
    
    on_radial = euc_kf.Integral_on_Manny(mean_curvature_radial - mean_curvature_cartesian, manifold_radial, manifold_radial.lebedev_info)
    on_cartesian = euc_kf.Integral_on_Manny(mean_curvature_radial - mean_curvature_cartesian, manifold_cartesian, manifold_cartesian.lebedev_info)

    return 0.5 * (on_radial + on_cartesian)