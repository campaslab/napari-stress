# -*- coding: utf-8 -*-
from napari.types import VectorsData, PointsData

from ..types import (_METADATAKEY_H_E123_ELLIPSOID,
                     _METADATAKEY_MEAN_CURVATURE,
                     _METADATAKEY_H0_SURFACE_INTEGRAL,
                     _METADATAKEY_ANISO_STRESS_TISSUE,
                     _METADATAKEY_ANISO_STRESS_CELL,
                     _METADATAKEY_STRESS_TENSOR_CART,
                     _METADATAKEY_STRESS_TENSOR_ELLI)

from .._stress.manifold_SPB import manifold
from .utils import naparify_measurement

import numpy as np

@naparify_measurement
def tissue_and_cell_scale_stress(pointcloud: PointsData,
                                 ellipsoid: VectorsData,
                                 quadrature_points_on_ellipsoid: manifold,
                                 quadrature_points_on_droplet: manifold,
                                 gamma: float = 26.0,
                                 ) -> (np.ndarray, dict, dict):
    """
    Calculate anisotropic cell/tissue-scale stresses.

    Parameters
    ----------
    pointcloud: PointsData
        Input point cloud on droplet surface
    quadrature_points_on_ellipsoid: manifold
        Result of spherical harmonics expansion and subsequent lebedev
        quadrature of points on the surface of a fitted ellipsoid
    quadrature_points_on_droplet: manifold
        Result of spherical harmonics expansion and subsequent lebedev
        quadrature of points on the surface of the input pointcloud
    gamma: Interfacial surface tension of droplet in mN/m. The default is
        26mN/m. See also [1]
    alpha: Lower percentile to be excluded for calculation of total anisotropic
        stresses. Default is 0.05 (=5%)

    Returns
    -------
    np.ndarray:
        None
    dict:
        Dictionaries with keys `anisotropic_stress_tissue` and
        `anisotropic_stress_cell` holding the anisotropic stresses for every
        point on the droplet surface
    dict:
        Dictionary with keys `Tissue_stress_tensor_elliptical` and
        `Tissue_stress_tensor_cartesian` holding the anisotropic stress along
        the major/minor axes of the droplet and the cardinal image directions.

    See Also
    --------
    [1] Campàs, Otger, et al. "Quantifying cell-generated mechanical forces
    within living embryonic tissues." Nature methods 11.2 (2014): 183-189.

    """
    from .. import measurements, approximation

    # fit elliposid and get point on ellipsoid surface
    ellipsoid = approximation.least_squares_ellipsoid(pointcloud)
    ellipsoid_points = approximation.expand_points_on_ellipse(ellipsoid, pointcloud)

    # collect features: H_E123 (stress along ellipsoid axis)
    _, _, metadata = measurements.curvature_on_ellipsoid(ellipsoid,
                                                          ellipsoid_points)
    H_ellipsoid_major_medial_minor = metadata[_METADATAKEY_H_E123_ELLIPSOID]

    # collect features: H0_ellipsoid from spherical harmonics expansion on ellipsoid
    _, _features, _metadata = measurements.calculate_mean_curvature_on_manifold(
        quadrature_points_on_ellipsoid)

    H0_ellipsoid = _metadata[_METADATAKEY_H0_SURFACE_INTEGRAL]

    # =========================================================================
    # Tissue scale
    # =========================================================================
    # use H0_Ellpsoid to calculate tissue stress projections:
    sigma_11_e = 2 * gamma * (H_ellipsoid_major_medial_minor[0] - H0_ellipsoid)
    sigma_22_e = 2 * gamma * (H_ellipsoid_major_medial_minor[1] - H0_ellipsoid)
    sigma_33_e = 2 * gamma * (H_ellipsoid_major_medial_minor[2] - H0_ellipsoid)

    # tissue stress tensor (elliptical coordinates)
    Tissue_Stress_Tensor_elliptical = np.zeros((3,3))
    Tissue_Stress_Tensor_elliptical[0,0] = sigma_11_e
    Tissue_Stress_Tensor_elliptical[1,1] = sigma_22_e
    Tissue_Stress_Tensor_elliptical[2,2] = sigma_33_e

    # get rotation matrix:
    # the normalized major/minor axis vectors used as column vectors compose
    # the orientation matrix of the ellipsoid
    Rotation_matrix = ellipsoid[:, 1].T
    Rotation_matrix = Rotation_matrix / np.linalg.norm(Rotation_matrix, axis=0)

    # cartesian tissue stress tensor:
    Tissue_Stress_Tensor_cartesian = np.dot(
        np.dot(Rotation_matrix.T ,Tissue_Stress_Tensor_elliptical),
        Rotation_matrix)

    # =========================================================================
    # Cell scale
    # =========================================================================
    # get quadrature points on ellipsoid surface that correspond to spherical
    # harmonics expansion of droplet
    quadrature_points_on_ellipsoid = approximation.expand_points_on_ellipse(
        ellipsoid,
        quadrature_points_on_droplet.get_coordinates())
    _, features, metadata = measurements.curvature_on_ellipsoid(
        ellipsoid, quadrature_points_on_ellipsoid)

    mean_curvature_ellipsoid_lebedev_points = features[
        _METADATAKEY_MEAN_CURVATURE]

    # get mean curvature on doplet expansion
    _, features, metadata = measurements.calculate_mean_curvature_on_manifold(
        quadrature_points_on_droplet)

    mean_curvature_spherical_harmonics = features[_METADATAKEY_MEAN_CURVATURE]
    H0_spherical_harmonics = metadata[_METADATAKEY_H0_SURFACE_INTEGRAL]

    anisotropic_stress = 2 * gamma * (
        mean_curvature_spherical_harmonics - H0_spherical_harmonics
        )

    anisotropic_stress_tissue = 2 * gamma * (
        mean_curvature_ellipsoid_lebedev_points - H0_ellipsoid
        )

    anisotropic_stress_cell = anisotropic_stress - anisotropic_stress_tissue

    features, metadata = {}, {}

    features[_METADATAKEY_ANISO_STRESS_TISSUE] = anisotropic_stress_tissue
    features[_METADATAKEY_ANISO_STRESS_CELL] = anisotropic_stress_cell
    metadata[_METADATAKEY_STRESS_TENSOR_ELLI] = Tissue_Stress_Tensor_elliptical
    metadata[_METADATAKEY_STRESS_TENSOR_CART] = Tissue_Stress_Tensor_cartesian

    return _, features, metadata


def _cumulative_density_analysis(data: np.ndarray, alpha: float = 0.05):
    """
    Analyze distribution of given data.

    Parameters
    ----------
    data : np.ndarray
    alpha : float
        percentile in the data to bve excluded from analysis

    Returns
    -------
    results: dict
        Dictionary with keys `lower_boundary`, `upper_boundary`,
        `excluded_data_points` and `distribution_histogram`.
        The `excluded_data_points` are 1 where stresses were above the given
        alpha threshold and -1 where stresses were below the given alpha
        threshold.

    """
    from scipy import stats

    data = np.sort(data.flatten())

    # from: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_histogram.html
    hist_Data_Field = np.histogram(data, bins='auto', density=True)
    hist_dist = stats.rv_histogram(hist_Data_Field)

    min_val_excl_Data_Field = hist_dist.ppf(alpha)
    max_val_excl_Data_Field = hist_dist.ppf(1. - alpha)

    # 0: where within .5- \delta of median, -1, where CDF<\delta, 1 where CDF>1-\delta
    curv_pts_excluded_Data_Field = np.zeros_like(data)
    curv_pts_excluded_Data_Field = np.where(data < min_val_excl_Data_Field, -1, curv_pts_excluded_Data_Field)
    curv_pts_excluded_Data_Field = np.where(data > max_val_excl_Data_Field, 1, curv_pts_excluded_Data_Field)

    results = {}
    results['lower_boundary'] = min_val_excl_Data_Field
    results['upper_boundary'] = max_val_excl_Data_Field
    results['excluded_data_points'] = curv_pts_excluded_Data_Field
    results['distribution_histogram'] = hist_dist

    return results
