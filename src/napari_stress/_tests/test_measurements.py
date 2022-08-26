# -*- coding: utf-8 -*-
import numpy as np


def test_curvature(make_napari_viewer):
    from napari_stress._spherical_harmonics.spherical_harmonics_napari import perform_lebedev_quadrature
    from napari_stress import measurements, get_droplet_point_cloud, fit_spherical_harmonics, types
    from magicgui import magicgui

    # We'll first create a spherical harmonics expansion and a lebedev
    # quadrature from scratch
    viewer = make_napari_viewer()

    pointcloud = get_droplet_point_cloud()[0]
    viewer.add_points(pointcloud[0][:, 1:], **pointcloud[1])

    expansion = fit_spherical_harmonics(viewer.layers[-1].data)
    viewer.add_points(expansion[0], **expansion[1])

    lebedev_points = perform_lebedev_quadrature(viewer.layers[-1], viewer=viewer)
    results_layer = viewer.layers[-1]
    assert types._METADATAKEY_MANIFOLD in results_layer.metadata

    # from code
    H, H0_arithmetic, H0_surface = measurements.calculate_mean_curvature_on_manifold(
        results_layer.metadata[types._METADATAKEY_MANIFOLD])
    H, H0_arithmetic, H0_surface = measurements.calculate_mean_curvature_on_manifold(
        results_layer)

    assert types._METADATAKEY_H0_ARITHMETIC in results_layer.metadata
    assert types._METADATAKEY_H0_SURFACE_INTEGRAL in results_layer.metadata
    assert types._METADATAKEY_MEAN_CURVATURE in results_layer.features

    # Test gauss-bonnet
    measurements.gauss_bonnet_test(results_layer)
    assert types._METADATAKEY_GAUSS_BONNET_ABS in results_layer.metadata
    assert types._METADATAKEY_GAUSS_BONNET_REL in results_layer.metadata

    absolute, relative = measurements.gauss_bonnet_test(results_layer,
                                                        viewer=viewer)
    assert types._METADATAKEY_GAUSS_BONNET_ABS in results_layer.metadata.keys()
    assert types._METADATAKEY_GAUSS_BONNET_REL in results_layer.metadata.keys()

def test_stresses():
    from napari_stress import lebedev_quadrature
    from napari_stress import (measurements, approximation,
                               get_droplet_point_cloud,
                               create_manifold, types)
    from napari_stress._spherical_harmonics.spherical_harmonics import stress_spherical_harmonics_expansion

    max_degree = 5
    gamma = 26.0

    # do sh expansion
    pointcloud = get_droplet_point_cloud()[0][0][:, 1:]
    _, coefficients = stress_spherical_harmonics_expansion(pointcloud, max_degree=max_degree)
    quadrature_points, lbdv_info = lebedev_quadrature(coefficients)
    manifold = create_manifold(quadrature_points, lbdv_info, max_degree=max_degree)
    H_i, _, H0 = measurements.calculate_mean_curvature_on_manifold(manifold)

    # do ellipsoidal expansion
    ellipsoid = approximation.least_squares_ellipsoid(pointcloud)
    ellipsoid_points = approximation.expand_points_on_ellipse(ellipsoid, pointcloud)
    curvature_ellipsoid = measurements.curvature_on_ellipsoid(ellipsoid, ellipsoid_points)

    _, ellipsoid_coefficients = stress_spherical_harmonics_expansion(ellipsoid_points, max_degree=max_degree)
    ellipsoid_quadrature_points, lbdv_info = lebedev_quadrature(ellipsoid_coefficients)
    ellipsoid_manifold = create_manifold(ellipsoid_quadrature_points, lbdv_info, max_degree=max_degree)
    H_i_ellipsoid, _, H0_ellipsoid = measurements.calculate_mean_curvature_on_manifold(ellipsoid_manifold)

    orientation_matrix = ellipsoid[:, 1].T
    orientation_matrix = orientation_matrix / np.linalg.norm(orientation_matrix, axis=0)
    orientation_matrix

    # tissue stress tensor
    ellptical, cartesian = measurements.tissue_stress_tensor(
        curvature_ellipsoid[1]['metadata'][types._METADATAKEY_H_E123_ELLIPSOID],
        H0_ellipsoid,
        orientation_matrix,
        gamma=gamma)

    measurements.anisotropic_stress(H_i, H0,
                                    H_i_ellipsoid, H0_ellipsoid,
                                    gamma)
    measurements.maximal_tissue_anisotropy(ellipsoid)


def test_compatibility_decorator():
    import inspect
    import numpy as np
    import napari_stress

    from napari_stress import types

    def my_function(manifold: napari_stress._stress.manifold_SPB.manifold, sigma: float = 1.0) -> (dict, dict):
        some_data = np.random.random((10,3))
        metadata = {'attribute': 1}
        metadata[types._METADATAKEY_MANIFOLD] = manifold
        features = {'attribute2': np.random.random(10)}
        return features, metadata

    function = napari_stress.measurements.utils.naparify_measurement(my_function)
    sig = inspect.signature(function)

    assert sig.parameters[types._METADATAKEY_MANIFOLD].annotation == 'napari.layers.Points'

# def test_compatibility_decorator2(make_napari_viewer):
#     import napari_stress
#     from napari_stress import measurements
#     from napari_stress._spherical_harmonics.spherical_harmonics_napari import perform_lebedev_quadrature

#     from napari_stress import types

#     viewer = make_napari_viewer()

#     pointcloud = napari_stress.get_droplet_point_cloud()[0]
#     viewer.add_points(pointcloud[0][:, 1:], **pointcloud[1])

#     expansion = napari_stress.fit_spherical_harmonics(viewer.layers[-1].data)
#     viewer.add_points(expansion[0], **expansion[1])

#     lebedev_points = perform_lebedev_quadrature(viewer.layers[-1], viewer=viewer)
#     results_layer = viewer.layers[-1]
#     assert types._METADATAKEY_MANIFOLD in results_layer.metadata

#     # pass layer to measurements function
#     measurements.calculate_mean_curvature_on_manifold(results_layer)
#     assert types._METADATAKEY_H0_ARITHMETIC in results_layer.metadata.keys()
#     assert types._METADATAKEY_H0_SURFACE_INTEGRAL in results_layer.metadata.keys()
#     assert types._METADATAKEY_MEAN_CURVATURE in results_layer.features.keys()
