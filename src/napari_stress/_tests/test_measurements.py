# -*- coding: utf-8 -*-
import numpy as np

def test_comprehenive_stress_toolbox(make_napari_viewer):
    from napari_stress import (get_droplet_point_cloud, measurements)

    viewer = make_napari_viewer()
    pointcloud = get_droplet_point_cloud()[0]
    viewer.add_points(pointcloud[0][:, 1:], **pointcloud[1])

    widget = measurements.stress_analysis_toolbox(viewer)
    viewer.window.add_dock_widget(widget)

    widget._run()

def test_comprehensive_stress_toolbox_4d(make_napari_viewer):
    from napari_stress import (get_droplet_point_cloud_4d, measurements)

    viewer = make_napari_viewer()
    pointcloud = get_droplet_point_cloud_4d()[0]
    viewer.add_points(pointcloud[0], **pointcloud[1])

    widget = measurements.stress_analysis_toolbox(viewer)
    viewer.window.add_dock_widget(widget)

    widget._run()


def test_curvature(make_napari_viewer):
    from napari_stress._spherical_harmonics.spherical_harmonics_napari import perform_lebedev_quadrature
    from napari_stress import measurements, get_droplet_point_cloud, fit_spherical_harmonics, types
    from magicgui import magicgui
    from napari.layers import Layer

    # We'll first create a spherical harmonics expansion and a lebedev
    # quadrature from scratch
    viewer = make_napari_viewer()

    pointcloud = get_droplet_point_cloud()[0]
    viewer.add_points(pointcloud[0][:, 1:], **pointcloud[1])

    expansion = fit_spherical_harmonics(viewer.layers[-1].data)
    viewer.add_points(expansion[0], **expansion[1])

    lebedev_points = perform_lebedev_quadrature(viewer.layers[-1], viewer=viewer)
    l = Layer.create(lebedev_points[0], lebedev_points[1], lebedev_points[2])
    viewer.add_layer(l)
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
    quadrature_points, lbdv_info = lebedev_quadrature(coefficients, number_of_quadrature_points=200,
                                                      use_minimal_point_set=False)
    manifold = create_manifold(quadrature_points, lbdv_info, max_degree=max_degree)
    H_i, _, H0 = measurements.calculate_mean_curvature_on_manifold(manifold)

    # do ellipsoidal expansion
    ellipsoid = approximation.least_squares_ellipsoid(pointcloud)
    ellipsoid_points = approximation.expand_points_on_ellipse(ellipsoid, pointcloud)
    curvature_ellipsoid = measurements.curvature_on_ellipsoid(ellipsoid, ellipsoid_points)

    _, ellipsoid_coefficients = stress_spherical_harmonics_expansion(ellipsoid_points, max_degree=max_degree)
    ellipsoid_quadrature_points, lbdv_info = lebedev_quadrature(ellipsoid_coefficients, number_of_quadrature_points=200,
                                                      use_minimal_point_set=False)
    ellipsoid_manifold = create_manifold(ellipsoid_quadrature_points, lbdv_info, max_degree=max_degree)
    H_i_ellipsoid, _, H0_ellipsoid = measurements.calculate_mean_curvature_on_manifold(ellipsoid_manifold)

    # tissue stress tensor
    ellptical, cartesian = measurements.tissue_stress_tensor(
        ellipsoid,
        H0_ellipsoid,
        gamma=gamma)

    stress, stress_tissue, stress_cell = measurements.anisotropic_stress(
        H_i, H0, H_i_ellipsoid, H0_ellipsoid, gamma)

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
