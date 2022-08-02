# -*- coding: utf-8 -*-

import vedo
import numpy as np

def test_curvature(make_napari_viewer):
    from napari_stress._spherical_harmonics.spherical_harmonics_napari import perform_lebedev_quadrature
    from napari_stress import measurements, get_droplet_point_cloud, fit_spherical_harmonics
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
    assert 'manifold' in results_layer.metadata

    # from code
    _, features, metadata = measurements.calculate_mean_curvature_on_manifold(
        results_layer.metadata['manifold']
        )

    assert 'H0_arithmetic_average' in metadata
    assert 'H0_surface_integral' in metadata
    assert 'Mean_curvature_at_lebedev_points' in features

    # from napari
    widget = magicgui(measurements.calculate_mean_curvature_on_manifold)
    viewer.window.add_dock_widget(widget)
    widget.manifold.value = results_layer
    widget()

    assert 'H0_arithmetic_average' in results_layer.metadata
    assert 'H0_surface_integral' in results_layer.metadata
    assert 'Mean_curvature_at_lebedev_points' in results_layer.features

    # Test gauss-bonnet
    measurements.gauss_bonnet_test(results_layer)
    assert 'Gauss_Bonnet_error' in results_layer.metadata
    assert 'Gauss_Bonnet_relative_error' in results_layer.metadata

    _, _, metadata = measurements.gauss_bonnet_test(results_layer.metadata['manifold'])
    assert 'Gauss_Bonnet_error' in metadata
    assert 'Gauss_Bonnet_relative_error' in metadata


def test_compatibility_decorator():
    import inspect
    import numpy as np
    import napari_stress

    def my_function(manifold: napari_stress._stress.manifold_SPB.manifold, sigma: float = 1.0) -> (dict, dict):
        some_data = np.random.random((10,3))
        metadata = {'attribute': 1}
        metadata['manifold'] = manifold
        features = {'attribute2': np.random.random(10)}
        return features, metadata

    function = napari_stress.measurements.utils.naparify_measurement(my_function)
    sig = inspect.signature(function)

    assert sig.parameters['manifold'].annotation == 'napari.layers.Points'

def test_compatibility_decorator2(make_napari_viewer):
    import napari_stress
    from napari_stress import measurements
    from napari_stress._spherical_harmonics.spherical_harmonics_napari import perform_lebedev_quadrature
    viewer = make_napari_viewer()

    pointcloud = napari_stress.get_droplet_point_cloud()[0]
    viewer.add_points(pointcloud[0][:, 1:], **pointcloud[1])

    expansion = napari_stress.fit_spherical_harmonics(viewer.layers[-1].data)
    viewer.add_points(expansion[0], **expansion[1])

    lebedev_points = perform_lebedev_quadrature(viewer.layers[-1], viewer=viewer)
    results_layer = viewer.layers[-1]
    assert 'manifold' in results_layer.metadata

    # pass layer to measurements function
    measurements.calculate_mean_curvature_on_manifold(results_layer)
    assert 'H0_arithmetic_average' in results_layer.metadata.keys()
    assert 'H0_surface_integral' in results_layer.metadata.keys()
    assert 'Mean_curvature_at_lebedev_points' in results_layer.features.keys()

    # pass manifold to measurement function
    _, features, metadata = measurements.calculate_mean_curvature_on_manifold(results_layer.metadata['manifold'])
    assert 'H0_arithmetic_average' in metadata.keys()
    assert 'H0_surface_integral' in metadata.keys()
    assert 'Mean_curvature_at_lebedev_points' in features.keys()
