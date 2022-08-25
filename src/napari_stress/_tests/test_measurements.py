# -*- coding: utf-8 -*-



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

#     # pass manifold to measurement function
#     _, features, metadata = measurements.calculate_mean_curvature_on_manifold(results_layer.metadata[types._METADATAKEY_MANIFOLD])
#     assert types._METADATAKEY_H0_ARITHMETIC in metadata.keys()
#     assert types._METADATAKEY_H0_SURFACE_INTEGRAL in metadata.keys()
#     assert types._METADATAKEY_MEAN_CURVATURE in features.keys()