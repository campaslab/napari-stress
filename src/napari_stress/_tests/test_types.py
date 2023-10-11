# -*- coding: utf-8 -*-
from napari import layers
import numpy as np
import magicgui

def test_custom_types(make_napari_viewer):
    from napari_stress.types import manifold, _METADATAKEY_MEAN_CURVATURE
    from napari_stress import (measurements, get_droplet_point_cloud,
                               fit_spherical_harmonics)
    from napari_stress._spherical_harmonics.spherical_harmonics_napari import perform_lebedev_quadrature

    def test_function(argument: manifold) -> manifold:

        layer = None
        if isinstance(argument, layers.Layer):
            layer = argument
            argument = argument.metadata[manifold.__name__]

        results = measurements.calculate_mean_curvature_on_manifold(argument)
        if layer is not None:
            layer.features[_METADATAKEY_MEAN_CURVATURE] = results[0]
        return results

    # create ayer with manifold
    viewer = make_napari_viewer()

    pointcloud = get_droplet_point_cloud()[0]
    viewer.add_points(pointcloud[0][:, 1:], **pointcloud[1])

    expansion = fit_spherical_harmonics(viewer.layers[-1].data)
    viewer.add_points(expansion[0], **expansion[1])

    lebedev_points = perform_lebedev_quadrature(viewer.layers[-1], viewer=viewer)
    l = layers.Layer.create(lebedev_points[0], lebedev_points[1], lebedev_points[2])
    viewer.add_layer(l)

    widget = magicgui.magicgui(test_function)
    viewer.window.add_dock_widget(widget)
    results = widget(viewer.layers[-1])

    assert _METADATAKEY_MEAN_CURVATURE in viewer.layers[-1].features.keys()


if __name__ == '__main__':
    import napari
    test_custom_types(napari.Viewer)
