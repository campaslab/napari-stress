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



if __name__ == '__main__':
    import napari
    test_curvature(napari.Viewer)
