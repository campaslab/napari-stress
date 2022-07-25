# -*- coding: utf-8 -*-

import vedo
import numpy as np
import napari_stress

def test_measurements(make_napari_viewer):
    from napari_stress._spherical_harmonics.spherical_harmonics_napari import perform_lebedev_quadrature

    # We'll first create a spherical harmonics expansion and a lebedev
    # quadrature from scratch
    viewer = make_napari_viewer()

    pointcloud = napari_stress.get_droplet_point_cloud()[0]
    viewer.add_points(pointcloud[0][:, 1:], **pointcloud[1])

    expansion = napari_stress.fit_spherical_harmonics(viewer.layers[-1].data)
    viewer.add_points(expansion[0], **expansion[1])

    lebedev_points = perform_lebedev_quadrature(viewer.layers[-1], viewer=viewer)
    results_layer = viewer.layers[-1]
    assert 'manifold' in results_layer.metadata

    # Now we can test the actual measurements
    result = napari_stress.measurements.calculate_mean_curvature_on_manifold(results_layer,
                                                                             viewer=viewer)

    features, metadata = napari_stress.measurements.calculate_mean_curvature_on_manifold(
        results_layer.metadata['manifold']
        )

    assert 'H0_arithmetic_average' in result.metadata
    assert 'H0_surface_integral' in result.metadata
    assert 'Mean_curvature_at_lebedev_points' in result.features
    assert 'H0_arithmetic_average' in metadata
    assert 'H0_surface_integral' in metadata
    assert 'Mean_curvature_at_lebedev_points' in features


if __name__ == '__main__':
    import napari
    test_measurements(napari.Viewer)
