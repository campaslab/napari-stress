# -*- coding: utf-8 -*-
import vedo
import numpy as np
import napari_stress

def test_spherical_harmonics():

    ellipse = vedo.shapes.Ellipsoid()

    # Test pyshtools implementation
    points = napari_stress.fit_spherical_harmonics(ellipse.points(), max_degree=3,
                                                   implementation='shtools')
    assert np.array_equal(ellipse.points().shape, points.shape)

    # Test stress implementation
    points = napari_stress.fit_spherical_harmonics(ellipse.points(), max_degree=3,
                                                   implementation='stress')
    assert np.array_equal(ellipse.points().shape, points.shape)

    # Test default implementations
    points = napari_stress.fit_spherical_harmonics(ellipse.points(), max_degree=3)
    assert np.array_equal(ellipse.points().shape, points.shape)

def test_quadrature(make_napari_viewer):
    points = napari_stress.get_dropplet_point_cloud()[0]

    lebedev_points = napari_stress.measure_curvature(points[0])

    viewer = make_napari_viewer()
    lebedev_points = napari_stress.measure_curvature(points[0], viewer=viewer)
    lebedev_points = napari_stress.measure_curvature(points[0],
                                                    use_minimal_point_set=True,
                                                    number_of_quadrature_points=50)
