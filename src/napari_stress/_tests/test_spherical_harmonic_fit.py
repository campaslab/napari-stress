# -*- coding: utf-8 -*-
import vedo
import numpy as np
import napari_stress

def test_spherical_harmonics():

    ellipse = vedo.shapes.Ellipsoid()

    # Test pyshtools implementation
    points = napari_stress.fit_spherical_harmonics(ellipse.points(), max_degree=3,
                                                   implementation='shtools')
    assert np.array_equal(ellipse.points().shape, points[:, 1:].shape)

    # Test stress implementation
    points = napari_stress.fit_spherical_harmonics(ellipse.points(), max_degree=3,
                                                   implementation='stress')
    assert np.array_equal(ellipse.points().shape, points[:, 1:].shape)

    # Test default implementations
    points = napari_stress.fit_spherical_harmonics(ellipse.points(), max_degree=3)
    assert np.array_equal(ellipse.points().shape, points[:, 1:].shape)

def test_quadrature(make_napari_viewer):
    points = napari_stress.get_dropplet_point_cloud()[0]
    
    lebedev_points = napari_stress.measure_curvature(points[0])
    
    viewer = make_napari_viewer()
    viewer.add_points(points[0], **points[1])
    viewer.add_points(lebedev_points, size=0.5, face_color='cyan')
    
if __name__ == '__main__':
    import napari
    test_quadrature(napari.Viewer)