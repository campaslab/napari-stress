# -*- coding: utf-8 -*-
import numpy as np

def test_spherical_harmonics():
    import napari_stress
    import vedo

    ellipse = vedo.shapes.Ellipsoid()
    points = napari_stress.fit_spherical_harmonics(ellipse.points(), max_degree=3)

    assert np.array_equal(ellipse.points().shape, points[:, 1:].shape)

def test_smoothing():
    from napari_stress import smoothMLS2D
    import vedo

    points = vedo.shapes.Sphere(res=30).points() * 10
    points += np.random.uniform(size=points.shape)

    smoothed_points = smoothMLS2D(points, factor=0.25, radius=0)
    smoothed_points = smoothMLS2D(points, factor=1.0, radius=4)

if __name__ == '__main__':
    test_smoothing()
