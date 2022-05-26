# -*- coding: utf-8 -*-
import numpy as np
import napari_stress

def test_spherical_harmonics():

    import vedo

    ellipse = vedo.shapes.Ellipsoid()
    points = napari_stress.fit_spherical_harmonics(ellipse.points(), max_degree=3)

    assert np.array_equal(ellipse.points().shape, points[:, 1:].shape)

def test_reconstruction():
    import vedo
    points = vedo.shapes.Ellipsoid().points() * 100

    surface = napari_stress.reconstruct_surface(points)

if __name__ == '__main__':
    test_reconstruction()
