# -*- coding: utf-8 -*-
import numpy as np

def test_spherical_harmonics():
    import napari_stress
    import vedo

    ellipse = vedo.shapes.Ellipsoid()
    points = napari_stress.fit_spherical_harmonics(ellipse.points(), max_degree=3)

    assert np.array_equal(ellipse.points().shape, points[:, 1:].shape)
