# -*- coding: utf-8 -*-
import vedo
import numpy as np


def test_fit():

    from napari_stress import spherical_harmonic_fit
    sphere = vedo.shapes.Sphere()
    points = sphere.points()

    fitted_points = spherical_harmonic_fit(points, fit_degree=1)
    errors = fitted_points[1]['features']['errors']
    assert np.array_equal(fitted_points[0].shape, points.shape)

    for fit_degree in range(2, 10):
        fitted_points = spherical_harmonic_fit(points, fit_degree)

        assert np.mean(fitted_points[1]['features']['errors'] < errors)
        errors = fitted_points[1]['features']['errors']

if __name__ == '__main__':
    test_fit()
