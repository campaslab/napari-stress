# -*- coding: utf-8 -*-
import vedo
import numpy as np


def test_fit():

    from napari_stress import spherical_harmonic_fit
    from napari_stress import TimelapseConverter
    from napari.types import PointsData

    # 3D case
    sphere = vedo.shapes.Sphere()
    points = sphere.points()

    fitted_points = spherical_harmonic_fit(points, fit_degree=1)
    errors = fitted_points[1]['features']['errors']
    assert np.array_equal(fitted_points[0].shape[0], points.shape[0])

    # Check that errors get smaller with higher fit degree
    for fit_degree in range(2, 10):
        fitted_points = spherical_harmonic_fit(points, fit_degree)

        assert np.mean(fitted_points[1]['features']['errors'] < errors)
        errors = fitted_points[1]['features']['errors']

    #4D case
    Converter = TimelapseConverter()
    points_list = [vedo.shapes.Sphere().points() * k for k in np.arange(1.9, 2.1, 0.1)]
    points_array = Converter.list_of_data_to_data(points_list, PointsData)

    fitted_points = spherical_harmonic_fit(points_array, fit_degree=3)
    assert np.array_equal(fitted_points[0].shape, points_array.shape)


if __name__ == '__main__':
    test_fit()
