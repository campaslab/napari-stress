# -*- coding: utf-8 -*-
import vedo
import numpy as np
import napari_stress

def test_frontend_spherical_harmonics():

    ellipse = vedo.shapes.Ellipsoid()

    # Test pyshtools implementation
    points1 = napari_stress.fit_spherical_harmonics(ellipse.points(), max_degree=3,
                                                   implementation='shtools')
    assert np.array_equal(ellipse.points().shape, points1[0].shape)

    # Test stress implementation
    points2 = napari_stress.fit_spherical_harmonics(ellipse.points(), max_degree=3,
                                                   implementation='stress')
    assert np.array_equal(ellipse.points().shape, points2[0].shape)

    # Test default implementations
    points = napari_stress.fit_spherical_harmonics(ellipse.points(), max_degree=3)[0]
    assert np.array_equal(ellipse.points().shape, points.shape)

def test_spherical_harmonics():
    from napari_stress._spherical_harmonics import spherical_harmonics as sh

    ellipse_points = vedo.shapes.Ellipsoid().points()

    pts, coeffs_pysh = sh.shtools_spherical_harmonics_expansion(ellipse_points)
    pts, coeffs_stress = sh.stress_spherical_harmonics_expansion(ellipse_points)

    lebedev_points, lebedev_info = sh.lebedev_quadrature(coeffs_stress)  # with pickle
    lebedev_points, lebedev_info = sh.lebedev_quadrature(coeffs_stress,
                                                         use_minimal_point_set=False)  # with pickle
    lebedev_points, lebedev_info = sh.lebedev_quadrature(coeffs_stress)  # without pickle

def test_interoperatibility():

    from napari_stress._spherical_harmonics import spherical_harmonics as sh
    from napari_stress._spherical_harmonics.sph_func_SPB import convert_coeffcient_matrix_to_pyshtools_format
    from pyshtools import SHCoeffs

    points = napari_stress.get_droplet_point_cloud()[0][0][:, 1:]

    pts_pysh, coeffs_pysh = sh.shtools_spherical_harmonics_expansion(points, max_degree=5)
    pts_stress, coeffs_stress = sh.stress_spherical_harmonics_expansion(points, max_degree=5)

    coeffs_x = SHCoeffs.from_array(convert_coeffcient_matrix_to_pyshtools_format(coeffs_stress[0]))
    coeffs_y = SHCoeffs.from_array(convert_coeffcient_matrix_to_pyshtools_format(coeffs_stress[1]))
    coeffs_z = SHCoeffs.from_array(convert_coeffcient_matrix_to_pyshtools_format(coeffs_stress[2]))


def test_quadrature(make_napari_viewer):
    points = napari_stress.get_droplet_point_cloud()[0]

    lebedev_points = napari_stress.measure_curvature(points[0])

    viewer = make_napari_viewer()
    lebedev_points = napari_stress.measure_curvature(points[0], viewer=viewer)
    lebedev_points = napari_stress.measure_curvature(points[0],
                                                    use_minimal_point_set=True,
                                                    number_of_quadrature_points=50)

if __name__ == '__main__':
    import napari
    test_interoperatibility()
