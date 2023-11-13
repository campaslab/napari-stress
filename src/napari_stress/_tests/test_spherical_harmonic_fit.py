# -*- coding: utf-8 -*-
import vedo
import numpy as np
import napari_stress


def test_frontend_spherical_harmonics(make_napari_viewer):
    from napari_stress._spherical_harmonics.spherical_harmonics_napari import (
        perform_lebedev_quadrature,
    )

    ellipse = vedo.shapes.Ellipsoid()

    # Test pyshtools implementation
    points1 = napari_stress.fit_spherical_harmonics(
        ellipse.points(), max_degree=10, implementation="shtools"
    )
    assert np.array_equal(ellipse.points().shape, points1[0].shape)

    # Test stress implementation
    points2 = napari_stress.fit_spherical_harmonics(
        ellipse.points(), max_degree=3, implementation="stress"
    )
    assert np.array_equal(ellipse.points().shape, points2[0].shape)

    # Test default implementations
    points = napari_stress.fit_spherical_harmonics(ellipse.points(), max_degree=3)
    assert np.array_equal(ellipse.points().shape, points[0].shape)

    # Test implementation with viewer
    viewer = make_napari_viewer()
    points_layer = viewer.add_points(points[0], **points[1])
    assert "spherical_harmonics_coefficients" in points_layer.metadata

    # Test quadrature
    lebedev_points = perform_lebedev_quadrature(points_layer, viewer=viewer)
    results_layer = viewer.layers[-1]
    assert "manifold" in lebedev_points[1]["metadata"]
    assert "spherical_harmonics_coefficients" in results_layer.metadata


def test_front_spherical_harmonics_4d(make_napari_viewer):
    from napari_stress._spherical_harmonics.spherical_harmonics_napari import (
        perform_lebedev_quadrature,
    )
    from napari_stress import get_droplet_point_cloud_4d

    viewer = make_napari_viewer()
    pointcloud = get_droplet_point_cloud_4d()[0]

    points = napari_stress.fit_spherical_harmonics(pointcloud[0])
    viewer.add_points(points[0], **points[1])

    # Test quadrature
    points_layer = viewer.layers[-1]
    lebedev_points = perform_lebedev_quadrature(points_layer, viewer=viewer)
    assert "manifold" in lebedev_points[1]["metadata"]


def test_spherical_harmonics():
    from napari_stress._spherical_harmonics import spherical_harmonics as sh

    ellipse_points = vedo.shapes.Ellipsoid().points()

    # test cartesian expansion
    pts, coeffs_pysh = sh.shtools_spherical_harmonics_expansion(
        ellipse_points, expansion_type="cartesian"
    )
    pts, coeffs_stress = sh.stress_spherical_harmonics_expansion(
        ellipse_points, expansion_type="cartesian"
    )

    lebedev_points, lebedev_info = sh.lebedev_quadrature(coeffs_stress)  # with pickle
    lebedev_points, lebedev_info = sh.lebedev_quadrature(
        coeffs_stress, use_minimal_point_set=False
    )  # with pickle
    lebedev_points, lebedev_info = sh.lebedev_quadrature(
        coeffs_stress
    )  # without pickle

    # test radial expansion
    pts, coeffs_stress = sh.stress_spherical_harmonics_expansion(
        ellipse_points, expansion_type="radial"
    )
    pts, coeffs_stress = sh.shtools_spherical_harmonics_expansion(
        ellipse_points, expansion_type="radial"
    )
    assert pts.shape[1] == 3
    assert pts.shape[0] == len(ellipse_points)


def test_interoperatibility():
    from napari_stress._spherical_harmonics import spherical_harmonics as sh
    from napari_stress._stress.sph_func_SPB import (
        convert_coeffcients_stress_to_pyshtools,
        convert_coefficients_pyshtools_to_stress,
    )

    deg = 10
    # get stress expansion of pointcloud
    points = napari_stress.get_droplet_point_cloud()[0][0][:, 1:]
    pts_stress, coeffs_stress = sh.stress_spherical_harmonics_expansion(
        points, max_degree=deg
    )

    # convert the x/y/z expansions to pysh format
    coeffs_pysh_x = convert_coeffcients_stress_to_pyshtools(coeffs_stress[0])
    coeffs_pysh_y = convert_coeffcients_stress_to_pyshtools(coeffs_stress[1])
    coeffs_pysh_z = convert_coeffcients_stress_to_pyshtools(coeffs_stress[2])

    # convert coeffs back to stress format
    coeffs_str_x = convert_coefficients_pyshtools_to_stress(coeffs_pysh_x)
    coeffs_str_y = convert_coefficients_pyshtools_to_stress(coeffs_pysh_y)
    coeffs_str_z = convert_coefficients_pyshtools_to_stress(coeffs_pysh_z)

    # check if coeffs are still the same
    _coeffs_stress = np.stack([coeffs_str_x, coeffs_str_y, coeffs_str_z])
    assert all((_coeffs_stress - coeffs_stress).flatten() == 0)


def test_lebedev_points():
    from napari_stress._stress.lebedev_write_SPB import LebFunc

    for i in [
        6,
        14,
        26,
        38,
        50,
        74,
        86,
        110,
        146,
        170,
        194,
        230,
        266,
        302,
        350,
        434,
        590,
        770,
        974,
        1202,
        1454,
        1730,
        2030,
        2354,
        2702,
        3074,
        3470,
        3890,
        4334,
        4802,
        5294,
        5810,
    ]:
        print("    %d : [" % i)
        lf = LebFunc[i]()

        assert lf is not None
