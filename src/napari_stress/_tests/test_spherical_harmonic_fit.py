import vedo

import napari_stress


def test_front_spherical_harmonics_4d(make_napari_viewer):
    from napari_stress import get_droplet_point_cloud_4d
    from napari_stress._spherical_harmonics.spherical_harmonics_napari import (
        perform_lebedev_quadrature,
    )

    viewer = make_napari_viewer()
    pointcloud = get_droplet_point_cloud_4d()[0][0]
    pointcloud = pointcloud[
        pointcloud[:, 0] < 3, :
    ]  # take only first timepoints

    points = napari_stress.fit_spherical_harmonics(pointcloud)
    viewer.add_points(points[0], **points[1])

    # Test quadrature
    points_layer = viewer.layers[-1]
    lebedev_points = perform_lebedev_quadrature(points_layer, viewer=viewer)
    assert "manifold" in lebedev_points[1]["metadata"]


def test_spherical_harmonics():
    from napari_stress._spherical_harmonics import spherical_harmonics as sh

    ellipse_points = vedo.shapes.Ellipsoid().vertices

    pts, coeffs_stress = sh.stress_spherical_harmonics_expansion(
        ellipse_points, expansion_type="cartesian"
    )

    lebedev_points, lebedev_info = sh.lebedev_quadrature(
        coeffs_stress
    )  # with pickle
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
    assert pts.shape[1] == 3
    assert pts.shape[0] == len(ellipse_points)


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
