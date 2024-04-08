# -*- coding: utf-8 -*-
import numpy as np


def generate_pointclouds(
    a0: float = 10, a1: float = 20, a2: float = 30, x0: tuple = (0, 0, 0)
):
    import vedo

    ellipsoid = vedo.Ellipsoid(
        pos=x0,
        axis1=(a0, 0, 0),
        axis2=(0, a1, 0),
        axis3=(0, 0, a2),
    )

    points = ellipsoid.vertices

    # rotate all points around the z-axis by random angle between
    # 0 and 360 degrees using only numpy
    angle = np.radians(np.random.randint(0, 360))
    c, s = np.cos(angle), np.sin(angle)
    R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
    points = points @ R

    # rotate also around x-axis by random angle between
    # 0 and 180 degrees using only numpy
    angle = np.radians(np.random.randint(0, 180))
    c, s = np.cos(angle), np.sin(angle)
    R = np.array(((1, 0, 0), (0, c, -s), (0, s, c)))
    points = points @ R

    return points


def test_lsq_ellipsoid0(n_tests=10):
    from napari_stress import approximation

    a0 = 10
    a1 = 20
    a2 = 30
    x0 = np.asarray([0, 0, 0])

    for i in range(n_tests):
        points = generate_pointclouds(a0=a0, a1=a1, a2=a2, x0=x0)

        expander = approximation.EllipsoidExpander()
        expander.fit(points)

        expanded_points = expander.expand(points)
        center = expander.center_
        axes_lengths = np.sort(expander.axes_)

        assert np.allclose(center, x0)
        assert np.allclose(a2, axes_lengths[2])
        assert np.allclose(a1, axes_lengths[1])
        assert np.allclose(a0, axes_lengths[0])

        assert np.allclose(expander.properties.mean(), 0, atol=0.01)
        assert len(expanded_points) == len(points)


def test_lsq_ellipsoid():
    from napari_stress import approximation, get_droplet_point_cloud

    pointcloud = get_droplet_point_cloud()[0][0][:, 1:]
    ellipsoid = approximation.least_squares_ellipsoid(pointcloud)

    fitted_points = approximation.expand_points_on_ellipse(ellipsoid, pointcloud)

    assert fitted_points is not None


def test_lsq_ellipsoid2():
    import vedo
    from napari_stress import approximation
    from napari_stress._utils.coordinate_conversion import (
        _center_from_ellipsoid,
        _axes_lengths_from_ellipsoid,
    )

    from scipy.spatial.transform import Rotation as R

    center = np.zeros((3, 3))
    l1 = 1
    l2 = 2
    l3 = 3
    a1 = np.array([l1, 0, 0])
    a2 = np.array([0, l2, 0])
    a3 = np.array([0, 0, l3])

    points = vedo.Ellipsoid().points()

    fitted_ellipsoid = approximation.least_squares_ellipsoid(points)
    x0 = _center_from_ellipsoid(fitted_ellipsoid)
    lengths = _axes_lengths_from_ellipsoid(fitted_ellipsoid)

    assert np.allclose(center, x0, atol=0.01)
    assert np.allclose(a1.max() / 2, lengths[0])
    assert np.allclose(a2.max() / 2, lengths[1])
    assert np.allclose(a3.max() / 2, lengths[2])

    # rotate axis and check if the result still holds
    rotation_matrix = R.from_euler("z", 90, degrees=True).as_matrix()
    rotated_points = np.dot(points, rotation_matrix)

    fitted_ellipsoid = approximation.least_squares_ellipsoid(rotated_points)
    x0 = _center_from_ellipsoid(fitted_ellipsoid)
    lengths = _axes_lengths_from_ellipsoid(fitted_ellipsoid)

    assert np.allclose(center, x0, atol=0.01)
    assert np.allclose(l2 / 2, lengths[0])
    assert np.allclose(l1 / 2, lengths[1])
    assert np.allclose(l3 / 2, lengths[2])

    rotation_matrix = R.from_euler("xz", [45, 45], degrees=True).as_matrix()
    rotated_points = np.dot(points, rotation_matrix)

    fitted_ellipsoid = approximation.least_squares_ellipsoid(rotated_points)
    expanded_points = approximation.expand_points_on_ellipse(
        fitted_ellipsoid, rotated_points
    )

    assert np.allclose(rotated_points, expanded_points)


def test_ellipse_normals():
    from napari_stress import approximation, get_droplet_point_cloud

    pointcloud = get_droplet_point_cloud()[0][0][:, 1:]

    normals = approximation.normals_on_ellipsoid(pointcloud)

    assert normals is not None


def test_curvature_on_ellipsoid(make_napari_viewer):
    from napari_stress import (
        approximation,
        measurements,
        types,
        get_droplet_point_cloud,
    )

    pointcloud = get_droplet_point_cloud()[0][0][:, 1:]
    ellipsoid_stress = approximation.least_squares_ellipsoid(pointcloud)
    fitted_points_stress = approximation.expand_points_on_ellipse(
        ellipsoid_stress, pointcloud
    )
    data, features, metadata = measurements.curvature_on_ellipsoid(
        ellipsoid_stress, fitted_points_stress
    )

    assert data is not None
    assert features is not None
    assert metadata is not None

    viewer = make_napari_viewer()
    viewer.add_points(fitted_points_stress)
    viewer.add_vectors(ellipsoid_stress)
    results = measurements.curvature_on_ellipsoid(
        ellipsoid_stress, fitted_points_stress
    )

    assert types._METADATAKEY_H_E123_ELLIPSOID in results[1]["metadata"].keys()
    assert types._METADATAKEY_H0_ELLIPSOID in results[1]["metadata"].keys()
    assert types._METADATAKEY_MEAN_CURVATURE in results[1]["features"].keys()
