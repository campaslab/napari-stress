# -*- coding: utf-8 -*-
import numpy as np
import pytest


@pytest.fixture
def ellipsoid_points():
    """
    Generates points on the surface of an ellipsoid.
    """
    a, b, c = 5, 3, 2  # Semi-axes of the ellipsoid
    u = np.linspace(0, 2 * np.pi, 30)  # Longitude-like angles
    v = np.linspace(0, np.pi, 30)  # Latitude-like angles
    u, v = np.meshgrid(u, v)
    x = a * np.cos(u) * np.sin(v)
    y = b * np.sin(u) * np.sin(v)
    z = c * np.cos(v)
    return np.column_stack((x.ravel(), y.ravel(), z.ravel()))


@pytest.fixture
def droplet_points():
    """
    Create a droplet point cloud.
    """
    from napari_stress import sample_data

    pointcloud = sample_data.get_droplet_point_cloud()[0][0][:, 1:]

    return pointcloud


def test_spherical_harmonics_fit(droplet_points):
    """
    Test fitting spherical harmonics to ellipsoid points.
    """
    from napari_stress import approximation

    expander = approximation.SphericalHarmonicsExpander(
        max_degree=5, expansion_type="cartesian"
    )
    expander.fit(droplet_points)

    assert expander.coefficients_ is not None
    assert expander.coefficients_.shape == (3, (5 + 1), (5 + 1))


def test_spherical_harmonics_expand(droplet_points):
    """
    Test expanding points using spherical harmonics.
    """
    from napari_stress import approximation

    expander = approximation.SphericalHarmonicsExpander(
        max_degree=10, expansion_type="cartesian"
    )
    expander.fit(droplet_points)
    expanded_points = expander.expand(droplet_points)

    # Assert the expanded points are close to the original points
    np.testing.assert_allclose(expanded_points, droplet_points, rtol=0.01)

    expander_radial = approximation.SphericalHarmonicsExpander(
        max_degree=10, expansion_type="radial"
    )
    expander_radial.fit(droplet_points)
    expanded_points_radial = expander_radial.expand(droplet_points)

    # Assert the expanded points are close to the original points
    np.testing.assert_allclose(expanded_points_radial, droplet_points, rtol=0.01)


def generate_ellipsoidal_pointclouds(
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

    max_mean_curvatures = []
    min_mean_curvatures = []
    for i in range(n_tests):
        points = generate_ellipsoidal_pointclouds(a0=a0, a1=a1, a2=a2, x0=x0)

        expander = approximation.EllipsoidExpander()
        expander.fit(points)

        expanded_points = expander.expand(points)
        center = expander.center_
        axes_lengths = np.sort(expander.axes_)

        max_mean_curvatures.append(expander.properties["maximum_mean_curvature"])
        min_mean_curvatures.append(expander.properties["minimum_mean_curvature"])

        assert np.allclose(center, x0)
        assert np.allclose(a2, axes_lengths[2])
        assert np.allclose(a1, axes_lengths[1])
        assert np.allclose(a0, axes_lengths[0])

        assert np.allclose(expander.properties["residuals"].mean(), 0, atol=0.01)
        assert len(expanded_points) == len(points)

    # all elements in either list should be the same
    assert np.allclose(max_mean_curvatures, max_mean_curvatures[0])
    assert np.allclose(min_mean_curvatures, min_mean_curvatures[0])


def test_lsq_ellipsoid1():
    "Test whether pproperties in class are correctly set."
    from napari_stress import approximation

    expander = approximation.EllipsoidExpander()

    random_coefficients = np.random.random((3, 2, 3))
    expander.coefficients_ = random_coefficients

    assert np.array_equal(expander.center_, random_coefficients[0, 0])
    assert np.array_equal(
        expander.axes_, np.linalg.norm(random_coefficients[:, 1], axis=1)
    )


def test_lsq_ellipsoid():
    from napari_stress import approximation, get_droplet_point_cloud

    pointcloud = get_droplet_point_cloud()[0][0][:, 1:]
    ellipsoid = approximation.least_squares_ellipsoid(pointcloud)

    fitted_points = approximation.expand_points_on_ellipse(ellipsoid, pointcloud)

    assert fitted_points is not None


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
