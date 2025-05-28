import numpy as np
import pytest


@pytest.fixture
def generate_pointclouds():
    """
    Generates synthetic point clouds for testing.
    """

    def ellipsoid_points(a=5, b=3, c=2):
        u = np.linspace(0, 2 * np.pi, 30)  # Longitude-like angles
        v = np.linspace(0, np.pi, 30)  # Latitude-like angles
        u, v = np.meshgrid(u, v)
        x = a * np.cos(u) * np.sin(v)
        y = b * np.sin(u) * np.sin(v)
        z = c * np.cos(v)
        return np.column_stack((x.ravel(), y.ravel(), z.ravel()))

    def droplet_points():
        from napari_stress import sample_data

        return sample_data.get_droplet_point_cloud()[0][0][:, 1:]

    def random_ellipsoid_points(a0=10, a1=20, a2=30, x0=(0, 0, 0)):
        import vedo

        ellipsoid = vedo.Ellipsoid(
            pos=x0,
            axis1=(a0, 0, 0),
            axis2=(0, a1, 0),
            axis3=(0, 0, a2),
        )
        points = ellipsoid.vertices
        # Random rotations
        angle_z = np.radians(np.random.randint(0, 360))
        angle_x = np.radians(np.random.randint(0, 180))
        R_z = np.array(
            [
                [np.cos(angle_z), -np.sin(angle_z), 0],
                [np.sin(angle_z), np.cos(angle_z), 0],
                [0, 0, 1],
            ]
        )
        R_x = np.array(
            [
                [1, 0, 0],
                [0, np.cos(angle_x), -np.sin(angle_x)],
                [0, np.sin(angle_x), np.cos(angle_x)],
            ]
        )
        return points @ R_z @ R_x

    return ellipsoid_points, droplet_points, random_ellipsoid_points


def test_spherical_harmonics(generate_pointclouds):
    """
    Test spherical harmonics fitting and expansion.
    """
    from napari_stress import approximation

    _, droplet_points, _ = generate_pointclouds
    points = droplet_points()
    degree = 20

    # Test fitting
    expander = approximation.SphericalHarmonicsExpander(
        max_degree=degree, expansion_type="cartesian"
    )
    expander.fit(points)
    assert expander.coefficients_ is not None
    assert expander.coefficients_.shape == (3, (degree + 1), (degree + 1))

    # Test expansion
    expanded_points = expander.expand(points)
    np.testing.assert_allclose(expanded_points, points, rtol=0.01)

    # Test radial expansion
    expander_radial = approximation.SphericalHarmonicsExpander(
        max_degree=10, expansion_type="radial"
    )
    expander_radial.fit(points)
    expanded_points_radial = expander_radial.expand(points)
    np.testing.assert_allclose(expanded_points_radial, points, rtol=0.01)


def test_ellipsoid_fitting(generate_pointclouds):
    """
    Test ellipsoid fitting and properties.
    """
    from napari_stress import approximation

    _, _, random_ellipsoid_points = generate_pointclouds
    a0, a1, a2 = 10, 20, 30
    x0 = np.asarray([0, 0, 0])

    points = random_ellipsoid_points(a0, a1, a2, x0)

    # Test fitting
    expander = approximation.EllipsoidExpander()
    expander.fit(points)
    expanded_points = expander.expand(points)
    center = expander.center_
    axes_lengths = np.sort(expander.axes_)

    # Validate properties
    assert np.allclose(center, x0)
    assert np.allclose(a2, axes_lengths[2])
    assert np.allclose(a1, axes_lengths[1])
    assert np.allclose(a0, axes_lengths[0])
    assert np.allclose(expander.properties["residuals"].mean(), 0, atol=0.01)
    assert len(expanded_points) == len(points)

    # Validate mean curvature
    max_curvature = expander.properties["maximum_mean_curvature"]
    min_curvature = expander.properties["minimum_mean_curvature"]
    axes_sorted = np.sort(expander.axes_)
    a, b, c = axes_sorted[2], axes_sorted[1], axes_sorted[0]
    h_max_theory = a / (2 * c**2) + a / (2 * b**2)
    h_min_theory = c / (2 * b**2) + c / (2 * a**2)
    assert np.allclose(max_curvature, h_max_theory)
    assert np.allclose(min_curvature, h_min_theory)
    assert max_curvature > min_curvature


def test_curvature_and_normals(generate_pointclouds, make_napari_viewer):
    """
    Test curvature and normals on ellipsoid.
    """
    from napari_stress import approximation, measurements, types

    _, droplet_points, _ = generate_pointclouds
    points = droplet_points()

    # Fit ellipsoid
    ellipsoid = approximation.least_squares_ellipsoid(points)
    fitted_points = approximation.expand_points_on_ellipse(ellipsoid, points)

    # Test curvature
    data, features, metadata = measurements.curvature_on_ellipsoid(
        ellipsoid, fitted_points
    )
    assert data is not None
    assert features is not None
    assert metadata is not None

    # Test normals
    normals = approximation.normals_on_ellipsoid(points)
    assert normals is not None

    # Validate metadata keys
    viewer = make_napari_viewer()
    viewer.add_points(fitted_points)
    viewer.add_vectors(ellipsoid)
    results = measurements.curvature_on_ellipsoid(ellipsoid, fitted_points)
    assert types._METADATAKEY_H_E123_ELLIPSOID in results[1]["metadata"]
    assert types._METADATAKEY_H0_ELLIPSOID in results[1]["metadata"]
    assert types._METADATAKEY_MEAN_CURVATURE in results[1]["features"]
