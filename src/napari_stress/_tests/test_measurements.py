import os
import shutil

import numpy as np
import vedo

from napari_stress import (
    approximation,
    get_droplet_point_cloud,
    get_droplet_point_cloud_4d,
    measurements,
    reconstruction,
    sample_data,
)


def cartesian_to_spherical(cartesian_coords):
    """
    Convert Cartesian coordinates (z, y, x) to spherical coordinates (r, theta, phi).

    Parameters:
    cartesian_coords (np.ndarray): Nx3 array of Cartesian coordinates.

    Returns:
    np.ndarray: Nx3 array of spherical coordinates.
    """
    z, y, x = (
        cartesian_coords[:, 0],
        cartesian_coords[:, 1],
        cartesian_coords[:, 2],
    )
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)  # Azimuthal angle
    phi = np.arccos(z / r)  # Polar angle
    return np.column_stack((r, theta, phi))


def spherical_to_cartesian(spherical_coords):
    """
    Convert spherical coordinates (r, theta, phi) to Cartesian coordinates (z, y, x).

    Parameters:
    spherical_coords (np.ndarray): Nx3 array of spherical coordinates.

    Returns:
    np.ndarray: Nx3 array of Cartesian coordinates.
    """
    r, theta, phi = (
        spherical_coords[:, 0],
        spherical_coords[:, 1],
        spherical_coords[:, 2],
    )
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.column_stack((z, y, x))


def theoretical_mean_curvature_on_pointcloud(a, b, c, pointcloud) -> float:
    """Theoretical mean curvature of an ellipsoid.

    Parameters
    ----------
    a : float
        Length of the first semi-axis.
    b : float
        Length of the second semi-axis.
    c : float
        Length of the third semi-axis.
    elevation : float
        Angular position of sample on ellipsoid surface
    azimuth : float
        Equatorial position of sample on ellipsoid surface
    """
    import vedo

    pointcloud = pointcloud - pointcloud.mean(axis=0)[None, :]

    pointcloud_spherical = vedo.transformations.cart2spher(
        pointcloud[:, 2], pointcloud[:, 1], pointcloud[:, 0]
    )
    elevation = pointcloud_spherical[1]
    azimuth = pointcloud_spherical[2]

    above = (
        a
        * b
        * c
        * (
            3 * (a**2 + b**2)
            + 2 * c**2
            + (a**2 + b**2 - 2 * c**2) * np.cos(2 * elevation)
            - 2 * (a**2 - b**2) * np.cos(2 * azimuth) * np.sin(elevation) ** 2
        )
    )
    below = 8 * (
        a**2 * b**2 * np.cos(elevation) ** 2
        + c**2
        * (b**2 * np.cos(azimuth) ** 2 + a**2 * np.sin(azimuth) ** 2)
        * np.sin(elevation) ** 2
    ) ** (3 / 2)
    return above / below


def project_point_on_ellipse_surface(
    query_point: np.ndarray, a: float, b: float, c: float, center: np.ndarray
) -> np.ndarray:
    """Project a point on the surface of an ellipsoid.

    Parameters
    ----------
    query_point : np.ndarray
        Point to project on the surface of the ellipsoid.
    a : float
        Length of the first axis of the ellipsoid.
    b : float
        Length of the second axis of the ellipsoid.
    c : float
        Length of the third axis of the ellipsoid.
    center : np.ndarray
        Center of the ellipsoid.

    Returns
    -------
    np.ndarray
        Point on the surface of the ellipsoid.
    """
    # transformation matrix to turn ellipsoid into a sphere
    T = np.array([[1 / a, 0, 0], [0, 1 / b, 0], [0, 0, 1 / c]])

    # transform query point to coordinate system in which ellipsoid is a sphere
    transformed_query_point = np.dot(query_point - center, T)

    # transform query point into spherical coordinates
    transformed_query_point_spherical = cartesian_to_spherical(
        transformed_query_point
    )

    # replace radius of query point with radius of sphere, which in this coordinate
    # system is always 0.5
    point_on_transformed_ellipse_surface = np.stack(
        [
            np.ones(transformed_query_point_spherical.shape[0]) * 0.5,
            transformed_query_point_spherical[:, 1],
            transformed_query_point_spherical[:, 2],
        ]
    ).T

    # transform point on sphere back to cartesian coordinates
    point_on_transformed_ellipse_surface = spherical_to_cartesian(
        point_on_transformed_ellipse_surface
    )
    point_on_ellipse_surface = (
        np.dot(point_on_transformed_ellipse_surface, np.linalg.inv(T)) + center
    )

    return point_on_ellipse_surface


def test_curvature():
    """Test curvature-related functionality."""
    # Test patch-fitted curvature
    sphere = vedo.Sphere()
    sphere = (sphere.vertices, np.asarray(sphere.cells))
    result = (
        measurements.curvature._calculate_patch_fitted_curvature_on_surface(
            sphere, search_radius=0.25
        )
    )
    features = result[1]["metadata"]["features"]
    assert "mean_curvature" in features
    assert np.allclose(features["mean_curvature"].mean(), 1.0, atol=0.1)

    # Test mean curvature on ellipsoid
    size = 128
    b, c = 0.5, 0.5
    number_of_ellipsoids = 4
    for a in np.linspace(0.2, 0.8, number_of_ellipsoids):
        ellipsoid = sample_data.make_blurry_ellipsoid(
            a, b, c, size, definition_width=10
        )[0]
        results_reconstruction = reconstruction.reconstruct_droplet(
            ellipsoid,
            voxelsize=np.array([1, 1, 1]),
            use_dask=False,
            resampling_length=4.5,
            interpolation_method="linear",
            trace_length=10,
            sampling_distance=1,
            return_intermediate_results=True,
        )
        distances = np.linalg.norm(
            results_reconstruction[2][0]
            - project_point_on_ellipse_surface(
                results_reconstruction[2][0],
                a * size,
                b * size,
                c * size,
                np.array([size / 2, size / 2, size / 2]),
            ),
            axis=1,
        )
        assert np.mean(distances) < 1

    # Test curvature on ellipsoid
    pointcloud = get_droplet_point_cloud_4d()[0][0]
    expander = approximation.EllipsoidExpander()
    expander.fit(pointcloud[:, 1:])
    ellipsoid = expander.coefficients_
    approximated_pointcloud = expander.expand(pointcloud[:, 1:])
    curvature = measurements.curvature_on_ellipsoid(
        ellipsoid, approximated_pointcloud
    )
    assert curvature is not None


def test_stress_toolbox(make_napari_viewer):
    """Test stress analysis toolbox for 3D and 4D data."""
    from napari_stress import measurements

    viewer = make_napari_viewer()

    # Test 3D data
    pointcloud = get_droplet_point_cloud()[0]
    viewer.add_points(pointcloud[0][:, 1:], **pointcloud[1])
    widget = measurements.toolbox.stress_analysis_toolbox(viewer)
    widget.comboBox_quadpoints.setCurrentIndex(4)
    viewer.window.add_dock_widget(widget)
    widget._run()
    widget._export_settings(file_name="test.yaml")
    widget2 = measurements.toolbox.stress_analysis_toolbox(viewer)
    widget2._import_settings(file_name="test.yaml")
    assert widget2.comboBox_quadpoints.currentText() == "50"

    # Test 4D data
    pointcloud_4d = get_droplet_point_cloud_4d()[0]
    viewer.add_points(pointcloud_4d[0])
    widget._run()
    assert os.path.isdir(widget.save_directory)
    assert os.path.isfile(
        os.path.join(widget.save_directory, "raw_values", "stress_data.csv")
    )
    shutil.rmtree(widget.save_directory)


def test_distances():
    """Test geodesics, haversine, and k-nearest neighbors."""
    # Test haversine distances
    distance_matrix = measurements.haversine_distances(
        degree_lebedev=10, n_lebedev_points=434
    )
    assert np.allclose(distance_matrix.max(), np.pi / 2)

    # Test k-nearest neighbors
    points = np.random.rand(100, 3)
    df = measurements.distance_to_k_nearest_neighbors(points)
    assert df.shape[0] == points.shape[0]


def test_spatiotemporal_autocorrelation():
    from napari.types import SurfaceData

    from napari_stress import (
        TimelapseConverter,
        create_manifold,
        get_droplet_point_cloud,
        lebedev_quadrature,
        measurements,
        reconstruction,
    )
    from napari_stress._spherical_harmonics.spherical_harmonics import (
        stress_spherical_harmonics_expansion,
    )

    max_degree = 5
    # do sh expansion
    pointcloud = get_droplet_point_cloud()[0][0][:, 1:]
    _, coefficients = stress_spherical_harmonics_expansion(
        pointcloud, max_degree=max_degree, expansion_type="radial"
    )
    quadrature_points, lbdv_info = lebedev_quadrature(
        coefficients,
        number_of_quadrature_points=170,
        use_minimal_point_set=False,
    )
    manifold = create_manifold(
        quadrature_points, lbdv_info, max_degree=max_degree
    )
    quadrature_points = manifold.get_coordinates()
    surface = reconstruction.reconstruct_surface_from_quadrature_points(
        quadrature_points
    )

    Converter = TimelapseConverter()
    surfaces_4d = Converter.list_of_data_to_data(
        data=[surface, surface, surface], layertype=SurfaceData
    )

    # get distance matrix and divide by volume-integrated H0
    H0 = measurements.average_mean_curvatures_on_manifold(manifold)[2]
    distance_matrix = measurements.haversine_distances(max_degree, 170) / H0

    measurements.spatio_temporal_autocorrelation(
        surfaces=surfaces_4d, distance_matrix=distance_matrix
    )


def test_temporal_autocorrelation():
    import numpy as np
    import pandas as pd

    from napari_stress import measurements

    n_frames = 10
    n_measurements = 100

    np.random.seed(1)
    # create a set of features in multiple timepoints and add a bit more noise
    # in every frame - the autocorrelaiton should thus decrease monotonously
    frames = np.repeat(np.arange(n_frames), n_measurements)
    feature = np.ones(n_measurements)
    features = []
    features.append(feature)
    for i in range(1, n_frames):
        features.append(
            features[i - 1] + np.random.normal(size=n_measurements, scale=0.3)
        )

    df = pd.DataFrame(np.concatenate(features), columns=["feature"])
    df["frame"] = frames

    gradient = np.gradient(
        measurements.temporal_autocorrelation(df, feature="feature")
    )
    assert np.all(gradient < 0)
