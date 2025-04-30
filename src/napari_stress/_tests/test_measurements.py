import numpy as np
import os
import shutil
from pathlib import Path
import pandas as pd
import vedo
from napari_stress import (
    measurements,
    reconstruction,
    sample_data,
    approximation,
    get_droplet_point_cloud,
    get_droplet_point_cloud_4d,
    create_manifold,
    lebedev_quadrature,
)
from napari_stress._spherical_harmonics.spherical_harmonics import (
    stress_spherical_harmonics_expansion,
)


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
    import vedo

    # transformation matrix to turn ellipsoid into a sphere
    T = np.array([[1 / a, 0, 0], [0, 1 / b, 0], [0, 0, 1 / c]])

    # transform query point to coordinate system in which ellipsoid is a sphere
    transformed_query_point = np.dot(query_point - center, T)

    # transform query point into spherical coordinates
    transformed_query_point_spherical = vedo.transformations.cart2spher(
        transformed_query_point[2],
        transformed_query_point[1],
        transformed_query_point[0],
    )

    # replace radius of query point with radius of sphere, which in this coordinate
    # system is always 0.5
    point_on_transformed_ellipse_surface = np.array(
        [
            0.5,
            transformed_query_point_spherical[1],
            transformed_query_point_spherical[2],
        ]
    )

    # transform point on sphere back to cartesian coordinates
    point_on_transformed_ellipse_surface = vedo.transformations.spher2cart(
        *point_on_transformed_ellipse_surface
    )[::-1]
    point_on_ellipse_surface = (
        np.dot(point_on_transformed_ellipse_surface, np.linalg.inv(T)) + center
    )

    return point_on_ellipse_surface


def test_curvature():
    """Test curvature-related functionality."""
    # Test patch-fitted curvature
    sphere = vedo.Sphere()
    sphere = (sphere.vertices, np.asarray(sphere.cells))
    result = measurements.curvature._calculate_patch_fitted_curvature_on_surface(
        sphere, search_radius=0.25
    )
    features = result[1]["metadata"]["features"]
    assert "mean_curvature" in features.keys()
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
            resampling_length=3.5,
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
    widget = measurements.toolbox.stress_analysis_toolbox(
        viewer
    )
    widget.comboBox_quadpoints.setCurrentIndex(4)
    viewer.window.add_dock_widget(widget)
    widget._run()
    widget._export_settings(file_name="test.yaml")
    widget2 = measurements.toolbox.stress_analysis_toolbox(
        viewer
    )
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


def test_autocorrelation():
    """Test temporal and spatiotemporal autocorrelation."""
    # Temporal autocorrelation
    n_frames = 10
    n_measurements = 100
    np.random.seed(1)
    frames = np.repeat(np.arange(n_frames), n_measurements)
    features = [np.ones(n_measurements)]
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

    # Spatiotemporal autocorrelation
    max_degree = 5
    pointcloud = get_droplet_point_cloud()[0][0][:, 1:]
    _, coefficients = stress_spherical_harmonics_expansion(
        pointcloud, max_degree=max_degree, expansion_type="radial"
    )
    quadrature_points, lbdv_info = lebedev_quadrature(
        coefficients,
        number_of_quadrature_points=434,
        use_minimal_point_set=False,
    )
    manifold = create_manifold(
        quadrature_points, lbdv_info, max_degree=max_degree
    )
    distance_matrix = measurements.haversine_distances(max_degree, 434)
    measurements.spatio_temporal_autocorrelation(
        surfaces=[manifold, manifold], distance_matrix=distance_matrix
    )
