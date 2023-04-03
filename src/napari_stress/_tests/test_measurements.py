# -*- coding: utf-8 -*-
import numpy as np


def test_mean_curvature_on_ellipsoid():
    """Test that the mean curvature is computed correctly."""
    from napari_stress import reconstruction, sample_data, measurements
    size = 128
    b, c = 0.5, 0.5
    number_of_ellipsoids = 4

    for a in np.linspace(0.2, 0.8, number_of_ellipsoids):
        ellipsoid = sample_data.make_blurry_ellipsoid(a, b, c, size,
                                                      definition_width=10)[0]
        results_reconstruction = reconstruction.reconstruct_droplet(
            ellipsoid, voxelsize=np.array([1, 1, 1]), use_dask=False,
            resampling_length=3.5, interpolation_method='linear',
            trace_length=10, sampling_distance=1)

        # calculate each point's distance to the surface of the ellipsoid
        distances = []
        ellipse_points = []
        for pt in results_reconstruction[3][0]:
            ellipse_points.append(
                project_point_on_ellipse_surface(
                    pt, a*size, b*size, c*size, np.array([size/2, size/2, size/2])
                )
            )
            distances.append(np.linalg.norm(pt - ellipse_points[-1]))
        distances = np.array(distances).flatten()

        # relative position error in x, y and z direction
        relative_position_error = np.abs(
            results_reconstruction[3][0] - ellipse_points) / ellipse_points
        relative_position_error = relative_position_error.flatten()

        assert np.mean(relative_position_error) < 0.01
        assert np.mean(distances) < 1

        # calculate mean curvature for two aspect ratios
        # average relative mean curvature error must be < 10%
        if 1 < a/b and a/b < 1.4:
            results_measurement = measurements.comprehensive_analysis(
                results_reconstruction[3][0],
                max_degree=20,
                n_quadrature_points=434,
                gamma=5)

            mean_curvature_measured = results_measurement[3][1][
                'features']['mean_curvature']
            mean_curvature_theoretical = theoretical_mean_curvature_on_pointcloud(
                c * size/2, b * size/2, a * size/2,
                results_measurement[4][0])
            relative_difference = np.abs(mean_curvature_measured -
                                         mean_curvature_theoretical) / mean_curvature_theoretical
            assert np.mean(relative_difference) < 0.1


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

    pointcloud_spherical = vedo.cart2spher(pointcloud[:, 2],
                                           pointcloud[:, 1],
                                           pointcloud[:, 0])
    elevation = pointcloud_spherical[1]
    azimuth = pointcloud_spherical[2]

    above = a * b * c * (3 * (a**2 + b**2) + 2*c**2 + (a**2 + b**2 - 2*c**2) *
                         np.cos(2*elevation) - 2*(a**2 - b**2) *
                         np.cos(2*azimuth) * np.sin(elevation)**2)
    below = 8 * (a**2 * b**2 * np.cos(elevation)**2 + c**2 *
                 (b**2 * np.cos(azimuth)**2 + a**2 * np.sin(azimuth)**2) *
                 np.sin(elevation)**2)**(3/2)
    return above/below


def project_point_on_ellipse_surface(query_point: np.ndarray,
                                     a: float,
                                     b: float,
                                     c: float,
                                     center: np.ndarray) -> np.ndarray:
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
    T = np.array([[1/a, 0, 0],
                  [0, 1/b, 0],
                  [0, 0, 1/c]])

    # transform query point to coordinate system in which ellipsoid is a sphere
    transformed_query_point = np.dot(query_point - center, T)

    # transform query point into spherical coordinates
    transformed_query_point_spherical = vedo.utils.cart2spher(transformed_query_point[2],
                                                              transformed_query_point[1],
                                                              transformed_query_point[0])

    # replace radius of query point with radius of sphere, which in this coordinate 
    # system is always 0.5
    point_on_transformed_ellipse_surface = np.array([0.5,
                                                     transformed_query_point_spherical[1],
                                                     transformed_query_point_spherical[2]])

    # transform point on sphere back to cartesian coordinates
    point_on_transformed_ellipse_surface = vedo.utils.spher2cart(*point_on_transformed_ellipse_surface)[::-1]
    point_on_ellipse_surface = np.dot(point_on_transformed_ellipse_surface, np.linalg.inv(T)) + center

    return point_on_ellipse_surface


def test_k_nearest_neighbors():
    from napari_stress import measurements
    from napari.types import PointsData

    points = PointsData(np.random.rand(100, 3))
    df = measurements.distance_to_k_nearest_neighbors(points)

    assert df.shape[0] == points.shape[0]

    points = PointsData(np.array([[0, 0],
                                  [1, 1],
                                  [0.5, 0.5],
                                  [0, 1],
                                  [1, 0]]))
    df = measurements.distance_to_k_nearest_neighbors(points, k=4)
    assert df.loc[2].to_numpy() == np.linalg.norm([0.5, 0.5])


def test_spatiotemporal_autocorrelation():
    from napari_stress import lebedev_quadrature
    from napari_stress import (measurements, reconstruction,
                               get_droplet_point_cloud,
                               create_manifold, TimelapseConverter)
    from napari_stress._spherical_harmonics.spherical_harmonics import stress_spherical_harmonics_expansion
    from napari.types import SurfaceData

    max_degree = 5
    # do sh expansion
    pointcloud = get_droplet_point_cloud()[0][0][:, 1:]
    _, coefficients = stress_spherical_harmonics_expansion(pointcloud,
                                                           max_degree=max_degree,
                                                            expansion_type='radial')
    quadrature_points, lbdv_info = lebedev_quadrature(coefficients, number_of_quadrature_points=434,
                                                      use_minimal_point_set=False)
    manifold = create_manifold(quadrature_points, lbdv_info, max_degree=max_degree)
    quadrature_points = manifold.get_coordinates()
    surface = reconstruction.reconstruct_surface_from_quadrature_points(quadrature_points)
    
    Converter = TimelapseConverter()
    surfaces_4d = Converter.list_of_data_to_data(data=[surface, surface, surface], layertype=SurfaceData)

    # get distance matrix and divide by volume-integrated H0
    H0 = measurements.average_mean_curvatures_on_manifold(manifold)[2]
    distance_matrix = measurements.haversine_distances(max_degree, 434)/H0
    

    measurements.spatio_temporal_autocorrelation(surfaces=surfaces_4d,
                                                 distance_matrix=distance_matrix)

def test_autocorrelation():
    from napari_stress import measurements
    import numpy as np
    import pandas as pd
    
    n_frames = 10
    n_measurements = 100

    # create a set of features in multiple timepoints and add a bit more noise
    # in every frame - the autocorrelaiton should thus decrease monotonously
    frames = np.repeat(np.arange(n_frames), n_measurements)
    feature = np.ones(n_measurements)
    features = []
    features.append(feature)
    for i in range(1, n_frames):
        features.append(features[i-1] + np.random.normal(size=n_measurements, scale=0.3))

    df = pd.DataFrame(np.concatenate(features), columns=['feature'])
    df['frame'] = frames

    gradient = np.gradient(measurements.temporal_autocorrelation(df, feature='feature'))
    assert np.all(gradient < 0)

def test_haversine():
    from napari_stress import measurements

    distance_matrix = measurements.haversine_distances(degree_lebedev=10, n_lebedev_points=434)

    # the biggest possible distance on a unit sphere is pi/2
    assert np.allclose(distance_matrix.max(), np.pi/2)


def test_geodesics():
    import vedo
    from napari_stress import measurements, approximation

    sphere = vedo.Ellipsoid(pos=(0, 0, 0),
                            axis1=(1, 0, 0),
                            axis2=(0, 2, 0),
                            axis3=(0, 0, 3))
    ellipsoid = approximation.least_squares_ellipsoid(sphere.points())
    expansion = approximation.expand_points_on_ellipse(ellipsoid,
                                                       sphere.points())

    curvature = measurements.curvature_on_ellipsoid(ellipsoid, expansion)
    surface = (sphere.points(), np.asarray(sphere.faces()),
               curvature[1]['features']['mean_curvature'])

    GDM = measurements.geodesic_distance_matrix(surface)
    results = measurements.local_extrema_analysis(surface,
                                                  distance_matrix=GDM)

    maxima_points = results[0][1]['features']['local_max_and_min']

    assert sum(maxima_points == 1) == 2
    assert sum(maxima_points == -1) == 2

    geodesic_vectors = measurements.geodesic_path(surface, 0, 1)

    assert geodesic_vectors.shape[0] == 44
    assert geodesic_vectors.shape[1] == 2
    assert geodesic_vectors.shape[2] == 3


def test_comprehenive_stress_toolbox(make_napari_viewer):
    from napari_stress import (get_droplet_point_cloud, measurements)

    viewer = make_napari_viewer()
    pointcloud = get_droplet_point_cloud()[0]
    viewer.add_points(pointcloud[0][:, 1:], **pointcloud[1])

    widget = measurements.stress_analysis_toolbox(viewer)
    viewer.window.add_dock_widget(widget)

    widget._run()

def test_comprehensive_stress_toolbox_4d(make_napari_viewer):
    from napari_stress import (get_droplet_point_cloud_4d, measurements)

    viewer = make_napari_viewer()
    pointcloud = get_droplet_point_cloud_4d()[0]
    viewer.add_points(pointcloud[0], **pointcloud[1])

    widget = measurements.stress_analysis_toolbox(viewer)
    viewer.window.add_dock_widget(widget)

    widget._run()


def test_curvature(make_napari_viewer):
    from napari_stress._spherical_harmonics.spherical_harmonics_napari import perform_lebedev_quadrature
    from napari_stress import measurements, get_droplet_point_cloud, fit_spherical_harmonics, types
    from magicgui import magicgui
    from napari.layers import Layer

    # We'll first create a spherical harmonics expansion and a lebedev
    # quadrature from scratch
    viewer = make_napari_viewer()

    pointcloud = get_droplet_point_cloud()[0]
    viewer.add_points(pointcloud[0][:, 1:], **pointcloud[1])

    expansion = fit_spherical_harmonics(viewer.layers[-1].data)
    viewer.add_points(expansion[0], **expansion[1])

    lebedev_points = perform_lebedev_quadrature(viewer.layers[-1], viewer=viewer)
    l = Layer.create(lebedev_points[0], lebedev_points[1], lebedev_points[2])
    viewer.add_layer(l)
    results_layer = viewer.layers[-1]

    assert types._METADATAKEY_MANIFOLD in results_layer.metadata

    # from code
    H, H0_arithmetic, H0_surface = measurements.calculate_mean_curvature_on_manifold(
        results_layer.metadata[types._METADATAKEY_MANIFOLD])
    H, H0_arithmetic, H0_surface = measurements.calculate_mean_curvature_on_manifold(
        results_layer)

    assert types._METADATAKEY_H0_ARITHMETIC in results_layer.metadata
    assert types._METADATAKEY_H0_SURFACE_INTEGRAL in results_layer.metadata
    assert types._METADATAKEY_MEAN_CURVATURE in results_layer.features

    # Test gauss-bonnet
    measurements.gauss_bonnet_test(results_layer)
    assert types._METADATAKEY_GAUSS_BONNET_ABS in results_layer.metadata
    assert types._METADATAKEY_GAUSS_BONNET_REL in results_layer.metadata

    absolute, relative = measurements.gauss_bonnet_test(results_layer,
                                                        viewer=viewer)
    assert types._METADATAKEY_GAUSS_BONNET_ABS in results_layer.metadata.keys()
    assert types._METADATAKEY_GAUSS_BONNET_REL in results_layer.metadata.keys()

    results = measurements.average_mean_curvatures_on_manifold(
        results_layer.metadata[types._METADATAKEY_MANIFOLD])
    results = measurements.average_mean_curvatures_on_manifold(
        results_layer)

    assert types._METADATAKEY_H0_ARITHMETIC in results_layer.metadata
    assert types._METADATAKEY_H0_SURFACE_INTEGRAL in results_layer.metadata
    assert types._METADATAKEY_MEAN_CURVATURE in results_layer.features
    assert types._METADATAKEY_H0_VOLUME_INTEGRAL in results_layer.metadata
    assert types._METADATAKEY_S2_VOLUME_INTEGRAL in results_layer.metadata

    # There shouldn't be a volume integral unless a radial manifold was used
    assert results_layer.metadata[types._METADATAKEY_S2_VOLUME_INTEGRAL] is None
    assert results_layer.metadata[types._METADATAKEY_H0_VOLUME_INTEGRAL] is None

def test_curvature2(make_napari_viewer):
    from napari_stress._spherical_harmonics.spherical_harmonics_napari import perform_lebedev_quadrature
    from napari_stress import measurements, get_droplet_point_cloud, fit_spherical_harmonics, types
    from napari_stress._spherical_harmonics.spherical_harmonics_napari import expansion_types
    from magicgui import magicgui
    from napari.layers import Layer

    # We'll first create a spherical harmonics expansion and a lebedev
    # quadrature from scratch
    viewer = make_napari_viewer()

    # Same as other test but we'll use a radial expansion
    pointcloud = get_droplet_point_cloud()[0]
    viewer.add_points(pointcloud[0][:, 1:], **pointcloud[1])

    expansion = fit_spherical_harmonics(viewer.layers[-1].data, expansion_type=expansion_types.radial)
    viewer.add_points(expansion[0], **expansion[1])

    lebedev_points = perform_lebedev_quadrature(viewer.layers[-1], viewer=viewer)
    l = Layer.create(lebedev_points[0], lebedev_points[1], lebedev_points[2])
    viewer.add_layer(l)
    results_layer = viewer.layers[-1]

    assert types._METADATAKEY_MANIFOLD in results_layer.metadata
    # from code
    H, H0_arithmetic, H0_surface = measurements.calculate_mean_curvature_on_manifold(
        results_layer.metadata[types._METADATAKEY_MANIFOLD])
    H, H0_arithmetic, H0_surface = measurements.calculate_mean_curvature_on_manifold(
        results_layer)

    results = measurements.average_mean_curvatures_on_manifold(
        results_layer.metadata[types._METADATAKEY_MANIFOLD])
    results = measurements.average_mean_curvatures_on_manifold(
        results_layer)

    assert types._METADATAKEY_H0_ARITHMETIC in results_layer.metadata
    assert types._METADATAKEY_H0_SURFACE_INTEGRAL in results_layer.metadata
    assert types._METADATAKEY_MEAN_CURVATURE in results_layer.features
    assert types._METADATAKEY_H0_VOLUME_INTEGRAL in results_layer.metadata
    assert types._METADATAKEY_S2_VOLUME_INTEGRAL in results_layer.metadata

    # There should be a volume integral because a radial manifold was used
    assert results_layer.metadata[types._METADATAKEY_S2_VOLUME_INTEGRAL] is not None
    assert results_layer.metadata[types._METADATAKEY_H0_VOLUME_INTEGRAL] is not None
    
def test_curvature3():
    """Tests mean curvatures on ellipsoid"""
    from napari_stress import get_droplet_point_cloud_4d
    from napari_stress import approximation, measurements

    pointcloud = get_droplet_point_cloud_4d()[0][0]

    ellipsoid = approximation.least_squares_ellipsoid(pointcloud)
    approximated_pointcloud = approximation.expand_points_on_ellipse(ellipsoid, pointcloud)
    curvature = measurements.curvature_on_ellipsoid(ellipsoid, approximated_pointcloud)


def test_stresses():
    from napari_stress import lebedev_quadrature
    from napari_stress import (measurements, approximation,
                               get_droplet_point_cloud,
                               create_manifold, types)
    from napari_stress._spherical_harmonics.spherical_harmonics import stress_spherical_harmonics_expansion

    max_degree = 5
    gamma = 26.0

    # do sh expansion
    pointcloud = get_droplet_point_cloud()[0][0][:, 1:]
    _, coefficients = stress_spherical_harmonics_expansion(pointcloud, max_degree=max_degree)
    quadrature_points, lbdv_info = lebedev_quadrature(coefficients, number_of_quadrature_points=200,
                                                      use_minimal_point_set=False)
    manifold = create_manifold(quadrature_points, lbdv_info, max_degree=max_degree)
    H_i, _, H0 = measurements.calculate_mean_curvature_on_manifold(manifold)

    # do ellipsoidal expansion
    ellipsoid = approximation.least_squares_ellipsoid(pointcloud)
    ellipsoid_points = approximation.expand_points_on_ellipse(ellipsoid, pointcloud)
    curvature_ellipsoid = measurements.curvature_on_ellipsoid(ellipsoid, ellipsoid_points)

    _, ellipsoid_coefficients = stress_spherical_harmonics_expansion(ellipsoid_points, max_degree=max_degree)
    ellipsoid_quadrature_points, lbdv_info = lebedev_quadrature(ellipsoid_coefficients, number_of_quadrature_points=200,
                                                      use_minimal_point_set=False)
    ellipsoid_manifold = create_manifold(ellipsoid_quadrature_points, lbdv_info, max_degree=max_degree)
    H_i_ellipsoid, _, H0_ellipsoid = measurements.calculate_mean_curvature_on_manifold(ellipsoid_manifold)

    # tissue stress tensor
    ellptical, cartesian = measurements.tissue_stress_tensor(
        ellipsoid,
        H0_ellipsoid,
        gamma=gamma)

    stress, stress_tissue, stress_cell = measurements.anisotropic_stress(
        H_i, H0, H_i_ellipsoid, H0_ellipsoid, gamma)

    measurements.anisotropic_stress(H_i, H0,
                                    H_i_ellipsoid, H0_ellipsoid,
                                    gamma)
    measurements.maximal_tissue_anisotropy(ellipsoid)

def test_compatibility_decorator():
    import inspect
    import numpy as np
    import napari_stress

    from napari_stress import types

    def my_function(manifold: napari_stress._stress.manifold_SPB.manifold, sigma: float = 1.0) -> (dict, dict):
        some_data = np.random.random((10,3))
        metadata = {'attribute': 1}
        metadata[types._METADATAKEY_MANIFOLD] = manifold
        features = {'attribute2': np.random.random(10)}
        return features, metadata

    function = napari_stress.measurements.utils.naparify_measurement(my_function)
    sig = inspect.signature(function)

    assert sig.parameters[types._METADATAKEY_MANIFOLD].annotation == 'napari.layers.Points'


if __name__ == '__main__':
    test_geodesics()