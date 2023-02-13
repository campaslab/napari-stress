# -*- coding: utf-8 -*-
import numpy as np

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
    from napari_stress import measurements

    sphere = vedo.Sphere(r=10)
    values = np.zeros(sphere.N())
    values[0] = -1
    values[-1] = 1
    surface = (sphere.points(), np.asarray(sphere.faces()), values)

    GDM = measurements.geodesic_distance_matrix(surface)
    geodesic_vectors = measurements.geodesic_path(surface, 1, 2)
    results = measurements.local_extrema_analysis(surface,
                                                distance_matrix=GDM)

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
    test_k_nearest_neighbors()