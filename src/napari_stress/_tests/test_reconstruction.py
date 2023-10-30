# -*- coding: utf-8 -*-
import numpy as np


def test_reconstruction(make_napari_viewer):
    from napari_stress import reconstruction, get_droplet_4d
    from napari.layers import Layer
    import napari_stress

    viewer = make_napari_viewer()
    image = get_droplet_4d()[0][0]
    viewer.add_image(image, scale=[1.1, 1.1, 1.1], name="image")

    widget = napari_stress._reconstruction.toolbox.droplet_reconstruction_toolbox(
        viewer
    )
    viewer.window.add_dock_widget(widget)

    assert widget.doubleSpinBox_voxelsize_x.value() == 1.1
    assert widget.doubleSpinBox_voxelsize_y.value() == 1.1
    assert widget.doubleSpinBox_voxelsize_z.value() == 1.1

    results = reconstruction.reconstruct_droplet(
        image,
        voxelsize=np.asarray([4, 2, 2]),
        interpolation_method="linear",
        target_voxelsize=2,
        use_dask=True,
    )

    for result in results:
        layer = Layer.create(result[0], result[1], result[2])
        viewer.add_layer(layer)

    # test with dask
    results = reconstruction.reconstruct_droplet(
        image,
        voxelsize=np.asarray([1.93, 1, 1]),
        target_voxelsize=1,
        interpolation_method="linear",
        resampling_length=1,
        use_dask=True,
    )

    # test saving/loading settings
    widget._export_settings(file_name="test.yaml")
    widget2 = napari_stress._reconstruction.toolbox.droplet_reconstruction_toolbox(
        viewer
    )
    widget2._import_settings(file_name="test.yaml")
    assert widget2.doubleSpinBox_voxelsize_x.value() == 1.1
    assert widget2.doubleSpinBox_voxelsize_y.value() == 1.1
    assert widget2.doubleSpinBox_voxelsize_z.value() == 1.1


def test_quadrature_point_reconstuction(make_napari_viewer):
    from napari_stress import get_droplet_point_cloud, fit_spherical_harmonics
    from napari_stress import reconstruction
    from napari_stress._spherical_harmonics.spherical_harmonics_napari import (
        perform_lebedev_quadrature,
    )

    pointcloud = get_droplet_point_cloud()[0]
    # Expansion
    viewer = make_napari_viewer()
    points = fit_spherical_harmonics(pointcloud[0], max_degree=3)
    points_layer = viewer.add_points(points[0], **points[1])

    # quadrature
    lebedev_points = perform_lebedev_quadrature(points_layer, viewer=viewer)[0]
    reconstruction.reconstruct_surface_from_quadrature_points(lebedev_points)


def test_surface_tracing():
    from napari_stress import reconstruction
    from skimage import filters, morphology
    from vedo import shapes

    true_radius = 30

    # Make a blurry sphere first
    image = np.zeros([100, 100, 100])
    image[50 - 30 : 50 + 31, 50 - 30 : 50 + 31, 50 - 30 : 50 + 31] = morphology.ball(
        radius=true_radius
    )
    blurry_sphere = filters.gaussian(image, sigma=5)

    # Put surface points on a slightly larger radius and add a bit of noise
    surf_points = shapes.Sphere().points()
    surf_points += (surf_points * true_radius + 2) + 50

    # Test different fit methods (fancy/quick)
    fit_type = "quick"
    results = reconstruction.trace_refinement_of_surface(
        blurry_sphere,
        surf_points,
        trace_length=20,
        sampling_distance=1.0,
        selected_fit_type=fit_type,
        interpolation_method="linear",
        remove_outliers=False,
    )
    traced_points = results[0][0]

    radial_vectors = np.array([50, 50, 50])[None, :] - traced_points
    mean_radii = np.linalg.norm(radial_vectors, axis=1).mean()
    assert np.allclose(true_radius, mean_radii, atol=1.5)

    fit_type = "fancy"
    results = reconstruction.trace_refinement_of_surface(
        blurry_sphere,
        surf_points,
        trace_length=10,
        selected_fit_type=fit_type,
        interpolation_method="linear",
        remove_outliers=False,
    )
    traced_points = results[0][0]
    radial_vectors = np.array([50, 50, 50])[None, :] - traced_points
    mean_radii = np.linalg.norm(radial_vectors, axis=1).mean()

    assert np.allclose(true_radius, mean_radii, atol=1)

    # Test outlier identification
    surf_points[0] += [0, 0, 10]
    results = reconstruction.trace_refinement_of_surface(
        blurry_sphere,
        surf_points,
        trace_length=10,
        selected_fit_type=fit_type,
        interpolation_method="linear",
        remove_outliers=True,
    )
    traced_points = results[0][0]
    assert len(traced_points.squeeze()) < len(surf_points)

def test_fit_and_create_pointcloud():
    from .._reconstruction.patches import fit_and_create_pointcloud
    # Mock input data
    mock_pointcloud = np.array([[1, 2, 3],
                                [4, 5, 6],
                                [7, 8, 9]])

    # Expected result calculated manually or by a reference implementation
    # expected_fitting_params = np.array([0, 0, 0, 0, 0, 0]) # Example coefficients
    expected_fitted_pointcloud = np.array([[1, 2, 3],
                                           [4, 5, 6],
                                           [7, 8, 9]]) # Example expected pointcloud

    # Call the function to test
    fitted_pointcloud = fit_and_create_pointcloud(mock_pointcloud)

    # Assert that the output is as expected
    np.testing.assert_array_almost_equal(
        fitted_pointcloud, expected_fitted_pointcloud,
        decimal=5,
        err_msg='Fitted pointcloud did not match expected values'
    )

def test_fit_quadratic_surface():
    from .._reconstruction.patches import fit_quadratic_surface
    # Mock input data
    x_coords = np.array([0, 1, 2])
    y_coords = np.array([0, 1, 2])
    z_coords = np.array([1, 1, 1])  # Flat surface along z = 1

    # Expected result (a flat surface would just have the constant term)
    expected_fitting_params = np.array([1, 0, 0, 0, 0, 0])

    # Call the function to test
    fitting_params = fit_quadratic_surface(x_coords, y_coords, z_coords)

    # Assert that the output is as expected
    np.testing.assert_array_almost_equal(
        fitting_params, expected_fitting_params,
        decimal=5,
        err_msg='Fitting parameters did not match expected values'
    )

def test_create_fitted_coordinates():
    from .._reconstruction.patches import create_fitted_coordinates
    # Mock input data
    x_coords = np.array([0, 1, 2])
    y_coords = np.array([0, 1, 2])
    fitting_params = np.array([1, 0, 0, 0, 0, 0])  # Flat surface along z = 1

    # Expected result
    expected_zyx_pointcloud = np.array([[1, 0, 0], [1, 1, 1], [1, 2, 2]])

    # Call the function to test
    zyx_pointcloud = create_fitted_coordinates(x_coords, y_coords, fitting_params)

    # Assert that the output is as expected
    np.testing.assert_array_almost_equal(
        zyx_pointcloud, expected_zyx_pointcloud,
        decimal=5,
        err_msg='Created pointcloud did not match expected values'
    )