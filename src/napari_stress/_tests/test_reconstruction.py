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


def test_quadrature_point_reconstruction(make_napari_viewer):
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
    from napari_stress._reconstruction.patches import _fit_and_create_pointcloud

    # Mock input data
    mock_pointcloud = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Expected result calculated manually or by a reference implementation
    # expected_fitting_params = np.array([0, 0, 0, 0, 0, 0]) # Example coefficients
    expected_fitted_pointcloud = np.array(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    )  # Example expected pointcloud

    # Call the function to test
    fitted_pointcloud = _fit_and_create_pointcloud(mock_pointcloud)

    # Assert that the output is as expected
    np.testing.assert_array_almost_equal(
        fitted_pointcloud,
        expected_fitted_pointcloud,
        decimal=5,
        err_msg="Fitted pointcloud did not match expected values",
    )


def test_fit_quadratic_surface():
    from napari_stress._reconstruction.patches import _fit_quadratic_surface

    pointcloud = np.array(
        [
            [1, 0, 0],
            [1, 1, 2],
            [1, 1, 2],
        ]
    )

    # Expected result (a flat surface would just have the constant term)
    expected_fitting_params = np.array([1, 0, 0, 0, 0, 0])

    # Call the function to test
    fitting_params = _fit_quadratic_surface(pointcloud)

    # Assert that the output is as expected
    np.testing.assert_array_almost_equal(
        fitting_params,
        expected_fitting_params,
        decimal=5,
        err_msg="Fitting parameters did not match expected values",
    )


def test_create_fitted_coordinates():
    from napari_stress._reconstruction.patches import _create_fitted_coordinates

    # Mock input data
    x_coords = np.array([0, 1, 2])
    y_coords = np.array([0, 1, 2])
    fitting_params = np.array([1, 0, 0, 0, 0, 0])  # Flat surface along z = 1

    # Expected result
    expected_zyx_pointcloud = np.stack([[1, 0, 0], [1, 1, 2], [1, 1, 2]])

    # Call the function to test
    zyx_pointcloud = _create_fitted_coordinates(
        np.stack([np.zeros(3), y_coords, x_coords]), fitting_params
    )

    # Assert that the output is as expected
    np.testing.assert_array_almost_equal(
        zyx_pointcloud,
        expected_zyx_pointcloud,
        decimal=5,
        err_msg="Created pointcloud did not match expected values",
    )


def test_patch_fitting():
    from napari_stress._reconstruction.fit_utils import _fibonacci_sampling
    from napari_stress._reconstruction.patches import (
        _find_neighbor_indices,
        _orient_patch,
        _calculate_mean_curvature_on_patch,
        _fit_quadratic_surface,
        fit_patches,
    )

    # first create fibonacci pointcloud
    pointcloud = _fibonacci_sampling(number_of_points=256)

    # assert that distance of every point to origin is 1
    assert np.allclose(np.linalg.norm(pointcloud, axis=1), 1)

    # make sure every point has one neighbor (itself)
    patch_indices = _find_neighbor_indices(pointcloud, patch_radius=0.1)
    assert np.allclose(np.asarray(patch_indices).squeeze(), np.arange(256))

    # make sure all points are neighbors of each other for large radius
    patch_indices = _find_neighbor_indices(pointcloud, patch_radius=2)
    assert np.allclose(np.asarray([len(x) for x in patch_indices]), np.repeat(256, 256))

    # extract a patch around the first point
    # assert that all of these points have a distance of less than radius to the first point
    radius = 0.3
    patch_indices = _find_neighbor_indices(pointcloud, patch_radius=radius)
    assert all(
        np.linalg.norm(pointcloud[patch_indices[0]] - pointcloud[0][None, :], axis=1)
        < radius
    )

    # get patch and orient it
    patch = pointcloud[patch_indices[0]]
    transformed_patch, _, orient_matrix = _orient_patch(patch, patch[0])
    assert np.array_equal(transformed_patch.shape, patch.shape)
    assert np.allclose(transformed_patch.mean(axis=0), 0)  # patch center should be zero

    # make sure it's aligned with the first axis
    _, _, _orient_matrix = _orient_patch(transformed_patch, [0, 0, 0])
    assert np.allclose(_orient_matrix[:, 0], [1, 0, 0])

    # calculate principal curvatures on patch
    fit_params = _fit_quadratic_surface(transformed_patch)
    curvature, principal_curvatures = _calculate_mean_curvature_on_patch(
        transformed_patch[0], fit_params
    )

    assert abs(curvature[0] - 1) < 0.1  # curvature error should be smaller than 10%

    # calculate all principal curvatures
    radius = 0.5
    patch_indices = _find_neighbor_indices(pointcloud, patch_radius=radius)
    k1 = []
    k2 = []
    curvatures = []

    for i, p in enumerate(pointcloud):
        patch = pointcloud[patch_indices[i]]
        transformed_patch, _, orient_matrix = _orient_patch(patch, patch[0])
        fit_params = _fit_quadratic_surface(transformed_patch)
        curvature, principal_curvatures = _calculate_mean_curvature_on_patch(
            transformed_patch[0], fit_params
        )
        k1.append(principal_curvatures[0][0])
        k2.append(principal_curvatures[0][1])
        curvatures.append(curvature[0])

    # The principal curvatures should be identical as it's a sphere
    # curvature error should be smaller than 10%
    assert np.allclose(np.asarray(k1), np.asarray(k2), atol=0.02)
    assert np.std(np.asarray(curvatures)) < 0.05

    # Lastly, apply patch fitting and iterative methods
    # should still be a sphere afterwards
    fitted_pointcloud = fit_patches(pointcloud, search_radius=radius)
    assert np.allclose(np.linalg.norm(fitted_pointcloud, axis=1), 1, atol=0.01)

    # # iterative method - result should still be a sphere
    # fitted_pointcloud = iterative_curvature_adaptive_patch_fitting(pointcloud)
    # assert np.allclose(np.linalg.norm(fitted_pointcloud, axis=1), 1)
