import numpy as np


def test_surface_tracing():
    """Test surface tracing and refinement."""
    from skimage import filters, morphology
    from vedo import shapes

    from napari_stress import reconstruction

    true_radius = 30

    # Make a blurry sphere
    image = np.zeros([100, 100, 100])
    image[50 - 30 : 50 + 31, 50 - 30 : 50 + 31, 50 - 30 : 50 + 31] = (
        morphology.ball(radius=true_radius)
    )
    blurry_sphere = filters.gaussian(image, sigma=5)

    # Put surface points on a slightly larger radius and add noise
    surf_points = shapes.Sphere().vertices
    surf_points += (surf_points * true_radius + 2) + 50

    # Test different fit methods (quick and fancy)
    for fit_type, atol in [("quick", 1.5), ("fancy", 1)]:
        results = reconstruction.trace_refinement_of_surface(
            blurry_sphere,
            surf_points,
            trace_length=20 if fit_type == "quick" else 10,
            sampling_distance=1.0,
            selected_fit_type=fit_type,
            interpolation_method="linear",
            remove_outliers=False,
        )
        traced_points = results[0][0]
        radial_vectors = np.array([50, 50, 50])[None, :] - traced_points
        mean_radii = np.linalg.norm(radial_vectors, axis=1).mean()
        assert np.allclose(true_radius, mean_radii, atol=atol)

    # Test outlier identification
    surf_points[0] += [0, 0, 10]
    results = reconstruction.trace_refinement_of_surface(
        blurry_sphere,
        surf_points,
        trace_length=10,
        selected_fit_type="fancy",
        interpolation_method="linear",
        remove_outliers=True,
    )
    traced_points = results[0][0]
    assert len(traced_points.squeeze()) < len(surf_points)


def test_patch_fitting():
    """Test patch fitting and related utilities."""
    from napari_stress._reconstruction.fit_utils import _fibonacci_sampling
    from napari_stress._reconstruction.patches import (
        _calculate_mean_curvature_on_patch,
        _create_fitted_coordinates,
        _find_neighbor_indices,
        _fit_and_create_pointcloud,
        _fit_quadratic_surface,
        _orient_patch,
        fit_patches,
    )

    # Test Fibonacci sampling
    pointcloud = _fibonacci_sampling(number_of_points=256)
    assert np.allclose(np.linalg.norm(pointcloud, axis=1), 1)

    # Test neighbor finding
    patch_indices = _find_neighbor_indices(pointcloud, patch_radius=0.1)
    assert np.allclose(np.asarray(patch_indices).squeeze(), np.arange(256))

    patch_indices = _find_neighbor_indices(pointcloud, patch_radius=2)
    assert np.allclose(
        np.asarray([len(x) for x in patch_indices]), np.repeat(256, 256)
    )

    # Test patch orientation
    radius = 0.3
    patch_indices = _find_neighbor_indices(pointcloud, patch_radius=radius)
    patch = pointcloud[patch_indices[0]]
    transformed_patch, _, orient_matrix = _orient_patch(patch, patch[0])
    assert np.array_equal(transformed_patch.shape, patch.shape)
    assert np.allclose(transformed_patch.mean(axis=0), 0)

    # Test quadratic surface fitting
    fit_params = _fit_quadratic_surface(transformed_patch)
    curvature, principal_curvatures = _calculate_mean_curvature_on_patch(
        transformed_patch[0], fit_params
    )
    assert abs(curvature[0] - 1) < 0.1

    # Test patch fitting
    fitted_pointcloud = fit_patches(pointcloud, search_radius=radius)
    assert np.allclose(np.linalg.norm(fitted_pointcloud, axis=1), 1, atol=0.01)

    # Test _fit_and_create_pointcloud
    mock_pointcloud = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    fitted_pointcloud = _fit_and_create_pointcloud(mock_pointcloud)
    np.testing.assert_array_almost_equal(
        fitted_pointcloud,
        mock_pointcloud,
        decimal=5,
        err_msg="Fitted pointcloud did not match expected values",
    )

    # Test _create_fitted_coordinates
    x_coords = np.array([0, 1, 2])
    y_coords = np.array([0, 1, 2])
    fitting_params = np.array([1, 0, 0, 0, 0, 0])  # Flat surface along z = 1
    zyx_pointcloud = _create_fitted_coordinates(
        np.stack([np.zeros(3), y_coords, x_coords]), fitting_params
    )
    expected_zyx_pointcloud = np.stack([[1, 0, 0], [1, 1, 2], [1, 1, 2]])
    np.testing.assert_array_almost_equal(
        zyx_pointcloud,
        expected_zyx_pointcloud,
        decimal=5,
        err_msg="Created pointcloud did not match expected values",
    )
