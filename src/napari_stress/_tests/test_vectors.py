import numpy as np

def test_intensity_measurement_on_normals():
    from napari_stress import vectors, sample_data, frame_by_frame
    import napari_segment_blobs_and_things_with_membranes as nsbatwm
    import napari_process_points_and_surfaces as nppas

    droplet = sample_data.get_droplet_4d()[0][0]
    droplet_rescaled = frame_by_frame(nsbatwm.rescale)(droplet, scale_x=1, scale_y=1, scale_z=2)
    droplet_binary = frame_by_frame(nsbatwm.threshold_otsu)(droplet_rescaled)

    surface = frame_by_frame(nppas.label_to_surface)(droplet_binary, 1)
    surface_smooth = frame_by_frame(nppas.smooth_surface)(surface)

    # measure intensity
    normals = vectors.normal_vectors_on_surface(surface_smooth, length_multiplier=5, center=True)
    vectors_LDtuple = vectors._sample_intensity_along_vector(normals, droplet_rescaled)

    assert len(vectors_LDtuple[0]) == len(normals)
    assert 'intensity_mean' in vectors_LDtuple[1]['features'].keys()


def test_normal_vectors_on_pointcloud():
    import vedo
    from napari_stress import vectors

    sphere = vedo.Sphere(r=10, res=100).points()
    normal_vectors = vectors.normal_vectors_on_pointcloud(sphere)
    assert len(normal_vectors) == len(sphere)

    # test length multiplier
    multiplier = 2
    normal_vectors = vectors.normal_vectors_on_pointcloud(
        sphere, length_multiplier=multiplier)
    vector_legth = np.linalg.norm(normal_vectors[:, 1], axis=1)
    assert np.allclose(vector_legth, multiplier)

    # if centering is False, base points should be the same as input points
    center = False
    normal_vectors = vectors.normal_vectors_on_pointcloud(
        sphere, center=center)
    assert np.allclose(normal_vectors[:, 0], sphere)


def test_normal_vectors_on_surface():
    import vedo
    from napari_stress import vectors
    import numpy as np

    sphere = vedo.Sphere(r=10, res=100)
    sphere = (sphere.points(), np.asarray(sphere.faces()))
    normal_vectors = vectors.normal_vectors_on_surface(sphere)
    assert len(normal_vectors) == len(sphere[0])

    # test length multiplier
    multiplier = 2
    normal_vectors = vectors.normal_vectors_on_surface(
        sphere, length_multiplier=multiplier)
    vector_legth = np.linalg.norm(normal_vectors[:, 1], axis=1)
    assert np.allclose(vector_legth, multiplier)

    # Test centering
    center = False
    normal_vectors = vectors.normal_vectors_on_surface(
        sphere, center=center)
    assert np.allclose(normal_vectors[:, 0], sphere[0])


def test_pairwise_point_distances():
    import vedo
    from napari_stress import vectors
    import numpy as np

    sphere1 = vedo.Sphere(r=10, res=100).points()
    sphere2 = vedo.Sphere(r=5, res=100).points()
    pairwise_point_distances = vectors.pairwise_point_distances(
        sphere1, sphere2)

    vector_legth = np.linalg.norm(pairwise_point_distances[:, 1], axis=1)
    assert np.allclose(vector_legth, 5)


def test_move_points():
    import vedo
    from napari_stress import vectors
    import numpy as np

    sphere = vedo.Sphere(r=10, res=100).points()
    normals = vectors.normal_vectors_on_pointcloud(sphere)
    moved_points_relative = vectors.relative_move_points_along_vector(
        sphere, normals, 0.5)
    moved_points_absolute = vectors.absolute_move_points_along_vector(
        sphere, normals, 2)
    moved_points_relative_length = np.linalg.norm(
        moved_points_relative - sphere, axis=1)
    moved_points_absolute_length = np.linalg.norm(
        moved_points_absolute - sphere, axis=1)

    assert np.allclose(moved_points_relative_length, 0.5)
    assert np.allclose(moved_points_absolute_length, 2)