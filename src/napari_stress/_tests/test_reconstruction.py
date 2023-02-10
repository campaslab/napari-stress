# -*- coding: utf-8 -*-

def test_reconstruction(make_napari_viewer):
    from napari_stress import reconstruction, get_droplet_4d
    import napari_process_points_and_surfaces as nppas
    from skimage import measure
    from napari.layers import Layer

    import numpy as np

    viewer = make_napari_viewer()


    image = get_droplet_4d()[0][0]
    viewer.add_image(image)

    widget = reconstruction.droplet_reconstruction_toolbox(viewer)
    viewer.window.add_dock_widget(widget)

    results = reconstruction.reconstruct_droplet(
        image,
        voxelsize=np.asarray([1.93, 1, 1]),
        target_voxelsize=1
        )

    for result in results:
        layer = Layer.create(result[0], result[1], result[2])
        viewer.add_layer(layer)

    # test with dask
    results = reconstruction.reconstruct_droplet(
        image,
        voxelsize=np.asarray([1.93, 1, 1]),
        target_voxelsize=1,
        resampling_length=1,
        use_dask=True
        )

def test_quadrature_point_reconstuction(make_napari_viewer):
    from napari_stress import get_droplet_point_cloud, fit_spherical_harmonics
    from napari_stress import reconstruction
    from napari_stress._spherical_harmonics.spherical_harmonics_napari import perform_lebedev_quadrature

    pointcloud = get_droplet_point_cloud()[0]
    # Expansion
    viewer = make_napari_viewer()
    points = fit_spherical_harmonics(pointcloud[0], max_degree=3)
    points_layer = viewer.add_points(points[0], **points[1])

    # quadrature
    lebedev_points = perform_lebedev_quadrature(points_layer, viewer=viewer)[0]
    reconstruction.reconstruct_surface_from_quadrature_points(lebedev_points)

def test_vector_tools():
    from napari_stress import vectors
    import vedo
    import numpy as np

    sphere = vedo.Sphere(r=10, pos=(50, 50, 50))
    image = np.random.rand(100, 100, 100)
    sampling_distance = 0.25

    vectors.normal_vectors_on_pointcloud(sphere.points())
    normal_vectors = vectors.normal_vectors_on_surface((sphere.points(), np.asarray(sphere.faces())))
    df_intensity = vectors.sample_intensity_along_vector(normal_vectors, image, sampling_distance=sampling_distance)

    assert df_intensity.shape[0] == normal_vectors.shape[0]
    assert df_intensity.shape[1] == 1/sampling_distance


if __name__ == "__main__":
    test_vector_tools()
