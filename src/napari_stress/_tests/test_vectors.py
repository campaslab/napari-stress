def test_normal_vectors_on_pointcloud():
    import vedo
    from napari_stress import vectors

    sphere = vedo.Sphere(r=10, res=100).points()
    normal_vectors = vectors.normal_vectors_on_pointcloud(sphere)
    assert len(normal_vectors) == len(sphere)


def test_normal_vectors_on_surface():
    import vedo
    from napari_stress import vectors
    import numpy as np

    sphere = vedo.Sphere(r=10, res=100)
    sphere = (sphere.points(), np.asarray(sphere.faces()))
    normal_vectors = vectors.normal_vectors_on_surface(sphere)
    assert len(normal_vectors) == len(sphere[0])


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
