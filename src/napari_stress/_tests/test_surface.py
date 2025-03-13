import napari
import numpy as np
import vedo

import napari_stress


def test_reconstruction():
    points = vedo.shapes.Ellipsoid().vertices * 100

    surface = napari_stress.reconstruct_surface(points)

    assert isinstance(surface, tuple)


def test_surface_to_points():
    ellipse = vedo.shapes.Ellipsoid()

    surface = (ellipse.vertices, np.asarray(ellipse.faces()))
    points = napari_stress.extract_vertex_points(surface)

    assert isinstance(points, np.ndarray)


def test_ellipsoid_points():
    pointcloud = (
        np.random.normal(size=(1000, 3)) * 10 * np.array([1, 2, 3])[None, :]
    )
    ellipse_points = napari_stress.fit_ellipsoid_to_pointcloud_points(
        pointcloud, inside_fraction=0.5
    )

    assert np.array_equal(ellipse_points.shape, (1058, 3))

    axis = napari_stress.fit_ellipsoid_to_pointcloud_vectors(
        pointcloud, inside_fraction=0.5
    )

    assert np.array_equal(axis.shape, (3, 2, 3))

    axis = napari_stress.fit_ellipsoid_to_pointcloud_vectors(
        pointcloud, inside_fraction=0.5, normalize=True
    )

    assert np.array_equal(axis.shape, (3, 2, 3))

    pointcloud = vedo.shapes.Ellipsoid().vertices * 10
    ellipse_points = napari_stress.fit_ellipsoid_to_pointcloud_points(
        pointcloud, inside_fraction=0.5
    )

    # test 4d handling
    Converter = napari_stress.TimelapseConverter()
    pointcloud_4d = Converter.list_of_data_to_data(
        [pointcloud, pointcloud + 1], napari.types.PointsData
    )
    vectors_4d = napari_stress.fit_ellipsoid_to_pointcloud_vectors(
        pointcloud_4d
    )

    assert np.array_equal(vectors_4d.shape, (6, 2, 4))
