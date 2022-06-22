# -*- coding: utf-8 -*-
import numpy as np
import napari_stress
import vedo
import napari

def test_reconstruction():
    points = vedo.shapes.Ellipsoid().points() * 100

    surface = napari_stress.reconstruct_surface(points)

def test_surface_to_points():
    ellipse = vedo.shapes.Ellipsoid()

    surface = (ellipse.points(), np.asarray(ellipse.faces()))
    points = napari_stress.extract_vertex_points(surface)

def test_ellipsoid_points():
    pointcloud = np.random.normal(size=(1000, 3)) * 10 * np.array([1,2,3])[None, :]
    ellipse_points = napari_stress.fit_ellipsoid_to_pointcloud_points(pointcloud, pvalue=0.5)
    axis = napari_stress.fit_ellipsoid_to_pointcloud_vectors(pointcloud, pvalue=0.5)
    axis = napari_stress.fit_ellipsoid_to_pointcloud_vectors(pointcloud, pvalue=0.5, normalize=True)

    pointcloud = vedo.shapes.Ellipsoid().points() * 10
    ellipse_points = napari_stress.fit_ellipsoid_to_pointcloud_points(pointcloud, pvalue=0.5)

if __name__ == '__main__':
    test_ellipsoid_points()
