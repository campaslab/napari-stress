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

def test_ellipsoid():
    pointcloud = vedo.shapes.Ellipsoid().points()
    ellipse_points = napari_stress.fit_ellipsoid(pointcloud, pvalue=0.5)

    viewer = napari.Viewer()
    viewer.add_points(pointcloud, size=0.1, face_color='orange')
    viewer.add_points(ellipse_points, size=0.1, face_color='cyan')

if __name__ == '__main__':
    test_ellipsoid()
