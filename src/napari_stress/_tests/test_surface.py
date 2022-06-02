# -*- coding: utf-8 -*-
import numpy as np
import napari_stress
import vedo

def test_spherical_harmonics():

    ellipse = vedo.shapes.Ellipsoid()
    points = napari_stress.fit_spherical_harmonics(ellipse.points(), max_degree=3)

    assert np.array_equal(ellipse.points().shape, points[:, 1:].shape)

def test_smoothing():
    from napari_stress import smoothMLS2D, smooth_sinc
    import vedo

    sphere = vedo.shapes.Sphere(res=30)
    points = sphere.points() * 10
    faces = np.asarray(sphere.faces())
    points += np.random.uniform(size=points.shape)

    smoothed_points = smoothMLS2D(points, factor=0.25, radius=0)
    smoothed_points = smooth_sinc((points, faces))
    
    
def test_reconstruction():
    points = vedo.shapes.Ellipsoid().points() * 100

    surface = napari_stress.reconstruct_surface(points)

def test_surface_to_points():
    ellipse = vedo.shapes.Ellipsoid()

    surface = (ellipse.points(), np.asarray(ellipse.faces()))
    points = napari_stress.extract_vertex_points(surface)

