# -*- coding: utf-8 -*-

import numpy as np
import napari_process_points_and_surfaces as nppas
from napari.types import LabelsData, SurfaceData, PointsData, VectorsData
from napari_stress._utils.frame_by_frame import frame_by_frame
from napari_tools_menu import register_function

import vedo

@register_function(menu="Points > Fit ellipsoid to pointcloud (vedo, n-STRESS)")
@frame_by_frame
def fit_ellipsoid_to_pointcloud_points(points: PointsData,
                                       inside_fraction: float = 0.673) -> PointsData:
    """
    Fit an ellipsoid to a pointcloud an retrieve surface pointcloud.

    Parameters
    ----------
    points : PointsData
    inside_fraction : float, optional
        Fraction of points to be inside the fitted ellipsoid. The default is 0.673.

    Returns
    -------
    PointsData

    """
    ellipsoid = vedo.pca_ellipsoid(vedo.pointcloud.Points(points),
                                   pvalue=inside_fraction)

    output_points = ellipsoid.points()

    return output_points

@register_function(menu="Points > Fit ellipsoid to pointcloud (vedo, n-STRESS)")
@frame_by_frame
def fit_ellipsoid_to_pointcloud_vectors(points: PointsData,
                                        inside_fraction: float = 0.673,
                                        normalize: bool = False) -> VectorsData:
    """
    Fit an ellipsoid to a pointcloud an retrieve the major axises as vectors.

    Parameters
    ----------
    points : PointsData
    inside_fraction : float, optional
        Fraction of points to be inside the fitted ellipsoid. The default is 0.673.
    normalize : bool, optional
        Normalize the resulting vectors. The default is False.

    Returns
    -------
    VectorsData

    """
    ellipsoid = vedo.pca_ellipsoid(vedo.pointcloud.Points(points),
                                   pvalue=inside_fraction)

    vectors = np.stack([ellipsoid.axis1 * ellipsoid.va,
                        ellipsoid.axis2 * ellipsoid.vb,
                        ellipsoid.axis3 * ellipsoid.vc])

    if normalize:
        vectors = vectors/np.linalg.norm(vectors, axis=0)[None, :]

    base_points = np.stack([ellipsoid.center, ellipsoid.center, ellipsoid.center])
    vectors = np.stack([base_points, vectors]).transpose((1,0,2))

    return vectors

@register_function(menu="Points > Create surface from points (flying edges, vedo, n-STRESS)")
@frame_by_frame
def reconstruct_surface(points: PointsData,
                        radius: float = 1.0,
                        holeFilling: bool = True,
                        padding: float = 0.05
                        ) -> SurfaceData:
    """
    Reconstruct a surface from a given pointcloud.

    Parameters
    ----------
    points : PointsData
    radius : float
        Radius within which to search for neighboring points.
    holeFilling : bool, optional
        The default is True.
    padding : float, optional
        Whether or not to thicken the surface by a given margin.
        The default is 0.05.

    Returns
    -------
    SurfaceData
    """
    pointcloud = vedo.pointcloud.Points(points)

    surface = pointcloud.reconstruct_surface(radius=radius,
                                             sample_size=None,
                                             hole_filling=holeFilling,
                                             padding=padding)

    return (surface.points(), np.asarray(surface.faces(), dtype=int))

@register_function(menu="Points > Create points from surface vertices (n-STRESS)")
@frame_by_frame
def extract_vertex_points(surface: SurfaceData) -> PointsData:
    """
    Return only the vertex points of an input surface.

    Parameters
    ----------
    surface : SurfaceData

    Returns
    -------
    PointsData

    """
    return surface[0]


@register_function(menu="Surfaces > Smoothing (Windowed Sinc, vedo, n-STRESS)")
@frame_by_frame
def smooth_sinc(surface: SurfaceData,
                niter: int = 15,
                passBand: float = 0.1,
                edgeAngle: float = 15,
                feature_angle: float = 60,
                boundary: bool = False) -> SurfaceData:

    mesh = vedo.mesh.Mesh((surface[0], surface[1]))
    mesh.smooth(niter=niter, passBand=passBand,
                edgeAngle=edgeAngle, featureAngle=feature_angle,
                boundary=boundary)
    return (mesh.points(), np.asarray(mesh.faces(), dtype=int))

@register_function(menu="Surfaces > Smoothing (MLS2D, vedo, n-STRESS)")
@frame_by_frame
def smoothMLS2D(points: PointsData,
                factor: float = 0.5,
                radius: float = None) -> PointsData:

    pointcloud = vedo.pointcloud.Points(points)
    pointcloud.smoothMLS2D(f=factor, radius=radius)

    if radius is not None:
        return pointcloud.points()[pointcloud.info['isvalid']]
    else:
        return pointcloud.points()

@register_function(menu="Surfaces > Simplify (decimate, vedo, n-STRESS)")
@frame_by_frame
def decimate(surface: SurfaceData,
             fraction: float = 0.1) -> SurfaceData:

    mesh = vedo.mesh.Mesh((surface[0], surface[1]))

    n_vertices = mesh.N()
    n_vertices_target = n_vertices * fraction

    while mesh.N() > n_vertices_target:
        _fraction = n_vertices_target/mesh.N()
        mesh.decimate(fraction=_fraction)

    return (mesh.points(), np.asarray(mesh.faces()))

