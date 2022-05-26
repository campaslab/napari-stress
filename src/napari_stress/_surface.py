# -*- coding: utf-8 -*-

import numpy as np
import napari_process_points_and_surfaces as nppas
from napari.types import LabelsData, SurfaceData, PointsData
from napari_stress._utils.frame_by_frame import frame_by_frame

import vedo
import typing
import pyshtools

@frame_by_frame
def resample_points(points: PointsData) -> PointsData:
    """Redistributes points in a pointcloud in a homogeneous manner"""
    pointcloud = vedo.pointcloud.Points(points)
    surface = pointcloud.reconstructSurface()
    points = nppas.sample_points_poisson_disk((surface.points(), np.asarray(surface.faces())),
                                              number_of_points=pointcloud.N())
    return points

@frame_by_frame
def fit_spherical_harmonics(points: PointsData,
                            max_degree: int = 5) -> PointsData:
    """
    Approximate a surface by spherical harmonics expansion

    Parameters
    ----------
    points : PointsData
    max_degree : int
        Order up to which spherical harmonics should be included for the approximation.

    Returns
    -------
    PointsData
        Pointcloud on surface of a spherical harmonics expansion at the same
        latitude/longitude as the input points.

    See also
    --------
    [1] https://en.wikipedia.org/wiki/Spherical_harmonics#/media/File:Spherical_Harmonics.png

    """
    # Convert points coordinates relative to center
    center = points.mean(axis=0)
    relative_coordinates = points - center[np.newaxis, :]

    # Convert point coordinates to spherical coordinates (in degree!)
    spherical_coordinates = vedo.cart2spher(relative_coordinates[:, 0],
                                            relative_coordinates[:, 1],
                                            relative_coordinates[:, 2])
    radius = spherical_coordinates[0]
    latitude = np.rad2deg(spherical_coordinates[1])
    longitude = np.rad2deg(spherical_coordinates[2])

    # Find spherical harmonics expansion coefficients until specified degree
    opt_fit_params = pyshtools._SHTOOLS.SHExpandLSQ(radius, latitude, longitude,
                                                    lmax = max_degree)[1]
    # Sample radius values at specified latitude/longitude
    spherical_harmonics_coeffcients = pyshtools.SHCoeffs.from_array(opt_fit_params)
    values = spherical_harmonics_coeffcients.expand(lat=latitude, lon=longitude)

    # Convert points back to cartesian coordinates
    points = vedo.spher2cart(values,
                             np.deg2rad(latitude),
                             np.deg2rad(longitude))
    return points.transpose() + center[np.newaxis, :]


@frame_by_frame
def reconstruct_surface(points: PointsData,
                        dims: list = [100, 100, 100],
                        radius: float = None,
                        sampleSize: int = None,
                        holeFilling: bool = True) -> SurfaceData:
    """
    Reconstruct a surface from a set of points.

    Parameters
    ----------
    points : PointsData (napari.types.PointsData)

    Returns
    -------
    SurfaceData: napari.types.SurfaceData

    """
    # Catch magicgui default values
    if radius == 0:
        radius = None

    if sampleSize == 0:
        sampleSize = None

    pointcloud = vedo.pointcloud.Points(points)
    surf = pointcloud.reconstructSurface(dims=dims,
                                         radius=radius,
                                         sampleSize=sampleSize,
                                         holeFilling=holeFilling)

    return (surf.points(), np.asarray(surf.faces(), dtype=int))

@frame_by_frame
def smooth_laplacian(surface: SurfaceData,
                     niter: int = 15,
                     relax_factor: float = 0.1,
                     edge_angle: float = 15,
                     feature_angle: float = 60,
                     boundary: bool = False) -> SurfaceData:
    mesh = vedo.mesh.Mesh((surface[0], surface[1]))
    mesh.smoothLaplacian(niter=niter, relaxfact=relax_factor,
                         edgeAngle=edge_angle,
                         featureAngle=feature_angle,
                         boundary=boundary)

    return (mesh.points(), np.asarray(mesh.faces()))

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

@frame_by_frame
def surface_from_label(label_image: LabelsData,
                       scale: typing.Union[np.ndarray, list] = np.array([1.0, 1.0, 1.0])
                       ) -> SurfaceData:

    surf = list(nppas.label_to_surface(label_image))
    surf[0] = surf[0] * np.asarray(scale)[None, :]

    return surf

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


@frame_by_frame
def adjust_surface_density(surface: SurfaceData,
                           density_target: float) -> SurfaceData:
    """Adjust the number of vertices of a surface to a defined density"""
    import open3d

    mesh = vedo.mesh.Mesh((surface[0], surface[1]))
    n_vertices_target = int(mesh.area() * density_target)

    # sample desired number of vertices from surface
    points = nppas.sample_points_poisson_disk(surface, n_vertices_target)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)

    # measure distances between points
    distances = np.array(pcd.compute_nearest_neighbor_distance())
    radius = np.median(distances)
    delta = 2 * distances.std()

    # reconstruct the surface
    surface = nppas.surface_from_point_cloud_ball_pivoting(points,
                                                           radius=radius,
                                                           delta_radius=delta)
    # Fix holes
    mesh = vedo.mesh.Mesh((surface[0], surface[1]))
    mesh.fillHoles(size=(radius+delta)**2)

    return (mesh.points(), np.asarray(mesh.faces()))
