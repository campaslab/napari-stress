# -*- coding: utf-8 -*-

import numpy as np
import napari_process_points_and_surfaces as nppas
from napari.types import LabelsData, SurfaceData, PointsData
from napari_stress._utils.frame_by_frame import frame_by_frame

import vedo
import typing

import pyshtools

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


def reconstruct_surface(points: PointsData,
                        radius: float = None,
                        sampleSize: int = None,
                        holeFilling: bool = True,
                        bounds: tuple = (),
                        padding: float = 0.05
                        ):
    pointcloud = vedo.pointcloud.Points(points)

    dims = np.max(pointcloud.points(), axis=0) - np.min(pointcloud.points(), axis=0)
    surf = pointcloud.reconstructSurface(dims=dims.astype(int),
                                         radius=radius,
                                         sampleSize=sampleSize,
                                         holeFilling=holeFilling,
                                         bounds=bounds,
                                         padding=padding)

    return (surf.points(), np.asarray(surf.faces(), dtype=int))

@frame_by_frame
def smooth_sinc(surface: SurfaceData,
                n_iterations: int = 15,
                passBand: float = 0.1,
                edgeAngle: float = 15,
                feature_angle: float = 60,
                boundary: bool = False) -> SurfaceData:
    """
    Adjust mesh point positions using the Windowed Sinc function interpolation kernel.

    Parameters
    ----------
    surface : SurfaceData
        DESCRIPTION.
    n_iterations : int, optional
        Number of iteratios. The default is 15.
    passBand : float, optional
        Passband of sinc filter. The default is 0.1.
    edgeAngle : float, optional
        Edge angle to control smoothing along edges. The default is 15.
    feature_angle : float, optional
        Specifies the feature angle for sharp edge identification. The default is 60.
    boundary : bool, optional
        The default is False.

    Returns
    -------
    SurfaceData

    See also
    --------
    https://vedo.embl.es/autodocs/content/vedo/mesh.html#vedo.mesh.Mesh.smooth

    """

    mesh = vedo.mesh.Mesh((surface[0], surface[1]))
    mesh.smooth(niter=n_iterations,
                passBand=passBand,
                edgeAngle=edgeAngle,
                featureAngle=feature_angle,
                boundary=boundary)
    return (mesh.points(), np.asarray(mesh.faces(), dtype=int))

@frame_by_frame
def smoothMLS2D(points: PointsData,
                factor: float = 0.25,
                radius: float = 1.0) -> PointsData:
    """
    Smooth points with a Moving Least Squares algorithm variant.

    Parameters
    ----------
    points : PointsData
    f : float, optional
        Smoothing factor - typical range is [0,2]. Will be ignored if radius is
        different from 0. The default is 0.25.
    radius : float, optional
        Search radius for neighboring points to identify isolated points.
        Set this value to zero to ignore it. The default is 1.0.

    Returns
    -------
    PointsData

    See also
    --------
    https://vedo.embl.es/autodocs/content/vedo/pointcloud.html#vedo.pointcloud.Points.smoothMLS2D

    """
    if radius == 0:
        radius = None

    pointcloud = vedo.pointcloud.Points(points)
    pointcloud.smoothMLS2D(f=factor, radius=radius)

    if radius is None:
        return pointcloud.points()
    else:
        return pointcloud.points()[pointcloud.info['isvalid']]

def surface_from_label(label_image: LabelsData,
                       scale: typing.Union[list, np.ndarray]) -> SurfaceData:

    if isinstance(scale, list):
        scale = np.array(scale)

    n_frames = label_image.shape[0]

    surfs = []
    for t in range(n_frames):
        surf = nppas.label_to_surface(label_image[t])
        surfs.append(vedo.mesh.Mesh((surf[0] * scale[None, :], surf[1])))

    return surfs


def adjust_surface_density(surface: SurfaceData,
                           density_target: float) -> vedo.mesh.Mesh:

    mesh = vedo.mesh.Mesh((surface[0], surface[1]))
    n_vertices_target = int(mesh.area() * density_target)

    while mesh.N() < n_vertices_target:
        mesh.subdivide()

    mesh.decimate(N=n_vertices_target)

    return (mesh.points(), np.asarray(mesh.faces(), dtype=int))
