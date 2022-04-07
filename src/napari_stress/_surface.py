# -*- coding: utf-8 -*-

import numpy as np
import napari_process_points_and_surfaces as nppas
from napari.types import LabelsData, SurfaceData, PointsData

import vedo
import typing


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

def smoothMLS2D(points: PointsData,
                f: float = 0.2,
                radius: float = None) -> PointsData:

    pointcloud = vedo.pointcloud.Points(points)
    pointcloud.smoothMLS2D(f=f, radius=radius)

    return pointcloud.Points()[pointcloud.info['isvalid']]

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
