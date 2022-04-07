# -*- coding: utf-8 -*-

import numpy as np
import napari_process_points_and_surfaces as nppas
from napari.types import LabelsData, SurfaceData

import vedo
import typing


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


def adjust_surface_density(mesh: vedo.mesh.Mesh,
                           density_target: float) -> vedo.mesh.Mesh:


    n_vertices_target = int(mesh.area() * density_target)

    while mesh.N() < n_vertices_target:
        mesh.subdivide()

    mesh.decimate(N=n_vertices_target)

    return mesh

def list_of_surfaces_to_surface(surfs: list) -> tuple:
    """
    Convert vedo surface object to napari-diggestable data format.

    Parameters
    ----------
    surfs : typing.Union[vedo.mesh.Mesh, list]
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if isinstance(surfs[0], vedo.mesh.Mesh):
        surfs = [(s.points(), s.faces()) for s in surfs]


    vertices = []
    faces = []
    n_verts = 0
    for idx, surf in enumerate(surfs):
        # Add time dimension to points coordinate array
        t = np.ones((surf[0].shape[0], 1)) * idx
        vertices.append(np.hstack([t, surf[0]]))  # add time dimension to points

        # Offset indices in faces list by previous amount of points
        faces.append(n_verts + np.array(surf[1]))

        # Add number of vertices in current surface to n_verts
        n_verts += surf[0].shape[0]

    if len(vertices) > 1:
        vertices = np.vstack(vertices)
        faces = np.vstack(faces)
    else:
        vertices = vertices[0]
        faces = faces[0]

    return (vertices, faces)
