# -*- coding: utf-8 -*-

import numpy as np

from ._utils import pointcloud_to_vertices4D
import napari_process_points_and_surfaces as nppas
from napari.types import LabelsData, SurfaceData, PointsData

import vedo
import tqdm
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

def reconstruct_surface_from_points(points: typing.Union[np.ndarray, list],
                                    dims: np.ndarray,
                                    surf_density: float = 10.0,
                                    n_smooth:int = 15) -> list:

    if isinstance(points, list):
        points = pointcloud_to_vertices4D(points)

    # Check if data is 4D and reformat into list of arrays for every frame
    if points.shape[1] == 4:
        timepoints = np.unique(points[:, 0])
        _points = [points[np.where(points[:, 0] == i)][:, 1:] for i in timepoints]
    else:
        _points = [points]

    surfs = [None] * len(_points)
    for idx, pts in tqdm.tqdm(enumerate(_points), desc='Reconstructing surfaces',
                              total=len(_points)):

        # Get points and filter
        pts4vedo = vedo.Points(pts).clean(tol=0.02).densify(targetDistance=0.25)
        pts_filtered = vedo.pointcloud.removeOutliers(pts4vedo, radius=4)

        # # Smooth surface with moving least squares
        pts_filtered.smoothMLS2D(radius=2)

        # Reconstruct surface
        surf = vedo.pointcloud.recoSurface(pts_filtered, dims=dims)
        surf.smooth(niter=n_smooth)

        surf = adjust_surface_density(surf, density_target=surf_density)
        surfs[idx] = surf

    return surfs

def adjust_surface_density(surf: vedo.mesh.Mesh,
                           density_target: float) -> vedo.mesh.Mesh:
    n_vertices_target = int(surf.area() * density_target)

    if surf.N() > n_vertices_target:
        surf.decimate(N=n_vertices_target)
    else:
        surf.subdivide().decimate(N=n_vertices_target)

    return surf

def list_of_surfaces_to_surface(surfs: typing.Union[vedo.mesh.Mesh, list]) -> tuple:
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
    if isinstance(surfs, vedo.mesh.Mesh):
        surfs = [surfs]

    vertices = []
    faces = []
    n_verts = 0
    for idx, surf in enumerate(surfs):
        # Add time dimension to points coordinate array
        t = np.ones((surf.points().shape[0], 1)) * idx
        vertices.append(np.hstack([t, surf.points()]))  # add time dimension to points

        # Offset indices in faces list by previous amount of points
        faces.append(n_verts + np.array(surf.faces()))

        # Add number of vertices in current surface to n_verts
        n_verts += surf.N()

    if len(vertices) > 1:
        vertices = np.vstack(vertices)
        faces = np.vstack(faces)
    else:
        vertices = vertices[0]
        faces = faces[0]

    return (vertices, faces)
