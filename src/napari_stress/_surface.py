# -*- coding: utf-8 -*-

import numpy as np

from ._utils import pointcloud_to_vertices4D

import vedo
import tqdm
import typing


def reconstruct_surface(points: typing.Union[np.ndarray, list],
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

def surface2layerdata(surfs: typing.Union[vedo.mesh.Mesh, list],
                      value_key: str = 'Spherefit_curvature') -> tuple:
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
        print('I did this')
        surfs = [surfs]

    vertices = []
    faces = []
    values = []
    n_verts = 0
    for idx, surf in enumerate(surfs):
        # Add time dimension to points coordinate array
        t = np.ones((surf.points().shape[0], 1)) * idx
        vertices.append(np.hstack([t, surf.points()]))
        
        # Offset indices in faces list by previous amount of points
        faces.append(n_verts + np.array(surf.faces()))
        values.append(surf.pointdata[value_key])
        
        # Add number of vertices in current surface to n_verts
        n_verts += surf.N()
        
    if len(vertices) > 1:
        vertices = np.vstack(vertices)
        faces = np.vstack(faces)
        _values = []
        for lst in values:
            _values += list(lst)
        values = np.asarray(_values)
    else:
        vertices = vertices[0]
        faces = faces[0]
        values = values[0]
    
    return (vertices, faces, values)


def calculate_curvatures(surf: typing.Union[vedo.mesh.Mesh, list],
                         radius: float = 1) -> list:
    """
    Calculate the curvature for each point on a surface.
    
    This function iterates over every vertex of and retrieves all points within
    a defined neighborhood range. A sphere is then fitted to these patches. The
    local curvature then corresponds to the squared inverse radius (1/r**2) of
    the sphere.

    Parameters
    ----------
    surf : vedo.mesh
        DESCRIPTION.
    radius : int, optional
        Radius within which points will be considered to be neighbors.
        The default is 1.

    Returns
    -------
    surf : list of vedo mesh objects. 
    
        The curvature of each surface in the list is stored in 
        surface.pointdata['Spherefit_curvature'].

    See also
    --------
    https://github.com/marcomusy/vedo/issues/610
    """
    # Turn input into a list if a single surface was passed
    if isinstance(surf, vedo.mesh.Mesh):
        surf = [surf]
    
    for _surf in surf:
        curvature = np.zeros(_surf.N())  # allocate
        residues = np.zeros(_surf.N())  # allocate
        for idx in tqdm.tqdm(range(_surf.N()), desc='Fitting surface'):

            patch = _surf.closestPoint(_surf.points()[idx], radius=radius)
            patch = vedo.pointcloud.Points(patch)  # make it a vedo object
            
            # If the patch radius is too small, the curvature can not be measured
            try:
                s = vedo.pointcloud.fitSphere(patch)
                curvature[idx] = 1/(s.radius)**2
                residues[idx] = s.residue
            except Exception:
                curvature[idx] = 0
                residues[idx] = 0
            
        _surf.pointdata['Spherefit_curvature'] = curvature
        _surf.pointdata['residues'] = residues
    
    return surf       


def fibonacci_sphere(semiAxesLengths, rot_matrix, center, samples=2**8):

    samples = int(samples)

    # distribute fibonacci points
    z = np.linspace(1 - 1.0/samples, 1.0/samples - 1, samples)
    radius = np.sqrt(1.0 - z**2)
    goldenAngle = np.pi * (3.0 - np.sqrt(5.0))
    theta = goldenAngle * np.arange(0, samples, 1)  # golden angle increment

    if semiAxesLengths is None:
        semiAxesLengths = np.asarray([1.0, 1.0, 1.0])

    ZYX_local = np.zeros((samples, 3))
    ZYX_local[:, 2] = semiAxesLengths[2] * radius * np.cos(theta)
    ZYX_local[:, 1] = semiAxesLengths[1] * radius * np.sin(theta)
    ZYX_local[:, 0] = semiAxesLengths[0] * z

    # rotate ellipse points
    if rot_matrix is not None:
        XYZ_rot = np.dot(ZYX_local, rot_matrix)
    else:
        XYZ_rot = ZYX_local

    # translate ellipse points from 0 to center
    if center is not None:
        t = np.asarray([center for i in range(0, samples)])
        ZYX = XYZ_rot + t
    else:
        ZYX = XYZ_rot

    return ZYX
