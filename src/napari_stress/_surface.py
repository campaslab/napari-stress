# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import cKDTree, Delaunay

from ._utils import cart2sph

import vedo
import tqdm
import typing


def reconstruct_surface(points: np.ndarray,
                        dims: np.ndarray,
                        n_smooth:int = 5) -> list:

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

        # Smooth surface with moving least squares
        pts_filtered.smoothMLS2D(radius=2)
        
        # Reconstruct surface
        surf = vedo.pointcloud.recoSurface(pts_filtered, dims=dims)
        surf.smooth().computeNormals()

        surfs[idx] = surf
    
    return surfs

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
        surfs = [surfs]

    # add surfaces to viewer
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
        
    vertices = np.vstack(vertices)
    faces = np.vstack(faces)
    values = np.concatenate(values)
    
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
        for idx in tqdm.tqdm(range(_surf.N()), desc='Fitting surface'):
            patch = _surf.closestPoint(_surf.points()[idx], radius=radius)
            patch = vedo.pointcloud.Points(patch)  # make it a vedo object
            s = vedo.pointcloud.fitSphere(patch)
            
            curvature[idx] = 1/(s.radius)**2
            
        _surf.pointdata['Spherefit_curvature'] = curvature
    
    return surf    


def get_patch(points, idx_query, center, norm=True):
    """
    Transforms a patch consisting of a query point and its neighbours into a
    coordinate system with the surface normal pointing towards direction (0, 0, 1) (upwards)

    Parameters
    ----------
    points : pandas array with all points on the surface.
        Must have properties XYZ and Normals
    idx_query : int
        index of the point in the dataframe that is queried. The neighbourhood of this
        point will be transformed into a coordinate system with the normal vector being (0,0,1).
    center : length-3 array
        coordinates of dropplet center. Used to determine the orientation of the normal vector

    Returns
    -------
    Nx3 numpy array

    """

    XYZ = np.vstack(points.loc[points.Neighbours[idx_query]].XYZ)
    ctr_patch = np.asarray([XYZ.mean(axis=0)]).repeat(XYZ.shape[0], axis=0)
    Xq = points.XYZ.loc[idx_query]

    # center coordinates (points and query point)
    _XYZ = XYZ - ctr_patch
    ctr_Xq = Xq - ctr_patch[0]

    # get orientation matrix
    S = 1/len(_XYZ) *np.dot(_XYZ.conjugate().T, _XYZ)
    D, R =  np.linalg.eig(S)  # R is orientation matrix

    # Transform to new coordinate frame
    _XYZ_rot = np.dot(_XYZ, R)
    _Xq_rot = np.dot(ctr_Xq, R)

    # Make sure z-normal points upwards
    _S = 1/len(_XYZ) *np.dot(_XYZ_rot.conjugate().T, _XYZ_rot)
    D, R =  np.linalg.eig(_S)

    # If normal is pointing to (0,0,-1), flip orientation upside-down
    if np.sum(R[:, 2]) < 0:
        R_flip  = np.asarray([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        _XYZ_rot = np.dot(_XYZ_rot, R_flip)
        _Xq_rot = np.dot(_Xq_rot, R_flip)

    return _XYZ_rot, _Xq_rot


def triangulate_surface(points):
    """
    Function to triangulate a mesh from a list of points.

    Parameters
    ----------
    points : Nx3 array
        Array holding point coordinates.

    Returns
    -------
    mesh

    """
    CoM = np.mean(points, axis=0)

    _phi, _theta, _r = cart2sph(points[:, 0] - CoM[0],
                                points[:, 1] - CoM[1],
                                points[:, 2] - CoM[2])

    points_spherical = np.vstack([_phi, _theta]).transpose((1,0))
    tri = Delaunay(points_spherical)

    return tri


def get_neighbours(points, patch_radius):
    """
    Find neighbours of each point in 3D within a certain radius

    Parameters
    ----------
    points : Nx3 array
    patch_radius : float
        range around one point within which points are counted as neighbours

    Returns
    -------
    list of indeces that refer to the indices of neighbour points for every point.

    """

    neighbours = []

    tree = cKDTree(points)

    # browse all points and append the indeces of its neighbours to a list
    for idx in range(points.shape[0]):
        neighbours.append(tree.query_ball_point(points[idx], patch_radius))

    N_neighbours = [len(x) for x in neighbours]

    return neighbours, N_neighbours


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
