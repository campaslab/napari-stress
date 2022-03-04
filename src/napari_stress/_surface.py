# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import cKDTree, Delaunay

from ._utils import cart2sph

import vedo



def reconstruct_surface(points: np.ndarray,
                        dims: np.ndarray,
                        n_smooth:int = 5) -> list:
    
    # Check if data is 4D and reformat into list of arrays for every frame
    if points.shape[1] == 4:
        timepoints = np.unique(points[:, 0])
        _points = [points[np.where(points[:, 0] == i)][:, 1:] for i in timepoints]
    else:
        _points = [points]        
        
    surfs = []
    for pts in _points:
        
        # Get points and filter
        pts4vedo = vedo.Points(pts)
        pts_filtered = vedo.pointcloud.removeOutliers(pts4vedo, radius=4)
        
        # Reconstruct surface
        surf = vedo.pointcloud.recoSurface(pts_filtered, dims=dims)
        surf.clean(tol=0.05).smooth().computeNormals()
        
        # Calculate curvature
        surf.addCurvatureScalars(method=1)  #0-gaussian, 1-mean, 2-max, 3-min curvature.
        
        surfs.append(surf.clone())
    
    return surfs


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
