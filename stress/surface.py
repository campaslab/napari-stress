# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 13:31:56 2021

@author: johan
"""

from skimage.transform import rescale, resize
import numpy as np
import tqdm
from scipy import interpolate
from scipy.spatial import cKDTree
import pandas as pd

from stress.utils import cart2sph, get_IQs

def get_local_normals(points, **kwargs):
    """
    
    Calculate normal vectors for points on a curved surface
    Parameters
    ----------
    points : TYPE
        DESCRIPTION.
    patch_radius : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    patch_radius = kwargs.get('patch_radius', 2.5)
    
    points = get_neighbours(points, patch_radius=patch_radius)
    center = np.asarray([points.X.mean(), points.Y.mean(), points.Z.mean()])
    
    # find the patch center for each point and its neighbours
    normals = []
    for i, point in tqdm.tqdm(points.iterrows(), desc='Finding normals'):
        neighbours = point.Neighbours
        x_nghb = points.iloc[neighbours].X
        y_nghb = points.iloc[neighbours].Y
        z_nghb = points.iloc[neighbours].Z
        
        # get the center in space of each patch
        patch_center_x = x_nghb.mean()
        patch_center_y = y_nghb.mean()
        patch_center_z = z_nghb.mean()
        
        # calculate centered coordinates
        x = x_nghb - patch_center_x
        y = y_nghb - patch_center_y
        z = z_nghb - patch_center_z
        X = np.asarray([x, y, z]).T  # must be Nx3
        
        # calculate eigenvalues for patch coordinates to find normals
        S = 1/len(x) * np.dot(X.conjugate().T, X)
        D, R = np.linalg.eig(S)
        
        # check all identifed vectors to see which one points out
        cross_prod = []
        direction = []
        for j in range(3):
            v = R[:, j]
            
            # calculate vector from center to patch center
            vcp = np.asarray([patch_center_x, patch_center_y, patch_center_z]) - center
            vcp = vcp/np.linalg.norm(vcp)
            
            # calculate dot product to see if vector points outward
            direction.append(np.sign(np.dot(vcp, v)))
            
            # calculate norm of cross product to see if eigenvector is aligned with vector from center
            cross_prod.append(np.linalg.norm(np.cross(v, vcp)))
            
        idx = np.argmin(cross_prod)
        if direction[idx] >= 0:
            nV = R[:, idx]
        else:
            nV = -R[:, idx]
        
        normals.append(nV)
    
    points['Normals'] = normals
    
    return points
    

def resample_surface(STRESS):
    
    # Parse input
    points = STRESS.points
    sampling_length = STRESS.surface_sampling_density
    
    # First find center of currently provided coordinates
    center = STRESS.get_center()
    
    # calculate centered coordinates
    x = STRESS.get_x() - center[0]
    y = STRESS.get_y() - center[1]
    z = STRESS.get_z() - center[2]
    
    _phi, _theta, _r = cart2sph(x, y, z)
    
    # Assume all points are on a sphere:
    surface_area = 4 * np.pi * _r.mean()**2
    N_points = surface_area//(sampling_length**2)
    
    #Create new fibonacci sphere with more points
    XYZ = fibonacci_sphere(semiAxesLengths=None,
                           rot_matrix=None,
                           center=None,
                           samples=N_points)
    xnew, ynew,znew = XYZ[:,0], XYZ[:,1], XYZ[:,2]
    
    # We are looking for the radii of fibonacci-distributed points, which we can interpolate from the
    # ray-traced surface points. Instead of interpolating the radius at a higher surface density,
    # interpolate the componentns (x,y,z) of the radius at higher surface density. The interpolation 
    # is done in spherical coordinates.
    Fx = interpolate.LinearNDInterpolator(np.vstack([_phi, _theta]).transpose(), x)
    Fy = interpolate.LinearNDInterpolator(np.vstack([_phi, _theta]).transpose(), y)
    Fz = interpolate.LinearNDInterpolator(np.vstack([_phi, _theta]).transpose(), z)
    
    phi, theta, r = cart2sph(xnew, ynew, znew)
    x = Fx(phi, theta)
    y = Fy(phi, theta)
    z = Fz(phi, theta)
    
    # Return as dataframe
    _points = pd.DataFrame(columns=points.columns)
    
    # Translate to non-centered coordinates
    x = x + center[0]
    y = y + center[1]
    z = z + center[2]
    _points['X'] = x
    _points['Y'] = y
    _points['Z'] = z
    _points.dropna(subset=['X', 'Y', 'Z'], axis=0, inplace=True)
    _points['XYZ'] = list(np.vstack([_points.X, _points.Y, _points.Z]).transpose())
    
    return _points
    

# def clean_points(STRESS):
    
#     # Get input
#     points = STRESS.points
#     patch_radius = STRESS.patch_radius
#     scale = STRESS.point_filter_scale

#     N = len(points)
    
#     # allocate index arrays that will show whether a point will be accpeted or not
#     idx_neighbours = [True] * N
#     idx_error = [True] * N
#     idx_fitparams = [True] * N

def clean_coordinates(STRESS):
    """
    Browse all coordinates and identify their respective neighbouring points within a certain radius
        1. If provided, error values from the curve fit process are used to eliminate
            points with errors above median + interquartile range
        2. If provided fit parameters from the previous step are used to elimate points
            with fit parameter outside the interquartile range

    Parameters
    ----------
    STRESS : stres job instance

    Returns
    -------
    STRESS : job instance

    """
    
    points = STRESS.points
    patch_radius = STRESS.patch_radius
    scale = STRESS.point_filter_scale
    
    # patch_radius = kwargs.get('patch_radius', 2)
    # scale = kwargs.get('scale', 1.5)
    
    # get neighbours of points
    points = get_neighbours(points, patch_radius=patch_radius)
    
    N = len(points)
    
    # allocate index arrays that will show whether a point will be accpeted or not
    idx_neighbours = [True] * N
    idx_error = [True] * N
    idx_fitparams = [True] * N
    idx_curvature = [True] * N
    idx_accept = [True] * N
    
    # Filter according to errors. If no errors are measured, they are all zero.
    if np.sum(points.FitErrors) != 0:
        I25, median, I75, IQ = get_IQs(np.log(points.FitErrors))
        idx_error = np.log(points.FitErrors) <= I75 + scale * IQ  # remove points with outlier errors
        idx_accept *= idx_error
        
    # Filter according to fit params, if provided
    if points.FitParams.sum() != 0:
        
        fitparams = np.asarray(points.FitParams)
        N_params = np.min(fitparams.shape)
        for i in range(N_params):
            
            params = fitparams[:, i]  # look at i-th fit parameter
            I25, median, I75, IQ = get_IQs(params)  # calc quartiles
            idx_fitparams *= (params <= I75 + scale*IQ) * (params >= I25 - scale*IQ)  # all fitparams must be in check.
        idx_accept *= idx_fitparams
            
    # Filter according to curvature
    if 'Curvature' in points.columns:
        if points['Curvature'].sum() != 0:
            curvs = np.vstack(points.Curvature).squeeze()
            I25, median, I75, IQ = get_IQs(curvs)  # calc quartiles
            idx_curvature *= (curvs <= I75 + scale*IQ) * (curvs >= I25 - scale*IQ)
            
            idx_accept *= idx_curvature
    
    # Filter according to neighbourhood (remove points with hardly any neighbours)
    N_neighbours = np.asarray(points.N_neighbours)
    I25, median, I75, IQ = get_IQs(N_neighbours)
    idx_neighbours = N_neighbours >= I25 - scale*IQ
    idx_accept *= idx_neighbours
            
    points = points[idx_accept]
    
    print(f'{np.sum(idx_accept)} out of {N} points were accepted for further processing.')
    
    return points.reset_index(drop=True)
            
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
    
    
    

def get_neighbours(points, patch_radius):
    """
    Find neighbours of each point in 3D within a certain radius

    Parameters
    ----------
    points : pandas dataframe
        dataframe with columns X, Y and Z
    patch_radius : float
        range around one point within which points are counted as neighbours

    Returns
    -------
    list of indeces that refer to the indices of neighbour points for every point.

    """
    
    neighbours = []
    
    XYZ = np.vstack(points.XYZ)
    tree = cKDTree(XYZ)
    
    # browse all points and append the indeces of its neighbours to a list
    for i, point in points.iterrows():
        neighbours.append(tree.query_ball_point(point.XYZ, patch_radius))
    
    # Add the indeces of the neighbours and the number of neighbours to the dataframe
    points['Neighbours'] = neighbours
    points['N_neighbours'] = [len(x) for x in points['Neighbours']]
    
    return points
    
    

def resample(image, vsx, vsy, vsz, res_mode='high'):
    """
    Resamples an image with anistropic voxels of size vsx, vsy and vsz to isotropic
    voxel size of smallest or largest resolution
    """
    
    # choose final voxel size
    if res_mode == 'high':
        vs = np.min([vsx, vsy, vsz])
        
    elif res_mode == 'low':
        vs = np.max([vsx, vsy, vsz])
        
    factor = np.asarray([vsx, vsy, vsz])/vs
    
    image_rescaled = rescale(image, factor, anti_aliasing=True)
    
    return image_rescaled
    
    
def fibonacci_sphere(semiAxesLengths, rot_matrix, center, samples=2**8):
    
    samples = int(samples)
    
    # distribute fibonacci points
    z = np.linspace(1 - 1.0/samples, 1.0/samples - 1, samples)
    radius = np.sqrt(1.0 - z**2)
    goldenAngle = np.pi * (3.0 - np.sqrt(5.0))
    theta = goldenAngle * np.arange(0, samples, 1)  # golden angle increment
    
    if semiAxesLengths is None:
        semiAxesLengths = np.asarray([1.0, 1.0, 1.0])
    
    XYZ_local = np.zeros((samples, 3))
    XYZ_local[:, 0] = semiAxesLengths[0] * radius * np.cos(theta)
    XYZ_local[:, 1] = semiAxesLengths[1] * radius * np.sin(theta)
    XYZ_local[:, 2] = semiAxesLengths[2] * z
    
    # rotate ellipse points
    if rot_matrix is not None:    
        XYZ_rot = np.dot(rot_matrix, XYZ_local.T)
    else:
        XYZ_rot = XYZ_local

    # translate ellipse points from 0 to center    
    if center is not None:
        t = np.asarray([center for i in range(0, samples)])
        XYZ = XYZ_rot + t.T
    else:
        XYZ = XYZ_rot
        
    if XYZ.shape[0] == 3:
        XYZ = XYZ.transpose()
    
    return XYZ