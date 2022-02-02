# -*- coding: utf-8 -*-

from skimage.transform import rescale
import numpy as np
import tqdm
from scipy import interpolate
from scipy.spatial import cKDTree

from ._utils import cart2sph, get_IQs

import pandas as pd

# def resample_surface(image, points, center, n_refinements=2, **kwargs):
#     """
#     Resamples points on the dropplet surface to higher density
#     """

#     trace_fit_method = kwargs.get('trace_fit_method', 'quick_edge')
#     n_refinements = kwargs.get('n_refinements', 2)
#     fluorescence = kwargs.get('fluorescence', 'interior')

#     print('\n---- Refinement-----')

#     for i in range(n_refinements):
#         print(f'Iteration #{i+1}:')

#         self.points = resample_surface(self)

#         # Calculate new center of mass
#         XYZ = np.vstack(self.points.XYZ)
#         self.CoM = np.asarray([XYZ[:,0].mean(), XYZ[:,1].mean(), XYZ[:,2].mean()])

#         self.points = get_local_normals(self.points)

#         # Calculate starting points for advanced tracing
#         start_pts = self.points.XYZ - 2*self.points.Normals

#         self.points = tracing.get_traces(self.image,
#                                          self.points,
#                                          start_pts=start_pts,
#                                          target_pts=self.points.XYZ,
#                                          detection=self.trace_fit_method,
#                                          fluorescence=self.fluorescence)

#         self.points = surface.clean_coordinates(self)

#     # Raise flags for provided data
#     self.has_normals = True

def get_local_normals(points, **kwargs):
    """

    Calculate normal vectors for points on a curved surface
    Parameters
    ----------
    points : Nx3 numpy array
        3D coordinates of points
    patch_radius : radius within which
        DESCRIPTION.

    Returns
    -------
    None.

    """

    patch_radius = kwargs.get('patch_radius', 2.5)

    neighbours, n_neighbours = get_neighbours(points, patch_radius=patch_radius)
    center = np.mean(points, axis=0)

    df = pd.DataFrame(columns = ['Z', 'Y', 'X', 'neighbours', 'n_neighbours'])
    df['Z'] = points[:, 0]
    df['Y'] = points[:, 1]
    df['X'] = points[:, 2]
    df['neighbours'] = neighbours
    df['n_neighbours'] = n_neighbours

    # find the patch center for each point and its neighbours
    normals = []
    for idx, point in tqdm.tqdm(df.iterrows(), desc='Finding normals'):
        neighbours = point.neighbours
        x_nghb = df.iloc[neighbours].X
        y_nghb = df.iloc[neighbours].Y
        z_nghb = df.iloc[neighbours].Z

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

    normals = np.vstack(normals)

    # Transpose ax order if not Nx3
    if normals.shape[0] == 3:
        normals = normals.transpose()

    return normals


def resample_points(points, **kwargs):
    """
    Resample input points on a surface to a predefined density on the surface

    Parameters
    ----------
    points : Nx3 array
        Input array with point coordinates in 3D

    **surface_sampling_density**: designated density of points on the surface. Default = 1

    Returns
    -------
    points : Nx3 array
        Array with coordinates of new, resampled points

    """
    # Parse input
    sampling_length = kwargs.get('surface_sampling_density', 1)

    # First find center of currently provided coordinates
    center = np.mean(points, axis=0)

    # calculate centered coordinates
    x = points[:, 0] - center[0]
    y = points[:, 1] - center[1]
    z = points[:, 2] - center[2]

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

    # Translate to non-centered coordinates
    x = x + center[0]
    y = y + center[1]
    z = z + center[2]

    # Convert to Nx3 array
    points = np.vstack([x, y, z]).transpose()

    # Make sure there are no nans
    points = points[~np.isnan(points).any(axis=1), :]

    return points


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

# def clean_coordinates(points, **kwargs):
#     """
#     Browse all coordinates and identify their respective neighbouring points within a certain radius
#         1. If provided, error values from the curve fit process are used to eliminate
#             points with errors above median + interquartile range
#         2. If provided fit parameters from the previous step are used to elimate points
#             with fit parameter outside the interquartile range

#     Parameters
#     ----------
#     STRESS : stres job instance

#     Returns
#     -------
#     STRESS : job instance

#     """


#     patch_radius = kwargs.get('patch_radius', 2)
#     scale = kwargs.get('scale', 1.5)

#     # get neighbours of points
#     neighbours, n_neighbours = get_neighbours(utils.df2ZYX(points), patch_radius=patch_radius)

#     N = len(points)

#     # allocate index arrays that will show whether a point will be accpeted or not
#     idx_neighbours = [True] * N
#     idx_error = [True] * N
#     idx_fitparams = [True] * N
#     idx_curvature = [True] * N
#     idx_accept = [True] * N

#     # Filter according to errors. If no errors are measured, they are all zero.
#     if np.sum(points.FitErrors) != 0:
#         I25, median, I75, IQ = get_IQs(np.log(points.FitErrors))
#         idx_error = np.log(points.FitErrors) <= I75 + scale * IQ  # remove points with outlier errors
#         idx_accept *= idx_error

#     # Filter according to fit params, if provided
#     if points.FitParams.sum() != 0:

#         fitparams = np.asarray(points.FitParams)
#         N_params = np.min(fitparams.shape)
#         for i in range(N_params):

#             params = fitparams[:, i]  # look at i-th fit parameter
#             I25, median, I75, IQ = get_IQs(params)  # calc quartiles
#             idx_fitparams *= (params <= I75 + scale*IQ) * (params >= I25 - scale*IQ)  # all fitparams must be in check.
#         idx_accept *= idx_fitparams

#     # Filter according to curvature
#     if 'Curvature' in points.columns:
#         if points['Curvature'].sum() != 0:
#             curvs = np.vstack(points.Curvature).squeeze()
#             I25, median, I75, IQ = get_IQs(curvs)  # calc quartiles
#             idx_curvature *= (curvs <= I75 + scale*IQ) * (curvs >= I25 - scale*IQ)

#             idx_accept *= idx_curvature

#     # Filter according to neighbourhood (remove points with hardly any neighbours)
#     N_neighbours = np.asarray(points.N_neighbours)
#     I25, median, I75, IQ = get_IQs(N_neighbours)
#     idx_neighbours = N_neighbours >= I25 - scale*IQ
#     idx_accept *= idx_neighbours

#     points = points[idx_accept]

#     print(f'{np.sum(idx_accept)} out of {N} points were accepted for further processing.')

#     return points.reset_index(drop=True)

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
