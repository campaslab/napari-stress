# -*- coding: utf-8 -*-

from napari.types import PointsData, VectorsData
from .._utils.frame_by_frame import frame_by_frame
from napari_tools_menu import register_function

import numpy as np

@register_function(menu="Points > Fit ellipsoid to pointcloud (n-STRESS)")
@frame_by_frame
def least_squares_ellipsoid(points: PointsData
                            ) -> VectorsData:
    """
    Fit ellipsoid to points with a last-squares approach.

    Parameters
    ----------
    points : PointsData

    Returns
    -------
    VectorsData: Major/minor axis of the ellipsoid
    """
    from .._utils.coordinate_conversion import polynomial_to_parameters3D
    coefficients = solve_ellipsoid_polynomial(points)    

    # convert results to VectorsData
    center, axes, R, R_inverse = polynomial_to_parameters3D(coefficients=coefficients)
    direction = R * axes[:, None]
    origin = np.stack(3 * [center])  # cheap repeat
    vector = np.stack([origin, direction]).transpose((1,0,2))

    return vector

@register_function(menu="Points > Calculate normals on ellipsoid (n-STRESS)")
@frame_by_frame
def normals_on_ellipsoid(points: PointsData) -> VectorsData:
    """Retrieve normal vector on ellipsoid points."""
    coefficients = solve_ellipsoid_polynomial(points)

    A = coefficients.flatten()[0]
    B = coefficients.flatten()[1]	
    C = coefficients.flatten()[2]	
    D = coefficients.flatten()[3]	
    E = coefficients.flatten()[4]	
    F = coefficients.flatten()[5]	
    G = coefficients.flatten()[6]	
    H = coefficients.flatten()[7]	
    I = coefficients.flatten()[8]		

    xx = points[:, 0][:, None]
    yy = points[:, 1][:, None]
    zz = points[:, 2][:, None]

    grad_F_x = 2.*A*xx + D*yy + E*zz + G
    grad_F_y = 2.*B*yy + D*xx + F*zz + H 
    grad_F_z = 2.*C*zz + E*xx + F*yy + I

    grad_F_X = np.hstack(( grad_F_x, grad_F_y, grad_F_z )) 
    Vec_Norms = np.sqrt(np.sum(np.multiply(grad_F_X, grad_F_X), axis = 1)).reshape(len(xx), 1)
    grad_F_X_normed = np.divide(grad_F_X, Vec_Norms)
	
    return np.stack([points, grad_F_X_normed]).transpose((1,0,2))


def solve_ellipsoid_polynomial(points: PointsData) -> np.ndarray:
    """
    Fit ellipsoid polynomial equation.
    
    finds best fit ellipsoid. Found at http://www.juddzone.com/ALGORITHMS/least_squares_3D_ellipsoid.html
    least squares fit to a 3D-ellipsoid
     Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz  = 1
    
    Note that sometimes it is expressed as a solution to
     Ax^2 + By^2 + Cz^2 + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz  = 1
    where the last six terms have a factor of 2 in them
    This is in anticipation of forming a matrix with the polynomial coefficients.
    Those terms with factors of 2 are all off diagonal elements.  These contribute
    two terms when multiplied out (symmetric) so would need to be divided by two
    """    

    # change xx from vector of length N to Nx1 matrix so we can use hstack
    x = points[:, 0, np.newaxis]
    y = points[:, 1, np.newaxis]
    z = points[:, 2, np.newaxis]

    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz = 1
    J = np.hstack((x*x,y*y,z*z,x*y,x*z,y*z, x, y, z))
    K = np.ones_like(x) #column of ones

    #np.hstack performs a loop over all samples and creates
    #a row in J for each x,y,z sample:
    # J[ix,0] = x[ix]*x[ix]
    # J[ix,1] = y[ix]*y[ix]
    # etc.

    JT=J.transpose()
    JTJ = np.dot(JT,J)
    InvJTJ = np.linalg.inv(JTJ);
    ABC= np.dot(InvJTJ, np.dot(JT,K)) #!!!! LOOK AT RESIDUALS TO GET ELLIPSOID ERRORS !!!!#

    # Rearrange, move the 1 to the other side
    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz - 1 = 0
    #    or
    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz + J = 0
    #  where J = -1
    eansa = np.append(ABC,-1)

    return eansa

    


@register_function(menu="Points > Expand point locations on ellipsoid (n-STRESS)")
@frame_by_frame
def expand_points_on_ellipse(fitted_ellipsoid: VectorsData,
                             pointcloud: PointsData) -> PointsData:
    """
    Expand a pointcloud on the surface of a fitted ellipse.

    This function takes a ellipsoid (in the form of the major axes) and a
    pointcloud from which the ellipsoid was derived. The coordinates of the
    points in the pointcloud are then transformed to their corresponding
    locations on the surface of the fitted ellipsoid.

    Parameters
    ----------
    fitted_ellipsoid : VectorsData
    pointcloud : PointsData

    Returns
    -------
    PointsData

    """
    from .._utils.coordinate_conversion import (
        cartesian_to_elliptical,
        elliptical_to_cartesian
        )

    U, V = cartesian_to_elliptical(fitted_ellipsoid, pointcloud)
    points_on_fitted_ellipse =  elliptical_to_cartesian(U, V, fitted_ellipsoid)

    return points_on_fitted_ellipse

@register_function(menu="Points > Pairwise distances (n-STRESS)")
@frame_by_frame
def pairwise_point_distances(points: PointsData,
                             fitted_points: PointsData) -> VectorsData:
    """
    Calculate pairwise distances between pointclouds.

    Parameters
    ----------
    points : PointsData
    fitted_points : PointsData

    Raises
    ------
    ValueError
        Both pointclouds must have the same number of points.

    Returns
    -------
    VectorsData

    """
    if not len(points) == len(fitted_points):
        raise ValueError('Both pointclouds must have same length, but had'
                         f'{len(points)} and {len(fitted_points)}.')

    delta = points - fitted_points
    return np.stack([fitted_points, delta]).transpose((1,0,2))
