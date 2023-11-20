# -*- coding: utf-8 -*-

from napari.types import PointsData, VectorsData
from .._utils.frame_by_frame import frame_by_frame
from napari_tools_menu import register_function
import numpy as np


@register_function(menu="Points > Fit ellipsoid to pointcloud (n-STRESS)")
@frame_by_frame
def least_squares_ellipsoid(points: PointsData) -> VectorsData:
    """
    Fit ellipsoid to points with a last-squares approach.

    This function takes a pointcloud and fits an ellipsoid to it using a
    least-squares approach. The ellipsoid is returned as a set of vectors
    representing the major and minor axes of the ellipsoid.

    Parameters
    ----------
    points : PointsData

    Returns
    -------
    VectorsData: Major/minor axis of the ellipsoid
    """
    from .._utils.coordinate_conversion import polynomial_to_parameters3D

    # # Formulate the problem as a linear equation Ax = b
    # A = np.column_stack((points[:, 0]**2,  # A
    #                     points[:, 1]**2,  # B
    #                     points[:, 2]**2,  # C
    #                     2 * points[:, 0] * points[:, 1],  # D
    #                     2 * points[:, 0] * points[:, 2],  # E
    #                     2 * points[:, 1] * points[:, 2],  # F
    #                     2 * points[:, 0],  # G
    #                     2 * points[:, 1],  # H
    #                     2 * points[:, 2]))  # I

    # b = np.ones(points.shape[0])

    # # Use least squares solver to find the coefficients
    # coefficients, _, _, _= np.linalg.lstsq(A, b, rcond=None)

    # # Extract the ellipsoid parameters
    # A, B, C, D, E, F, G, H, I = coefficients

    # # Calculate the center of the ellipsoid
    # center = np.array([-G / A, -H / B, -I / C])

    # # Calculate the major axes lengths
    # a = 1 / np.sqrt(A)
    # b = 1 / np.sqrt(B)
    # c = 1 / np.sqrt(C)

    # # Calculate the major axes directions
    # cov_matrix = np.cov(points, rowvar=False)
    # eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # # Sort the eigenvalues and eigenvectors in descending order
    # sorted_indices = np.argsort(eigenvalues)[::-1]
    # eigenvalues = eigenvalues[sorted_indices]
    # eigenvectors = eigenvectors[:, sorted_indices]

    # # Assemble the (3x2x3) array
    # result_matrix = np.zeros((3, 2, 3))

    # for i in range(3):
    #     result_matrix[i, 0, 0] = center[2]  # Base point (z)
    #     result_matrix[i, 0, 1] = center[1]  # Base point (y)
    #     result_matrix[i, 0, 2] = center[0]  # Base point (x)

    #     result_matrix[i, 1, 0] = eigenvectors[i, 2] * a  # Direction (z)
    #     result_matrix[i, 1, 1] = eigenvectors[i, 1] * b  # Direction (y)
    #     result_matrix[i, 1, 2] = eigenvectors[i, 0] * c  # Direction (x)

    coefficients = _solve_ellipsoid_polynomial(points)

    # convert results to VectorsData
    center, axes, R, R_inverse = polynomial_to_parameters3D(coefficients=coefficients)
    direction = R * axes[:, None]
    origin = np.stack(3 * [center])  # cheap repeat
    vector = np.stack([origin, direction]).transpose((1, 0, 2))

    return vector


@register_function(menu="Points > Calculate normals on ellipsoid (n-STRESS)")
@frame_by_frame
def normals_on_ellipsoid(points: PointsData) -> VectorsData:
    """
    Fits an ellipsoid and calculates the normals vectors.

    This function takes a pointcloud and calculates the normals on the
    ellipsoid fitted to the pointcloud.

    Parameters
    ----------
    points : PointsData

    Returns
    -------
    VectorsData: Normals on the ellipsoid
    """
    coefficients = _solve_ellipsoid_polynomial(points)

    A = coefficients.flatten()[0]
    B = coefficients.flatten()[1]
    C = coefficients.flatten()[2]
    D = coefficients.flatten()[3]
    E = coefficients.flatten()[4]
    F = coefficients.flatten()[5]
    G = coefficients.flatten()[6]
    H = coefficients.flatten()[7]
    J = coefficients.flatten()[8]

    xx = points[:, 0][:, None]
    yy = points[:, 1][:, None]
    zz = points[:, 2][:, None]

    grad_F_x = 2.0 * A * xx + D * yy + E * zz + G
    grad_F_y = 2.0 * B * yy + D * xx + F * zz + H
    grad_F_z = 2.0 * C * zz + E * xx + F * yy + J

    grad_F_X = np.hstack((grad_F_x, grad_F_y, grad_F_z))
    Vec_Norms = np.sqrt(np.sum(np.multiply(grad_F_X, grad_F_X), axis=1)).reshape(
        len(xx), 1
    )
    grad_F_X_normed = np.divide(grad_F_X, Vec_Norms)

    return np.stack([points, grad_F_X_normed]).transpose((1, 0, 2))


def _solve_ellipsoid_polynomial(points: PointsData) -> np.ndarray:
    """
    Fit ellipsoid polynomial equation.

    Found at http://www.juddzone.com/ALGORITHMS/least_squares_3D_ellipsoid.html
    least squares fit to a 3D-ellipsoid
     Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz  = 1

    Note that sometimes it is expressed as a solution to
     Ax^2 + By^2 + Cz^2 + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz  = 1
    where the last six terms have a factor of 2 in them
    This is in anticipation of forming a matrix with the polynomial
    coefficients. Those terms with factors of 2 are all off diagonal
    elements. These contribute two terms when multiplied out
    (symmetric) so would need to be divided by two
    """

    # change xx from vector of length N to Nx1 matrix so we can use hstack
    x = points[:, 0, np.newaxis]
    y = points[:, 1, np.newaxis]
    z = points[:, 2, np.newaxis]

    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz = 1
    J = np.hstack((x * x, y * y, z * z, x * y, x * z, y * z, x, y, z))
    K = np.ones_like(x)  # column of ones

    JT = J.transpose()
    JTJ = np.dot(JT, J)
    InvJTJ = np.linalg.inv(JTJ)
    ABC = np.dot(InvJTJ, np.dot(JT, K))

    # Rearrange, move the 1 to the other side
    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz - 1 = 0
    #    or
    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz + J = 0
    #  where J = -1
    eansa = np.append(ABC, -1)

    return eansa


@register_function(menu="Points > Expand point locations on ellipsoid (n-STRESS)")
@frame_by_frame
def expand_points_on_ellipse(
    fitted_ellipsoid: VectorsData, pointcloud: PointsData
) -> PointsData:
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
        elliptical_to_cartesian,
    )

    U, V = cartesian_to_elliptical(fitted_ellipsoid, pointcloud)
    points_on_fitted_ellipse = elliptical_to_cartesian(U, V, fitted_ellipsoid)

    return points_on_fitted_ellipse
