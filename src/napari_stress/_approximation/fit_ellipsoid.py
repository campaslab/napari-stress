# -*- coding: utf-8 -*-

from napari.types import PointsData, VectorsData
from .._utils.frame_by_frame import frame_by_frame
from .. import __version__
from napari_tools_menu import register_function
import numpy as np

import deprecation


@deprecation.deprecated(
    deprecated_in="0.3.3",
    current_version=__version__,
    removed_in="0.4.0",
    details="Use approximateion.EllipseExpander instead.",
)
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
    coefficients = fit_ellipsoid_to_points(points)
    center, axes_lengths, eigenvectors = fit_ellipsoid(coefficients)

    vectors = (
        eigenvectors / np.linalg.norm(eigenvectors, axis=1)[:, np.newaxis]
    ).T * axes_lengths[:, None]
    base = np.stack([center] * 3)
    ellipsoid = np.stack([base, vectors], axis=1)

    return ellipsoid


@deprecation.deprecated(
    deprecated_in="0.3.3",
    current_version=__version__,
    removed_in="0.4.0",
    details="Use approximateion.EllipseExpander instead.",
)
def fit_ellipsoid_to_points(
    points: "napari.types.PointsData",
) -> "napari.types.VectorsData":
    """
    Fit a 3D ellipsoid to given points using least squares fitting.
    The ellipsoid equation is: Ax^2 + By^2 + Cz^2 + Dxy + Exz + Fyz + Gx + Hy + Iz = 1

    :param points: A numpy array of shape (N, 3) representing the 3D points.
    :return: A numpy array containing the coefficients [A, B, C, D, E, F, G, H, I, J] of the ellipsoid equation.
    """
    # Extract x, y, z coordinates from points and reshape to column vectors
    x = points[:, 0, np.newaxis]
    y = points[:, 1, np.newaxis]
    z = points[:, 2, np.newaxis]

    # Construct the design matrix for the ellipsoid equation
    design_matrix = np.hstack((x**2, y**2, z**2, x * y, x * z, y * z, x, y, z))
    column_of_ones = np.ones_like(x)  # Column vector of ones

    # Perform least squares fitting to solve for the coefficients
    transposed_matrix = design_matrix.transpose()
    matrix_product = np.dot(transposed_matrix, design_matrix)
    inverse_matrix = np.linalg.inv(matrix_product)
    coefficients = np.dot(inverse_matrix, np.dot(transposed_matrix, column_of_ones))

    # Append -1 to the coefficients to represent the constant term on the right side of the equation
    ellipsoid_coefficients = np.append(coefficients, -1)

    return ellipsoid_coefficients


@deprecation.deprecated(
    deprecated_in="0.3.3",
    current_version=__version__,
    removed_in="0.4.0",
    details="Use approximateion.EllipseExpander instead.",
)
def fit_ellipsoid(coefficients):
    # Construct the augmented matrix from the coefficients
    Amat = np.array(
        [
            [
                coefficients[0],
                coefficients[3] / 2.0,
                coefficients[4] / 2.0,
                coefficients[6] / 2.0,
            ],
            [
                coefficients[3] / 2.0,
                coefficients[1],
                coefficients[5] / 2.0,
                coefficients[7] / 2.0,
            ],
            [
                coefficients[4] / 2.0,
                coefficients[5] / 2.0,
                coefficients[2],
                coefficients[8] / 2.0,
            ],
            [coefficients[6] / 2.0, coefficients[7] / 2.0, coefficients[8] / 2.0, -1],
        ]
    )

    # Extract the quadratic part and find its inverse
    A3 = Amat[:3, :3]
    A3inv = np.linalg.inv(A3)

    # Compute the center of the ellipsoid
    ofs = coefficients[6:9] / 2.0
    center = -np.dot(A3inv, ofs)

    # Transform the matrix to center the ellipsoid at the origin
    Tofs = np.eye(4)
    Tofs[3, :3] = center
    R = np.dot(Tofs, np.dot(Amat, Tofs.T))

    # Extract the transformed quadratic part
    R3 = R[:3, :3]

    # Perform eigendecomposition to find axes and orientation
    eigenvalues, eigenvectors = np.linalg.eig(R3 / -R[3, 3])

    # Compute the lengths of the axes
    axes_lengths = np.sqrt(1.0 / np.abs(eigenvalues))

    return center, axes_lengths, eigenvectors


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


@deprecation.deprecated(
    deprecated_in="0.3.3",
    current_version=__version__,
    removed_in="0.4.0",
    details="Use approximateion.EllipseExpander instead.",
)
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


@deprecation.deprecated(
    deprecated_in="0.3.3",
    current_version=__version__,
    removed_in="0.4.0",
    details="Use approximateion.EllipseExpander instead.",
)
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

    U, V = cartesian_to_elliptical(fitted_ellipsoid, pointcloud, invert=True)
    points_on_fitted_ellipse = elliptical_to_cartesian(
        U, V, fitted_ellipsoid, invert=True
    )

    return points_on_fitted_ellipse
