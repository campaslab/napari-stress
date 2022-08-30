# -*- coding: utf-8 -*-
import numpy as np
from napari.types import PointsData, VectorsData
from typing import Tuple

from scipy.spatial.transform import Rotation

import vedo


def _center_from_ellipsoid(ellipsoid: VectorsData) -> np.ndarray:
    return ellipsoid[:, 0].mean(axis=0)

def _axes_lengths_from_ellipsoid(ellipsoid: VectorsData) -> np.ndarray:
    # first, get lengths of ellipsoid axes:
    return np.linalg.norm(ellipsoid[:, 1], axis=1)

def _orientation_from_ellipsoid(ellipsoid: VectorsData) -> np.ndarray:
    lengths = _axes_lengths_from_ellipsoid(ellipsoid)
    return (ellipsoid[:, 1]/ lengths[:, None])

def polynomial_to_parameters3D(coefficients: np.ndarray):
    #gets 3D parameters of an ellipsoid. Found at http://www.juddzone.com/ALGORITHMS/least_squares_3D_ellipsoid.html
    # convert the polynomial form of the 3D-ellipsoid to parameters
    # center, axes, and transformation matrix
    # vec is the vector whose elements are the polynomial
    # coefficients A..J
    # returns (center, axes, rotation matrix)

    #Algebraic form: X.T * Amat * X --> polynomial form


    Amat=np.array(
        [
        [ coefficients[0],     coefficients[3]/2.0, coefficients[4]/2.0, coefficients[6]/2.0 ],
        [ coefficients[3]/2.0, coefficients[1],     coefficients[5]/2.0, coefficients[7]/2.0 ],
        [ coefficients[4]/2.0, coefficients[5]/2.0, coefficients[2],     coefficients[8]/2.0 ],
        [ coefficients[6]/2.0, coefficients[7]/2.0, coefficients[8]/2.0, coefficients[9]     ]
        ]
    )

    #See B.Bartoni, Preprint SMU-HEP-10-14 Multi-dimensional Ellipsoidal Fitting
    # equation 20 for the following method for finding the center
    A3 = Amat[0:3,0:3]
    A3inv = np.linalg.inv(A3)
    ofs = coefficients[6:9]/2.0
    center = -np.dot(A3inv,ofs)

    # Center the ellipsoid at the origin
    Tofs = np.eye(4)
    Tofs[3,0:3] = center
    R = np.dot(Tofs,np.dot(Amat,Tofs.T))

    R3 = R[0:3,0:3]
    R3test = R3/R3[0,0]
    # print('normed \n',R3test)
    s1 = -R[3, 3]
    R3S = R3/s1
    (el,ec) = np.linalg.eig(R3S)

    recip = 1.0/np.abs(el)
    axes=np.sqrt(recip)

    inve=np.linalg.inv(ec) #inverse is actually the transpose here

    return (center, axes, inve, ec)

# Convert input R^3 points into Ellipsoidal coors:
def cartesian_to_elliptical(ellipsoid: VectorsData,
                            points: PointsData) -> np.ndarray:

    # first, get lengths of ellipsoid axes:
    lengths = np.linalg.norm(ellipsoid[:, 1], axis=1)

    # get center of ellipsoid
    center = _center_from_ellipsoid(ellipsoid)
    lengths = _axes_lengths_from_ellipsoid(ellipsoid)
    R_inverse = _orientation_from_ellipsoid(ellipsoid).T

    n_points = len(points)
    U_coors_calc = np.zeros(( n_points, 1 ))
    V_coors_calc = np.zeros(( n_points, 1 ))

    for pt_numb in range(n_points):
        y_tilde_pt = np.linalg.solve(R_inverse,
                               points[pt_numb, :].reshape(3,1) - center.reshape(3, 1)
                               )

        yt_0 = y_tilde_pt[0,0]
        yt_1 = y_tilde_pt[1,0]
        yt_2 = y_tilde_pt[2,0]

        U_pt = np.arctan2(yt_1 * lengths[0],
                          yt_0 * lengths[1] )

        if(U_pt < 0):
            U_pt = U_pt + 2.*np.pi

        U_coors_calc[pt_numb] = U_pt

        cylinder_r = np.sqrt(yt_0**2 + yt_1**2)    # r in cylinderical coors for y_tilde
        cyl_r_exp = np.sqrt(
            (lengths[0] * np.cos(U_pt))**2 + (lengths[1]*np.sin(U_pt))**2
            )

        V_pt = np.arctan2(cylinder_r * lengths[2],
                          yt_2 * cyl_r_exp )

        if(V_pt < 0):
            V_pt = V_pt + 2.*np.pi

        V_coors_calc[pt_numb] = V_pt

    return U_coors_calc, V_coors_calc

# Convert Ellipsoidal Coordinates to R^3 points:
def elliptical_to_cartesian(U_pts_cloud: np.ndarray,
                            V_pts_cloud: np.ndarray,
                            ellipsoid: VectorsData) -> PointsData:

    R_inverse = _orientation_from_ellipsoid(ellipsoid).T
    lengths = _axes_lengths_from_ellipsoid(ellipsoid)
    center = _center_from_ellipsoid(ellipsoid)

    num_pts_used = len(U_pts_cloud)
    X_LS_Ellps_calc_pts = np.zeros(( num_pts_used, 1 ))
    Y_LS_Ellps_calc_pts = np.zeros(( num_pts_used, 1 ))
    Z_LS_Ellps_calc_pts = np.zeros(( num_pts_used, 1 ))

    for pt_test in range(num_pts_used): #pt_test_theta in range(num_test_pts_per_dim):
        theta_tp =  U_pts_cloud[pt_test, 0] #theta_pts[pt_test_theta]
        phi_tp = V_pts_cloud[pt_test, 0] #phi_pts[pt_test_phi]
        pt = np.array([[np.sin(phi_tp) * np.cos(theta_tp)],
                       [np.sin(phi_tp) * np.sin(theta_tp)],
                       [np.cos(phi_tp)] ])
        Test_pt = np.dot(R_inverse, np.multiply(pt, lengths.reshape(3, 1) )  ) +\
            center.reshape(3, 1)

        X_LS_Ellps_calc_pts[pt_test, 0] = Test_pt[0, 0]
        Y_LS_Ellps_calc_pts[pt_test, 0] = Test_pt[1, 0]
        Z_LS_Ellps_calc_pts[pt_test, 0] = Test_pt[2, 0]

    pts = np.stack([X_LS_Ellps_calc_pts, Y_LS_Ellps_calc_pts, Z_LS_Ellps_calc_pts])
    return pts.squeeze().transpose()
