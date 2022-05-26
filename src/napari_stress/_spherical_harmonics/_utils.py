# -*- coding: utf-8 -*-
import numpy as np
from napari.types import PointsData
from typing import Tuple

from scipy.spatial.transform import Rotation

import vedo

from .._spherical_harmonics import sph_func_SPB as sph_f
from .._spherical_harmonics import lbdv_info_SPB as lbdv_i
from .._spherical_harmonics import euc_k_form_SPB as euc_kf
from .._spherical_harmonics import manifold_SPB as mnfd

# return least squares harmonic fit to point cloud, given choice of basis and degree:
def Least_Squares_Harmonic_Fit(fit_degree: int,
                               points_ellipse_coords: tuple,
                               input_points: PointsData,
                               use_true_harmonics: bool = True) -> np.ndarray:
    """
    Perform least squares fit of spherical harmonic basis functions.
    """

    U, V = points_ellipse_coords[0], points_ellipse_coords[1]

    All_Y_mn_pt_in = []

    for n in range(fit_degree + 1):
        for m in range(-1*n, n+1):
            Y_mn_coors_in = []
            Y_mn_coors_in = lbdv_i.Eval_SPH_Basis(m, n, U, V)
            All_Y_mn_pt_in.append(Y_mn_coors_in)

    All_Y_mn_pt_in_mat = np.hstack(( All_Y_mn_pt_in ))
    x1 = np.linalg.lstsq(All_Y_mn_pt_in_mat, input_points[:, 0])[0]
    x2 = np.linalg.lstsq(All_Y_mn_pt_in_mat, input_points[:, 1])[0]
    x3 = np.linalg.lstsq(All_Y_mn_pt_in_mat, input_points[:, 2])[0]

    return np.vstack([x1, x2, x3]).transpose()

# Use same lbdv_fit to analyze ellipsoid on lbdv points:
def Ellipsoid_LBDV(LBDV_Fit,
                   fit_degree: int,
                   points_ellipse_coords: tuple,
                   ellipse_points: PointsData,
                   use_true_harmonics: bool = False):

    U, V = points_ellipse_coords[0], points_ellipse_coords[1]

    fit_params = Least_Squares_Harmonic_Fit(fit_degree, U, V, ellipse_points, use_true_harmonics)

    X_ellps_fit_sph_coef_mat = sph_f.Un_Flatten_Coef_Vec(fit_params[:, 0], fit_degree)
    Y_ellps_fit_sph_coef_mat = sph_f.Un_Flatten_Coef_Vec(fit_params[:, 1], fit_degree)
    Z_ellps_fit_sph_coef_mat = sph_f.Un_Flatten_Coef_Vec(fit_params[:, 2], fit_degree)

    # Create SPH_func to represent X, Y, Z:
    X_ellps_fit_sph = sph_f.sph_func(X_ellps_fit_sph_coef_mat, fit_degree)
    Y_ellps_fit_sph = sph_f.sph_func(Y_ellps_fit_sph_coef_mat, fit_degree)
    Z_ellps_fit_sph = sph_f.sph_func(Z_ellps_fit_sph_coef_mat, fit_degree)

    # Get {X,Y,Z} Ellps Coordinates at lebedev points, so we can leverage our code:
    X_lbdv_ellps_pts = euc_kf.Extract_Quad_Pt_Vals_From_SPH_Fn(X_ellps_fit_sph, LBDV_Fit, 'A')
    Y_lbdv_ellps_pts = euc_kf.Extract_Quad_Pt_Vals_From_SPH_Fn(Y_ellps_fit_sph, LBDV_Fit, 'A')
    Z_lbdv_ellps_pts = euc_kf.Extract_Quad_Pt_Vals_From_SPH_Fn(Z_ellps_fit_sph, LBDV_Fit, 'A')

    # create manifold of Ellps, to calculate H and average H of Ellps:
    Ellps_Manny_Dict = {}
    Ellps_Manny_Name_Dict = {} # sph point cloud at lbdv
    Ellps_Manny_Name_Dict['X_lbdv_pts'] = X_lbdv_ellps_pts
    Ellps_Manny_Name_Dict['Y_lbdv_pts'] = Y_lbdv_ellps_pts
    Ellps_Manny_Name_Dict['Z_lbdv_pts'] = Z_lbdv_ellps_pts

    Ellps_Manny_Dict['Pickle_Manny_Data'] = False # BJG: until it works don't pickle
    Ellps_Manny_Dict['Maniold_lbdv'] = LBDV_Fit

    Ellps_Manny_Dict['Manifold_SPH_deg'] = fit_degree
    Ellps_Manny_Dict['use_manifold_name'] = False # we are NOT using named shapes in these tests
    Ellps_Manny_Dict['Maniold_Name_Dict'] = Ellps_Manny_Name_Dict # sph point cloud at lbdv

    Ellps_Manny = mnfd.manifold(Ellps_Manny_Dict)

    # Use Gauss-Bonnet to test our resolution of the manifold:
    Ellps_K_lbdv_pts = euc_kf.Combine_Chart_Quad_Vals(Ellps_Manny.K_A_pts, Ellps_Manny.K_B_pts, LBDV_Fit)
    Gauss_Bonnet_Err_Ellps = euc_kf.Integral_on_Manny(Ellps_K_lbdv_pts, Ellps_Manny, LBDV_Fit) - 4*np.pi
    Gauss_Bonnet_Rel_Err_Ellps = abs(Gauss_Bonnet_Err_Ellps)/(4*np.pi)

    Mean_Curv_Ellps_lbdv_pts_unsigned = euc_kf.Combine_Chart_Quad_Vals(Ellps_Manny.H_A_pts, Ellps_Manny.H_B_pts, LBDV_Fit)

    # We estimate H0 on Ellps: Integratal of H_ellps (on surface) divided by Ellps surface area:
    Ones_pts = 1.*(np.ones_like(Mean_Curv_Ellps_lbdv_pts_unsigned))
    H0_Ellps_Int_of_H_over_Area = euc_kf.Integral_on_Manny(Mean_Curv_Ellps_lbdv_pts_unsigned, Ellps_Manny, LBDV_Fit) / euc_kf.Integral_on_Manny(Ones_pts, Ellps_Manny, LBDV_Fit)

    abs_H0_Ellps_Int_of_H_over_Area = abs(H0_Ellps_Int_of_H_over_Area) # should be positive
    Mean_Curv_Ellps_lbdv_pts = Mean_Curv_Ellps_lbdv_pts_unsigned*abs_H0_Ellps_Int_of_H_over_Area/H0_Ellps_Int_of_H_over_Area # flip sign, if needed

    points = np.hstack([X_lbdv_ellps_pts, Y_lbdv_ellps_pts, Z_lbdv_ellps_pts])
    results = {}
    results['abs_H0_Ellps_Int_of_H_over_Area'] = abs_H0_Ellps_Int_of_H_over_Area
    results['Gauss_Bonnet_Rel_Err_Ellps'] = Gauss_Bonnet_Rel_Err_Ellps
    results['Mean_Curv_Ellps_lbdv_pts'] = Mean_Curv_Ellps_lbdv_pts

    results['lbdv_ellps_pts'] = points

    return results


def Conv_3D_pts_to_Elliptical_Coors(points: PointsData
                                    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit an ellipse to points and convert R^3 points to ellipsoidal coordinates.

    Parameters
    ----------
    points : PointsData
        DESCRIPTION.

    Returns
    -------
    U_coors_calc : np.ndarray
        1st component of points in ellipsoidal coordinate system
    V_coors_calc : np.ndarray
        2nd component of points in ellipsoidal coordinate system

    """
    _pts = vedo.pointcloud.Points(points)
    ellipsoid = vedo.pcaEllipsoid(_pts)

    center = ellipsoid.center
    r = Rotation.from_euler('xyz', ellipsoid.GetOrientation())
    inv_rot_mat = np.linalg.inv(r.as_matrix())

    num_pts_used = _pts.N()
    U_coors_calc = np.zeros(( num_pts_used, 1 ))
    V_coors_calc = np.zeros(( num_pts_used, 1 ))

    for pt_numb in range(num_pts_used):
        y_tilde_pt = np.linalg.solve(inv_rot_mat, points[pt_numb, :].reshape(3,1) - center.reshape(3, 1)  )

        yt_0 = y_tilde_pt[0,0]
        yt_1 = y_tilde_pt[1,0]
        yt_2 = y_tilde_pt[2,0]

        U_pt = np.arctan2( yt_1 * ellipsoid.va, yt_0 * ellipsoid.vb)

        if(U_pt < 0):
            U_pt = U_pt + 2. * np.pi

        U_coors_calc[pt_numb] = U_pt

        cylinder_r = np.sqrt(yt_0**2 + yt_1**2)    # r in cylinderical coors for y_tilde
        cyl_r_exp = np.sqrt( (ellipsoid.va * np.cos(U_pt))**2 + (ellipsoid.vb * np.sin(U_pt))**2 )

        V_pt = np.arctan2( cylinder_r * ellipsoid.vc, yt_2 * cyl_r_exp )

        if(V_pt < 0):
            V_pt = V_pt + 2.*np.pi

        V_coors_calc[pt_numb] = V_pt

    return U_coors_calc, V_coors_calc
