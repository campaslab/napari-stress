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


def cartesian_to_elliptical_coordinates(points: PointsData
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
