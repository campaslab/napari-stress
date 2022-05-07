# -*- coding: utf-8 -*-
import numpy as np
from napari.types import PointsData

from scipy.special import sph_harm

import lbdv_info_SPB as lbdv_i
import sph_func_SPB as sph_f
import euc_k_form_SPB as euc_kf
import manifold_SPB as mnfd

# return least squares harmonic fit to point cloud, given choice of basis and degree:
def Least_Squares_Harmonic_Fit(fit_degree: int,
                               points_ellipse_coords: tuple,
                               input_points: PointsData,
                               use_true_harmonics: bool = True) -> np.ndarray:

    U, V = points_ellipse_coords[0], points_ellipse_coords[1]

    All_Y_mn_pt_in = []

    for n in range(fit_degree + 1):
        for m in range(-1*n, n+1):
            Y_mn_coors_in = []

            # we can use actual harmonics or our basis:
            if(use_true_harmonics == True):
                Y_mn_coors_in = sph_harm(m, n, U, V)
            else:
                Y_mn_coors_in = lbdv_i.Eval_SPH_Basis(m, n, U, V)

            All_Y_mn_pt_in.append(Y_mn_coors_in)

    All_Y_mn_pt_in_mat = np.hstack(( All_Y_mn_pt_in ))
    x1 = np.linalg.lstsq(All_Y_mn_pt_in_mat, input_points[:, 0])[0]
    x2 = np.linalg.lstsq(All_Y_mn_pt_in_mat, input_points[:, 1])[0]
    x3 = np.linalg.lstsq(All_Y_mn_pt_in_mat, input_points[:, 2])[0]

    return np.hstack([x1, x2, x3])

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

# Convert input R^3 points into Ellipsoidal coors:
def Conv_3D_pts_to_Elliptical_Coors(point_cloud: PointsData,
                                    ellipse_params: dict):

    major_axis = ellipse_params['ellipse_axes']
    center = ellipse_params['center']
    inv_rot_mat = ellipse_params['inverse_rotation_matrix']

    num_pts_used = len(point_cloud)
    U_coors_calc = np.zeros(( num_pts_used, 1 ))
    V_coors_calc = np.zeros(( num_pts_used, 1 ))

    for pt_numb in range(num_pts_used):
        y_tilde_pt = np.linalg.solve(inv_rot_mat, point_cloud[pt_numb, :].reshape(3,1) - center.reshape(3, 1)  )

        yt_0 = y_tilde_pt[0,0]
        yt_1 = y_tilde_pt[1,0]
        yt_2 = y_tilde_pt[2,0]

        U_pt = np.arctan2( yt_1 * major_axis[0], yt_0 * major_axis[1])

        if(U_pt < 0):
            U_pt = U_pt + 2. * np.pi

        U_coors_calc[pt_numb] = U_pt

        cylinder_r = np.sqrt(yt_0**2 + yt_1**2)    # r in cylinderical coors for y_tilde
        cyl_r_exp = np.sqrt( (major_axis[0] * np.cos(U_pt))**2 + (major_axis[1] * np.sin(U_pt))**2 )

        V_pt = np.arctan2( cylinder_r * major_axis[2], yt_2 * cyl_r_exp )

        if(V_pt < 0):
            V_pt = V_pt + 2.*np.pi

        V_coors_calc[pt_numb] = V_pt

    return U_coors_calc, V_coors_calc

def pointcloud_relative_coords(points: PointsData) -> PointsData:
    center = points.mean(axis=0)
    rel_coords = points - center[None, :]
    return rel_coords

def leastsquares_ellipsoid(pointcloud: PointsData):
    #finds best fit ellipsoid. Found at http://www.juddzone.com/ALGORITHMS/least_squares_3D_ellipsoid.html
    #least squares fit to a 3D-ellipsoid
    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz  = 1
    #
    # Note that sometimes it is expressed as a solution to
    #  Ax^2 + By^2 + Cz^2 + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz  = 1
    # where the last six terms have a factor of 2 in them
    # This is in anticipation of forming a matrix with the polynomial coefficients.
    # Those terms with factors of 2 are all off diagonal elements.  These contribute
    # two terms when multiplied out (symmetric) so would need to be divided by two

    # change xx from vector of length N to Nx1 matrix so we can use hstack
    x = pointcloud[:, 0, np.newaxis]
    y = pointcloud[:, 0, np.newaxis]
    z = pointcloud[:, 0, np.newaxis]

    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz = 1
    J = np.hstack((x*x,y*y,z*z,x*y,x*z,y*z, x, y, z))
    K = np.ones_like(x) #column of ones

    #np.hstack performs a loop over all samples and creates
    #a row in J for each x,y,z sample:
    # J[ix,0] = x[ix]*x[ix]
    # J[ix,1] = y[ix]*y[ix]
    # etc.

    JT = J.transpose()
    JTJ = np.dot(JT,J)
    InvJTJ = np.linalg.inv(JTJ);
    ABC = np.dot(InvJTJ, np.dot(JT,K)) #!!!! LOOK AT RESIDUALS TO GET ELLIPSOID ERRORS !!!!#

    # Rearrange, move the 1 to the other side
    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz - 1 = 0
    #    or
    #  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz + J = 0
    #  where J = -1
    eansa=np.append(ABC,-1)

    return (eansa)

# For above Ellipsoid Code:
def polyToParams3D(vec, verbose: bool = True):
    """
    gets 3D parameters of an ellipsoid. Found at http://www.juddzone.com/ALGORITHMS/least_squares_3D_ellipsoid.html
    convert the polynomial form of the 3D-ellipsoid to parameters
    center, axes, and transformation matrix
    vec is the vector whose elements are the polynomial
    coefficients A..J


    Parameters
    ----------
    vec : TYPE
        DESCRIPTION.
    verbose : bool, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    returns (center, axes, rotation matrix)

    """
    #Algebraic form: X.T * Amat * X --> polynomial form

    if verbose: print('\npolynomial\n',vec)

    Amat = np.array([
        [ vec[0],     vec[3]/2.0, vec[4]/2.0, vec[6]/2.0 ],
        [ vec[3]/2.0, vec[1],     vec[5]/2.0, vec[7]/2.0 ],
        [ vec[4]/2.0, vec[5]/2.0, vec[2],     vec[8]/2.0 ],
        [ vec[6]/2.0, vec[7]/2.0, vec[8]/2.0, vec[9]     ]
    ])

    if verbose: print('\nAlgebraic form of polynomial\n', Amat)

    #See B.Bartoni, Preprint SMU-HEP-10-14 Multi-dimensional Ellipsoidal Fitting
    # equation 20 for the following method for finding the center
    A3 = Amat[0:3,0:3]
    A3inv=np.linalg.inv(A3)
    ofs=vec[6:9]/2.0
    center=-np.dot(A3inv,ofs)
    if verbose: print('\nCenter at:',center)

    # Center the ellipsoid at the origin
    Tofs=np.eye(4)
    Tofs[3,0:3]=center
    R = np.dot(Tofs,np.dot(Amat,Tofs.T))
    if verbose: print('\nAlgebraic form translated to center\n',R,'\n')

    R3=R[0:3,0:3]
    R3test=R3/R3[0,0]
    # print('normed \n',R3test)
    s1=-R[3, 3]
    R3S=R3/s1
    (el,ec)=np.linalg.eig(R3S)

    recip=1.0/np.abs(el)
    axes=np.sqrt(recip)
    if verbose: print('\nAxes are\n',axes  ,'\n')

    inve=np.linalg.inv(ec) #inverse is actually the transpose here
    if verbose: print('\nRotation matrix\n',inve)

    results = {}
    results['center'] = center
    results['ellipse_axes'] = axes
    results['inverse'] = inve
    results['eigenvectors'] = ec
    return results
