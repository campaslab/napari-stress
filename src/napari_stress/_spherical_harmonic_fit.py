import numpy as np
from napari.types import SurfaceData, PointsData
import vedo

from ._spherical_harmonics import lbdv_info_SPB as lbdv_i
from ._spherical_harmonics import sph_func_SPB as sph_f

from ._spherical_harmonics._utils import fit_ellipsoid,\
    Conv_3D_pts_to_Elliptical_Coors,\
    pointcloud_relative_coords,\
    Ellipsoid_LBDV,\
    Least_Squares_Harmonic_Fit

def spherical_harmonic_fit(points: PointsData,
                           mean_curvature: np.ndarray,
                           fit_degree: int = 3,
                           cap_number_of_points: bool = True,
                           tension_gamma_value: float = 0.5,
                           use_true_harmonics: bool = False):

    # Tension: \gamma  =.5, by default, so we multiply by this to get anistropic tension for curvature
    TwoGammaVal = 2. * tension_gamma_value

    # Get info from point cloud input for calculations we want to do:
    #1st way we estimate H0: mean of input curvatures Elijah calculated
    H0_avg_input_curvs = np.average(mean_curvature)

    points_relative = pointcloud_relative_coords(points)

    #JM: This part is important
    # get LS Ellipsoid estimate and get ellipsoid 3D parameters of original points
    ellipse_params = fit_ellipsoid(points)

    # Put our point cloud in LS Ellipsoid coordinates:
    U_coors_pts_in = np.zeros_like(mean_curvature)
    V_coors_pts_in = np.zeros_like(mean_curvature)

    U, V = Conv_3D_pts_to_Elliptical_Coors(points, ellipse_params)

    # self.LS_Ellps_Mean_Curvs = Mean_Curvs_on_Ellipsoid(a0, a1, a2, self.U_coors_pts_in, self.V_coors_pts_in)

    # # Compute Anis Stress/ Cell Stress from INPUT Curvatures:
    # self.Anis_Stress_pts_UV_input = self.TwoGammaVal*self.mean_curvs_input
    # H0_ellps_avg_ellps_UV_curvs = np.average(self.LS_Ellps_Mean_Curvs, axis=0) # 1st method of H0 computation, for Ellipsoid in UV points
    # self.Anis_Cell_Stress_pts_UV_input = self.TwoGammaVal*(self.mean_curvs_input - self.LS_Ellps_Mean_Curvs - (self.H0_avg_input_curvs -  H0_ellps_avg_ellps_UV_curvs))

    # # See LS ellipsoid in R^3 coordinates, compare to input points:
    # self.X_LS_Ellps_pts, self.Y_LS_Ellps_pts, self.Z_LS_Ellps_pts = Conv_Elliptical_Coors_to_3D_pts(self.U_coors_pts_in, self.V_coors_pts_in, self.LS_Ellps_axes, self.LS_Ellps_Inv_Rot_Mat, self.LS_Ellps_center)

    # # OUTWARD Normal vectors of Ellipsoid:
    # self.LS_Ellps_normals = Ellipsoid_Level_Set_Normals(self.X_LS_Ellps_pts, self.Y_LS_Ellps_pts, self.Z_LS_Ellps_pts, self.LS_Ellps_fit_Coef)

    # LS_Ellps_Err_X = self.X_LS_Ellps_pts - self.X_orig_in
    # LS_Ellps_Err_Y = self.Y_LS_Ellps_pts - self.Y_orig_in
    # LS_Ellps_Err_Z = self.Z_LS_Ellps_pts - self.Z_orig_in

    # self.LS_Ellps_Err_vecs = np.hstack(( LS_Ellps_Err_X, LS_Ellps_Err_Y, LS_Ellps_Err_Z ))

    # # We want to find largest Ellipsoid Axis to compare with final Droplet orientation:
    # axis_1 = np.vstack(( Conv_Elliptical_Coors_to_3D_pts( np.array([[0.]]), np.array([[np.pi/2.]]), self.LS_Ellps_axes, self.LS_Ellps_Inv_Rot_Mat, np.zeros_like(self.LS_Ellps_center)) )) # x-axis
    # len_axis_1 = np.linalg.norm(axis_1)

    # axis_2 = np.vstack(( Conv_Elliptical_Coors_to_3D_pts( np.array([[np.pi/2.]]), np.array([[np.pi/2.]]), self.LS_Ellps_axes, self.LS_Ellps_Inv_Rot_Mat, np.zeros_like(self.LS_Ellps_center)) )) # y_axis
    # len_axis_2 = np.linalg.norm(axis_2)

    # axis_3 = np.vstack(( Conv_Elliptical_Coors_to_3D_pts( np.array([[0.]]), np.array([[0.]]), self.LS_Ellps_axes, self.LS_Ellps_Inv_Rot_Mat, np.zeros_like(self.LS_Ellps_center)) )) # z_axis
    # len_axis_3 = np.linalg.norm(axis_3)

    # self.Maj_Min_Axis = np.zeros((3)) # 1 for major axis, -1 for minor axis, for {X,Y,Z}

    # if(len_axis_1 >= len_axis_2 and len_axis_1 >= len_axis_3):
    #     self.major_orientation_axis = axis_1
    #     self.Maj_Min_Axis[0] = 1.
    #     self.ellps_semi_axis_a = len_axis_1

    #     if(len_axis_2 <= len_axis_3):
    #         self.Maj_Min_Axis[1] = -1.
    #         self.ellps_semi_axis_c = len_axis_2
    #         self.minor_orientation_axis = axis_2
    #         self.ellps_semi_axis_b = len_axis_3
    #         self.medial_orientation_axis = axis_3
    #     else:
    #         self.Maj_Min_Axis[2] = -1.
    #         self.ellps_semi_axis_c = len_axis_3
    #         self.minor_orientation_axis = axis_3
    #         self.ellps_semi_axis_b = len_axis_2
    #         self.medial_orientation_axis = axis_2

    # elif(len_axis_2 >= len_axis_1 and len_axis_2 >= len_axis_3):
    #     self.major_orientation_axis = axis_2
    #     self.Maj_Min_Axis[1] = 1.
    #     self.ellps_semi_axis_a = len_axis_2

    #     if(len_axis_1 <= len_axis_3):
    #         self.Maj_Min_Axis[0] = -1.
    #         self.ellps_semi_axis_c = len_axis_1
    #         self.minor_orientation_axis = axis_1
    #         self.ellps_semi_axis_b = len_axis_3
    #         self.medial_orientation_axis = axis_3
    #     else:
    #         self.Maj_Min_Axis[2] = -1.
    #         self.ellps_semi_axis_c = len_axis_3
    #         self.minor_orientation_axis = axis_3
    #         self.ellps_semi_axis_b = len_axis_1
    #         self.medial_orientation_axis = axis_1

    # elif(len_axis_3 >= len_axis_2 and len_axis_3 >= len_axis_1):
    #     self.major_orientation_axis = axis_3
    #     self.Maj_Min_Axis[2] = 1.
    #     self.ellps_semi_axis_a = len_axis_3

    #     if(len_axis_2 <= len_axis_1):
    #         self.Maj_Min_Axis[1] = -1.
    #         self.ellps_semi_axis_c = len_axis_2
    #         self.minor_orientation_axis = axis_2
    #         self.ellps_semi_axis_b = len_axis_1
    #         self.medial_orientation_axis = axis_1
    #     else:
    #         self.Maj_Min_Axis[0] = -1.
    #         self.ellps_semi_axis_c = len_axis_1
    #         self.minor_orientation_axis = axis_1
    #         self.ellps_semi_axis_b = len_axis_2
    #         self.medial_orientation_axis = axis_2

    # else:
    #     print("\n"+"Error, major axis not discerned"+"\n")

    # # Get matrix to change basis from cartesian to ellipsoidal coors:
    # self.A_basis_mat = np.zeros((3, 3))
    # self.A_basis_mat[0,0] = self.major_orientation_axis[0, 0] /self.ellps_semi_axis_a
    # self.A_basis_mat[0,1] = self.major_orientation_axis[1, 0] /self.ellps_semi_axis_a
    # self.A_basis_mat[0,2] = self.major_orientation_axis[2, 0] /self.ellps_semi_axis_a
    # self.A_basis_mat[1,0] = self.medial_orientation_axis[0, 0] /self.ellps_semi_axis_b
    # self.A_basis_mat[1,1] = self.medial_orientation_axis[1, 0] /self.ellps_semi_axis_b
    # self.A_basis_mat[1,2] = self.medial_orientation_axis[2, 0] /self.ellps_semi_axis_b
    # self.A_basis_mat[2,0] = self.minor_orientation_axis[0, 0] /self.ellps_semi_axis_c
    # self.A_basis_mat[2,1] = self.minor_orientation_axis[1, 0] /self.ellps_semi_axis_c
    # self.A_basis_mat[2,2] = self.minor_orientation_axis[2, 0] /self.ellps_semi_axis_c

    # self.H_ellps_e_1 = self.ellps_semi_axis_a/(2.*self.ellps_semi_axis_b**2) +  self.ellps_semi_axis_a/(2.*self.ellps_semi_axis_c**2)
    # self.H_ellps_e_2 = self.ellps_semi_axis_b/(2.*self.ellps_semi_axis_a**2) +  self.ellps_semi_axis_b/(2.*self.ellps_semi_axis_c**2)
    # self.H_ellps_e_3 = self.ellps_semi_axis_c/(2.*self.ellps_semi_axis_a**2) +  self.ellps_semi_axis_c/(2.*self.ellps_semi_axis_b**2)

    # #!!!! ASSUMES WE USE PYCOMPADRE, could redo with signed dists to center !!!!#
    # #self.LS_Ellps_Err_signed = -1.* np.sum( np.multiply(self.PyCompadre_Normals_used, self.LS_Ellps_Err_vecs), axis = 1 )
    # self.LS_Ellps_Err_signed = -1.*np.sum( np.multiply(self.LS_Ellps_normals, self.LS_Ellps_Err_vecs), axis = 1 )

    # Least-Squares Fit to Harmonics in {X,Y,Z}, using ellipsoidal coordinates:
    # If we use 5810 points or hyper-interpolation quad pts
    if cap_number_of_points:
        num_lbdv_pts_fit = 5810
    else:
        num_lbdv_pts_fit = lbdv_i.look_up_lbdv_pts(fit_degree + 1)


    # use H0_Ellpsoid to calculate tissue stress projections:
    # LBDV_Fit = lbdv_i.lbdv_info(fit_degree, num_lbdv_pts_fit)
    # results = Ellipsoid_LBDV(LBDV_Fit)
    # H0_Ellpsoid = results['abs_H0_Ellps_Int_of_H_over_Area']
    # self.sigma_11_e = self.TwoGammaVal*(self.H_ellps_e_1 - self.H0_Ellpsoid)
    # self.sigma_22_e = self.TwoGammaVal*(self.H_ellps_e_2 - self.H0_Ellpsoid)
    # self.sigma_33_e = self.TwoGammaVal*(self.H_ellps_e_3 - self.H0_Ellpsoid)

    # # tissue stress tensor, elliptical coors:
    # self.Tissue_Stress_Tens_Ellp_Coors = np.zeros((3,3))
    # self.Tissue_Stress_Tens_Ellp_Coors[0,0] = self.sigma_11_e
    # self.Tissue_Stress_Tens_Ellp_Coors[1,1] = self.sigma_22_e
    # self.Tissue_Stress_Tens_Ellp_Coors[2,2] = self.sigma_33_e
    # tr_sigma_ellps = self.sigma_11_e + self.sigma_22_e + self.sigma_33_e

    # # cartesian tissue stress tensor:
    # self.Tissue_Stress_Tens_Cart_Coors = np.dot( np.dot(self.A_basis_mat.T ,self.Tissue_Stress_Tens_Ellp_Coors), self.A_basis_mat)
    # self.sigma_11_tissue_x =  self.Tissue_Stress_Tens_Cart_Coors[0,0]
    # self.sigma_22_tissue_y =  self.Tissue_Stress_Tens_Cart_Coors[1,1]
    # self.sigma_33_tissue_z =  self.Tissue_Stress_Tens_Cart_Coors[2,2]

    popt = Least_Squares_Harmonic_Fit(fit_degree=fit_degree,
                                      points_ellipse_coords = (U, V),
                                      input_points = points,
                                      use_true_harmonics=use_true_harmonics)

    X_fit_sph_coef_mat = sph_f.Un_Flatten_Coef_Vec(popt[:, 0], fit_degree)
    Y_fit_sph_coef_mat = sph_f.Un_Flatten_Coef_Vec(popt[:, 1], fit_degree)
    Z_fit_sph_coef_mat = sph_f.Un_Flatten_Coef_Vec(popt[:, 2], fit_degree)

    # Create SPH_func to represent X, Y, Z:
    X_fit_sph = sph_f.sph_func(X_fit_sph_coef_mat, fit_degree)
    Y_fit_sph = sph_f.sph_func(Y_fit_sph_coef_mat, fit_degree)
    Z_fit_sph = sph_f.sph_func(Z_fit_sph_coef_mat, fit_degree)

    X_fit_sph_UV_pts = X_fit_sph.Eval_SPH(U, V)
    Y_fit_sph_UV_pts = Y_fit_sph.Eval_SPH(U, V)
    Z_fit_sph_UV_pts = Z_fit_sph.Eval_SPH(U, V)

    return np.hstack((X_fit_sph_UV_pts, Y_fit_sph_UV_pts, Z_fit_sph_UV_pts ))
