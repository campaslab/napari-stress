import numpy as np
from napari.types import LayerDataTuple, PointsData

from ._spherical_harmonics import sph_func_SPB as sph_f

from ._spherical_harmonics._utils import Conv_3D_pts_to_Elliptical_Coors,\
    Least_Squares_Harmonic_Fit

def spherical_harmonic_fit(points: PointsData,
                           fit_degree: int = 3) -> LayerDataTuple:
    """
    Approximate a pointcloud by fitting a base of spherical harmonic functions

    Parameters
    ----------
    points : PointsData
    fit_degree : int, optional
        Degree of fitted polynomial functions. The higher the number, the
        smaller the deviation between fitted points and input points.
        The default is 3.
    use_true_harmonics : bool, optional
        Whether or not to use the scipy implementation for spherical harmonic
        functions. Currently not supported. The default is False.

    Returns
    -------
    PointsData
    """

    # get LS Ellipsoid estimate and get ellipsoid 3D parameters of original points
    U, V = Conv_3D_pts_to_Elliptical_Coors(points)

    popt = Least_Squares_Harmonic_Fit(fit_degree=fit_degree,
                                      points_ellipse_coords = (U, V),
                                      input_points = points,
                                      use_true_harmonics=False)

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

    popt_points = np.hstack((X_fit_sph_UV_pts, Y_fit_sph_UV_pts, Z_fit_sph_UV_pts ))
    errors = np.linalg.norm(popt_points - points, axis=1)

    properties = {'errors': errors}

    return (popt_points, {'properties': properties}, 'points')
