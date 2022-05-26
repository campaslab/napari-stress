# -*- coding: utf-8 -*-
from napari.types import PointsData
import vedo
import numpy as np

import pyshtools
from . import sph_func_SPB as sph_f

from typing import Tuple

from ._utils import Conv_3D_pts_to_Elliptical_Coors,\
    Least_Squares_Harmonic_Fit

def shtools_spherical_harmonics_expansion(points: PointsData,
                                          max_degree: int = 5
                                          ) -> Tuple[PointsData, np.ndarray]:
    """
    Approximate a surface by spherical harmonics expansion with pyshtools implementation.

    Parameters
    ----------
    points : PointsData
    max_degree : int, optional
        Degree of spherical harmonics to fit to the data. The default is 5.

    Returns
    -------
    PointsData

    """
    # Convert points coordinates relative to center
    center = points.mean(axis=0)
    relative_coordinates = points - center[np.newaxis, :]

    # Convert point coordinates to spherical coordinates (in degree!)
    spherical_coordinates = vedo.cart2spher(relative_coordinates[:, 0],
                                            relative_coordinates[:, 1],
                                            relative_coordinates[:, 2])
    radius = spherical_coordinates[0]
    longitude = np.rad2deg(spherical_coordinates[1])
    latitude = np.rad2deg(spherical_coordinates[2])

    # Find spherical harmonics expansion coefficients until specified degree
    opt_fit_params = pyshtools._SHTOOLS.SHExpandLSQ(radius, latitude, longitude,
                                                    lmax = max_degree)[1]
    # Sample radius values at specified latitude/longitude
    spherical_harmonics_coeffcients = pyshtools.SHCoeffs.from_array(opt_fit_params)
    values = spherical_harmonics_coeffcients.expand(lat=latitude, lon=longitude)

    # Convert points back to cartesian coordinates
    points = vedo.spher2cart(values,
                             spherical_coordinates[1],
                             spherical_coordinates[2]).transpose()

    points = points + center[np.newaxis, :]
    return points, spherical_harmonics_coeffcients.to_array()

def stress_spherical_harmonics_expansion(points: PointsData,
                                         max_degree: int = 5) -> Tuple[PointsData, np.ndarray]:
    """
    Approximate a surface by spherical harmonics expansion with stress implementation.

    Parameters
    ----------
    points : PointsData
    max_degree : int, optional
        Degree of spherical harmonics to fit to the data. The default is 5.

    Returns
    -------
    PointsData
    """

    # get LS Ellipsoid estimate and get point cordinates in elliptical coordinates
    points_elliptical_1, points_elliptical_2 = Conv_3D_pts_to_Elliptical_Coors(points)

    # This implementation fits a superposition of three sets of spherical harmonics
    # to the data, one for each cardinal direction (x/y/z).
    optimal_fit_parameters = Least_Squares_Harmonic_Fit(
        fit_degree=max_degree,
        points_ellipse_coords = (points_elliptical_1, points_elliptical_2),
        input_points = points)

    X_fit_sph_coef_mat = sph_f.Un_Flatten_Coef_Vec(optimal_fit_parameters[:, 0], max_degree)
    Y_fit_sph_coef_mat = sph_f.Un_Flatten_Coef_Vec(optimal_fit_parameters[:, 1], max_degree)
    Z_fit_sph_coef_mat = sph_f.Un_Flatten_Coef_Vec(optimal_fit_parameters[:, 2], max_degree)

    spherical_harmonics_coeffcients = np.stack([X_fit_sph_coef_mat, Y_fit_sph_coef_mat, Z_fit_sph_coef_mat])

    # Create SPH_func to represent X, Y, Z:
    X_fit_sph = sph_f.sph_func(spherical_harmonics_coeffcients[0], max_degree)
    Y_fit_sph = sph_f.sph_func(spherical_harmonics_coeffcients[1], max_degree)
    Z_fit_sph = sph_f.sph_func(spherical_harmonics_coeffcients[2], max_degree)

    X_fit_sph_UV_pts = X_fit_sph.Eval_SPH(points_elliptical_1, points_elliptical_2)
    Y_fit_sph_UV_pts = Y_fit_sph.Eval_SPH(points_elliptical_1, points_elliptical_2)
    Z_fit_sph_UV_pts = Z_fit_sph.Eval_SPH(points_elliptical_1, points_elliptical_2)

    fitted_points = np.hstack((X_fit_sph_UV_pts, Y_fit_sph_UV_pts, Z_fit_sph_UV_pts ))

    """
    #TODO: Add code from campas stress repo
    Link: https://github.com/campaslab/STRESS/blob/29c6627cb4c95330567cde5d0189238e3b95d7ab/Refactored_Droplet_Class.py#L888

    This code simplifies the surface points to a set of lebedev points which allows to calculate
    mean curvatures very easily.
    """

    return fitted_points, spherical_harmonics_coeffcients
