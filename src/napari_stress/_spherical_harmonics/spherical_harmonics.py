# -*- coding: utf-8 -*-
"""
This file bundles up spherical harmonics functionality to have all "behind-the-
scenes" functionality in one place. This will allow to have all fron-end-related
functions (e.g., those functions that are visible to napari) in a separated place.

"""

from napari.types import PointsData
from typing import Tuple
import numpy as np
import vedo
import pyshtools
import warnings

from .._utils.fit_utils import Least_Squares_Harmonic_Fit
from .._stress import sph_func_SPB as sph_f
from .._stress import manifold_SPB as mnfd
from .._stress import euclidian_k_form_SPB as euc_kf
from .._stress import lebedev_info_SPB as lebedev_info

def shtools_spherical_harmonics_expansion(points: PointsData,
                                          max_degree: int = 5,
                                          expansion_type: str ='radial'
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
    from .._stress.sph_func_SPB import convert_coefficients_pyshtools_to_stress
    # Convert points coordinates relative to center
    center = points.mean(axis=0)
    relative_coordinates = points - center[np.newaxis, :]

    # Convert point coordinates to spherical coordinates (in degree!)
    spherical_coordinates = vedo.cart2spher(relative_coordinates[:, 0],
                                            relative_coordinates[:, 1],
                                            relative_coordinates[:, 2])
    radius = spherical_coordinates[0]
    latitude = np.rad2deg(spherical_coordinates[1])
    longitude = np.rad2deg(spherical_coordinates[2])

    if expansion_type == 'radial':

        # Find spherical harmonics expansion coefficients until specified degree
        opt_fit_params = pyshtools._SHTOOLS.SHExpandLSQ(radius, latitude, longitude,
                                                        lmax = max_degree)[1]
        # Sample radius values at specified latitude/longitude
        coefficients = pyshtools.SHCoeffs.from_array(opt_fit_params)
        values = coefficients.expand(lat=latitude, lon=longitude)

        # Convert points back to cartesian coordinates
        points = vedo.spher2cart(values,
                                spherical_coordinates[1],
                                spherical_coordinates[2]).transpose()

        points = points + center[np.newaxis, :]
        coefficients = convert_coefficients_pyshtools_to_stress(coefficients)
        return points, coefficients

    elif expansion_type == 'cartesian':
        
        optimal_fit_params = []
        fitted_coordinates = np.zeros_like(points)
        for i in range(3):
            params = pyshtools._SHTOOLS.SHExpandLSQ(points[:, i], latitude, longitude,
                                                    lmax = max_degree)[1]
            coefficients = pyshtools.SHCoeffs.from_array(params)
            fitted_coordinates[:, i] = coefficients.expand(lat=latitude, lon=longitude)
            optimal_fit_params.append(convert_coefficients_pyshtools_to_stress(coefficients))
        
        return fitted_coordinates, np.stack(optimal_fit_params)
            


def stress_spherical_harmonics_expansion(points: PointsData,
                                         max_degree: int = 5,
                                         expansion_type: str ='cartesian'
                                         ) -> Tuple[PointsData, np.ndarray]:
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
    from .. import _approximation as approximation
    from .._utils.coordinate_conversion import cartesian_to_elliptical
    from .._stress.charts_SPB import Cart_To_Coor_A

    if expansion_type == 'cartesian':
        # get LS Ellipsoid estimate and get point cordinates in elliptical coordinates
        ellipsoid = approximation.least_squares_ellipsoid(points)
        longitude, latitude = cartesian_to_elliptical(ellipsoid, points)

        # This implementation fits a superposition of three sets of spherical harmonics
        # to the data, one for each cardinal direction (x/y/z).
        optimal_fit_parameters = []
        for i in range(3):
            params = Least_Squares_Harmonic_Fit(
                fit_degree=max_degree,
                sample_locations = (longitude, latitude),
                values = points[:, i])
            optimal_fit_parameters.append(params)
        optimal_fit_parameters = np.vstack(optimal_fit_parameters).transpose()

        X_fit_sph_coef_mat = sph_f.Un_Flatten_Coef_Vec(optimal_fit_parameters[:, 0], max_degree)
        Y_fit_sph_coef_mat = sph_f.Un_Flatten_Coef_Vec(optimal_fit_parameters[:, 1], max_degree)
        Z_fit_sph_coef_mat = sph_f.Un_Flatten_Coef_Vec(optimal_fit_parameters[:, 2], max_degree)

        coefficients = np.stack([X_fit_sph_coef_mat, Y_fit_sph_coef_mat, Z_fit_sph_coef_mat])

        # Create SPH_func to represent X, Y, Z:
        X_fit_sph = sph_f.spherical_harmonics_function(coefficients[0], max_degree)
        Y_fit_sph = sph_f.spherical_harmonics_function(coefficients[1], max_degree)
        Z_fit_sph = sph_f.spherical_harmonics_function(coefficients[2], max_degree)

        X_fit_sph_UV_pts = X_fit_sph.Eval_SPH(longitude, latitude)
        Y_fit_sph_UV_pts = Y_fit_sph.Eval_SPH(longitude, latitude)
        Z_fit_sph_UV_pts = Z_fit_sph.Eval_SPH(longitude, latitude)

        fitted_points = np.hstack((X_fit_sph_UV_pts, Y_fit_sph_UV_pts, Z_fit_sph_UV_pts ))
        return fitted_points, coefficients
    
    if expansion_type == 'radial':
        # This implementation fits a spherical harmonics expansion
        # to the data to describe radius as a function of latitude/longitutde
        center = points.mean(axis=0)
        points_relative = points - center[None, :]

        radii = np.sqrt(points_relative[:, 0]**2 + points_relative[:, 1]**2 + points_relative[:, 2]**2)
        longitude, latitude = Cart_To_Coor_A(points_relative[:, 0], points_relative[:, 1], points_relative[:, 2])

        optimal_fit_parameters = Least_Squares_Harmonic_Fit(
            fit_degree=max_degree,
            sample_locations = (longitude[:, None], latitude[:, None]),
            values = radii)

        # Add a singleton dimension to be consistent with coefficient array shape
        coefficients = sph_f.Un_Flatten_Coef_Vec(optimal_fit_parameters, max_degree)[None, :]

        # expand radii
        r_fit_sph = sph_f.spherical_harmonics_function(coefficients[0], max_degree)
        r_fit_sph_UV_pts = r_fit_sph.Eval_SPH(longitude, latitude).squeeze()

        fitted_points = vedo.spher2cart(r_fit_sph_UV_pts,
                                        latitude,
                                        longitude).transpose() + center[None, :]
        

        return fitted_points, coefficients

    

def lebedev_quadrature(coefficients: np.ndarray,
                       number_of_quadrature_points: int = 500,
                       use_minimal_point_set: bool = True
                       ) -> Tuple[PointsData, lebedev_info.lbdv_info]:
    """
    Calculate lebedev quadrature points for a given spherical harmonics expansion.

    Parameters
    ----------
    coefficients : np.ndarray
        Spherical harmonics coefficient matrix.
    number_of_quadrature_points : int, optional
        Number of quadrature points to retrieve on the passed set of spherical
        harmonics functions. The default is 500.
    use_minimal_point_set : bool, optional
        Depending on the degree a defined minimal set of quadrature points is
        sufficient to integrate exactly. The default is True.

    Returns
    -------
    lebedev_points : PointsData
    LBDV_Fit : lebedev_info
        Data container storing points and quick access for other parameters.

    """
    # Clip number of quadrature points
    if number_of_quadrature_points > 5810:
        number_of_quadrature_points = 5810

    # Coefficient matrix should be [DIM, DEG, DEG]; if DIM=1 this corresponds
    # to a radial spherical harmonics expansion
    if len(coefficients.shape) == 2:
        coefficients = coefficients[None, :]

    # An expansion of degree 3 will have an Nx4x4 coefficient matrix
    max_degree = coefficients.shape[-1] - 1

    possible_n_points = np.asarray(list(lebedev_info.pts_of_lbdv_lookup.values()))
    index_correct_n_points = np.argmin(abs(possible_n_points - number_of_quadrature_points))
    number_of_quadrature_points = possible_n_points[index_correct_n_points]

    if use_minimal_point_set:
        number_of_quadrature_points = lebedev_info.look_up_lbdv_pts(max_degree + 1)

    # Only a specific amount of points are needed for spherical harmonics of a
    # given order. Using more points will not give more precise results.
    if number_of_quadrature_points > lebedev_info.look_up_lbdv_pts(max_degree + 1):
        warnings.warn(r'Note: Only {necessary_n_points} are required for exact results.')

    # Create spherical harmonics functions to represent z/y/x
    fit_functions = [
        sph_f.spherical_harmonics_function(x, max_degree) for x in coefficients
        ]

    # Get {Z/Y/X} Coordinates at lebedev points, so we can leverage our code more efficiently (and uniformly) on surface:
    LBDV_Fit = lebedev_info.lbdv_info(max_degree, number_of_quadrature_points)
    lebedev_points  = [
        euc_kf.get_quadrature_points_from_sh_function(f, LBDV_Fit, 'A') for f in fit_functions
        ]
    lebedev_points = np.stack(lebedev_points).squeeze().transpose()

    return lebedev_points, LBDV_Fit

def create_manifold(points: PointsData,
                    lebedev_fit: lebedev_info.lbdv_info,
                    max_degree: int) -> mnfd.manifold:

    # add information to manifold on what type of expansion was used
    Manny_Dict = {}
    Manny_Name_Dict = {} # sph point cloud at lbdv

    manifold_type = None
    if len(points.shape) == 1:
        manifold_type = 'radial'
        coordinates = np.stack([lebedev_fit.X.squeeze() * points,
                                lebedev_fit.Y.squeeze() * points,
                                lebedev_fit.Z.squeeze() * points]).T
        Manny_Name_Dict['coordinates'] = coordinates
        
    elif len(points.shape) == 2:
        manifold_type = 'cartesian'
        Manny_Name_Dict['coordinates'] = points

    # create manifold to calculate H, average H:
    Manny_Dict['Pickle_Manny_Data'] = False
    Manny_Dict['Maniold_lbdv'] = lebedev_fit

    Manny_Dict['Manifold_SPH_deg'] = max_degree
    Manny_Dict['use_manifold_name'] = False # we are NOT using named shapes in these tests
    Manny_Dict['Maniold_Name_Dict'] = Manny_Name_Dict # sph point cloud at lbdv

    return mnfd.manifold(Manny_Dict, manifold_type=manifold_type, raw_coordinates=points)

def get_normals_on_manifold(manifold: mnfd.manifold) -> np.ndarray:
    """
    Calculate the outwards pointing normal vectors on a manifold.

    Parameters
    ----------
    manifold : mnfd.manifold

    Returns
    -------
    np.ndarray: (Nx3) - sized normal vectors

    """


    normal_X_lbdv_pts = euc_kf.Combine_Chart_Quad_Vals(manifold.Normal_Vec_X_A_Pts,
                                                       manifold.Normal_Vec_X_B_Pts,
                                                       manifold.lebedev_info)
    normal_Y_lbdv_pts = euc_kf.Combine_Chart_Quad_Vals(manifold.Normal_Vec_Y_A_Pts,
                                                       manifold.Normal_Vec_Y_B_Pts,
                                                       manifold.lebedev_info)
    normal_Z_lbdv_pts = euc_kf.Combine_Chart_Quad_Vals(manifold.Normal_Vec_Z_A_Pts,
                                                       manifold.Normal_Vec_Z_B_Pts,
                                                       manifold.lebedev_info)

    normals_lbdv_points = np.stack([normal_X_lbdv_pts, normal_Y_lbdv_pts, normal_Z_lbdv_pts])
    return normals_lbdv_points.squeeze().transpose()

def calculate_mean_curvature_on_manifold(lebedev_points: PointsData,
                                         lebedev_fit: lebedev_info.lbdv_info,
                                         max_degree: int) -> np.ndarray:
    """
    Calculate mean curvatures for a set of lebedev points.

    Parameters
    ----------
    lebedev_points : PointsData
    lebedev_fit : lebedev_info.lbdv_info
        lebedev info object - provides info which point is defined by which chart.
    max_degree : int
        Degree of spherical harmonics expansion.

    Returns
    -------
    np.ndarray
        Mean curvature value for every lebedev point.

    """
    manifold = create_manifold(lebedev_points, lebedev_fit, max_degree)
    normals = get_normals_on_manifold(manifold, lebedev_fit)

    # Test orientation:
    points = manifold.get_coordinates()
    centered_lbdv_pts = points - points.mean(axis=0)[None, :]

    # Makre sure orientation is inward, so H is positive (for Ellipsoid, and small deviations):
    Orientations = [np.dot(x, y) for x, y in zip(centered_lbdv_pts,  normals)]
    num_pos_orr = np.sum(np.asarray(Orientations).flatten() > 0)

    Orientation = 1. # unchanged (we want INWARD)
    if(num_pos_orr > .5 * len(centered_lbdv_pts)):
        Orientation = -1.

    return Orientation*euc_kf.Combine_Chart_Quad_Vals(manifold.H_A_pts, manifold.H_B_pts, lebedev_fit).squeeze()
