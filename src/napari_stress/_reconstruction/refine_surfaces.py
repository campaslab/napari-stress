# -*- coding: utf-8 -*-

import vedo
from napari.types import SurfaceData, ImageData, PointsData, LayerDataTuple

from .._utils.fit_utils import _sigmoid, _gaussian, _function_args_to_list, _detect_max_gradient, _detect_maxima
from .._utils.frame_by_frame import frame_by_frame

from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit

import numpy as np
import tqdm
import pandas as pd

from enum import Enum

from napari_tools_menu import register_function
from typing import List

import warnings
warnings.filterwarnings('ignore')

class fit_types(Enum):
    quick_edge_fit = 'quick'
    fancy_edge_fit = 'fancy'

class edge_functions(Enum):
    interior = {'fancy': _sigmoid, 'quick': _detect_max_gradient}
    surface = {'fancy': _gaussian, 'quick': _detect_maxima}


@register_function(menu="Points > Trace-refine points (n-STRESS)")
@frame_by_frame
def trace_refinement_of_surface(intensity_image: ImageData,
                                points: PointsData,
                                selected_fit_type: fit_types = fit_types.fancy_edge_fit,
                                selected_edge: edge_functions = edge_functions.interior,
                                trace_length: float = 10.0,
                                sampling_distance: float = 0.5,
                                scale_z: float = 1.0,
                                scale_y: float = 1.0,
                                scale_x: float = 1.0,
                                remove_outliers: bool = True,
                                outlier_tolerance: float = 1.5,
                                show_progress: bool = True,
                                )-> List[LayerDataTuple]:
    """
    Generate intensity profiles along traces.

    This function receives an intensity image and a pointcloud with points on
    the surface of an object in the intensity image. It assumes that the
    pointcloud corresponds to the vertices on a surface around that object.

    As a first step, the function calculates normals on the surface and
    multiplies the length of this vector with `trace_length`. The intensity of
    the input image is then sampled along this vector perpendicular to the
    surface with a distance of `sampling distance` between each point along the
    normal vector.

    The location of the object's surface is then determined by fitting a
    selected function to the intensity profile along the prolonged normal vector.

    Parameters
    ----------
    intensity_image : ImageData
    points : PointsData
    selected_fit_type : fit_types, optional
        Which fit types to choose from. Can be `fit_types.fancy_edge_fit`/`"fancy"` or
        `fit_types.quick_edge_fit`/`"quick"`.
    selected_edge : edge_functions, optional
        Depending on the fluorescence of the intensity image, a different fit
        function is required. Can be either of `edge_functions.interior` or
        edge_functions.surface. The default is `edge_functions.interior`.
    trace_length : float, optional
        Length of the normal vector perpendicular to the surface. The default is 2.0.
    sampling_distance : float, optional
        Distance between two sampled intensity values along the normal vector.
        The default is 0.1.
    scale_z : float
        Voxel size in z
    scale_y: float
        Voxel size in y
    scale_x: float
        Voxel size in x
    show_progress : bool, optional
        The default is False.
    remove_outliers : bool, optional
        If this is set to true, the function will evaluate the fit residues of
        the chosen function and remove points that are classified as outliers.
        The default is True.
    outlier_tolerance : float, optional
        Determines how strict the outlier removal will be. Values with
        `value > Q75 + interquartile_factor * IQR` are classified as outliers,
        whereas `Q75` and `IQR` denote the 75% quartile and the interquartile
        range, respectively.
        The default is 1.5.

    Returns
    -------
    PointsData

    """
    from .. import _vectors as vectors
    from .._measurements.measurements import distance_to_k_nearest_neighbors
    from scipy.spatial import KDTree
    if isinstance(selected_fit_type, str):
        selected_fit_type = fit_types(selected_fit_type)

    if isinstance(selected_edge, str):
        edge_detection_function = edge_functions.__members__[selected_edge].value[selected_fit_type.value]
    else:
        edge_detection_function = selected_edge.value[selected_fit_type.value]

    # Convert to mesh and calculate outward normals
    unit_normals = vectors.normal_vectors_on_pointcloud(points)[:, 1] * (-1)

    # Define start and end points for the surface tracing vectors
    scale = np.asarray([scale_z, scale_y, scale_x])
    n_samples = int(trace_length/sampling_distance)
    n_points = len(points)

    # Define trace vectors (full length and single step
    start_points = points/scale[None, :] - 0.5 * trace_length * unit_normals
    trace_vectors = trace_length * unit_normals
    vector_step = trace_vectors/n_samples  # vector of length sampling distance

    # measure intensity along the vectors
    intensity_along_vector = vectors.sample_intensity_along_vector(
        np.stack([start_points, trace_vectors]).transpose((1, 0, 2)),
        intensity_image,
        sampling_distance=sampling_distance,
        interpolation_method='cubic')

    # Allocate arrays for results (location of object border, fit parameters,
    # fit errors, and intensity profiles)
    fit_parameters = _function_args_to_list(edge_detection_function)[1:]
    fit_errors = [p + '_err' for p in fit_parameters]
    columns = ['idx_of_border'] +\
        fit_parameters + fit_errors

    if len(fit_parameters) == 1:
        fit_parameters, fit_errors = [fit_parameters[0]], [fit_errors[0]]

    # create empty dataframe to keep track of results
    fit_data = pd.DataFrame(columns=columns, index=np.arange(n_points))

    if show_progress:
        iterator = tqdm.tqdm(range(n_points), desc = 'Processing vertices...')
    else:
        iterator = range(n_points)

    # Iterate over all provided target points
    for idx in iterator:

        array = np.array(intensity_along_vector.loc[idx].to_numpy())
        # Simple or fancy fit?
        if selected_fit_type == fit_types.quick_edge_fit:
            idx_of_border = edge_detection_function(array)
            perror = 0
            popt = 0
            MSE = np.array([0, 0])

        elif selected_fit_type == fit_types.fancy_edge_fit:
            popt, perror = _fancy_edge_fit(array, selected_edge_func=edge_detection_function)
            idx_of_border = popt[0]
            
            # calculate fit errors
            MSE = _mean_squared_error(
                fit_function=edge_detection_function,
                x=np.arange(len(array)),
                y=array,
                fit_params=popt)

        new_point = (start_points[idx] + idx_of_border * vector_step[idx]) * scale

        fit_data.loc[idx, fit_errors] = perror
        fit_data.loc[idx, fit_parameters] = popt
        fit_data.loc[idx, 'idx_of_border'] = idx_of_border
        fit_data.loc[idx, 'surface_points_x'] = new_point[0]
        fit_data.loc[idx, 'surface_points_y'] = new_point[1]
        fit_data.loc[idx, 'surface_points_z'] = new_point[2]
        fit_data.loc[idx, 'mean_squared_error'] = MSE[0]
        fit_data.loc[idx, 'fraction_variance_unexplained'] = MSE[1]
        fit_data.loc[idx, 'fraction_variance_unexplained_log'] = np.log(MSE[1])

    fit_data['start_points'] = list(start_points)

    if remove_outliers:
        # Remove outliers
        good_points = _identify_outliers(
            fit_data, 
            column_names=['fraction_variance_unexplained_log'],
            which=['above'],
            factor=outlier_tolerance)
        fit_data = fit_data[good_points]
        intensity_along_vector = intensity_along_vector[good_points]

    # remove NaNs from reconstructed points
    no_nan_idx = ~np.isnan(np.stack(fit_data['start_points'].to_numpy()).squeeze()).any(axis=1)
    fit_data = fit_data[no_nan_idx]
    intensity_along_vector = intensity_along_vector[no_nan_idx]

    # measure distance to nearest neighbor
    fit_data['distance_to_nearest_neighbor'] = distance_to_k_nearest_neighbors(
        fit_data[['surface_points_x',
                  'surface_points_y',
                  'surface_points_z']].to_numpy(), k=15
    )

    # reformat to layerdatatuple: points
    feature_names = fit_parameters + fit_errors +\
        ['distance_to_nearest_neighbor',
         'mean_squared_error',
        'fraction_variance_unexplained', 
        'fraction_variance_unexplained_log'] +\
        ['idx_of_border']
    features = fit_data[feature_names].to_dict('list')
    metadata = {'intensity_profiles': intensity_along_vector}
    properties = {'name': 'Refined_points',
                  'size': 1,
                  'features': features,
                  'metadata': metadata,
                  'face_color': 'cyan'}
    data = fit_data[['surface_points_x',
                     'surface_points_y',
                     'surface_points_z']].to_numpy()
    fit_data.drop(columns=['surface_points_x',
                           'surface_points_y',
                           'surface_points_z'],
                  inplace=True)
    layer_points = (data, properties, 'points')

    # reformat to layerdatatuple: normal vectors
    start_points = np.stack(fit_data['start_points'].to_numpy()).squeeze()
    trace_vectors = trace_vectors[fit_data.index.to_numpy()]
    trace_vectors = np.stack([start_points, trace_vectors]).transpose((1, 0, 2))

    properties = {'name': 'Normals'}
    layer_normals = (trace_vectors, properties, 'vectors')

    return (layer_points, layer_normals)        

def _identify_outliers(
        table: pd.DataFrame,
        column_names: list,
        which: list,
        factor: float = 1.5,
        merge: float = 'and') -> np.ndarray:
    """
    Identify outliers in a table based on the IQR method.

    Parameters
    ----------
    table : pd.DataFrame
        Table containing the data.
    column_names : list
        List of column names to check for outliers.
    which : list
        List of strings indicating which outliers to identify. Options are
        'above', 'below', and 'both'.
    factor : float, optional
        Factor to multiply the IQR with. The default is 1.5.
    merge : float, optional
        Merge outliers identified in different columns. Options are 'and' and
        'or'. The default is 'and'.
    """
    if isinstance(table, dict):
        table = pd.DataFrame(table)

    # True if values are good, False if outliers
    indices = np.ones((len(column_names), len(table)), dtype=bool)
    indices[:, table[table.isnull().any(axis=1)].index] = False

    for idx, column in enumerate(column_names):
        Q1 = table[column].quantile(0.25)
        Q3 = table[column].quantile(0.75)
        IQR = Q3 - Q1
        if which[idx] == 'above':
            idx_outlier = table[column] > (Q3 + factor * IQR)
        elif which[idx] == 'below':
            idx_outlier = table[column] < (Q1 - factor * IQR)
        elif which[idx] == 'both':
            idx_outlier = (table[column] < (Q1 - factor * IQR)) + (table[column] > (Q3 + factor * IQR))
        indices[idx, table[idx_outlier].index] = False

    if merge == 'and':
        indices = np.all(indices, axis=0)
    elif merge == 'or':
        indices = ~np.any(~indices, axis=0)

    return indices


def _fancy_edge_fit(array: np.ndarray,
                    selected_edge_func: edge_functions = edge_functions.interior
                    ) -> float:
    """
    Fit a line profile with a gaussian normal curve or a sigmoidal function.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

    Parameters
    ----------
    profile : np.ndarray
        DESCRIPTION.
    mode : int, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    float
        DESCRIPTION.
    """
    params = _function_args_to_list(selected_edge_func)[1:]
    array = [x for x in array if not np.isnan(x)]  # filter out nans
    try:
        if selected_edge_func == _sigmoid:

            # trim begin of trace to get rid of rising intensity slope
            ind_max = np.argmax(array)
            array = array[ind_max:]

            
            amplitude_est = max(array)
            center_est = np.where(np.diff(array) == np.diff(array).min())[0][0]
            background_slope = 0
            slope_est = 1
            offset_est = min(array)
            parameter_estimate = [center_est,
                                    amplitude_est,
                                    slope_est,
                                    background_slope,
                                    offset_est]

            optimal_fit_parameters, _covariance = curve_fit(
                selected_edge_func, np.arange(len(array)), array, parameter_estimate
                )
            
            optimal_fit_parameters[0] = optimal_fit_parameters[0] + ind_max

        elif selected_edge_func == _gaussian:
            parameter_estimate = [len(array)/2,
                                  len(array)/2,
                                  max(array)]
            optimal_fit_parameters, _covariance = curve_fit(
                selected_edge_func, np.arange(0, len(array), 1), array, parameter_estimate
                )

        # retrieve errors from covariance matrix
        parameter_error = np.sqrt(np.diag(_covariance))

    # If fit fails, replace bad values with NaN
    except Exception:
        optimal_fit_parameters = np.repeat(np.nan, len(params))
        parameter_error = np.repeat(np.nan, len(params))

    return optimal_fit_parameters, parameter_error

def _mean_squared_error(fit_function: callable, x: np.ndarray, y: np.ndarray, fit_params: list) -> List[float]:
    """
    Calculate error parameters for a given fit functions and the determined parameters.

    Args:
        fit_function (callable): Used fit function
        x (np.ndarray): x-values
        y (np.ndarray): measured, corresponding y-values
        fit_params (list): determined fit parameters

    Returns:
        float: mean squared error
        float: fraction of variance unexplained
    """
    is_number = ~np.isnan(y)
    x = x[is_number].squeeze()  # filter out nans
    y = y[is_number].squeeze()  # filter out nans
    y_fit = fit_function(x, *fit_params)
    
    mean_squared_error = np.mean((y - y_fit)**2)
    fraction_variance_unexplained = mean_squared_error/np.var(y)

    return mean_squared_error, fraction_variance_unexplained
