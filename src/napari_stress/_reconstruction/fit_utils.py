import numpy as np
import pandas as pd

from enum import Enum
from scipy.optimize import curve_fit
from typing import List
import inspect


def _sigmoid(
    array: np.ndarray,
    center: float,
    amplitude: float,
    slope: float,
    background_slope: float,
    offset: float,
):
    """
    Sigmoidal fit function
    https://stackoverflow.com/questions/55725139/fit-sigmoid-function-s-shape-curve-to-data-using-python
    """
    return (amplitude + background_slope * (-1) * (array - center)) / (
        1 + np.exp((-1) * slope * (array - center))
    ) + offset


def _gaussian(array: np.ndarray, center: float, sigma: float, amplitude: float):
    """
    Gaussian normal fit function
    https://en.wikipedia.org/wiki/Normal_distribution
    """
    return (
        amplitude
        / np.sqrt((2 * np.pi * sigma**2))
        * np.exp(-((array - center) ** 2) / (2 * sigma**2))
    )


def _detect_maxima(array: np.ndarray, center: float = None):
    """
    Function to find the maximum's index of data with a single peak.
    """
    return np.argmax(array)


def _detect_max_gradient(array: np.ndarray, center: float = None):
    """
    Function to find the location of the steepest gradient for sigmoidal data
    """
    return np.argmax(np.abs(np.diff(array)))


def _function_args_to_list(function: callable) -> list:
    sig = inspect.signature(function)
    return list(sig.parameters.keys())


class fit_types(Enum):
    quick_edge_fit = "quick"
    fancy_edge_fit = "fancy"


class interpolation_types(Enum):
    linear = "linear"
    cubic = "cubic"
    nearest = "nearest"


class edge_functions(Enum):
    interior = {"fancy": _sigmoid, "quick": _detect_max_gradient}
    surface = {"fancy": _gaussian, "quick": _detect_maxima}


def _identify_outliers(
    table: pd.DataFrame,
    column_names: list,
    which: list,
    factor: float = 1.5,
    merge: float = "and",
) -> np.ndarray:
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
        if which[idx] == "above":
            idx_outlier = table[column] > (Q3 + factor * IQR)
        elif which[idx] == "below":
            idx_outlier = table[column] < (Q1 - factor * IQR)
        elif which[idx] == "both":
            idx_outlier = (table[column] < (Q1 - factor * IQR)) + (
                table[column] > (Q3 + factor * IQR)
            )
        indices[idx, table[idx_outlier].index] = False

    if merge == "and":
        indices = np.all(indices, axis=0)
    elif merge == "or":
        indices = ~np.any(~indices, axis=0)

    return indices


def _fibonacci_sampling(number_of_points: int = 256) -> "napari.types.PointsData":
    """
    Sample points on unit sphere according to fibonacci-scheme.

    See Also
    --------
    http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/

    Parameters
    ----------
    number_of_points : int, optional
        Number of points to be sampled. The default is 256.

    Returns
    -------
    PointsData

    """
    goldenRatio = (1 + 5**0.5) / 2
    i = np.arange(0, number_of_points)
    theta = 2 * np.pi * i / goldenRatio
    phi = np.arccos(1 - 2 * (i + 0.5) / number_of_points)
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)

    return np.stack([x, y, z]).T


def _fancy_edge_fit(
    array: np.ndarray, selected_edge_func: edge_functions = edge_functions.interior
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
            # estimate parameters and trim intensity array
            parameter_estimate, trimmed_array, trimmed_idx = estimate_fit_parameters(
                array
            )
            trimmed_indices = np.arange(trimmed_idx[0], trimmed_idx[1])

            boundaries = np.array(
                [
                    [0, len(array)],
                    [0.5 * max(array), max(array) * 1.5],
                    [-np.inf, np.inf],
                    [-np.inf, np.inf],
                    [0, 0.5 * max(array)],
                ]
            )

            # run fit
            optimal_fit_parameters, _covariance = curve_fit(
                selected_edge_func,
                trimmed_indices,
                trimmed_array,
                parameter_estimate,
                bounds=boundaries.T,
            )

        elif selected_edge_func == _gaussian:
            parameter_estimate = [len(array) / 2, len(array) / 2, max(array)]
            optimal_fit_parameters, _covariance = curve_fit(
                selected_edge_func,
                np.arange(0, len(array), 1),
                array,
                parameter_estimate,
            )

        # retrieve errors from covariance matrix
        parameter_error = np.sqrt(np.diag(_covariance))

    # If fit fails, replace bad values with NaN
    except Exception:
        optimal_fit_parameters = np.repeat(np.nan, len(params))
        parameter_error = np.repeat(np.nan, len(params))

    return optimal_fit_parameters, parameter_error


def estimate_fit_parameters(intensity):
    """
    Estimate starting parameters for the sigmoidal fit.

    Function definiton:
    f(x) = (amp+lambda*(x-mu))/(1 + exp(-k*(x-mu)) + offset

    Args:
        intensity (np.ndarray): intensity profile

    Returns:
        tuple: starting parameters
            k: slope of the sigmoid
            center: center of the sigmoid
            lambda_val: lambda value of the sigmoid
            amp: amplitude of the sigmoid
            offset: offset of the sigmoid
        np.ndarray: trimmed intensity profile
    """
    # Check if there are too few points
    if len(intensity) < 5:
        print("Warning: Too few points in trace (length < 5)")
        return [np.nan] * 5, intensity, intensity

    # Calculate the first derivative (dY)
    dY = 0.5 * (
        np.diff([intensity[0]] + intensity) + np.diff(intensity + [intensity[-1]])
    )

    # Identify X3 as the position of the maximum change
    indX3 = np.argmax(abs(dY))

    # Calculate the second derivative (ddY)
    ddY = 0.5 * (np.diff([dY[0]] + list(dY)) + np.diff(list(dY) + [dY[-1]]))

    # Define X2 and X4
    ddY_left, ddY_right = ddY.copy(), ddY.copy()
    ddY_left[indX3:] = np.nan
    ddY_right[:indX3] = np.nan
    indX2 = np.nanargmax(ddY_left)
    indX4 = np.nanargmax(ddY_right)

    # Refine indices for X2 and X4
    fullWidth = indX4 - indX2
    halfWidth = fullWidth / 2
    indX2 = int(indX3 - halfWidth)
    indX2 = max(1, min(len(intensity) - 2, indX2))
    indX4 = int(indX3 + halfWidth)
    indX4 = max(2, min(len(intensity) - 1, indX4))

    # Estimate parameters
    k = 4 / (indX4 - indX2)
    if intensity[indX4] > intensity[indX2]:
        k = -k
    center = indX3
    amp = 2 * intensity[indX3]

    if k > 0:
        offset = intensity[-1]
        lambda_val = (intensity[indX2] - intensity[0]) / (indX2 - 0)
    else:
        offset = intensity[0]
        lambda_val = (intensity[-1] - intensity[indX4]) / (len(intensity) - 1 - indX4)

    # Return starting parameters and trimmed intensity values
    return [center, amp, -k, lambda_val, offset], intensity[indX2:indX4], [indX2, indX4]


def _mean_squared_error(
    fit_function: callable, x: np.ndarray, y: np.ndarray, fit_params: list
) -> List[float]:
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

    mean_squared_error = np.mean((y - y_fit) ** 2)
    fraction_variance_unexplained = mean_squared_error / np.var(y)

    return mean_squared_error, fraction_variance_unexplained
