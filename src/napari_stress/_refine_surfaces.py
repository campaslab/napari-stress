# -*- coding: utf-8 -*-

import vedo

from napari_tools_menu import register_function

from napari.types import SurfaceData, LayerDataTuple, ImageData

from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit
import numpy as np
import tqdm
import pandas as pd

from enum import Enum
from typing import List

class fit_methods(Enum):
    quick_edge_fit = 0
    fancy_edge_fit = 1

class fluorescence_types(Enum):
    interior = {'mode': 0, 'n_params': 4}
    surface = {'mode': 1, 'n_params': 3}

@register_function(menu="Surfaces > Retrace surface vertices (vedo, nppas)")
def trace_refinement_of_surface(image: ImageData,
                                surface: SurfaceData,
                                trace_length: float = 2.0,
                                sampling_distance: float = 0.1,
                                fit_method: fit_methods = fit_methods.fancy_edge_fit,
                                fluorescence: fluorescence_types = fluorescence_types.interior,
                                scale: np.ndarray = np.array([1.0, 1.0, 1.0]),
                                show_progress: bool = True
                                )-> List[LayerDataTuple]:
    """
    Generate intensity profiles along traces.

    The profiles are interpolated from the input image with linear interpolation
    Parameters
    ----------
    image : np.ndarray
        Input intensity image from which traces will be calculated
    start_pts : np.ndarray
        Nx3 or 1x3 array of points that should be used as base points for trace
        vectors. If vector is 1x3-sized, the same coordinate will be used as
        base point for all trace vectors.
    target_pts : np.ndarray
        Nx3-sized array of points that are used as direction vectors for each
        trace. Traces will be calculated along the line from the base points
        `start_pts` towards `target_pts`
    sampling_distance : float, optional
        Distance between sampled intensity along a trace. Default is 1.0
    detection : str, optional
        Detection method, can be `quick_edge` or `advanced`. The default is'quick_edge'.
    fluorescence : str, optional
        Fluorescence type of water droppled, can be `interior` or `surface`.
        The default is 'interior'.

    Returns
    -------
    surface_points : np.ndarray
        Nx3 array of points on the surface of the structure in the image.
    errors : np.ndarray
        Nx1 array of error values for the points on the surface.
    FitParams : np.ndarray
        Nx3 array with determined fit parameters if detection was set to `advanced`
    """

    # parse inputs
    if isinstance(fit_method, int):
        fit_method = fit_methods(fit_method)

    if isinstance(fluorescence, int):
        fluorescence = fluorescence_types(fluorescence)

    # Convert to mesh and calculate normals
    mesh = vedo.mesh.Mesh((surface[0], surface[1]))
    mesh.computeNormals()

    # Define start and end points for the surface tracing vectors
    n_samples = int(trace_length/sampling_distance)
    start_pts = mesh.points()/scale[None, :] - 0.5 * trace_length * mesh.pointdata['Normals']

    # Define trace vectors for full length and for single step
    vectors = trace_length * mesh.pointdata['Normals']
    v_step = vectors/n_samples

    # Create coords for interpolator
    X1 = np.arange(0, image.shape[0], 1)
    X2 = np.arange(0, image.shape[1], 1)
    X3 = np.arange(0, image.shape[2], 1)
    rgi = RegularGridInterpolator((X1, X2, X3), image, bounds_error=False,
                                  fill_value=image.min())

    # Allocate arrays for results
    surface_points = np.zeros_like(surface[0])
    idx_of_border = np.zeros(mesh.N())
    fit_errors = []
    fit_params = []
    profiles = []


    if show_progress:
        tk = tqdm.tqdm(range(mesh.N(), desc = 'Processing vertices...'))
    else:
        tk = range(mesh.N())

    # Iterate over all provided target points
    for idx in tk:

        profile_coords = [start_pts[idx] + k * v_step[idx] for k in range(n_samples)]
        profiles.append(rgi(profile_coords))

        if fit_method == fit_methods.quick_edge_fit:
            _idx_of_border = _quick_edge_fit(profiles[idx],
                                             mode=fluorescence.value['mode'])
            perror = np.array(0)
            popt = np.array(0)
        elif fit_method == fit_methods.fancy_edge_fit:
            try:
                popt, perror = _fancy_edge_fit(profiles[idx],
                                               mode=fluorescence.value['mode'])
            except:
                popt = np.repeat(np.nan, fluorescence.value['n_params'])
                perror = np.repeat(np.nan, fluorescence.value['n_params'])

            _idx_of_border = popt[0]

        idx_of_border[idx] = _idx_of_border
        fit_errors.append(perror)
        fit_params.append(popt)

    # convert to arrays
    fit_errors = np.vstack(fit_errors)
    fit_params = np.vstack(fit_params)

    # Filter points to remove points with high fit errors
    good_pts = _remove_outliers_by_index(pd.DataFrame(fit_errors))

    start_pts = start_pts[good_pts]
    idx_of_border = idx_of_border[good_pts]
    v_step = v_step[good_pts]
    fit_params = fit_params[good_pts, :]
    fit_errors = fit_errors[good_pts, :]
    profiles = np.array(profiles)[good_pts]

    # get coordinate of surface point and errors (respect scale for this)
    surface_points = (start_pts + idx_of_border[:, None] * v_step) * scale[None, :]
    vectors = np.array([surface_points, idx_of_border[:, None] * v_step]).transpose((1, 0, 2))

    #TODO
    # Turn relevant parameters into single dataframe to attach to returned points layer
    # A viewer on such properties can then visualize this

    properties = {'fit_errors': fit_errors, 'fit_params': fit_params, 'profiles': profiles}

    return [(surface_points, {'properties': properties}, 'Points'),
            (vectors, {'name': 'Trace vectors'}, 'Vectors')]

def _remove_outliers_by_index(df):
    "Filter all rows that qualify as outliers based on column-statistics."
    indices = np.ones(len(df), dtype=bool)
    df = df.dropna()
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        indices[df[df[col] > (Q3 + 1.5 * IQR)].index] = False
    return indices


def _fancy_edge_fit(profile: np.ndarray, mode: int = 0) -> float:
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
    if mode == 0:

        # Make sure that intensity goes up along ray so that fit can work
        if profile[0] > profile[-1]:
            profile = profile[::-1]

        p0 = [max(profile), len(profile/2), np.diff(profile).mean(), min(profile)]
        popt, pcov = curve_fit(_sigmoid, np.arange(0, len(profile), 1), profile, p0)
        perr = np.sqrt(np.diag(pcov))
        return popt, perr

    elif mode == 1:
        p0 = [len(profile)/2, len(profile)/2, max(profile)]
        popt, pcov = curve_fit(_gaussian, np.arange(0, len(profile), 1), profile, p0)
        perr = np.sqrt(np.diag(pcov))
        return popt, perr

def _quick_edge_fit(profile: np.ndarray, mode = 0) -> int:

    if mode == 0:
        return np.argmax(np.abs(np.diff(profile)))
    elif mode == 1:
        return np.argmax(profile)

def _sigmoid(x, center, amplitude, slope, offset):
    "https://stackoverflow.com/questions/55725139/fit-sigmoid-function-s-shape-curve-to-data-using-python"
    return amplitude / (1 + np.exp(-slope*(x-center))) + offset

def _gaussian(x, center, sigma, amplitude):
    return amplitude/np.sqrt((2*np.pi*sigma**2)) * np.exp(-(x - center)**2 / (2*sigma**2))
