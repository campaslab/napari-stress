# -*- coding: utf-8 -*-

import vedo

from napari_tools_menu import register_function

from napari.types import SurfaceData, ImageData, LayerDataTuple
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
    interior = 0
    surface = 1

@register_function(menu="Surfaces > Retrace surface vertices (vedo, nppas)")
def trace_refinement_of_surface(image: ImageData,
                                surface: SurfaceData,
                                trace_length: float = 2.0,
                                sampling_distance: float = 0.1,
                                fit_method: fit_methods = fit_methods.quick_edge_fit,
                                fluorescence: fluorescence_types = fluorescence_types.interior
                                )-> List[LayerDataTuple]:

    # parse inputs
    if isinstance(fit_method, int):
        pass
    else:
        fit_method = fit_method.value

    if isinstance(fluorescence, int):
        pass
    else:
        fluorescence = fluorescence.value

    # Convert to mesh and calculate normals
    mesh = vedo.mesh.Mesh((surface[0], surface[1]))
    mesh.computeNormals()

    # Define start and end points for the surface tracing vectors
    trace_length = np.repeat(trace_length, mesh.N())
    start_points = mesh.points() - (1/2)*trace_length[:, None] * mesh.pointdata['Normals']
    target_points = mesh.points() + (1/2)*trace_length[:, None] * mesh.pointdata['Normals']

    new_points, properties = _get_traces(image,
                                         start_points,
                                         target_points,
                                         sampling_distance=sampling_distance,
                                         fit_method=fit_method,
                                         fluorescence=fluorescence)
    surf = ((new_points,  properties['fit_errors']),
            {'colormap': 'magma'},
            'Points')
    vector_data = np.array([start_points, target_points - start_points])
    vectors = (vector_data.transpose((1,0,2)), {}, 'Vectors')
    return [surf, vectors]

def _get_traces(image: np.ndarray,
               start_pts: np.ndarray,
               target_pts: np.ndarray,
               sampling_distance: float =  0.1,
               fit_method: int = 0,
               fluorescence: int = 0) -> [np.ndarray, dict]:
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
    # allocate vector for origin points, repeat N times if only one coordinate is provided
    if len(start_pts.shape) == 1:
        start_pts = np.asarray([start_pts]).repeat(target_pts.shape[0], axis=0)

    # Calculate trace lengths for given start/target points and calculate
    # number of sampling points given the passed sampling rate
    vectors = target_pts - start_pts

    trace_lengths = np.linalg.norm(vectors, axis=1)
    n_samples = np.asarray(trace_lengths/sampling_distance, dtype=int)
    v_step = vectors/n_samples[:, None]

    # Create coords for interpolator
    X1 = np.arange(0, image.shape[0], 1)
    X2 = np.arange(0, image.shape[1], 1)
    X3 = np.arange(0, image.shape[2], 1)
    rgi = RegularGridInterpolator((X1, X2, X3), image)

    # Allocate arrays for results
    surface_points = np.zeros((len(start_pts), 3))
    idx_of_border = np.zeros(len(start_pts))
    fit_errors = []
    fit_params = []
    profiles = []

    # Iterate over all provided target points
    for idx in tqdm.tqdm(range(target_pts.shape[0])):

        profile_coords = [start_pts[idx] + k * v_step[idx] for k in range(n_samples[idx])]
        profiles.append(rgi(profile_coords))

        if fit_method == 0:
            _idx_of_border = _quick_edge_fit(profiles[idx], mode=fluorescence)
            perror = np.array(0)
            popt = np.array(0)
        elif fit_method == 1:
            popt, perror = _fancy_edge_fit(profiles[idx], mode=fluorescence)
            _idx_of_border = popt[0]

        idx_of_border[idx] = _idx_of_border
        fit_errors.append(perror)
        fit_params.append(popt)

    # convert to arrays
    fit_errors = np.asarray(fit_errors)
    fit_params = np.asarray(fit_params)
    params_matrix = np.concatenate([fit_errors, fit_params], axis=1)
    df = pd.DataFrame(params_matrix, columns=[''])

    # Filter bad points
    good_pts = _remove_outliers_by_index(df)
    start_pts = start_pts[good_pts]
    idx_of_border = idx_of_border[good_pts]
    v_step = v_step[good_pts]
    fit_params = fit_params[good_pts, :]
    fit_errors = fit_errors[good_pts, :]
    profiles = profiles[good_pts]

    # get coordinate of surface point and errors
    surface_points = start_pts + idx_of_border[:, None] * v_step

    return surface_points, {'fit_errors': fit_errors,
                            'fit_params': fit_params,
                            'profiles': profiles}


def _fancy_edge_fit(profile: np.ndarray, mode: int = 0) -> float:
    "https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html"

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

def _sigmoid(x, x0, L, k, b):
    "https://stackoverflow.com/questions/55725139/fit-sigmoid-function-s-shape-curve-to-data-using-python"
    return L / (1 + np.exp(-k*(x-x0))) + b

def _gaussian(x, x0, sigma, A):
    return A/np.sqrt((2*np.pi*sigma**2)) * np.exp(-(x - x0)**2 / (2*sigma**2))
