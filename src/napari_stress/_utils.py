# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 10:31:29 2021

@author: johan
"""

import numpy as np
import pandas

from scipy.spatial import Delaunay

def filter_dataframe(df, columns = [], criteria=[], **kwargs):
    """
    Filter a dataframe with regard to values in columns and provided criteria.
    The rows in the dataframe that do not pass the criteria are then removed or
    marked for removal in an additional columns

    Parameters
    ----------
    df : pandas dataframe

    columns : list of dataframe columns: [column1, column2, ...]
        DESCRIPTION. The default is [].
    criteria : list of filter criteria, which can be either ['above', 'below', 'within']
        DESCRIPTION. List of criteria by which the values in the corresponding
        column should be filtered.

    **inplace** default True. Removes elements that do not satisfy the filters from the dataframe.
        False: An additional column 'pass' will be added to the dataframe .

    **scale** multiplication factor of interquartile distance: If a value exceeds
        I75 + scale * interquartile_distance, it is considered an outlier.

    Returns
    -------
    pandas dataframe

    """

    inplace = kwargs.get('inplace', True)
    scale = kwargs.get('scale', 1.5)
    verbose = kwargs.get('verbose', True)

    if len(columns) != len(criteria):
        raise Exception('Number of criteria ({len(criteria)} must match number of filtered columns ({len(columns)})! ')

    # Allocate vector referring to pass/fail of df entry
    idx_pass = [True] * len(df)

    # iterate over all supplied filters
    for criterion, column in zip(criteria, columns):

        # skip column if all values are zero.
        if np.sum(df[column]) == 0:
            pass

        I25, median, I75, IQ = get_IQs(df[column])

        if criterion == 'above':
            threshold = I25 - scale * IQ
            idx_pass_crit = df[column] > threshold

            if verbose:
                print(f'Removed {np.sum(~idx_pass_crit)} points based on {column}-criterion')

        elif criterion == 'below':
            threshold = I25 - scale * IQ
            idx_pass_crit = df[column] < threshold

            if verbose:
                print(f'Removed {np.sum(~idx_pass_crit)} points based on {column}-criterion')

        elif criterion == 'within':
            threshold_low = I25 - scale * IQ
            threshold_upp = I75 + scale * IQ
            idx_pass_crit = (df[column] > threshold_low) * (df[column] < threshold_upp)

            if verbose:
                print(f'Removed {np.sum(~idx_pass_crit)} points based on {column}-criterion')

        # apply new filter to filter vector
        idx_pass = idx_pass * idx_pass_crit

    if inplace:
        df = df[idx_pass].reset_index(drop=True)
    else:
        df['pass'] = idx_pass

    return df


def get_IQs(array):
    """Calculate quantiles and interquartile distance for a given array"""

    array = np.asarray(array).flatten()

    median = np.median(array)
    I25 = np.quantile(array, 0.25)
    I75 = np.quantile(array, 0.75)
    IQ = I75 - I25

    return I25, median, I75, IQ


def cart2sph(x, y, z):

    radius = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/radius)
    phi = np.zeros_like(theta)

    for i in range(x.shape[0]):
        if x[i] >= 0:
            phi[i] = np.arctan(y[i]/x[i])
        else:
            phi[i] = np.arctan(y[i]/x[i]) + np.pi

    return phi, theta, radius


def df2ZYX(df):
    "Convert the XYZ columns of a dataframe to an Nx3 array"

    return np.vstack([df.Z, df.Y, df.X]).transpose()
