from .._stress import lebedev_info_SPB as lebedev_info
import numpy as np


def Least_Squares_Harmonic_Fit(
    fit_degree: int, sample_locations: tuple, values: np.ndarray
) -> np.ndarray:
    """
    Perform least squares harmonic fit on input points.

    Parameters
    ----------

    fit_degree: int
    sample_locations: tuple
        Input points in elliptical coordinates - required least squares
        fit to find ellipsoid major/minor axes
    values: np.ndarray
        Values to be expanded on the surface. Can be cartesian point coordinates
        (x/y/z) or radii for a radial expansion.

    Returns
    -------
    coefficients: np.ndarray
        Numpy array holding spherical harmonics expansion coefficients. The size
        on the type of expansion (cartesian or radial).
    """

    U, V = sample_locations[0], sample_locations[1]

    All_Y_mn_pt_in = []

    for n in range(fit_degree + 1):
        for m in range(-1 * n, n + 1):
            Y_mn_coors_in = []
            Y_mn_coors_in = lebedev_info.Eval_SPH_Basis(m, n, U, V)
            All_Y_mn_pt_in.append(Y_mn_coors_in)
    All_Y_mn_pt_in_mat = np.hstack((All_Y_mn_pt_in))

    coefficients = np.linalg.lstsq(All_Y_mn_pt_in_mat, values)[0]
    return coefficients
