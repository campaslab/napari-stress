from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari

import numpy as np
import pandas as pd


def anisotropic_stress(
    mean_curvature_droplet: np.ndarray,
    H0_droplet: float,
    mean_curvature_ellipsoid: np.ndarray,
    H0_ellipsoid: float,
    gamma: float = 26.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate anisotropic stress from mean and averaged curvatures.

    Parameters
    ----------
    mean_curvature_droplet : np.ndarray
        mean curvature at every point on the surface of a droplet
    H0_droplet : float
        surface-integrated surface curvature on droplet
    mean_curvature_ellipsoid : np.ndarray
        mean curvature at every point on the surface of an ellipsoid that was
        fitted to a droplet. The droplet positions must correspond to the
        point locations on the droplet surface in terms of latitude and
        longitude
    H0_ellipsoid : float
        surface-integrated surface curvature on ellipsoid
    gamma : float, optional
        interfacial surface tension in mN/m. The default is 26.0. See also [1].

    Returns
    -------
    stress : np.ndarray
        raw anisotropic stress on every point on the droplet surface
    stress_tissue : np.ndarray
        tissue-scale anisotropic stress on the droplet surface
    stress_droplet : np.ndarray
        cell-scale anisotropic stress on the droplet surface


    References
    ----------
    .. [1] Campàs, Otger, et al. "Quantifying cell-generated mechanical forces
           within living embryonic tissues." Nature Methods 11.2 (2014): 183-189.

    """
    stress = 2 * gamma * (mean_curvature_droplet - H0_droplet)
    stress_tissue = 2 * gamma * (mean_curvature_ellipsoid - H0_ellipsoid)
    stress_droplet = stress - stress_tissue

    return stress, stress_tissue, stress_droplet


def maximal_tissue_anisotropy(
    ellipsoid: "napari.types.VectorsData", gamma: float
) -> float:
    """
    Calculate maximaum stress anisotropy on ellipsoid.

    Parameters
    ----------
    ellipsoid : 'napari.types.VectorsData'
    gamma : float, optional
        Interfacial surface tension in mN/m.

    Returns
    -------
    maximal_tissue_anisotropy : float
        Maximum stress anisotropy on ellipsoid

    """
    from .._approximation import EllipsoidExpander

    expander = EllipsoidExpander()
    expander.coefficients_ = ellipsoid
    expander._measure_max_min_curvatures()

    return (
        2
        * gamma
        * (
            expander.properties["maximum_mean_curvature"]
            - expander.properties["minimum_mean_curvature"]
        )
    )


def tissue_stress_tensor(
    ellipsoid: "napari.types.VectorsData", H0_ellipsoid: float, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate tissue stress tensor(s).

    Parameters
    ----------
    ellipsoid : 'napari.types.VectorsData'
        Ellipsoid that was fitted to a droplet.
    H0_ellipsoid : float
        averaged mean curvature of the ellipsoid.
    gamma : float
        droplet interfacial tension in mN/m

    Returns
    -------
    Tissue_Stress_Tensor_elliptical : np.ndarray
        3x3 orientation matrix with stresses along ellipsoid axes
    Tissue_Stress_Tensor_cartesian : np.ndarray
        3x3 orientation matrix with stresses projected onto cartesian axes
    """
    from .._measurements import mean_curvature_on_ellipse_cardinal_points
    from .._utils.coordinate_conversion import _orientation_from_ellipsoid

    cardinal_curvatures = mean_curvature_on_ellipse_cardinal_points(ellipsoid)
    orientation_matrix = _orientation_from_ellipsoid(ellipsoid)

    # use H0_Ellpsoid to calculate tissue stress projections:
    sigma_11_e = 2 * gamma * (cardinal_curvatures[0] - H0_ellipsoid)
    sigma_22_e = 2 * gamma * (cardinal_curvatures[1] - H0_ellipsoid)
    sigma_33_e = 2 * gamma * (cardinal_curvatures[2] - H0_ellipsoid)

    # tissue stress tensor (elliptical coordinates)
    Tissue_Stress_Tensor_elliptical = np.zeros((3, 3))
    Tissue_Stress_Tensor_elliptical[0, 0] = sigma_11_e
    Tissue_Stress_Tensor_elliptical[1, 1] = sigma_22_e
    Tissue_Stress_Tensor_elliptical[2, 2] = sigma_33_e

    # cartesian tissue stress tensor:
    Tissue_Stress_Tensor_cartesian = np.dot(
        np.dot(orientation_matrix.T, Tissue_Stress_Tensor_elliptical),
        orientation_matrix,
    )

    return Tissue_Stress_Tensor_elliptical, Tissue_Stress_Tensor_cartesian


def calculate_anisotropy(
    df: pd.DataFrame,
    column: str,
    alpha: float = 0.05,
    group_column: str = "time",
) -> pd.DataFrame:
    """
    Calculate anisotropy of a column in a dataframe.

    The dataframe is assumed to contain multiple groups,
    which are defined by the values in the group_column.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the data to analyze
    column : str, optional
        Column name to analyze. The column is assumed to contain numerical
        data.
    alpha : float, optional
        Lower and upper percentile of the data to exclude when calculating
        the anisotropy, by default 0.05
    group_column : str, optional
        Column name to use for grouping the data, by default 'time'

    Returns
    -------
    pd.DataFrame
        Dataframe containing the anisotropy of the data in the column for
        every group in the dataframe:
        - `column` + '_lower': lower percentile of the data
        - `column` + '_upper': upper percentile of the data
        - `column` + '_anisotropy': anisotropy of the data
    """

    # write a function to apply to every group in the dataframe
    def anisotropy(
        df: pd.DataFrame,
        alpha: float = 0.05,
        column: str = "anisotropic_stress",
    ):
        from scipy import stats

        hist_data = np.histogram(df[column], bins="auto", density=True)
        hist_dist = stats.rv_histogram(hist_data)

        smallest_excluded_value = hist_dist.ppf(alpha)
        largest_excluded_value = hist_dist.ppf(1.0 - alpha)
        return (
            smallest_excluded_value,
            largest_excluded_value,
            largest_excluded_value - smallest_excluded_value,
        )

    anisotropy_df = df.groupby(group_column).apply(
        anisotropy, alpha=alpha, column=column
    )
    anisotropy_df = anisotropy_df.apply(pd.Series)
    anisotropy_df.columns = [
        column + "_lower",
        column + "_upper",
        column + "_anisotropy",
    ]

    return anisotropy_df
