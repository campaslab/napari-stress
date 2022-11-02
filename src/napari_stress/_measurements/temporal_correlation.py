import numpy as np
from napari.types import SurfaceData
import pandas as pd

def temporal_autocorrelation(df: pd.DataFrame,
                             feature: str,
                             frame_column_name: str = 'frame'):
    """
    Calculate temporal autocorrelation for a list of features.

    Args:
        features (list): List of features - each entry corresponds to features
        of a single timeframe

    Returns:
        np.ndarray: temporal autocorrelation. The i-th entry denotes the correlation of
        features at time i with the feature at time 0.
    """

    # convert dataframe into list of features for every frame
    assert frame_column_name in df.columns
    features = [x[1][feature].to_numpy() for x in list(df.groupby(frame_column_name))]

    n_frames = len(features)
    inner_product = np.zeros((n_frames, n_frames))

    for i in range(n_frames):
        for j in range(i, n_frames):
            inner_product[i, j] = np.sum(features[j - i] * features[j])
    inner_product_sum = np.sum(inner_product, axis=1)

    temporal_autocorrelation = []
    for tau in range(n_frames):
        autocorrelation = ( inner_product_sum[tau]/(n_frames - tau) )/( inner_product_sum[0]/n_frames )
        temporal_autocorrelation.append(autocorrelation)

    return temporal_autocorrelation

def haversine_distances(degree_lebedev: int, n_lebedev_points: int):
    """
    Calculate geodesic (Great Circle) distance matrix on unit sphere, from haversine formula.

    See Also
    --------
    [0] https://en.wikipedia.org/wiki/Haversine_formula

    Args:
        deg_lbdv (_type_): _description_
        num_lbdv_pts (_type_): _description_

    Returns:
        _type_: _description_
    """
    from .._stress.lebedev_info_SPB import lbdv_info
    #num_lbdv_pts = LBDV_Input.lbdv_quad_pts
    
    LBDV_Input = lbdv_info(degree_lebedev, n_lebedev_points)
    distances = np.zeros(( n_lebedev_points, n_lebedev_points ))

    for pt_1 in range(n_lebedev_points):
        theta_1 = LBDV_Input.theta_pts[pt_1, 0]
        phi_1 = LBDV_Input.phi_pts[pt_1, 0]
        lat_1 = np.pi - phi_1 # latitude

        for pt_2 in range(pt_1+1, n_lebedev_points): # dist with self is 0
            theta_2 = LBDV_Input.theta_pts[pt_2, 0]
            phi_2 = LBDV_Input.phi_pts[pt_2, 0]
            lat_2 = np.pi - phi_2 # latitude
            
            lat_diff = lat_2 - lat_1
            long_diff = theta_2 - theta_1			
            
            h = np.sin(lat_diff/2.)**2 + np.cos(lat_1)*np.cos(lat_2)*(np.sin(long_diff**2/2.)**2)
            d_12 = np.arctan2(np.sqrt(h),np.sqrt(1. -h))
            
            distances[pt_1, pt_2] = d_12
            distances[pt_2, pt_1] = d_12

    return distances