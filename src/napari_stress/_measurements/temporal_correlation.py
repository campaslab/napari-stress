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

def spatio_temporal_autocorrelation(surfaces: SurfaceData,
                                    distance_matrix: np.ndarray,
                                    maximal_distance: float = None):
    """
    Spatio-temporal autocorrelation.

    Args:
        surfaces (SurfaceData): 4D-surface object with values
        distance_matrix (np.ndarray): Distance matrix to be used on the surface.
        maximal_distance (float, optional): _description_. Defaults to None.
    """
    # Calculate Spatio-Temporal Corrs of total stresses:
    from .geodesics import correlation_on_surface
    from .._utils.frame_by_frame import TimelapseConverter

    # Convert 4D surface into list of surfaces
    Converter = TimelapseConverter()
    list_of_surfaces = Converter.data_to_list_of_data(surfaces, layertype=SurfaceData)
    n_frames = len(list_of_surfaces)    

    # get bins for spatial correlation on surface
    result = correlation_on_surface(list_of_surfaces[0], list_of_surfaces[0], distance_matrix)
    distances = len(result['auto_correlations_distances'].flatten())
    inner_product = np.zeros(( n_frames, n_frames, distances ))

#	if maximal_distance is None:
#		maximal_distance = int(np.floor(max(dists_lbdv_non0)))

    for i in range(n_frames):
        for j in range(i, n_frames):
            result = correlation_on_surface(list_of_surfaces[j - i], list_of_surfaces[j], distance_matrix)
            inner_product[i, j, :] = result['auto_correlations_average'].flatten()

    # sum vals of cols for each row, gives us a matrix
    summed_inner_product = np.squeeze(np.sum(inner_product, axis=1)) 
    num_tau_samples = (np.arange(n_frames)+1)[::-1].reshape(n_frames, 1)
    
    avg_summed_inner_product = np.divide(summed_inner_product, num_tau_samples)
    norm_t_0 = np.sum(summed_inner_product[0, :].flatten() )
    
    normed_avg_summed_inner_product = avg_summed_inner_product/norm_t_0

    results = {'summed_spatiotemporal_correlations': summed_inner_product,
               'avg_summed_spatiotemporal_correlations': avg_summed_inner_product,
               'normed_avg_summed_spatiotemporal_correlations': normed_avg_summed_inner_product}

    return results

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