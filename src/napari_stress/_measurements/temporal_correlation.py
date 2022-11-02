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
