from napari.types import LayerDataTuple
from napari_tools_menu import register_function
import pandas as pd
import numpy as np
from .._utils.frame_by_frame import frame_by_frame


@register_function(
    menu="Measurement > Calculate intensities along normals (n-STRESS)")
@frame_by_frame
def _sample_intensity_along_vector(
    sample_vectors: "napari.types.VectorsData",
    image: "napari.types.ImageData",
    sampling_distance: float = 1.0,
    interpolation_method: str = "linear",
) -> LayerDataTuple:
    """
    Sample intensity along sample_vectors of equal length.

    This function exposes the `sample_intensity_along_vector` function
    to napari. It returns a tuple of (sample_vectors, properties, 'vectors').
    It requires the vectors to be of equal length.

    Parameters
    ----------
    sample_vectors : VectorsData
        sample_vectors along which to measure intensity
    image : ImageData
        Image to sample
    sampling_distance : float, optional
        Distance between samples, by default 1.0
    interpolation_method : str, optional
        Interpolation method, by default "linear"

    Returns
    -------
    LayerDataTuple
        Tuple of (sample_vectors, properties, 'vectors')
    """
    intensities = sample_intensity_along_vector(
        sample_vectors,
        image,
        sampling_distance=sampling_distance,
        interpolation_method=interpolation_method,
    )
    intensity_mean = intensities.mean(axis=1)
    intensity_std = intensities.std(axis=1)
    intensity_max = intensities.max(axis=1)
    intensity_min = intensities.min(axis=1)

    intensities['intensity_mean'] = intensity_mean
    intensities['intensity_std'] = intensity_std
    intensities['intensity_max'] = intensity_max
    intensities['intensity_min'] = intensity_min

    properties = {
        'features': intensities,
        'edge_color': 'intensity_max',
        'edge_width': 1,
        'edge_colormap': 'inferno'
    }
    return (sample_vectors, properties, 'vectors')


def sample_intensity_along_vector(
    sample_vectors: "napari.types.sample_vectorsData",
    image: "napari.types.ImageData",
    sampling_distance: float = 1.0,
    interpolation_method: str = "linear",
) -> pd.DataFrame:
    """
    Sample intensity along sample_vectors of equal length.

    Args:
        sample_vectors (napari.types.sample_vectorsData): sample_vectors
            along which to measure intensity
        image (napari.types.ImageData): Image to sample
        sampling_distance (float, optional): Distance between samples.
            Defaults to 1.0.
        interpolation_method (str, optional): Interpolation method.
            Defaults to 'linear'.
    """
    from scipy.interpolate import RegularGridInterpolator

    # Create coords for interpolator
    X1 = np.arange(0, image.shape[0], 1)
    X2 = np.arange(0, image.shape[1], 1)
    X3 = np.arange(0, image.shape[2], 1)
    interpolator = RegularGridInterpolator(
        (X1, X2, X3),
        image,
        bounds_error=False,
        fill_value=np.nan,
        method=interpolation_method,
    )

    # get start point and unit normal vector
    start_points = sample_vectors[:, 0]
    lengths = np.linalg.norm(sample_vectors[:, 1], axis=1)[:, np.newaxis]
    unit_vector = sample_vectors[:, 1] / lengths

    # calculate number of steps
    steps = np.round(lengths / sampling_distance).mean().astype(int)

    # sample intensity
    intensity = np.zeros((len(start_points), steps))

    coordinates_along_sample_vectors = []
    for idx in range(len(start_points)):
        _directions = [
            sampling_distance * unit_vector[idx] * i for i in np.arange(steps)
        ]
        _directions = np.stack(_directions)

        _sample_vectors = start_points[idx] + _directions
        coordinates_along_sample_vectors.append(_sample_vectors)

    for idx in range(len(start_points)):
        intensity[idx, :] = interpolator(coordinates_along_sample_vectors[idx])

    intensity = pd.DataFrame(np.stack(intensity))
    intensity.columns = [f"step_{idx}" for idx in range(steps)]

    return intensity
