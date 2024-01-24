from napari.types import LayerDataTuple
from napari_tools_menu import register_function
import pandas as pd
import numpy as np
from .._utils.frame_by_frame import frame_by_frame


@register_function(menu="Measurement > Measure intensities on surface (n-STRESS)")
@frame_by_frame
def _measure_intensity_on_surface(
    surface: "napari.types.SurfaceData",
    image: "napari.types.ImageData",
    measurement_range: float = 1.0,
    sampling_distance: float = 0.5,
    center: bool = True,
    interpolation_method: str = "linear",
) -> LayerDataTuple:
    """
    Measure intensity on surface.

    This function exposes the `measure_intensity_on_surface` function
    to napari. It returns a tuple of (surface, properties, 'surface').

    Parameters
    ----------
    surface : SurfaceData
    image : ImageData
    measurement_range : float, optional
        Range of measurement, by default 1.0. This determines in which
        range around the surface the intensity is measured.
    sampling_distance : float, optional
        Distance between samples, by default 0.5. This determines how many samples
        are taken in the measurement range. Needs to be smaller than measurement_range.
    center : bool, optional
        Center normal vectors on surface, by default True. If set to False, the
        normal vectors point away from the surface so that the intensity is measured
        only on one side of the surface.
    interpolation_method : str, optional
        Interpolation method, by default "linear"

    Returns
    -------
    LayerDataTuple
        Tuple of (surface, properties, 'surface')
    """
    intensities = measure_intensity_on_surface(
        surface,
        image,
        measurement_range=measurement_range,
        sampling_distance=sampling_distance,
        center=center,
        interpolation_method=interpolation_method,
    )

    intensity_mean = intensities.mean(axis=1)
    intensity_std = intensities.std(axis=1)
    intensity_max = intensities.max(axis=1)
    intensity_min = intensities.min(axis=1)

    intensities["intensity_mean"] = intensity_mean
    intensities["intensity_std"] = intensity_std
    intensities["intensity_max"] = intensity_max
    intensities["intensity_min"] = intensity_min

    properties = {"metadata": {"features": intensities}, "colormap": "inferno"}

    surface = list(surface)
    surface[2] = intensities["intensity_max"].values

    return (surface, properties, "surface")


def measure_intensity_on_surface(
    surface: "napari.types.SurfaceData",
    image: "napari.types.ImageData",
    measurement_range: float = 1.0,
    sampling_distance: float = 0.5,
    center: bool = True,
    interpolation_method: str = "linear",
) -> pd.DataFrame:
    """
    Measure intensity on surface.

    Parameters
    ----------
    surface : 'napari.types.SurfaceData'
    image : 'napari.types.ImageData'
    measurement_range : float, optional
        Range of measurement, by default 1.0. This determines in which
        range around the surface the intensity is measured.
    sampling_distance : float, optional
        Distance between samples, by default 0.5. This determines how many samples
        are taken in the measurement range. Needs to be smaller than measurement_range.
    center : bool, optional
        Center normal vectors on surface, by default True. If set to False, the
        normal vectors point away from the surface so that the intensity is measured
        only on one side of the surface.
    interpolation_method : str, optional
        Interpolation method, by default "linear"

    Returns
    -------
    pd.DataFrame
        Intensity values for each point on the surface.
    """
    from .._vectors import normal_vectors_on_surface

    normal_vectors = normal_vectors_on_surface(
        surface, length_multiplier=measurement_range, center=True
    )
    intensities = sample_intensity_along_vector(
        normal_vectors,
        image,
        sampling_distance=sampling_distance,
        interpolation_method=interpolation_method,
    )

    return intensities


@register_function(menu="Measurement > Measure intensities along normals (n-STRESS)")
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

    intensities["intensity_mean"] = intensity_mean
    intensities["intensity_std"] = intensity_std
    intensities["intensity_max"] = intensity_max
    intensities["intensity_min"] = intensity_min

    properties = {
        "features": intensities,
        "edge_color": "intensity_max",
        "edge_width": 1,
        "edge_colormap": "inferno",
    }
    return (sample_vectors, properties, "vectors")


def sample_intensity_along_vector(
    sample_vectors: "napari.types.VectorsData",
    image: "napari.types.ImageData",
    sampling_distance: float = 1.0,
    interpolation_method: str = "linear",
) -> pd.DataFrame:
    """
    Sample intensity along sample_vectors of equal length.

    Parameters
    ----------
    sample_vectors : 'napari.types.VectorsData'
        Vectors along which to measure intensity. Must be of equal length.
    image : 'napari.types.ImageData'
        Image to sample intensity from.
    sampling_distance : float, optional
        Distance between samples, by default 1.0
    interpolation_method : str, optional
        Interpolation method, by default "linear". See
        `scipy.interpolate.RegularGridInterpolator` for available methods.

    Returns
    -------
    pd.DataFrame
        Intensity values for each point on the sample_vectors.

    See Also
    --------
    `RegularGridInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html>`_
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
