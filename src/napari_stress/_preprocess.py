from napari.types import ImageData

from skimage import transform

import numpy as np
from ._utils.time_slicer import frame_by_frame


@frame_by_frame
def rescale(image: ImageData,
            scale_factors: np.ndarray) -> ImageData:
    """
    Rescale an image by a given set of scale factors

    Parameters
    ----------
    image : ImageData
    scale_factors : np.ndarray
        Factors by which image should be scaled. The length of the passed scale
        factors should match the image dimensions. For instance, rescaling a
        `[100x100x100]` image requires a `scale_factors` vector of length 3.

    Returns
    -------
    ImageData
    """

    return transform.rescale(image, scale=scale_factors)
