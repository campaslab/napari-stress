from napari.types import ImageData

from skimage import transform

import numpy as np
from ._utils.time_slicer import frame_by_frame


@frame_by_frame
def rescale(image: ImageData,
            dimension_1: float = 1.0,
            dimension_2: float = 1.0,
            dimension_3: float = 1.0) -> ImageData:
    """
    Rescale an image by a given set of scale factors.

    If the image is a 2D image, only the first two parameters
    (`dimension_1` and `dimension_2`) will be used.

    Parameters
    ----------
    image : ImageData
    dimension_1 : float, optional
        factor by which to scale the image along the first dimension. The default is 1.0.
    dimension_2 : float, optional
        factor by which to scale the image along the second dimension. The default is 1.0.
    dimension_3 : float, optional
        factor by which to scale the image along the third dimension. The default is 1.0.

    Returns
    -------
    ImageData

    """

    scale_factors = np.asarray([dimension_1, dimension_2, dimension_3])

    # Make sure to use the correct amount of factors if dimension of image is smaller than three
    scale_factors = scale_factors[:len(image.shape)]

    return transform.rescale(image, scale=scale_factors)
