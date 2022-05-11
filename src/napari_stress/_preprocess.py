from napari.types import ImageData

from skimage import transform

import numpy as np
from ._utils.time_slicer import frame_by_frame


@frame_by_frame
def resample(image: ImageData,
            scale_factors: np.ndarray) -> ImageData:
    """ Resample an image according to passed scale factors"""

    return transform.rescale(image, scale=scale_factors)
