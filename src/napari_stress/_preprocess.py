from napari_tools_menu import register_function
from napari.types import ImageData

from skimage.transform import rescale
from skimage import filters, measure

import pandas as pd
import numpy as np
import vedo


def reshape(image: np.ndarray,
            vsx: float,
            vsy: float,
            vsz: float,
            res_mode: str = 'high'):
    """
    Preprocesses an input 3D image for further processing. Preprocessing includes
    resampling to isotropic voxels.

    Parameters
    ----------
    image : ndarray
        3/4D image array with intensity values in each pixel
    vsx : float, optional
        pixel size in x-dimension. Default value: vsx = 2.076
    vsx : float, optional
        pixel size in x-dimension. Default value: vsx = 2.076
    vsx : float, optional
        pixel size in x-dimension. Default value: vsx = 3.998

    Returns
    -------
    image : MxNxK array
        binary image

    """
    image_resampled = []

    # resample every timepoint
    for t in range(image.shape[0]):
        _image_resampled = resample(image[t], vsx=vsx, vsy=vsy, vsz=vsz)
        image_resampled.append(_image_resampled)

    image_resampled = np.asarray(image_resampled)

    return image_resampled


@register_function(menu="Process > Resample image (skimage, ns)")
def resample(image: ImageData,
             vsz: float,
             vsy: float,
             vsx: float,
             res_mode='high') -> ImageData:
    """
    Resample an image with anistropic voxels of size vsx, vsy and vsz to isotropic
    voxel size of smallest or largest resolution
    """
    # choose final voxel size
    if res_mode == 'high':
        vs = np.min([vsx, vsy, vsz])

    elif res_mode == 'low':
        vs = np.max([vsx, vsy, vsz])

    factor = np.asarray([vsz, vsy, vsx])/vs
    image_rescaled = rescale(image, factor, anti_aliasing=True)

    return image_rescaled

# def fit_curvature():
#     """
#     Find curvature for every point
#     """

#     print('\n---- Curvature-----')
#     curv = []
#     for idx, point in tqdm.tqdm(self.points.iterrows(), desc='Measuring mean curvature', total=len(self.points)):
#         sXYZ, sXq = surface.get_patch(self.points, idx, self.CoM)
#         curv.append(curvature.surf_fit(sXYZ, sXq))

#     self.points['Curvature'] = curv
#     self.points = surface.clean_coordinates(self)

#     # Raise flags for provided data
#     self.has_curv = True
