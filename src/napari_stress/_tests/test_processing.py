# -*- coding: utf-8 -*-
import numpy as np


def test_rescaling4d():
    from napari_stress import rescale

    image = np.random.random(size=(2, 100, 100, 100))

    resampled_image = rescale(image,
                              dimension_1 = 1.0,
                              dimension_2 = 1.0,
                              dimension_3 = 1.0)
    assert np.array_equal(resampled_image.shape, image.shape)

    resampled_image = rescale(image,
                              dimension_1 = 0.5,
                              dimension_2 = 0.5,
                              dimension_3 = 0.5)
    assert np.array_equal(
        resampled_image.shape[1:], (np.asarray(image.shape)/2)[1:]
        )

def test_rescaling3D():
    from napari_stress import rescale

    image = np.random.random(size=(100, 100, 100))
    resampled_image = rescale(image,
                              dimension_1 = 1.0,
                              dimension_2 = 1.0,
                              dimension_3 = 1.0)
    assert np.array_equal(resampled_image.squeeze().shape, image.squeeze().shape)
