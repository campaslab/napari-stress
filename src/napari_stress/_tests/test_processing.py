# -*- coding: utf-8 -*-
import numpy as np
from napari_stress import rescale

def test_rescaling():

    image = np.random.random(size=(2, 100, 100, 100))

    resampled_image = rescale(image, scale_factors = np.asarray([1, 1, 1]))
    assert np.array_equal(resampled_image.shape, image.shape)

    resampled_image = rescale(image, scale_factors = np.asarray([0.5, 0.5, 0.5]))
    assert np.array_equal(
        resampled_image.shape[1:], (np.asarray(image.shape)/2)[1:]
        )

if __name__ == '_main__':
    test_rescaling()
