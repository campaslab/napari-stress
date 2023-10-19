# -*- coding: utf-8 -*-


def test_dropplet_point_cloud():
    from napari_stress import sample_data
    import numpy as np

    data = sample_data.get_droplet_point_cloud()[0]
    data_4d = sample_data.get_droplet_point_cloud_4d()[0]
    image_data_4d = sample_data.get_droplet_4d()[0]
    ellipsoid = sample_data.make_blurry_ellipsoid()[0]

    assert np.array_equal(data[0].shape, (1024, 4))
    assert data_4d[0].shape[0] == 26704
    assert len(image_data_4d[0]) == 21
    assert np.array_equal(ellipsoid.shape, (64, 64, 64))
