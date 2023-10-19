# -*- coding: utf-8 -*-


def test_dropplet_point_cloud():
    from napari_stress import sample_data
    import numpy as np

    data = sample_data.get_droplet_point_cloud()[0]
    data_4d = sample_data.get_droplet_point_cloud_4d()[0]
    image_data_4d = sample_data.get_droplet_4d()[0]
    ellipsoid = sample_data.make_blurry_ellipsoid()[0]

    assert np.arrayequal(data.shape, (1000, 3))
    assert data_4d.shape[0] == 4
    assert len(image_data_4d) == 21
    assert np.arrayequal(ellipsoid.shape, (64, 64, 64))