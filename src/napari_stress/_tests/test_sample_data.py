# -*- coding: utf-8 -*-

def test_dropplet_point_cloud():
    from napari_stress import sample_data

    data = sample_data.get_droplet_point_cloud()[0]
    data_4d = sample_data.get_droplet_point_cloud_4d()[0]
    image_data_4d = sample_data.get_droplet_4d()[0]
    ellipsoid = sample_data.make_binary_ellipsoid()[0]
