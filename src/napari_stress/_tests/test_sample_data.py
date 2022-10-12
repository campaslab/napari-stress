# -*- coding: utf-8 -*-

def test_dropplet_point_cloud(make_napari_viewer):
    from napari_stress import (get_droplet_point_cloud,
                               get_droplet_point_cloud_4d,
                               get_droplet_4d)

    data = get_droplet_point_cloud()[0]
    data_4d = get_droplet_point_cloud_4d()[0]
    image_data_4d = get_droplet_4d()[0]

    viewer = make_napari_viewer()
    viewer.add_points(data[0], **data[1])
    viewer.add_points(data_4d[0], **data_4d[1])

    viewer.add_image(image_data_4d[0])
