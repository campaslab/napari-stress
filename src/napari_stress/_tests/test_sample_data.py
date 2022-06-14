# -*- coding: utf-8 -*-

def test_dropplet_point_cloud(make_napari_viewer):
    from napari_stress import get_droplet_point_cloud
    
    data = get_droplet_point_cloud()[0]
    
    viewer = make_napari_viewer()
    viewer.add_points(data[0], **data[1])
