def test_dropplet_point_cloud(make_napari_viewer):
    viewer = make_napari_viewer()
    sample_keys = ["PC_1", "PC_2", "PC_3"]

    for key in sample_keys:
        viewer.layers.clear()
        viewer.open_sample("napari-stress", key)
        assert len(viewer.layers) == 1
