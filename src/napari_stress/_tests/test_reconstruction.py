# -*- coding: utf-8 -*-

def test_reconstruction(make_napari_viewer):
    from napari_stress import reconstruction, get_droplet_4d
    import napari_process_points_and_surfaces as nppas
    from skimage import measure
    from napari.layers import Layer

    import numpy as np

    viewer = make_napari_viewer()


    image = get_droplet_4d()[0][0]
    viewer.add_image(image)

    widget = reconstruction.droplet_reconstruction_toolbox(viewer)
    viewer.window.add_dock_widget(widget)

    results = reconstruction.reconstruct_droplet(
        image,
        voxelsize=np.asarray([1.93, 1, 1]),
        target_voxelsize=1
        )

    for result in results:
        layer = Layer.create(result[0], result[1], result[2])
        viewer.add_layer(layer)
