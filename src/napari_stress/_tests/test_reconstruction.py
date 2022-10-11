# -*- coding: utf-8 -*-

def test_reconstruction(make_napari_viewer):
    from napari_stress import reconstruction
    import napari_process_points_and_surfaces as nppas
    from skimage import measure
    from napari.layers import Layer

    import numpy as np

    viewer = make_napari_viewer()

    image = viewer.open(r'C:\Users\johan\Downloads\1_Drop.tif')[0].data[:5]

    widget = reconstruction.droplet_reconstruction_toolbox(viewer)
    viewer.window.add_dock_widget(widget)

    #widget._run()

    results = reconstruction.reconstruct_droplet(
        image,
        voxelsize=np.asarray([2, 1, 1]),
        target_voxelsize=2
        )

    for result in results:
        layer = Layer.create(result[0], result[1], result[2])
        viewer.add_layer(layer)

if __name__ == '__main__':
    import napari
    test_reconstruction(napari.Viewer)
