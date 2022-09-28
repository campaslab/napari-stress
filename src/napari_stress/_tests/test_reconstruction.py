# -*- coding: utf-8 -*-

def test_reconstruction(make_napari_viewer):
    from napari_stress import reconstruction
    viewer = make_napari_viewer()

    image = viewer.open(r'C:\Users\johamuel\Desktop\droplet_scaled.tif')
    binary = viewer.open(r'C:\Users\johamuel\Desktop\droplet_binary.tif')

    widget = reconstruction.droplet_reconstruction_toolbox(viewer)
    viewer.window.add_dock_widget(widget)

    widget._run()

if __name__ == '__main__':
    import napari
    test_reconstruction(napari.Viewer)
