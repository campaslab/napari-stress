# -*- coding: utf-8 -*-

def test_widget(make_napari_viewer):

    from napari_stress._stress import stress_widget

    viewer = make_napari_viewer()

    widget = stress_widget(viewer)
    viewer.window.add_dock_widget(widget)

    assert len(viewer.window._dock_widgets) == 1

if __name__ == '__main__':

    import napari
    test_widget(napari.Viewer)
