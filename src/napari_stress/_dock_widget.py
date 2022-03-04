from napari_plugin_engine import napari_hook_implementation
from ._stress import stress_widget


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return [
        stress_widget
    ]
