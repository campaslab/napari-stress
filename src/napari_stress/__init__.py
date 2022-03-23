
__version__ = "0.0.1"

from ._stress import stress_widget
from napari_plugin_engine import napari_hook_implementation

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return [
        stress_widget
    ]
