from ._surface import smooth_sinc
from napari_plugin_engine import napari_hook_implementation

@napari_hook_implementation
def napari_provide_sample_data():
    from . import get_droplet_point_cloud, get_droplet_point_cloud_4d, get_droplet_4d

    return {
        "Droplet pointcloud": get_droplet_point_cloud,
        "Droplet pointcloud (4D)": get_droplet_point_cloud_4d,
        "Droplet image (4D)": get_droplet_4d
    }
