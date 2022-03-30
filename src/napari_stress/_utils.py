import numpy as np
import vedo

from napari.types import PointsData

def pointcloud_to_vertices4D(surfs: list) -> np.ndarray:

    n_vertices = sum([surf.N() for surf in surfs])
    vertices_4d = np.zeros([n_vertices, 4])

    for idx, surf in enumerate(surfs):
        vertices_4d[idx * surf.N() : idx * surf.N() + surf.N(), 1:] = surf.points()
        vertices_4d[idx * surf.N() : idx * surf.N() + surf.N(), 0] = idx

    return vertices_4d

def vertices4d_to_pointcloud(vertices: np.ndarray) -> list:

    assert vertices.shape[1] == 4

    frames = np.unique(vertices[:, 0])

    surfs =  []
    for idx in frames:
        frame = vertices(np.where(vertices[:, 0] == idx))
        surfs.append(vedo.pointcloud.Points(frame))

    return surfs

def list_of_points_to_pointsdata(points: list) -> PointsData:
    """
    Convert list of pointData objects to single pointsdata object

    Parameters
    ----------
    points : list
        DESCRIPTION.

    Returns
    -------
    PointsData
        DESCRIPTION.
    """
    # First, split list in list of cordinates and properties
    list_of_properties = [pt[1]['properties'] for pt in points]
    list_of_verts = [pt[0] for pt in points]

    # Create time-index for each point in each frame
    time_frames = np.concatenate([
        [idx] * len(timepoint) for idx, timepoint in enumerate(list_of_verts)
        ])

    new_points = np.zeros((len(time_frames), 4))
    new_points[:, 0] = time_frames
    new_points[:, 1:] = np.vstack(list_of_verts)

    new_props = {}
    for key in list(list_of_properties[0].keys()):
        new_props[key] = np.vstack([tp[key] for tp in list_of_properties])

    return (new_points, {'properties': new_props}, 'Points')
