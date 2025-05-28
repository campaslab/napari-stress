from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import napari


def sanitize_faces(
    surface: "napari.types.SurfaceData",
) -> "napari.types.SurfaceData":
    """
    Sanitize the faces of a surface to ensure they are in the correct format.

    This function ensures that the faces of the surface are in the correct
    format and order. It also ensures that the vertices are in the correct
    format.

    Parameters
    ----------
    surface : 'napari.types.SurfaceData'
        The surface data to sanitize.

    Returns
    -------
    'napari.types.SurfaceData'
        The sanitized surface data.
    """
    # Ensure faces are in the correct format
    faces = surface[1]
    vertices = surface[0]
    center = np.mean(vertices, axis=0)

    for idx, triangle in enumerate(faces):
        v1 = vertices[int(triangle[0])]
        v2 = vertices[int(triangle[1])]
        v3 = vertices[int(triangle[2])]

        normal = np.cross(v2 - v1, v3 - v1)
        normal = normal / np.linalg.norm(normal)
        center_triangle = (v1 + v2 + v3) / 3

        if np.dot(normal, center_triangle - center) < 0:
            faces[idx] = faces[idx][::-1]

    return (vertices.astype(np.float64), faces.astype(np.int32))
