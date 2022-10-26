# -*- coding: utf-8 -*-
import numpy as np
from napari.types import PointsData, SurfaceData
from .._utils.frame_by_frame import frame_by_frame

from napari_tools_menu import register_function

@register_function(menu="Surfaces > Create surface from lebedev points (n-STRESS)")
@frame_by_frame
def reconstruct_surface_from_quadrature_points(points: PointsData
                                               ) -> SurfaceData:
    """
    Reconstruct the surface for a given set of quadrature points.

    Parameters
    ----------
    n_quadrature_points : int
        Number of used quadrature points

    Returns
    -------
    Sphere_Triangulation_Array : TYPE
        DESCRIPTION.

    """
    from scipy.spatial import Delaunay
    from .._stress import lebedev_write_SPB as lebedev_write

    n_quadrature_points = len(points)

    Lbdv_Cart_Pts_and_Wt_Quad = lebedev_write.Lebedev(n_quadrature_points)
    lbdv_coordinate_array = Lbdv_Cart_Pts_and_Wt_Quad[:,:-1]

    lbdv_plus_center = np.vstack(( lbdv_coordinate_array, np.array([0, 0, 0]) ))
    delauney_tetras = Delaunay(lbdv_plus_center)

    tetras = delauney_tetras.simplices
    num_tris = len(delauney_tetras.simplices)

    delauney_triangles = np.zeros(( num_tris, 3 ))

    for tri_i in range(num_tris):
        vert_ind = 0

        for tetra_vert in range(4):
            vertex = tetras[tri_i, tetra_vert]

            if(vertex != n_quadrature_points and vert_ind < 3):

                delauney_triangles[tri_i, vert_ind] = vertex
                vert_ind = vert_ind + 1

    return (points, delauney_triangles.astype(int))
