# -*- coding: utf-8 -*-

from napari.types import PointsData
from .._stress import manifold_SPB as mnfd

import numpy as np

def calculate_mean_curvature_on_manifold(manifold: mnfd.manifold,
                                         max_degree: int) -> np.ndarray:
    """
    Calculate mean curvatures for a set of lebedev points.

    Parameters
    ----------
    lebedev_points : PointsData
    lebedev_fit : lebedev_info.lbdv_info
        lebedev info object - provides info which point is defined by which chart.
    max_degree : int
        Degree of spherical harmonics expansion.

    Returns
    -------
    np.ndarray
        Mean curvature value for every lebedev point.

    """
    normals = get_normals_on_manifold(manifold, lebedev_fit)

    # Test orientation:
    points = manifold.get_coordinates()
    centered_lbdv_pts = points - points.mean(axis=0)[None, :]

    # Makre sure orientation is inward, so H is positive (for Ellipsoid, and small deviations):
    Orientations = [np.dot(x, y) for x, y in zip(centered_lbdv_pts,  normals)]
    num_pos_orr = np.sum(np.asarray(Orientations).flatten() > 0)

    Orientation = 1. # unchanged (we want INWARD)
    if(num_pos_orr > .5 * len(centered_lbdv_pts)):
        Orientation = -1.

    return Orientation*euc_kf.Combine_Chart_Quad_Vals(manifold.H_A_pts, manifold.H_B_pts, lebedev_fit).squeeze()
