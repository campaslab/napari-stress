# ruff: noqa: F821
import pandas as pd


def distance_to_k_nearest_neighbors(
    points: "napari.types.PointsData", k=5
) -> pd.DataFrame:
    """Calculate the distance to the k nearest neighbors for each point.

    Parameters
    ----------
    points : 'napari.types.PointsData'
        The points to calculate the distance to the k nearest neighbors for.
    k : int, optional
        The number of nearest neighbors to use for the calculation, by default 5.

    Returns
    -------
    pd.DataFrame
        A dataframe with the distance to the k nearest neighbors for each point.
    """
    from scipy.spatial import KDTree

    tree = KDTree(points)
    dist, _ = tree.query(points, k=k + 1)

    # measure distance to nearest neighbor
    tree = KDTree(points)
    dist, _ = tree.query(points, k=5)

    # calculate the mean distance to the k nearest neighbors
    df = pd.DataFrame(
        dist[:, 1:].mean(axis=1),
        columns=[f"distance_to_{k}_nearest_neighbor"],
    )
    return df
