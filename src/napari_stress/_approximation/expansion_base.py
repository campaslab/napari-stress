import numpy as np
import pandas as pd


class Expander:
    def __init__(self, max_degree: int = 5):
        self.max_degree = max_degree
        self.coefficients_ = None
        self.properties = pd.DataFrame()

    def fit(self, points: np.ndarray):
        self._data = points
        self.coefficients_ = self._fit(points)
        return self

    def expand(self, points: "napari.types.PointsData"):
        expanded_points = self._expand(points)
        self._measure_properties(points, expanded_points)
        return expanded_points

    def fit_expand(self, points: "napari.types.PointsData"):
        self.fit(points)
        return self.expand(points)

    def _fit(self, points: "napari.types.PointsData"):
        raise NotImplementedError

    def _expand(self, points: "napari.types.PointsData"):
        raise NotImplementedError

    def _measure_properties(self, input_points, output_points):
        raise NotImplementedError
