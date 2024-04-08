import numpy as np


class Expander:
    def __init__(self, get_measurements: bool = False):
        self.coefficients_ = None
        self.properties = dict()
        self._get_measurements = get_measurements

    def fit(self, points: np.ndarray):
        self._data = points
        self.coefficients_ = self._fit(points)
        return self

    def expand(self, points: "napari.types.PointsData"):
        expanded_points = self._expand(points)
        if self._get_measurements:
            self._measure_properties(points, expanded_points)
            self._measure_residuals()
        return expanded_points

    def fit_expand(self, points: "napari.types.PointsData"):
        self.fit(points)
        return self.expand(points)

    def _fit(self, points: "napari.types.PointsData"):
        raise NotImplementedError

    def _expand(self, points: "napari.types.PointsData"):
        raise NotImplementedError

    def _measure_properties(self, input_points, output_points):
    def _measure_residuals(self):
        raise NotImplementedError

        raise NotImplementedError
