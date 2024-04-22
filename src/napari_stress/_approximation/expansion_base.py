import numpy as np
from abc import ABC, abstractmethod


class Expander(ABC):
    def __init__(self):
        self.coefficients_ = None
        self.properties = {}

    def fit(self, points: "napari.types.PointsData"):
        self.coefficients_ = self._fit(points)
        return self

    def expand(self, points: "napari.types.PointsData"):
        expanded_points = self._expand(points)
        self._calculate_properties(points, expanded_points)
        return expanded_points

    def fit_expand(self, points: "napari.types.PointsData"):
        self.fit(points)
        return self.expand(points)

    @abstractmethod
    def _fit(self, points: "napari.types.PointsData"):
        raise NotImplementedError

    @abstractmethod
    def _expand(self, points: "napari.types.PointsData"):
        raise NotImplementedError

    @abstractmethod
    def _calculate_properties(self, input_points, output_points):
        raise NotImplementedError
