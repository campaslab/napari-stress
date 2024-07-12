from abc import ABC, abstractmethod


class Expander(ABC):
    """
    Abstract base class for an expander.

    This class should be subclassed when creating new types of expanders.
    Subclasses must implement the _fit, _expand, and _calculate_properties methods.
    """

    def __init__(self):
        """
        Initialize the expander.
        """
        self._coefficients = None
        self._properties = {}

    def fit(self, points: "napari.types.PointsData"):
        """
        Fit the expander to the given points.

        Parameters
        ----------
        points : napari.types.PointsData
            The points to fit the expander to.

        Returns
        -------
        self
        """
        self._coefficients = self._fit(points)
        return self

    def expand(self, points: "napari.types.PointsData"):
        """
        Expand the given points using the fitted expander.

        Parameters
        ----------
        points : napari.types.PointsData
            The points to expand.

        Returns
        -------
        expanded_points : napari.types.PointsData
            The expanded points.
        """
        expanded_points = self._expand(points)
        self._calculate_properties(points, expanded_points)
        return expanded_points

    def fit_expand(self, points: "napari.types.PointsData"):
        """
        Fit the expander to the given points and then expand them.

        Parameters
        ----------
        points : napari.types.PointsData
            The points to fit the expander to and then expand.

        Returns
        -------
        expanded_points : napari.types.PointsData
            The expanded points.
        """
        self.fit(points)
        return self.expand(points)

    @abstractmethod
    def _fit(self, points: "napari.types.PointsData"):
        """
        Fit the expander to the given points.

        This method should be implemented by subclasses.

        Parameters
        ----------
        points : napari.types.PointsData
            The points to fit the expander to.

        Returns
        -------
        coefficients : array-like
            The coefficients of the fitted expander.
        """
        raise NotImplementedError

    @abstractmethod
    def _expand(self, points: "napari.types.PointsData"):
        """
        Expand the given points using the fitted expander.

        This method should be implemented by subclasses.

        Parameters
        ----------
        points : napari.types.PointsData
            The points to expand.

        Returns
        -------
        expanded_points : napari.types.PointsData
            The expanded points.
        """
        raise NotImplementedError

    @abstractmethod
    def _calculate_properties(self, input_points, output_points):
        """
        Calculate properties of the expanded points.

        This method should be implemented by subclasses.

        Parameters
        ----------
        input_points : napari.types.PointsData
            The original points.
        output_points : napari.types.PointsData
            The expanded points.
        """
        raise NotImplementedError

    @property
    def coefficients_(self):
        """
        Get the coefficients of the fitted expander.

        Returns
        -------
        coefficients : array-like
            The coefficients of the fitted expander.
        """
        return self._coefficients

    @coefficients_.setter
    def coefficients_(self, value):
        """
        Set the coefficients of the fitted expander.

        Parameters
        ----------
        value : array-like
            The new coefficients for the expander.
        """
        self._coefficients = value
