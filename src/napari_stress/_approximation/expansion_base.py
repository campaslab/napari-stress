from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari


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

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
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

    @property
    def properties(self):
        """
        Get the properties of the fitted expander.

        Returns
        -------
        properties : dict
            The properties of the fitted expander.
        """
        return self._properties
