import numpy as np

from .expansion_base import Expander
from .._utils.frame_by_frame import frame_by_frame
from napari_tools_menu import register_function


class EllipsoidExpander(Expander):
    """
    Expand a set of points to fit an ellipsoid.

    Parameters
    ----------
    points : napari.types.PointsData
        The points to expand.
    """

    def __init__(self):
        super().__init__()

    def _fit(self, points: "napari.types.PointsData") -> "napari.types.VectorsData":
        """
        Fit a 3D ellipsoid to given points using least squares fitting.

        The ellipsoid equation is: Ax^2 + By^2 + Cz^2 + Dxy + Exz + Fyz + Gx + Hy + Iz = 1

        Parameters
        ----------
        points : napari.types.PointsData
            The points to fit an ellipsoid to.

        Returns
        -------
        ellipsoid_fitted_ : napari.types.VectorsData
            The fitted ellipsoid.
        """
        coefficients = self._fit_ellipsoid_to_points(points)
        self.center_, self.axes_, self._eigenvectors = self._extract_characteristics(
            coefficients
        )

        vectors = (
            self._eigenvectors
            / np.linalg.norm(self._eigenvectors, axis=1)[:, np.newaxis]
        ).T * self.axes_[:, None]
        base = np.stack([self.center_] * 3)
        ellipsoid_fitted_ = np.stack([base, vectors], axis=1)

        return ellipsoid_fitted_

    def _expand(self, points: "napari.types.PointsData"):
        """
        Expand a set of points to fit an ellipsoid.

        Parameters
        ----------
        points : napari.types.PointsData
            The points to expand.

        Returns
        -------
        expanded_points : napari.types.PointsData
            The expanded points.
        """
        from .._utils.coordinate_conversion import (
            cartesian_to_elliptical,
            elliptical_to_cartesian,
        )

        U, V = cartesian_to_elliptical(self.coefficients_, points, invert=True)
        expanded_points = elliptical_to_cartesian(U, V, self.coefficients_, invert=True)

        return expanded_points

    def _calculate_properties(self, input_points, output_points):
        """
        Measure properties of the expansion.

        Parameters
        ----------
        input_points : napari.types.PointsData
            The points before expansion.
        output_points : napari.types.PointsData
            The points after expansion.
        """

        distance = np.linalg.norm(input_points - output_points, axis=1)
        self.properties["residuals"] = distance

    def _fit_ellipsoid_to_points(
        self,
        points: "napari.types.PointsData",
    ) -> np.ndarray:
        """
        Fit an ellipsoid to a set of points.

        The used equation is of the form:
        x^2 / a^2 + y^2 / b^2 + z^2 / c^2 + 2xy / ab + 2xz / ac + 2yz / bc + 2dx / a + 2ey / b + 2fz / c = 1

        Parameters
        ----------
        points : napari.types.PointsData
            The points to fit an ellipsoid to.

        Returns
        -------
        ellipsoid_coefficients : np.ndarray
            The coefficients of the ellipsoid equation.
        """
        # Extract x, y, z coordinates from points and reshape to column vectors
        x = points[:, 0, np.newaxis]
        y = points[:, 1, np.newaxis]
        z = points[:, 2, np.newaxis]

        # Construct the design matrix for the ellipsoid equation
        design_matrix = np.hstack((x**2, y**2, z**2, x * y, x * z, y * z, x, y, z))
        column_of_ones = np.ones_like(x)  # Column vector of ones

        # Perform least squares fitting to solve for the coefficients
        transposed_matrix = design_matrix.transpose()
        matrix_product = np.dot(transposed_matrix, design_matrix)
        inverse_matrix = np.linalg.inv(matrix_product)
        coefficients = np.dot(inverse_matrix, np.dot(transposed_matrix, column_of_ones))

        # Append -1 to the coefficients to represent the constant term on the right side of the equation
        ellipsoid_coefficients = np.append(coefficients, -1)

        return ellipsoid_coefficients

    def _extract_characteristics(self, coefficients: np.ndarray):
        # Construct the augmented matrix from the coefficients
        Amat = np.array(
            [
                [
                    coefficients[0],
                    coefficients[3] / 2.0,
                    coefficients[4] / 2.0,
                    coefficients[6] / 2.0,
                ],
                [
                    coefficients[3] / 2.0,
                    coefficients[1],
                    coefficients[5] / 2.0,
                    coefficients[7] / 2.0,
                ],
                [
                    coefficients[4] / 2.0,
                    coefficients[5] / 2.0,
                    coefficients[2],
                    coefficients[8] / 2.0,
                ],
                [
                    coefficients[6] / 2.0,
                    coefficients[7] / 2.0,
                    coefficients[8] / 2.0,
                    -1,
                ],
            ]
        )

        # Extract the quadratic part and find its inverse
        A3 = Amat[:3, :3]
        A3inv = np.linalg.inv(A3)

        # Compute the center of the ellipsoid
        ofs = coefficients[6:9] / 2.0
        center = -np.dot(A3inv, ofs)

        # Transform the matrix to center the ellipsoid at the origin
        Tofs = np.eye(4)
        Tofs[3, :3] = center
        R = np.dot(Tofs, np.dot(Amat, Tofs.T))

        # Extract the transformed quadratic part
        R3 = R[:3, :3]

        # Perform eigendecomposition to find axes and orientation
        eigenvalues, eigenvectors = np.linalg.eig(R3 / -R[3, 3])

        # Compute the lengths of the axes
        axes_lengths = np.sqrt(1.0 / np.abs(eigenvalues))

        return center, axes_lengths, eigenvectors


@register_function(menu="Points > Fit ellipsoid to pointcloud (n-STRESS)")
@frame_by_frame
def fit_ellipsoid_to_pointcloud(
    points: "napari.types.PointsData",
) -> "napari.types.VectorsData":
    """
    Fit an ellipsoid to a set of points.

    Parameters
    ----------
    points : napari.types.PointsData
        The points to fit an ellipsoid to.

    Returns
    -------
    ellipsoid : napari.types.VectorsData
        The fitted ellipsoid.
    """
    expander = EllipsoidExpander()
    expander.fit(points)

    ellipsoid = expander.coefficients_

    return ellipsoid


@register_function(menu="Points > Expand point locations on ellipsoid (n-STRESS)")
@frame_by_frame
def expand_points_on_fitted_ellipsoid(
    points: "napari.types.PointsData",
) -> "napari.types.PointsData":
    """
    Project a set of points on a fitted ellipsoid.

    Parameters
    ----------
    points : napari.types.PointsData
        The points to project.

    Returns
    -------
    projected_points : napari.types.PointsData
        The projected points.
    """
    expander = EllipsoidExpander()
    expanded_points = expander.fit_expand(points)

    return expanded_points
