from typing import TYPE_CHECKING

import numpy as np

from .expansion_base import Expander

if TYPE_CHECKING:
    import napari


class EllipsoidExpander(Expander):
    """
    Expand a set of points to fit an ellipsoid using least squares fitting.

    The ellipsoid equation is of the form:

    .. math::
        Ax^2 + By^2 + Cz^2 + Dxy + Exz + Fyz + Gx + Hy + Iz = 1

    where A, B, C, D, E, F, G, H, I are the coefficients of the ellipsoid equation and
    x, y, z are the coordinates of the points. The parameters of this equation are
    fitted to the input points using least squares fitting.

    Methods
    -------
    fit(points: "napari.types.PointsData")
        Fit an ellipsoid to a set of points using leaast square fitting.

    expand(points: "napari.types.PointsData")
        Project a set of points onto their respective position on the fitted ellipsoid.

    fit_expand(points: "napari.types.PointsData")
        Fit an ellipsoid to a set of points and then expand them.

    Attributes
    ----------
    coefficients_ : napari.types.VectorsData
        Coefficients of the fitted ellipsoid. The coefficients are of the form
        (3, 2, 3); The first dimension represents the three axes of the ellipsoid
        (major, medial and minor). The second dimension represents the components of
        the ellipsoid vectors (base point and direction vector). The third dimension
        represents the dimension of the space (z, y, x).
    axes_ : np.ndarray
        Lengths of the axes of the ellipsoid.
    center_ : np.ndarray
        Center of the ellipsoid.
    properties : dict
        Dictionary containing properties of the expansion with following keys:
        - residuals: np.ndarray
            Residual euclidian distance between input points and expanded points.
        - normals: napari.types.VectorsData
            Normals on the ellipsoid at the input points.
        - mean_curvature: np.ndarray
            Calculate point-wise mean curvature :math:`H_i` for an ellipsoid.

            The formula for :math:`H_i` is given as:

            .. math::
                H(U, V) = \\frac{
                a_0 a_1 a_2 \\Bigg(
                3 \\left(a_0^2 + a_1^2\\right)
                + 2 \\left(a_2^2\\right)
                + \\left(a_0^2 + a_1^2 - 2 a_2^2\\right) \\cos(2V)
                - 2 \\left(a_0^2 - a_1^2\\right) \\cos(2U) \\sin^2(V)
                \\Bigg)
                }{
                8 \\Bigg(
                \\left(a_0 a_1 \\cos(V)\\right)^2
                + \\left(a_2 \\sin(V)\\right)^2
                \\Big(
                \\left(a_1 \\cos(U)\\right)^2 + \\left(a_0 \\sin(U)\\right)^2
                \\Big)
                \\Bigg)^{1.5}
                }

        - principal_curvatures1: np.ndarray
            First principal curvature at the input points. Calculated as the maximum curvature at the input points:
        - principal_curvatures2: np.ndarray
            Second principal curvature at the input points. Calculated as the minimum curvature at the input points.
        - h0_ellipsoid: np.ndarray
            Averaged mean curvature :math:`H_0` on the ellipsoid, calculated as the mean of the point-wise mean curvature :math:`H_i`.
        - h_e123_ellipsoid: np.ndarray
            Maximum, medial and minimum mean curvature of the ellipsoid.
        - maximum_mean_curvature: float
            Maximum mean curvature of the ellipsoid.
        - minimum_mean_curvature: float
            Minimum mean curvature of the ellipsoid.

            The maximum and minimum curvatures :math:`H_{max}` and :math:`H_{min}` are calculated as follows:

            .. math::
                H_{max} = a / (2 * c^2) + a / (2 * b^2)

                H_{min} = c / (2 * b^2) + c / (2 * a^2)

            where a, b and c are the lengths of the ellipsoid axes along the three spatial dimensions.


    Examples
    --------

    .. code-block:: python

        # Instantiate and fit an ellipsoid expander to a set of points
        expander  = EllipsoidExpander()
        expander.fit(points)

        # Expand the points on the fitted ellipsoid
        fitted_points = expander.expand(points)

    """

    def __init__(self):
        super().__init__()

    def fit(
        self, points: "napari.types.PointsData"
    ) -> "napari.types.VectorsData":
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
        self._ellipse_coefficients = self._fit_ellipsoid_to_points(points)
        self._center, self._axes, self._eigenvectors = (
            self._extract_characteristics(self._ellipse_coefficients)
        )

        vectors = (
            self._eigenvectors
            / np.linalg.norm(self._eigenvectors, axis=1)[:, np.newaxis]
        ).T * self._axes[:, None]
        base = np.stack([self._center] * 3)
        ellipsoid_fitted_ = np.stack([base, vectors], axis=1)

        self.coefficients_ = ellipsoid_fitted_

    def expand(self, points: "napari.types.PointsData"):
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

        U, V = cartesian_to_elliptical(self._coefficients, points, invert=True)
        expanded_points = elliptical_to_cartesian(
            U, V, self._coefficients, invert=True
        )

        self._calculate_properties(points, expanded_points)

        return expanded_points

    def fit_expand(
        self, points: "napari.types.PointsData"
    ) -> "napari.types.PointsData":
        """
        Fit an ellipsoid to a set of points and then expand them.

        Parameters
        ----------
        points : napari.types.PointsData
            The points to fit an ellipsoid to and then expand.

        Returns
        -------
        expanded_points : napari.types.PointsData
            The expanded points.
        """
        self.fit(points)
        return self.expand(points)

    def _calculate_properties(self, input_points, output_points):
        """
        Measure properties of the expansion.
        """
        self._measure_residuals(input_points, output_points)
        self._measure_max_min_curvatures()
        self._normals_on_ellipsoid(output_points)
        self._calculate_curvatures(output_points)

    @property
    def coefficients_(self):
        """
        The coefficients of the fitted ellipsoid

        Returns
        -------
        coefficients_ : napari.types.VectorsData
            The coefficients of the ellipsoid equation. The coefficients are of the form
            (3, 2, 3); The first dimension represents the three axes of the ellipsoid
            (major, medial and minor). The second dimension represents the components of
            the ellipsoid vectors (base point and direction vector). The third dimension
            represents the dimension of the space (z, y, x).
        """
        return super().coefficients_

    @coefficients_.setter
    def coefficients_(self, value: "napari.types.VectorsData"):
        """
        value: (3, 2, D) matrix representing the ellipsoid coefficients.
        """
        if value is not None:
            self._center = value[0, 0]
            self._axes = np.linalg.norm(value[:, 1], axis=1)
            self._coefficients = value

    @property
    def axes_(self):
        """
        The lengths of the axes of the ellipsoid.

        Returns
        -------
        axes_ : np.ndarray
            The lengths of the axes of the ellipsoid.
        """
        return self._axes

    @property
    def center_(self):
        """
        The center of the ellipsoid.

        Returns
        -------
        center_ : np.ndarray
            The center of the ellipsoid.
        """
        return self._center

    def _measure_max_min_curvatures(self):
        """
        Measure maximum and minimum curvatures of the ellipsoid.

        Returns
        -------
        None

        """
        # get and remove the largest, smallest and medial axis
        semi_axis_sorted = np.sort(self._axes)
        a = semi_axis_sorted[2]
        b = semi_axis_sorted[1]
        c = semi_axis_sorted[0]

        # accoording to paper (https://www.biorxiv.org/content/10.1101/2021.03.26.437148v1.full)
        maximum_mean_curvature = a / (2 * c**2) + a / (2 * b**2)
        minimum_mean_curvature = c / (2 * b**2) + c / (2 * a**2)

        self._properties["maximum_mean_curvature"] = maximum_mean_curvature
        self._properties["minimum_mean_curvature"] = minimum_mean_curvature

    def _measure_residuals(self, input_points, output_points):
        """
        Measure residuals of the expansion.

        Parameters
        ----------
        input_points : napari.types.PointsData
            The points before expansion.
        output_points : napari.types.PointsData
            The points after expansion.
        """
        distance = np.linalg.norm(input_points - output_points, axis=1)
        self._properties["residuals"] = distance

    def _normals_on_ellipsoid(self, points: "napari.types.PointsData"):
        """
        Calculate normals on the ellipsoid at given points.
        Parameters
        ----------
        points : napari.types.PointsData
            The points on the ellipsoid where normals are to be calculated.

        Returns
        -------
        None
        """

        A, B, C, D, E, F, G, H, J = self._ellipse_coefficients[:-1]
        xx = points[:, 0][:, None]
        yy = points[:, 1][:, None]
        zz = points[:, 2][:, None]

        grad_F_x = 2.0 * A * xx + D * yy + E * zz + G
        grad_F_y = 2.0 * B * yy + D * xx + F * zz + H
        grad_F_z = 2.0 * C * zz + E * xx + F * yy + J

        grad_F_X = np.hstack((grad_F_x, grad_F_y, grad_F_z))
        Vec_Norms = np.sqrt(
            np.sum(np.multiply(grad_F_X, grad_F_X), axis=1)
        ).reshape(len(xx), 1)
        grad_F_X_normed = np.divide(grad_F_X, Vec_Norms)

        self.properties["normals"] = np.stack(
            [points, grad_F_X_normed]
        ).transpose((1, 0, 2))

    def _calculate_curvatures(
        self,
        sample_points: "napari.types.PointsData",
    ):
        """
        Calculate mean curvature on ellipsoid.

        Parameters
        ----------
        sample_points : napari.types.PointsData
            Sample points to calculate mean curvature on.

        Returns
        -------
        result : dict
            Dictionary containing the mean curvature, principal curvatures and the averaged mean curvature
            on the ellipsoid.
        """
        result = _curvature_on_ellipsoid(
            self.coefficients_, sample_points, self.axes_
        )
        for key, value in result.items():
            self.properties[key] = value

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
        design_matrix = np.hstack(
            (x**2, y**2, z**2, x * y, x * z, y * z, x, y, z)
        )
        column_of_ones = np.ones_like(x)  # Column vector of ones

        # Perform least squares fitting to solve for the coefficients
        transposed_matrix = design_matrix.transpose()
        matrix_product = np.dot(transposed_matrix, design_matrix)
        inverse_matrix = np.linalg.inv(matrix_product)
        coefficients = np.dot(
            inverse_matrix, np.dot(transposed_matrix, column_of_ones)
        )

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
                    coefficients[9],
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


class EllipsoidImageExpander(Expander):
    """
    Expand a set of points to fit an ellipsoid estimated from a 3D image volume.
    The ellipsoid is estimated from the image volume using a thresholding method
    and least squares fitting.s

    The ellipsoid equation is of the form:

    .. math::
        Ax^2 + By^2 + Cz^2 + Dxy + Exz + Fyz + Gx + Hy + Iz = 1

    where A, B, C, D, E, F, G, H, I are the coefficients of the ellipsoid equation and
    x, y, z are the coordinates of the points. The parameters of this equation are
    fitted to the input image volume using least squares fitting.

    Parameters
    ----------
    fluorescence : str
        The type of fluorescence to use for estimating the ellipsoid.
        Can be either 'interior' or 'surface'.
        Default is 'interior'.

    Attributes
    ----------
    coefficients_ : napari.types.VectorsData
        Coefficients of the fitted ellipsoid. The coefficients are of the form
        (3, 2, 3); The first dimension represents the three axes of the ellipsoid
        (major, medial and minor). The second dimension represents the components of
        the ellipsoid vectors (base point and direction vector). The third dimension
        represents the dimension of the space (z, y, x).
    axes_ : np.ndarray
        Lengths of the axes of the ellipsoid.
    center_ : np.ndarray
        Center of the ellipsoid.
    properties : dict
        Dictionary containing properties of the expansion with following keys:
        - mean_curvature: np.ndarray
            Calculate point-wise mean curvature :math:`H_i` for an ellipsoid.

            The formula for :math:`H_i` is given as:

            .. math::
                H(U, V) = \\frac{
                a_0 a_1 a_2 \\Bigg(
                3 \\left(a_0^2 + a_1^2\\right)
                + 2 \\left(a_2^2\\right)
                + \\left(a_0^2 + a_1^2 - 2 a_2^2\\right) \\cos(2V)
                - 2 \\left(a_0^2 - a_1^2\\right) \\cos(2U) \\sin^2(V)
                \\Bigg)
                }{
                8 \\Bigg(
                \\left(a_0 a_1 \\cos(V)\\right)^2
                + \\left(a_2 \\sin(V)\\right)^2
                \\Big(
                \\left(a_1 \\cos(U)\\right)^2 + \\left(a_0 \\sin(U)\\right)^2
                \\Big)
                \\Bigg)^{1.5}
                }

        - principal_curvatures1: np.ndarray
            First principal curvature at the input points. Calculated as the maximum curvature at the input points:
        - principal_curvatures2: np.ndarray
            Second principal curvature at the input points. Calculated as the minimum curvature at the input points.
        - h0_ellipsoid: np.ndarray
            Averaged mean curvature :math:`H_0` on the ellipsoid, calculated as the mean of the point-wise mean curvature :math:`H_i`.
        - h_e123_ellipsoid: np.ndarray
            Maximum, medial and minimum mean curvature of the ellipsoid.

    fluorescence : str
        The type of fluorescence used for estimating the ellipsoid.
        Can be either 'interior' or 'surface'.
        Default is 'interior'.

    Methods
    -------
    fit(image: "napari.types.ImageData")
        Fit an ellipsoid to a 3D image volume using least squares fitting.
    expand(points: "napari.types.PointsData")
        Project a set of points onto their respective position on the fitted ellipsoid.
    fit_expand(image: "napari.types.ImageData")
        Fit an ellipsoid to a 3D image volume and then expand the points.

    Examples
    --------
    .. code-block:: python

        # Instantiate and fit an ellipsoid expander to a 3D image volume
        expander = EllipsoidImageExpander(fluorescence='interior')
        expander.fit(image)
        # Expand the points on the fitted ellipsoid
        fitted_points = expander.fit_expand(image, n_points=512)

    """

    def __init__(self, fluorescence: str = "interior"):
        super().__init__()

        self._fluorescence = fluorescence

    def fit(
        self,
        image: "napari.types.ImageData",
    ) -> "napari.types.VectorsData":
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

        est_params = self.est_ellipsoid_from_volume(image)
        self._center, self._axes, self._eigenvectors = est_params

        self.coefficients_ = np.stack(
            [
                np.asarray(self._center * 3).reshape(3, 3),
                self._eigenvectors * self._axes,
            ],
            axis=1,
        )

    def expand(self, n_points):
        from .._reconstruction.fit_utils import _fibonacci_sampling

        expanded_points = _fibonacci_sampling(n_points)
        expanded_points = (
            expanded_points @ self._eigenvectors * self._axes[None, :]
        )
        expanded_points += np.asarray(self._center).reshape(1, 3)
        return expanded_points

    def fit_expand(
        self, image: "napari.types.ImageData", n_points: int = 512
    ) -> "napari.types.PointsData":

        self.fit(image)
        output_points = self.expand(n_points)

        self._calculate_properties(output_points)
        return output_points

    def est_ellipsoid_from_volume(self, image: np.ndarray) -> tuple:
        """
        Estimate the ellipsoid parameters from a 3D volume.

        Parameters:
            V (numpy.ndarray): 3D volume data (Z, Y, X).
            fluorescence (str): 'interior' or 'surface'.
            testing (int, optional): Testing flag. Defaults to 0.

        Returns:
            est_center (list): Estimated center of the ellipsoid [zCOM, yCOM, xCOM].
            semi_axes_lengths (numpy.ndarray): Semi-axes lengths of the ellipsoid in ZYX order.
            rot_matrix (numpy.ndarray): Rotation matrix of the ellipsoid in ZYX order.
        """
        from skimage import filters

        # Get the dimensions of the volume
        z_dim, y_dim, x_dim = image.shape

        # Generate coordinate arrays
        Z, Y, X = np.meshgrid(
            np.arange(z_dim), np.arange(y_dim), np.arange(x_dim), indexing="ij"
        )

        # Threshold value
        threshold_value = filters.threshold_otsu(image)

        # Binary volume based on threshold
        V_bool = (image > threshold_value).astype(float)

        # Sum of binary volume
        sumV = np.sum(V_bool)

        # Center of mass (COM)
        zCOM = np.sum(Z * V_bool) / sumV
        yCOM = np.sum(Y * V_bool) / sumV
        xCOM = np.sum(X * V_bool) / sumV

        # Intensity-weighted centered coordinates
        ZYX = np.stack(
            [
                (Z - zCOM) * np.sqrt(V_bool),
                (Y - yCOM) * np.sqrt(V_bool),
                (X - xCOM) * np.sqrt(V_bool),
            ],
            axis=-1,
        ).reshape(-1, 3)

        # Covariance matrix
        S = (1 / sumV) * (ZYX.T @ ZYX)

        # Eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(S)

        # Rotation matrix
        rot_matrix = eigvecs

        # Semi-axes lengths
        if self._fluorescence == "interior":
            semi_axes_lengths = np.sqrt(5 * eigvals)
        elif self._fluorescence == "surface":
            semi_axes_lengths = np.sqrt(3 * eigvals)
        else:
            raise ValueError("Unexpected value for 'fluorescence'")

        # Estimated center in ZYX order
        est_center = [zCOM, yCOM, xCOM]

        return est_center, semi_axes_lengths, rot_matrix

    @property
    def coefficients_(self):
        """
        The coefficients of the fitted ellipsoid

        Returns
        -------
        coefficients_ : napari.types.VectorsData
            The coefficients of the ellipsoid equation. The coefficients are of the form
            (3, 2, 3); The first dimension represents the three axes of the ellipsoid
            (major, medial and minor). The second dimension represents the components of
            the ellipsoid vectors (base point and direction vector). The third dimension
            represents the dimension of the space (z, y, x).
        """
        return super().coefficients_

    @coefficients_.setter
    def coefficients_(self, value: "napari.types.VectorsData"):
        """
        value: (3, 2, D) matrix representing the ellipsoid coefficients.
        """
        if value is not None:
            self._center = value[0, 0]
            self._axes = np.linalg.norm(value[:, 1], axis=1)
            self._eigenvectors = (
                value[:, 1]
                / np.linalg.norm(value[:, 1], axis=1)[:, np.newaxis]
            )
            self._coefficients = value

    @property
    def axes_(self):
        """
        The lengths of the axes of the ellipsoid.

        Returns
        -------
        axes_ : np.ndarray
            The lengths of the axes of the ellipsoid.
        """
        return self._axes

    @property
    def center_(self):
        """
        The center of the ellipsoid.

        Returns
        -------
        center_ : np.ndarray
            The center of the ellipsoid.
        """
        return self._center

    def _calculate_properties(self, output_points):
        self._calculate_curvatures(output_points)

    def _calculate_curvatures(
        self,
        sample_points: "napari.types.PointsData",
    ):
        """
        Calculate mean curvature on ellipsoid.

        Parameters
        ----------
        sample_points : napari.types.PointsData
            Sample points to calculate mean curvature on.

        Returns
        -------
        result : dict
            Dictionary containing the mean curvature, principal curvatures and the averaged mean curvature
            on the ellipsoid.
        """
        result = _curvature_on_ellipsoid(
            self.coefficients_, sample_points, self.axes_
        )
        for key, value in result.items():
            self.properties[key] = value


def _curvature_on_ellipsoid(
    ellipsoid: "napari.types.VectorsData",
    sample_points: "napari.types.PointsData",
    lengths: np.ndarray,
) -> dict:
    """
    Calculate mean curvature on ellipsoid.

    Parameters
    ----------
    ellipsoid : VectorsData
        Ellipsoid major axes to calculate mean curvature for.
    sample_points : PointsData
        Sample points to calculate mean curvature on.

    Returns
    -------
    LayerDataTuple (tuple)
        The sample points, properties and layer type. The properties contain
        the mean curvature, principal curvatures and the averaged mean curvature
        on the ellipsoid.

    See Also
    --------

    `Mean curvature <https://en.wikipedia.org/wiki/Mean_curvature>`_.
        Mean curvature on Wikipedia.
    `Ellipsoid definition <https://mathworld.wolfram.com/Ellipsoid.html>`_
        Ellipsoid definition on Wolfram MathWorld.

    """
    from .._utils.coordinate_conversion import (
        cartesian_to_elliptical,
    )
    from ..types import (
        _METADATAKEY_H0_ELLIPSOID,
        _METADATAKEY_H_E123_ELLIPSOID,
        _METADATAKEY_MEAN_CURVATURE,
        _METADATAKEY_PRINCIPAL_CURVATURES1,
        _METADATAKEY_PRINCIPAL_CURVATURES2,
    )

    # lengths = conversion._axes_lengths_from_ellipsoid(ellipsoid)
    a0 = lengths[0]
    a1 = lengths[1]
    a2 = lengths[2]
    U, V = cartesian_to_elliptical(ellipsoid, sample_points, invert=True)

    # calculate point-wise mean curvature H_i
    num_H_ellps = (
        a0
        * a1
        * a2
        * (
            3.0 * (a0**2 + a1**2)
            + 2.0 * (a2**2)
            + (a0**2 + a1**2 - 2.0 * a2**2) * np.cos(2.0 * V)
            - 2.0 * (a0**2 - a1**2) * np.cos(2.0 * U) * np.sin(V) ** 2
        )
    )
    den_H_ellps = (
        8.0
        * (
            (a0 * a1 * np.cos(V)) ** 2
            + (a2 * np.sin(V)) ** 2
            * ((a1 * np.cos(U)) ** 2 + (a0 * np.sin(U)) ** 2)
        )
        ** 1.5
    )
    H_ellps_pts = (num_H_ellps / den_H_ellps).squeeze()

    # also calculate principal curvatures
    k_upstairs = a0**2 * a1**2 * a2**2
    K_downstairs = (
        a0**2 * a1**2 * np.cos(V) ** 2
        + a2**2
        * (a1**2 * np.cos(U) ** 2 + a0**2 * np.sin(U) ** 2)
        * np.sin(V) ** 2
    ) ** 2
    k = k_upstairs / K_downstairs.squeeze()
    k1 = H_ellps_pts + np.sqrt(H_ellps_pts**2 - k)
    k2 = H_ellps_pts - np.sqrt(H_ellps_pts**2 - k)

    # calculate averaged curvatures H_0:
    # 1st method of H0 computation, for Ellipsoid in UV points
    H0_ellps_avg_ellps_UV_curvs = H_ellps_pts.mean(axis=0)

    # calculate maximum/minimum mean curvatures from largest to shortest axis
    order = np.argsort(lengths)[::-1]
    lengths_sorted = lengths[order]
    a0 = lengths_sorted[0]
    a1 = lengths_sorted[1]
    a2 = lengths_sorted[2]

    H_ellps_e_1 = a0 / (2.0 * a1**2) + a0 / (2.0 * a2**2)
    H_ellps_e_2 = a1 / (2.0 * a0**2) + a1 / (2.0 * a2**2)
    H_ellps_e_3 = a2 / (2.0 * a0**2) + a2 / (2.0 * a1**2)

    result = {
        _METADATAKEY_MEAN_CURVATURE: H_ellps_pts,
        _METADATAKEY_PRINCIPAL_CURVATURES1: k1,
        _METADATAKEY_PRINCIPAL_CURVATURES2: k2,
        _METADATAKEY_H0_ELLIPSOID: H0_ellps_avg_ellps_UV_curvs,
        _METADATAKEY_H_E123_ELLIPSOID: np.array(
            [H_ellps_e_1, H_ellps_e_2, H_ellps_e_3]
        ),
    }

    return result
