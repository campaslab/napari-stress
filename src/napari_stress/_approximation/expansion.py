import numpy as np
from .expansion_base import Expander


class SphericalHarmonicsExpander(Expander):
    """
    Expand a set of points using spherical harmonics.

    The spherical harmonics expansion is performed using the following equation:

    .. math::
        f(\\theta, \\phi) = \\sum_{l=0}^{L} \\sum_{m=-l}^{l} a_{lm} Y_{lm}(\\theta, \\phi)

    where :math:`a_{lm}` are the spherical harmonics coefficients and :math:`Y_{lm}(\\theta, \\phi)`
    are the spherical harmonics functions. The expansion is performed in the spherical coordinate system.

    Parameters
    ----------
    max_degree : int
        Maximum degree of spherical harmonics expansion.
    expansion_type : str
        Type of expansion to perform. Can be either 'cartesian' or 'radial'.
    normalize_spectrum : bool
        Normalize power spectrum sum to 1.

    Methods
    -------
    fit(points: "napari.types.PointsData")
        Fit spherical harmonics to input data. If `expansion_type` is 'cartesian', the
        input points are converted to ellipsoidal coordinates (latitude/longitude)
        before fitting and each coordinate (z, y, z) is fitted separately. Hence,
        the derived coefficients are of shape `(3, max_degree + 1, max_degree + 1)`.
        If `expansion_type` is 'radial', the input points are converted to radial
        coordinates ($math:`\\rho, \\theta, \\phi$) before fitting. Only the radial
        coordinate is fitted, hence the derived coefficients are of shape
        `(1, max_degree + 1, max_degree + 1)`.

    expand(points: "napari.types.PointsData")
        Expand input points using spherical harmonics.

    fit_expand(points: "napari.types.PointsData")
        Fit spherical harmonics to input data and then expand them.

    """

    def __init__(
        self,
        max_degree: int = 5,
        expansion_type: str = "cartesian",
        normalize_spectrum: bool = True,
    ):

        super().__init__()
        self.expansion_type = expansion_type
        self.max_degree = max_degree
        self.normalize_spectrum = normalize_spectrum

    def _fit(self, points: "napari.types.PointsData"):
        """
        Fit spherical harmonics to input data.
        """

        if self.expansion_type == "cartesian":
            coefficients = self._fit_cartesian(points)
        elif self.expansion_type == "radial":
            coefficients = self._fit_radial(points)

        return coefficients

    def _fit_cartesian(self, points):
        from .._stress import sph_func_SPB as sph_f

        # Convert coordinates to ellipsoidal (latitude/longitude)
        longitude, latitude = self._cartesian_to_ellipsoidal_coordinates(points)

        # This implementation fits a superposition of three sets of spherical harmonics
        # to the data, one for each cardinal direction (x/y/z).
        optimal_fit_parameters = []
        for i in range(3):
            params = self._least_squares_harmonic_fit(
                fit_degree=self.max_degree,
                sample_locations=(longitude, latitude),
                values=points[:, i],
            )
            optimal_fit_parameters.append(params)
        optimal_fit_parameters = np.vstack(optimal_fit_parameters).transpose()

        fit_sph_coef_mats = np.empty((3, self.max_degree + 1, self.max_degree + 1))
        for idx in range(3):
            fit_sph_coef_mats[idx] = sph_f.Un_Flatten_Coef_Vec(
                optimal_fit_parameters[:, idx], self.max_degree
            )

        return fit_sph_coef_mats

    def _fit_radial(self, points):
        """
        Fit radial spherical harmonics to input data.
        """
        from .._stress import sph_func_SPB as sph_f

        radii, longitude, latitude = self._cartesian_to_radial_coordinates(points)

        optimal_fit_parameters = self._least_squares_harmonic_fit(
            fit_degree=self.max_degree,
            sample_locations=(longitude, latitude),
            values=radii,
        )

        # Add a singleton dimension to be consistent with coefficient array shape
        return sph_f.Un_Flatten_Coef_Vec(optimal_fit_parameters, self.max_degree)[
            None, :
        ]

    def _expand(self, points: "napari.types.PointsData"):
        """
        Expand spherical harmonics using input coefficients.
        """
        from .._stress import sph_func_SPB as sph_f

        if self.expansion_type == "cartesian":
            longitude, latitude = self._cartesian_to_ellipsoidal_coordinates(points)

        elif self.expansion_type == "radial":
            _, longitude, latitude = self._cartesian_to_radial_coordinates(points)

        fitted_points = np.array(
            [
                sph_f.spherical_harmonics_function(coef, self.max_degree).Eval_SPH(
                    longitude, latitude
                )
                for coef in self._coefficients
            ]
        ).T

        if self.expansion_type == "radial":
            import vedo

            fitted_points = (
                vedo.transformations.spher2cart(
                    fitted_points.squeeze(), latitude, longitude
                ).transpose()
                + points.mean(axis=0)[None, :]
            )

        return fitted_points

    @property
    def coefficients_(self):
        """
        The coefficients of the spherical harmonics expansion.

        Returns
        -------
        coefficients_ : np.ndarray
            The coefficients of the spherical harmonics expansion. The shape of the
            coefficients depends on the type of expansion. If the expansion type is
            'cartesian', the coefficients are of shape (3, max_degree + 1, max_degree + 1).
            If the expansion type is 'radial', the coefficients are of shape
            (1, max_degree + 1, max_degree + 1).
        """
        return self._coefficients

    @property
    def properties(self):
        """
        Get properties of the expansion.

        Returns
        -------
        properties : dict
            Dictionary containing properties of the expansion with following keys:

            - residuals: np.ndarray
                Residual euclidian distance between input points and expanded points.
            - power_spectrum: np.ndarray
                Power spectrum of spherical harmonics coefficients. If 'normalize_spectrum'
                is set to True, the power spectrum is normalized to sum to 1.

                The spectrum :math:`P_l` is calculated as follows:

                .. math::
                    P_l = \\sum_{m=-l}^{l} |a_{lm}|^2

                where :math:`a_{lm}` are the spherical harmonics coefficients.
        """
        return self._properties

    def _calculate_properties(self, input_points, expanded_points):
        """
        Calculate properties of the expansion.
        """
        self._calculate_residuals(input_points, expanded_points)
        self._calculate_power_spectrum(normalize=self.normalize_spectrum)

    def _calculate_residuals(self, input_points, expanded_points):
        """
        Calculate residuals between input points and expanded points.
        """
        # Calculate residuals between input points and expanded points
        residuals = np.linalg.norm(input_points - expanded_points, axis=1)
        self._properties["residuals"] = residuals

    def _calculate_power_spectrum(self, normalize=True):
        """
        Calculate power spectrum of spherical harmonics coefficients.

        Parameters
        ----------
        normalize : bool
            Normalize power spectrum to sum to 1.

        Returns
        -------
        power_spectrum : np.ndarray
            Power spectrum of spherical harmonics coefficients.
        """
        if self._coefficients is None:
            raise ValueError("No coefficients found. Run fit() first.")

        if len(self._coefficients.shape) > 2:
            # Handling multiple sets of coefficients without modifying the class state
            power_spectrum = np.zeros(
                self._coefficients.shape[1]
            )  # assuming the same degrees across all dimensions
            for i in range(self._coefficients.shape[0]):
                # Recursively calculate the spectrum for each set of coefficients, but avoid modifying self._coefficients
                temp_spectrum = self._calculate_power_spectrum_individual(
                    self._coefficients[i], normalize=False
                )
                power_spectrum += temp_spectrum
            if normalize:
                power_spectrum /= np.sum(power_spectrum)
        else:
            power_spectrum = self._calculate_power_spectrum_individual(
                self._coefficients, normalize=normalize
            )

        self._properties["power_spectrum"] = power_spectrum
        return power_spectrum

    def _least_squares_harmonic_fit(
        self, fit_degree: int, sample_locations: tuple, values: np.ndarray
    ) -> np.ndarray:
        """
        Perform least squares harmonic fit on input points.

        Parameters
        ----------

        fit_degree: int
        sample_locations: tuple
            Input points in elliptical coordinates - required least squares
            fit to find ellipsoid major/minor axes
        values: np.ndarray
            Values to be expanded on the surface. Can be cartesian point coordinates
            (x/y/z) or radii for a radial expansion.

        Returns
        -------
        coefficients: np.ndarray
            Numpy array holding spherical harmonics expansion coefficients. The size
            on the type of expansion (cartesian or radial).
        """
        from .._stress import lebedev_info_SPB as lebedev_info

        U, V = sample_locations[0], sample_locations[1]

        All_Y_mn_pt_in = []

        for n in range(fit_degree + 1):
            for m in range(-1 * n, n + 1):
                Y_mn_coors_in = []
                Y_mn_coors_in = lebedev_info.Eval_SPH_Basis(m, n, U, V)
                All_Y_mn_pt_in.append(Y_mn_coors_in)
        All_Y_mn_pt_in_mat = np.hstack((All_Y_mn_pt_in))

        coefficients = np.linalg.lstsq(All_Y_mn_pt_in_mat, values)[0]
        return coefficients

    def _calculate_power_spectrum_individual(self, coefficients, normalize=True):
        power_spectrum = np.zeros(coefficients.shape[0])
        for order in range(coefficients.shape[0]):
            # Assume the real coefficients are on the diagonal and immediately right (if exist)
            # and imaginary coefficients are symmetrically below the diagonal.
            real_parts = coefficients[order, order:]
            if order > 0:
                imag_parts = coefficients[:order, order]
            else:
                imag_parts = np.array([])

            coeffs = np.concatenate((real_parts, imag_parts))
            power_spectrum[order] = np.sum(np.abs(coeffs) ** 2)

        if normalize and np.sum(power_spectrum) != 0:
            power_spectrum /= np.sum(power_spectrum)

        return power_spectrum

    def _least_squares_harmonic_fit(self, fit_degree, sample_locations, values):
        """
        Perform least squares harmonic fit on input points using vectorized spherical harmonics.
        """
        from scipy.special import sph_harm

        U, V = sample_locations

        All_Y_mn_pt_in = []

        for n in range(fit_degree + 1):
            for m in range(-1 * n, n + 1):
                Y_mn_coors_in = []
                if m >= 0:
                    Y_mn_coors_in = sph_harm(m, n, U, V).real

                else:  # m<0, we use Y^(-m)_n.imag
                    Y_mn_coors_in = sph_harm(-1 * m, n, U, V).imag
                All_Y_mn_pt_in.append(Y_mn_coors_in)

        coefficients = np.linalg.lstsq(np.stack(All_Y_mn_pt_in).T, values)[0]

        return coefficients

    def _cartesian_to_ellipsoidal_coordinates(self, points):
        """
        Calculate ellipsoidal coordinates for a set of points and an ellipsoid.
        """
        from . import least_squares_ellipsoid

        # Calculate ellipsoid properties
        ellipsoid = least_squares_ellipsoid(points)
        ellipsoid_center = ellipsoid[:, 0].mean(axis=0)
        axis_lengths = np.linalg.norm(ellipsoid[:, 1], axis=1)
        rotation_matrix_inverse = (ellipsoid[:, 1] / axis_lengths[:, None]).T

        # Transform points to align with ellipsoid
        aligned_points = np.linalg.solve(
            rotation_matrix_inverse, (points - ellipsoid_center).T
        ).T

        # Calculate U coordinates (similar to longitude)
        u_coordinates = np.arctan2(
            aligned_points[:, 1] * axis_lengths[0],
            aligned_points[:, 0] * axis_lengths[1],
        )
        u_coordinates = np.where(
            u_coordinates < 0, u_coordinates + 2 * np.pi, u_coordinates
        )

        # Calculate V coordinates (similar to latitude)
        cylinder_radius = np.sqrt(aligned_points[:, 0] ** 2 + aligned_points[:, 1] ** 2)
        cylinder_radius_expanded = np.sqrt(
            (axis_lengths[0] * np.cos(u_coordinates)) ** 2
            + (axis_lengths[1] * np.sin(u_coordinates)) ** 2
        )

        v_coordinates = np.arctan2(
            cylinder_radius * axis_lengths[2],
            aligned_points[:, 2] * cylinder_radius_expanded,
        )
        v_coordinates = np.where(
            v_coordinates < 0, v_coordinates + 2 * np.pi, v_coordinates
        )

        return u_coordinates, v_coordinates

    def _cartesian_to_radial_coordinates(self, points):
        """
        Calculate radial coordinates for a set of points.
        """
        from .._stress.charts_SPB import Cart_To_Coor_A

        points_relative = points - points.mean(axis=0)[None, :]
        radii = np.sqrt(
            points_relative[:, 0] ** 2
            + points_relative[:, 1] ** 2
            + points_relative[:, 2] ** 2
        )
        longitude, latitude = Cart_To_Coor_A(
            points_relative[:, 0], points_relative[:, 1], points_relative[:, 2]
        )
        return radii, longitude, latitude


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
        self._center, self._axes, self._eigenvectors = self._extract_characteristics(
            coefficients
        )

        vectors = (
            self._eigenvectors
            / np.linalg.norm(self._eigenvectors, axis=1)[:, np.newaxis]
        ).T * self._axes[:, None]
        base = np.stack([self._center] * 3)
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

        U, V = cartesian_to_elliptical(self._coefficients, points, invert=True)
        expanded_points = elliptical_to_cartesian(U, V, self._coefficients, invert=True)

        return expanded_points

    def _calculate_properties(self, input_points, output_points):
        """
        Measure properties of the expansion.
        """
        self._measure_residuals(input_points, output_points)
        self._measure_max_min_curvatures()

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

    @property
    def properties(self):
        """
        Get properties of the expansion.

        Returns
        -------
        properties : dict
            Dictionary containing properties of the expansion with following keys:

            - residuals: np.ndarray
                Residual euclidian distance between input points and expanded points.
            - maximum_mean_curvature: float
                Maximum mean curvature of the ellipsoid.
            - minimum_mean_curvature: float
                Minimum mean curvature of the ellipsoid.

            The maximum and minimum curvatures :math:`H_{max}` and :math:`H_{min}` are calculated as follows:

            .. math::
                H_{max} = 1 / (2 * a^2) + 1 / (2 * b^2)

                H_{min} = 1 / (2 * b^2) + 1 / (2 * a^2)

            where a, b are the largest and smallest axes of the ellipsoid, respectively.
        """
        return self._properties

    def _measure_max_min_curvatures(self):
        """
        Measure maximum and minimum curvatures of the ellipsoid.

        Returns
        -------
        None

        """
        # get and remove the largest, smallest and medial axis
        axes = list(self._axes)
        largest_axis = max(axes)
        axes.remove(largest_axis)
        smallest_axis = min(axes)
        axes.remove(smallest_axis)
        medial_axis = axes[0]

        # accoording to paper (https://www.biorxiv.org/content/10.1101/2021.03.26.437148v1.full)
        maximum_mean_curvature = largest_axis / (
            2 * smallest_axis**2
        ) + largest_axis / (2 * medial_axis**2)
        minimum_mean_curvature = smallest_axis / (
            2 * medial_axis**2
        ) + smallest_axis / (2 * largest_axis**2)

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
        output_points = self._expand(input_points)

        distance = np.linalg.norm(input_points - output_points, axis=1)
        self._properties["residuals"] = distance

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
