from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import napari

from .expansion_base import Expander
from .expansion_ellipsoid import EllipsoidExpander


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

    Attributes
    ----------
    coefficients_ : np.ndarray
        Spherical harmonics coefficients of shape `(3, max_degree + 1, max_degree + 1)` for
        'cartesian' expansion type or `(1, max_degree + 1, max_degree + 1)` for 'radial' expansion type.
    properties : dict
        Dictionary containing properties of the expansion, including residuals and power spectrum.
        - residuals: np.ndarray
            Residual euclidian distance between input points and expanded points.
        - power_spectrum: np.ndarray
            Power spectrum of spherical harmonics coefficients. If 'normalize_spectrum'
            is set to True, the power spectrum is normalized to sum to 1.
            The spectrum :math:`P_l` is calculated as follows:

            .. math::
                P_l = \\sum_{m=-l}^{l} |a_{lm}|^2

            where :math:`a_{lm}` are the spherical harmonics coefficients.



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

    def fit(self, points: "napari.types.PointsData"):
        """
        Fit spherical harmonics to input data.
        """

        if self.expansion_type == "cartesian":
            self.coefficients_ = self._fit_cartesian(points)
        elif self.expansion_type == "radial":
            self.coefficients_ = self._fit_radial(points)

    def _fit_cartesian(self, points):
        from .._stress import sph_func_SPB as sph_f
        from .._utils.coordinate_conversion import (
            cartesian_to_elliptical,
        )

        self._ellipsoid_expander = EllipsoidExpander()
        self._ellipsoid_expander.fit(points)
        # get LS Ellipsoid estimate and get point cordinates in elliptical coordinates
        longitude, latitude = cartesian_to_elliptical(
            self._ellipsoid_expander.coefficients_, points, invert=True
        )
        longitude = longitude.squeeze()
        latitude = latitude.squeeze()

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

        X_fit_sph_coef_mat = sph_f.Un_Flatten_Coef_Vec(
            optimal_fit_parameters[:, 0], self.max_degree
        )
        Y_fit_sph_coef_mat = sph_f.Un_Flatten_Coef_Vec(
            optimal_fit_parameters[:, 1], self.max_degree
        )
        Z_fit_sph_coef_mat = sph_f.Un_Flatten_Coef_Vec(
            optimal_fit_parameters[:, 2], self.max_degree
        )

        coefficients = np.stack(
            [X_fit_sph_coef_mat, Y_fit_sph_coef_mat, Z_fit_sph_coef_mat]
        )

        return coefficients

    def _fit_radial(self, points):
        """
        Fit radial spherical harmonics to input data.
        """
        from .._stress import sph_func_SPB as sph_f

        radii, longitude, latitude = self._cartesian_to_radial_coordinates(
            points
        )

        optimal_fit_parameters = self._least_squares_harmonic_fit(
            fit_degree=self.max_degree,
            sample_locations=(longitude, latitude),
            values=radii,
        )

        # Add a singleton dimension to be consistent with coefficient array shape
        return sph_f.Un_Flatten_Coef_Vec(
            optimal_fit_parameters, self.max_degree
        )[None, :]

    def expand(self, points: "napari.types.PointsData"):
        """
        Expand spherical harmonics using input coefficients.
        """
        from .._stress import sph_func_SPB as sph_f
        from .._utils.coordinate_conversion import cartesian_to_elliptical

        if self.expansion_type == "cartesian":
            longitude, latitude = cartesian_to_elliptical(
                self._ellipsoid_expander.coefficients_, points, invert=True
            )
            # Create SPH_func to represent X, Y, Z:
            X_fit_sph = sph_f.spherical_harmonics_function(
                self._coefficients[0], self.max_degree
            )
            Y_fit_sph = sph_f.spherical_harmonics_function(
                self._coefficients[1], self.max_degree
            )
            Z_fit_sph = sph_f.spherical_harmonics_function(
                self._coefficients[2], self.max_degree
            )

            X_fit_sph_UV_pts = X_fit_sph.Eval_SPH(longitude, latitude)
            Y_fit_sph_UV_pts = Y_fit_sph.Eval_SPH(longitude, latitude)
            Z_fit_sph_UV_pts = Z_fit_sph.Eval_SPH(longitude, latitude)

            fitted_points = np.hstack(
                (X_fit_sph_UV_pts, Y_fit_sph_UV_pts, Z_fit_sph_UV_pts)
            )

        elif self.expansion_type == "radial":
            import vedo

            _, longitude, latitude = self._cartesian_to_radial_coordinates(
                points
            )
            r_fit_sph = sph_f.spherical_harmonics_function(
                self.coefficients_[0], self.max_degree
            )
            r_fit_sph_UV_pts = r_fit_sph.Eval_SPH(
                longitude, latitude
            ).squeeze()
            fitted_points = (
                vedo.transformations.spher2cart(
                    r_fit_sph_UV_pts, latitude, longitude
                ).transpose()
                + points.mean(axis=0)[None, :]
            )

        self._calculate_properties(points, fitted_points)

        return fitted_points

    def fit_expand(
        self, points: "napari.types.PointsData"
    ) -> "napari.types.PointsData":
        """
        Fit spherical harmonics to input data and then expand them.
        """
        self.fit(points)
        return self.expand(points)

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

    def _calculate_power_spectrum_individual(
        self, coefficients, normalize=True
    ):
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

    def _least_squares_harmonic_fit(
        self, fit_degree, sample_locations, values
    ):
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

    def _cartesian_to_radial_coordinates(self, points):
        """
        Convert Cartesian coordinates to radial coordinates.

        Parameters
        ----------
        points : napari.types.PointsData
            The points in Cartesian coordinates.

        Returns
        -------
        radii : np.ndarray
            The radial distances of the points.
        longitude : np.ndarray
            The longitudes of the points.
        latitude : np.ndarray
            The latitudes of the points.
        """
        from .._stress.charts_SPB import Cart_To_Coor_A

        center = points.mean(axis=0)
        points_relative = points - center[None, :]
        radii = np.sqrt(
            points_relative[:, 0] ** 2
            + points_relative[:, 1] ** 2
            + points_relative[:, 2] ** 2
        )

        longitude, latitude = Cart_To_Coor_A(
            points_relative[:, 0], points_relative[:, 1], points_relative[:, 2]
        )

        return radii, longitude, latitude
