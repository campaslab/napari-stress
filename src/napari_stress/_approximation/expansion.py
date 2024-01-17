import numpy as np
from .expansion_base import Expander
from .._utils import frame_by_frame
from napari_tools_menu import register_function


class SphericalHarmonicsExpander(Expander):
    def __init__(self, max_degree: int = 5, expansion_type: str = "cartesian"):
        super().__init__(max_degree=max_degree)
        self.expansion_type = expansion_type
        self._coordinates_ellipsoidal = None

    def calculate_power_spectrum(self, normalize=True):
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

        if len(self.coefficients_.shape) > 2:
            coeffs = self.coefficients_.copy()
            spectra = []
            for i in range(coeffs.shape[0]):
                self.coefficients_ = coeffs[i]
                spectra.append(self.calculate_power_spectrum(normalize=normalize))
            self.coefficients_ = coeffs
            return np.stack(spectra)

        if self.coefficients_ is None:
            raise ValueError("No coefficients found. Run fit() first.")

        power_spectrum = np.zeros(self.coefficients_.shape[0])

        for level in range(self.coefficients_.shape[0]):
            # Extract coefficients for degree l
            real_parts = np.concatenate(
                (
                    [self.coefficients_[level, level]],
                    self.coefficients_[level, level + 1 :],
                )
            )
            imag_parts = self.coefficients_[:level, level]

            # Combine real and imaginary parts
            coeffs = np.concatenate((real_parts, imag_parts))

            # Calculate power (sum of squares of magnitudes)
            power_spectrum[level] = np.sum(np.abs(coeffs) ** 2)

        if normalize:
            power_spectrum /= np.sum(power_spectrum)

        return power_spectrum

    def _fit(self, points: "napari.types.PointsData"):
        """
        Fit spherical harmonics to input data.
        """
        from .._stress import sph_func_SPB as sph_f

        # Convert coordinates to ellipsoidal (latitude/longitude)
        longitude, latitude = self._cartesian_to_ellipsoidal_coordinates(points)
        self._coordinates_ellipsoidal = np.stack([longitude, latitude], axis=0)

        if self.expansion_type == "cartesian":
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

            # Unflatten coefficients into a 3 x (max_degree + 1)^2 matrix
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

    def _expand(self, points: "napari.types.PointsData"):
        """
        Expand spherical harmonics using input coefficients.
        """
        from .._stress import sph_func_SPB as sph_f

        if self._coordinates_ellipsoidal is None:
            longitude, latitude = self._cartesian_to_ellipsoidal_coordinates(points)
            self._coordinates_ellipsoidal = np.stack([longitude, latitude], axis=0)

        # Create SPH_func to represent X, Y, Z:
        X_fit_sph = sph_f.spherical_harmonics_function(
            self.coefficients_[0], self.max_degree
        )
        Y_fit_sph = sph_f.spherical_harmonics_function(
            self.coefficients_[1], self.max_degree
        )
        Z_fit_sph = sph_f.spherical_harmonics_function(
            self.coefficients_[2], self.max_degree
        )

        longitude = self._coordinates_ellipsoidal[0]
        latitude = self._coordinates_ellipsoidal[1]

        X_fit_sph_UV_pts = X_fit_sph.Eval_SPH(longitude, latitude)
        Y_fit_sph_UV_pts = Y_fit_sph.Eval_SPH(longitude, latitude)
        Z_fit_sph_UV_pts = Z_fit_sph.Eval_SPH(longitude, latitude)

        fitted_points = np.stack(
            (X_fit_sph_UV_pts, Y_fit_sph_UV_pts, Z_fit_sph_UV_pts)
        ).T
        return fitted_points

    def _measure_properties(self, input_points, expanded_points):
        # Calculate distance between points and expanded points
        distance = np.linalg.norm(input_points - expanded_points, axis=1)
        self.properties["euclidian_distance"] = distance

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


@register_function(menu="Points > Fit spherical harmonics (n-STRESS")
@frame_by_frame
def expand_spherical_harmonics(
    points: "napari.types.PointsData",
    max_degree: int = 5,
) -> "napari.types.LayerDataTuple":
    """
    Expand points using spherical harmonics.

    Parameters
    ----------
    points : np.ndarray
        Points to be expanded.
    max_degree : int
        Maximum degree of spherical harmonics expansion.

    Returns
    -------
    expanded_points : np.ndarray
        Expanded points.
    """
    expander = SphericalHarmonicsExpander(max_degree=max_degree)
    expanded_points = expander.fit_expand(points)

    # convert dataframe expander.properties to dict so that one column is one key
    # and the values are the entries in the columns
    features = {}
    for key in expander.properties:
        features[key] = expander.properties[key].values

    points_layer = (
        expanded_points,
        {
            "features": features,
            "name": "Spherical Harmonics Expansion",
            "face_color": "euclidian_distance",
            "size": 0.5,
            "face_colormap": "inferno",
        },
        "points",
    )
    return points_layer
