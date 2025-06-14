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


class LebedevExpander(SphericalHarmonicsExpander):
    """
    Lebedev grid-based spherical harmonics expander.

    This class is a specialized version of the SphericalHarmonicsExpander that uses
    Lebedev grids for sampling points on the sphere.

    Parameters
    ----------
    max_degree : int
        Maximum degree of spherical harmonics expansion.
    n_quadrature_points : int
        Number of quadrature points to use for Lebedev sampling.
        Possible values are 6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302, 350, 434, 590, 770, 974, 1202, 1454, 1730, 2030, 2354, 2702, 3074, 3470, 3890, 4334, 4802, 5294, 5810.
    expansion_type : str
        Type of expansion to perform. Can be either 'cartesian' or 'radial'.
    use_minimal_point_set : bool
        If True, use a minimal point set for the Lebedev quadrature. For every degree, a
        corresponding number of quadrature points is used, which is the minimum number of points.
        The table below lists the expansion degree and the corresponding minimal number of points:

        | Degree | Points |
        |-------:|-------:|
        |      2 |      6 |
        |      3 |     14 |
        |      4 |     26 |
        |      5 |     38 |
        |      6 |     50 |
        |      7 |     74 |
        |      8 |     86 |
        |      9 |    110 |
        |     10 |    146 |
        |     11 |    170 |
        |     12 |    194 |
        |     13 |    230 |
        |     14 |    266 |
        |     15 |    302 |
        |     16 |    350 |
        |     18 |    434 |
        |     21 |    590 |
        |     24 |    770 |
        |     27 |    974 |
        |     30 |   1202 |
        |     33 |   1454 |
        |     36 |   1730 |
        |     39 |   2030 |
        |     42 |   2354 |
        |     45 |   2702 |
        |     48 |   3074 |
        |     51 |   3470 |
        |     54 |   3890 |
        |     57 |   4334 |
        |     60 |   4802 |
        |     63 |   5294 |
        |     66 |   5810 |

        If False, the number of quadrature points is set to the maximum number of points
        available for the given degree, which is 5810 for degree 66.
    normalize_spectrum : bool
        Normalize power spectrum sum to 1. If False, the power spectrum is not normalized.

    Attributes
    ----------
    coefficients_ : np.ndarray
        Spherical harmonics coefficients of shape `(3, max_degree + 1, max_degree + 1)` for
        'cartesian' expansion type or `(1, max_degree + 1, max_degree + 1)` for 'radial' expansion type.
    properties : dict
        Dictionary containing properties of the expansion.

        - normals: np.ndarray
            Outward normals at Lebedev quadrature points.
        - mean_curvature: np.ndarray
            Mean curvature :math:`H` at Lebedev quadrature points, computed as

            .. math::
                H = \\frac{1}{2}(k_1 + k_2)

            where :math:`k_1` and :math:`k_2` are the principal curvatures.
        - H0_arithmetic: float
            Arithmetic average of mean curvature:

            .. math::
                H_0^{\\mathrm{arith}} = \\frac{1}{N} \\sum_{i=1}^N H_i

            where :math:`N` is the number of quadrature points.
        - H0_surface_integral: float
            Surface-area-weighted average mean curvature:

            .. math::
                H_0^{\\mathrm{surf}} = \\frac{\\int_S H \\, dA}{\\int_S dA}

            where :math:`S` is the surface, :math:`H` is the mean curvature, and :math:`dA` is the surface element.
        - H0_volume_integral: float
            Average mean curvature estimated via the volume integral of the manifold:

            .. math::
                H_0^{\\mathrm{vol}} \\approx \\left( \\frac{4\\pi}{3V} \\right)^{1/3}

            where :math:`V` is the volume enclosed by the surface.
        - S2_volume_integral: float
            Volume of the unit sphere, typically :math:`\\frac{4}{3}\\pi`.
        - H0_radial_surface: float
            Surface-area-weighted average mean curvature on the radially expanded surface.
        - power_spectrum: np.ndarray
            Power spectrum of spherical harmonics coefficients. If 'normalize_spectrum'
            is set to True, the power spectrum is normalized to sum to 1.

            The spectrum :math:`P_l` is calculated as:

            .. math::
                P_l = \\sum_{m=-l}^{l} |a_{lm}|^2

            where :math:`a_{lm}` are the spherical harmonics coefficients.

    Methods
    -------
    fit(points: "napari.types.PointsData")
        Fit Lebedev quadrature points to input data.
    expand() -> "napari.types.SurfaceData"
        Expand spherical harmonics using Lebedev quadrature points.
        Calculates properties of the expansion as listed above.
    fit_expand(points: "napari.types.PointsData") -> "napari.types.SurfaceData"
        Fit Lebedev quadrature points to input data and then expand them.
    """

    def __init__(
            self,
            max_degree: int = 5,
            n_quadrature_points: int = 434,
            expansion_type: str = "cartesian",
            use_minimal_point_set: bool = False,
            normalize_spectrum: bool = True,):
        super().__init__(
            max_degree=max_degree,
            expansion_type=expansion_type,
            normalize_spectrum=normalize_spectrum,)
        
        # Clip number of quadrature points
        if n_quadrature_points > 5810:
            n_quadrature_points = 5810
        self.n_quadrature_points = n_quadrature_points
        self.use_minimal_point_set = use_minimal_point_set
        self._manifold = None

        possible_n_points = np.asarray(
            list(lebedev_info.pts_of_lbdv_lookup.values())
        )
        index_correct_n_points = np.argmin(
            abs(possible_n_points - self.n_quadrature_points)
        )
        self.n_quadrature_points = possible_n_points[index_correct_n_points]

        if self.use_minimal_point_set:
            self.n_quadrature_points = lebedev_info.look_up_lbdv_pts(
                self.max_degree + 1
            )
    def fit(self, points: "napari.types.PointsData"):
        super().fit(points)

    def expand(self, points: "napari.types.PointsData"):
        from .._stress import (
            lebedev_info_SPB as lebedev_info,
            sph_func_SPB as sph_f,
            euclidian_k_form_SPB as euc_kf,
            manifold_SPB as mnfd,
        )
        # Coefficient matrix should be [DIM, DEG, DEG]; if DIM=1 this corresponds
        # to a radial spherical harmonics expansion
        if len(self.coefficients_.shape) == 2:
            self.coefficients_ = self.coefficients_[None, :]

        # Create spherical harmonics functions to represent z/y/x
        fit_functions = [
            sph_f.spherical_harmonics_function(x, self.max_degree) for x in self.coefficients_
        ]

        # Get {Z/Y/X} Coordinates at lebedev points, so we can
        # leverage our code more efficiently (and uniformly) on surface:
        LBDV_Fit = lebedev_info.lbdv_info(self.max_degree, self.n_quadrature_points)
        lebedev_points = [
            euc_kf.get_quadrature_points_from_sh_function(f, LBDV_Fit, "A")
            for f in fit_functions
        ]
        lebedev_points = np.stack(lebedev_points).squeeze().transpose()

        # create manifold object for this quadrature
        # manifold = create_manifold(
        #     lebedev_points, lebedev_fit=LBDV_Fit, max_degree=self.max_degree
        # )
        Manny_Dict = {}
        Manny_Name_Dict = {}  # sph point cloud at lbdv

        manifold_type = None
        if self.expansion_type == "radial":
            manifold_type = "radial"
            coordinates = np.stack(
                [
                    LBDV_Fit.X * lebedev_points[:, None],
                    LBDV_Fit.Y * lebedev_points[:, None],
                    LBDV_Fit.Z * lebedev_points[:, None],
                ]
            ).T.squeeze()
            Manny_Name_Dict["coordinates"] = coordinates

        elif self.expansion_type == "cartesian":
            manifold_type = "cartesian"
            Manny_Name_Dict["coordinates"] = lebedev_points

        # create manifold to calculate H, average H:
        Manny_Dict["Pickle_Manny_Data"] = False
        Manny_Dict["Maniold_lbdv"] = LBDV_Fit

        Manny_Dict["Manifold_SPH_deg"] = self.max_degree
        Manny_Dict["use_manifold_name"] = (
            False  # we are NOT using named shapes in these tests
        )
        Manny_Dict["Maniold_Name_Dict"] = Manny_Name_Dict

        self._manifold = mnfd.manifold(
            Manny_Dict, manifold_type=manifold_type, raw_coordinates=lebedev_points
        )

        self._calculate_properties()

        # Turn lebedev points into surface
        surface = self._reconstruct_surface_from_quadrature_points(
            self._manifold.get_coordinates().squeeze()
        )

        return surface
    

    def _calculate_properties(self):
        """
        Calculate properties of the Lebedev expansion.
        """
        self._calculate_normals()
        self._calculate_mean_curvature()
        self._calculate_average_mean_curvatures()

    def _calculate_normals(self):
        """
        Calculate normals for the Lebedev quadrature points.
        """
        from .._stress import euclidian_k_form_SPB as euc_kf

        normal_X_lbdv_pts = euc_kf.Combine_Chart_Quad_Vals(
            self._manifold.Normal_Vec_X_A_Pts,
            self._manifold.Normal_Vec_X_B_Pts,
            self._manifold.lebedev_info,
        )
        normal_Y_lbdv_pts = euc_kf.Combine_Chart_Quad_Vals(
            self._manifold.Normal_Vec_Y_A_Pts,
            self._manifold.Normal_Vec_Y_B_Pts,
            self._manifold.lebedev_info,
        )
        normal_Z_lbdv_pts = euc_kf.Combine_Chart_Quad_Vals(
            self._manifold.Normal_Vec_Z_A_Pts,
            self._manifold.Normal_Vec_Z_B_Pts,
            self._manifold.lebedev_info,
        )

        normals_lbdv_points = np.stack(
            [normal_X_lbdv_pts, normal_Y_lbdv_pts, normal_Z_lbdv_pts]
        ).squeeze().T

        self.properties["normals"] = normals_lbdv_points
    
    def _calculate_mean_curvature(self):
        """
        Calculate mean curvature on quadrature points
        """
        from .._stress import euclidian_k_form_SPB as euc_kf
        from ..types import _METADATAKEY_MEAN_CURVATURE

        # Test orientation:
        points = self._manifold.get_coordinates().squeeze()
        centered_lbdv_pts = points - points.mean(axis=0)[None, :]

        # Makre sure orientation is inward,
        # so H is positive (for Ellipsoid, and small deviations):
        Orientations =  np.einsum('ij,ij->i', centered_lbdv_pts, self.properties['normals'])
        num_pos_orr = np.sum(np.asarray(Orientations).flatten() > 0)

        Orientation = 1.0  # unchanged (we want INWARD)
        if num_pos_orr > 0.5 * len(centered_lbdv_pts):
            Orientation = -1.0

        self.properties[_METADATAKEY_MEAN_CURVATURE] = (
            Orientation
            * euc_kf.Combine_Chart_Quad_Vals(
                self._manifold.H_A_pts,
                self._manifold.H_B_pts,
                self._manifold.lebedev_info,
            ).squeeze()
        )

    def _calculate_average_mean_curvatures(self):
        """
        Calculate averaged mean curvatures on manifold.

        The input can also be a layer with a manifold in its metadata. In this case
        the results are stored in the layer's metadata.

        Parameters
        ----------
        manifold: mnfd.manifold

        Returns
        -------
        H0_arithmetic: float
            Arithmetic average of mean curvature
        H0_surface_integral: float
            Averaged curvature by integrating surface area
        H0_volume_integral: float
            Averaged curvature by deriving volume-integral
            of manifold.
        S2_volume: float
            Volume of the unit sphere
        H0_radial_surface: float
            Averaged curvature on radially expanded surface.
            Only calculated if  `mnfd.manifold.manifold_type`
            is `radial`.
        """
        from ..types import (
            _METADATAKEY_MEAN_CURVATURE,
            _METADATAKEY_H0_ARITHMETIC,
            _METADATAKEY_H0_SURFACE_INTEGRAL,
            _METADATAKEY_H0_VOLUME_INTEGRAL,
            _METADATAKEY_S2_VOLUME_INTEGRAL,
            _METADATAKEY_H0_RADIAL_SURFACE
        )
        from .._stress import euclidian_k_form_SPB as euc_kf
        from .._stress.sph_func_SPB import S2_Integral

        mean_curvature = self.properties[_METADATAKEY_MEAN_CURVATURE]

        # Arithmetic average of curvature
        self.properties[_METADATAKEY_H0_ARITHMETIC] = mean_curvature.flatten().mean()

        # Integrating surface area - this is the one used for downstream analysis
        # Calculate mean curvature by integrating surface area.
        Integral_on_surface = euc_kf.Integral_on_Manny(
            mean_curvature,self._manifold, self._manifold.lebedev_info
        )
        Integral_on_sphere = euc_kf.Integral_on_Manny(
            np.ones_like(mean_curvature).astype(float),
            self._manifold,
            self._manifold.lebedev_info,
        )
        self.properties[_METADATAKEY_H0_SURFACE_INTEGRAL] = Integral_on_surface / Integral_on_sphere


        S2volume = None
        H0_from_Vol_Int = None
        H0_radial_int = None
        if self.expansion_type == "radial":
            radii = self._manifold.raw_coordinates
            lbdv_info = self._manifold.lebedev_info

            S2volume = S2_Integral(radii[:, None] ** 3 / 3, lbdv_info)
            H0_from_Vol_Int = ((4.0 * np.pi) / (3.0 * S2volume)) ** (
                1.0 / 3.0
            )  # approx ~1/R, for V ~ (4/3)*pi*R^3

            sphere_radii = np.ones_like(mean_curvature)

            area_radial_manifold = euc_kf.Integral_on_Manny(
                mean_curvature, self._manifold, self._manifold.lebedev_info
            )
            area_unit_sphere = euc_kf.Integral_on_Manny(
                sphere_radii, self._manifold, self._manifold.lebedev_info
            )
            H0_radial_int = area_radial_manifold / area_unit_sphere

        self.properties[_METADATAKEY_S2_VOLUME_INTEGRAL] = S2volume
        self.properties[_METADATAKEY_H0_VOLUME_INTEGRAL] = H0_from_Vol_Int
        self.properties[_METADATAKEY_H0_RADIAL_SURFACE] = H0_radial_int

    def _reconstruct_surface_from_quadrature_points(
            self,
            points: "napari.types.PointsData",
        ) -> "napari.types.SurfaceData":
        """
        Reconstruct the surface for a given set of quadrature points.

        Returns
        -------
        surface: "napari.types.SurfaceData"
            Tuple of points and faces

        """
        from scipy.spatial import Delaunay
        from .._stress import lebedev_write_SPB as lebedev_write

        Lbdv_Cart_Pts_and_Wt_Quad = lebedev_write.Lebedev(self.n_quadrature_points)
        lbdv_coordinate_array = Lbdv_Cart_Pts_and_Wt_Quad[:, :-1]

        lbdv_plus_center = np.vstack((lbdv_coordinate_array, np.array([0, 0, 0])))
        delauney_tetras = Delaunay(lbdv_plus_center)

        tetras = delauney_tetras.simplices
        num_tris = len(delauney_tetras.simplices)

        delauney_triangles = np.zeros((num_tris, 3))

        for tri_i in range(num_tris):
            vert_ind = 0

            for tetra_vert in range(4):
                vertex = tetras[tri_i, tetra_vert]

                if vertex != self.n_quadrature_points and vert_ind < 3:
                    delauney_triangles[tri_i, vert_ind] = vertex
                    vert_ind = vert_ind + 1

        return (points, delauney_triangles.astype(int))