import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import numpy as np
from magicgui.widgets import create_widget
from napari.layers import Layer, Points
from napari.types import LayerDataTuple, PointsData
from qtpy import uic
from qtpy.QtCore import QEvent, QObject
from qtpy.QtWidgets import QWidget

from .._stress import lebedev_info_SPB
from .._utils.frame_by_frame import frame_by_frame


class stress_analysis_toolbox(QWidget):
    """Comprehensive stress analysis of droplet points layer."""

    def __init__(self, napari_viewer):
        super().__init__()

        self.viewer = napari_viewer
        self.layer_select = create_widget(
            annotation=Points, label="points_layer"
        )
        uic.loadUi(os.path.join(Path(__file__).parent, "./toolbox.ui"), self)

        self.layout().addWidget(self.layer_select.native, 0, 1)
        self.installEventFilter(self)

        # populate quadrature dropdown: Only specific n_quadrature points
        # are allowed
        points_lookup = lebedev_info_SPB.quad_deg_lookUp
        for n_points in points_lookup:
            self.comboBox_quadpoints.addItem(str(n_points), n_points)

        # select default value corresponding to current max_degree
        minimal_point_number = lebedev_info_SPB.pts_of_lbdv_lookup[
            self.spinBox_max_degree.value()
        ]
        index = self.comboBox_quadpoints.findData(minimal_point_number)
        self.comboBox_quadpoints.setCurrentIndex(index)

        # connect buttons
        self.pushButton_run.clicked.connect(self._run)
        self.spinBox_max_degree.valueChanged.connect(
            self._check_minimal_point_number
        )
        self.pushButton_import.clicked.connect(self._import_settings)
        self.pushButton_export.clicked.connect(self._export_settings)
        self.lineEdit_export_location.setText(os.getcwd())
        self.pushButton_browse_export_location.clicked.connect(
            self._browse_export_location
        )
        self.checkBox_export.stateChanged.connect(self._on_checkbox_export)

    def _import_settings(self, file_name: str = None):
        """
        Import settings from yaml file.
        """
        from .._utils.import_export_settings import import_settings

        settings = import_settings(parent=self, file_name=file_name)
        if settings:
            self.spinBox_max_degree.setValue(settings["max_degree"])
            self.comboBox_quadpoints.setCurrentIndex(
                self.comboBox_quadpoints.findData(
                    settings["n_quadrature_points"]
                )
            )
            self.doubleSpinBox_gamma.setValue(settings["gamma"])

    def _export_settings(self, file_name: str = None):
        """
        Export settings to yaml file.
        """
        from .._utils.import_export_settings import export_settings

        settings = {
            "max_degree": self.spinBox_max_degree.value(),
            "n_quadrature_points": self.comboBox_quadpoints.currentData(),
            "gamma": self.doubleSpinBox_gamma.value(),
        }
        export_settings(settings, parent=self, file_name=file_name)

    def _browse_export_location(self):
        """Browse export location."""
        from qtpy.QtWidgets import QFileDialog

        file_name = QFileDialog.getExistingDirectory(
            self, "Select export location", os.getcwd()
        )

        if file_name:
            self.lineEdit_export_location.setText(file_name)

    def _on_checkbox_export(self):
        """Enable/disable export location."""
        if self.checkBox_export.isChecked():
            self.lineEdit_export_location.setEnabled(True)
            self.pushButton_browse_export_location.setEnabled(True)
        else:
            self.lineEdit_export_location.setEnabled(False)
            self.pushButton_browse_export_location.setEnabled(False)

    def eventFilter(self, obj: QObject, event: QEvent):
        """https://forum.image.sc/t/composing-workflows-in-napari/61222/3."""
        if event.type() == QEvent.ParentChange:
            self.layer_select.parent_changed.emit(self.parent())

        return super().eventFilter(obj, event)

    def _check_minimal_point_number(self) -> None:
        """Check if number of quadrature point complies with max_degree."""
        # lebedev_info_SPB.pts_of_lbdv_lookup is a dictionary of the form
        # {degree: number of quadrature points}
        # if no value for the given key exists, pick the next higher value.
        max_degree = self.spinBox_max_degree.value()
        lookup = lebedev_info_SPB.pts_of_lbdv_lookup
        for degree in range(max_degree, list(lookup.keys())[-1] + 1):
            if degree in lookup:
                minimal_point_number = lookup.get(degree)
                break

        if self.comboBox_quadpoints.currentData() < minimal_point_number:
            index = self.comboBox_quadpoints.findData(minimal_point_number)
            self.comboBox_quadpoints.setCurrentIndex(index)

        return None

    def _run(self):
        """Call analysis function."""
        # Prepare before analysis
        from .. import stress_backend

        _ = stress_backend.lbdv_info(
            Max_SPH_Deg=self.spinBox_max_degree.value(),
            Num_Quad_Pts=int(self.comboBox_quadpoints.currentData()),
        )
        import webbrowser

        from napari_stress import TimelapseConverter

        if self.checkBox_use_dask.isChecked():
            from dask.distributed import Client, get_client

            try:
                client = get_client()
            except ValueError:
                client = Client()
            webbrowser.open_new_tab(client.dashboard_link)

        # calculate number of frames
        Converter = TimelapseConverter()
        list_of_points = Converter.data_to_list_of_data(
            self.layer_select.value.data, layertype="napari.types.PointsData"
        )
        self.n_frames = len(list_of_points)

        # Run analysis
        results = comprehensive_analysis(
            self.layer_select.value.data,
            max_degree=self.spinBox_max_degree.value(),
            n_quadrature_points=int(self.comboBox_quadpoints.currentData()),
            gamma=self.doubleSpinBox_gamma.value(),
            use_dask=self.checkBox_use_dask.isChecked(),
        )

        for layer in results:
            _layer = Layer.create(
                data=layer[0], meta=layer[1], layer_type=layer[2]
            )
            _layer.translate = self.layer_select.value.translate
            self.viewer.add_layer(_layer)

        # Export results
        if self.checkBox_export.isChecked():
            self._export(results)

    def _export(self, results_stress_analysis):
        """Export results to csv file."""
        import datetime

        import napari

        from .. import plotting, utils

        # Compile data
        (
            df_over_time,
            df_nearest_pairs,
            df_all_pairs,
            df_autocorrelations,
            ellipsoid_contribution_matrix,
        ) = utils.compile_data_from_layers(
            results_stress_analysis,
            n_frames=self.n_frames,
            time_step=self.spinBox_timeframe.value(),
        )

        # export raw data to csv
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.save_directory = os.path.join(
            self.lineEdit_export_location.text(), "stress_analysis_" + now
        )

        figure_directory = os.path.join(self.save_directory, "figures")
        raw_values_directory = os.path.join(self.save_directory, "raw_values")
        pointcloud_directory = os.path.join(self.save_directory, "pointclouds")
        os.makedirs(figure_directory, exist_ok=True)
        os.makedirs(raw_values_directory, exist_ok=True)
        os.makedirs(pointcloud_directory, exist_ok=True)

        df_over_time.to_csv(
            os.path.join(raw_values_directory, "stress_data.csv")
        )
        df_nearest_pairs.to_csv(
            os.path.join(raw_values_directory, "nearest_pairs.csv")
        )
        df_all_pairs.to_csv(
            os.path.join(raw_values_directory, "all_pairs.csv")
        )
        df_autocorrelations.to_csv(
            os.path.join(raw_values_directory, "autocorrelations.csv")
        )

        # export ellipsoid contribution matrix
        np.save(
            os.path.join(
                raw_values_directory, "ellipsoid_contribution_matrix.npy"
            ),
            ellipsoid_contribution_matrix,
        )

        # Export figures
        figures_dict = plotting.create_all_stress_plots(
            results_stress_analysis,
            time_step=self.spinBox_timeframe.value(),
            n_frames=self.n_frames,
        )

        for fig in figures_dict:
            figure = figures_dict[fig]
            figure["figure"].tight_layout()
            figure["figure"].savefig(
                os.path.join(figure_directory, figure["path"])
            )

        # Export pointclouds
        for layer in results_stress_analysis:
            if layer[2] == "points":
                export_layer = napari.layers.Layer.create(*layer)
                napari.save_layers(
                    os.path.join(
                        pointcloud_directory,
                        f"pointcloud_{export_layer.name}.vtp",
                    ),
                    [export_layer],
                )


@frame_by_frame
def comprehensive_analysis(
    pointcloud: PointsData,
    max_degree: int = 5,
    n_quadrature_points: int = 110,
    maximal_distance: int = None,
    gamma: float = 26.0,
    verbose=False,
) -> list[LayerDataTuple]:
    """
    Comprehensive stress analysis of droplet points layer.

    Parameters
    ----------
    pointcloud : PointsData
        Points layer.
    max_degree : int, optional
        Maximum degree of spherical harmonics expansion, by default 5
    n_quadrature_points : int, optional
        Number of quadrature points, by default 110
    maximal_distance : int, optional
        Maximal distance for geodesic distance matrix, by default None
    gamma : float, optional
        Surface tension, by default 26.0
    verbose : bool, optional
        Show progress bar, by default False

    Returns
    -------
    list[LayerDataTuple]
        list of layer data tuples:
        - layer_spherical_harmonics: 'napari.types.PointsData'
            fitted spherical harmonics expansion
        - layer_fitted_ellipsoid_points: 'napari.types.PointsData'
            fitted ellipsoid points
        - layer_fitted_ellipsoid: 'napari.types.VectorsData'
            fitted ellipsoid major axes
        - layer_quadrature_ellipsoid: 'napari.types.PointsData'
            Lebedev quadrature points on ellipsoid
        - layer_quadrature: 'napari.types.PointsData'
            Lebedev quadrature points on spherical-harmonics fitted droplet
        - max_min_geodesics_total: 'napari.types.SurfaceData'
            geodesics on total stress from local maxima of total stress to
            local minima of total stress
        - min_max_geodesics_total: 'napari.types.SurfaceData'
            geodesics on total stress from local minima of total stress to
            local maxima of total stress
        - max_min_geodesics_cell: 'napari.types.SurfaceData'
            geodesics on cell stress from local maxima of cell stress to
            local minima of cell stress
        - min_max_geodesics_cell: 'napari.types.SurfaceData'
            geodesics on cell stress from local minima of cell stress to
            local maxima of cell stress
        - layer_surface_autocorrelation: 'napari.types.SurfaceData'
            Surface representation of autocorrelations of total stress.
    """
    from .. import approximation, measurements, vectors
    from ..types import (
        _METADATAKEY_ANGLE_ELLIPSOID_CART_E1_X1,
        _METADATAKEY_ANGLE_ELLIPSOID_CART_E1_X2,
        _METADATAKEY_ANGLE_ELLIPSOID_CART_E1_X3,
        _METADATAKEY_AUTOCORR_SPATIAL_CELL,
        _METADATAKEY_AUTOCORR_SPATIAL_TISSUE,
        _METADATAKEY_AUTOCORR_SPATIAL_TOTAL,
        _METADATAKEY_ELIPSOID_DEVIATION_CONTRIB,
        _METADATAKEY_EXTREMA_CELL_STRESS,
        _METADATAKEY_EXTREMA_TOTAL_STRESS,
        _METADATAKEY_FIT_RESIDUE,
        _METADATAKEY_GAUSS_BONNET_ABS,
        _METADATAKEY_GAUSS_BONNET_ABS_RAD,
        _METADATAKEY_GAUSS_BONNET_REL,
        _METADATAKEY_GAUSS_BONNET_REL_RAD,
        _METADATAKEY_H0_ARITHMETIC,
        _METADATAKEY_H0_ELLIPSOID,
        _METADATAKEY_H0_RADIAL_SURFACE,
        _METADATAKEY_H0_SURFACE_INTEGRAL,
        _METADATAKEY_H0_VOLUME_INTEGRAL,
        _METADATAKEY_MEAN_CURVATURE,
        _METADATAKEY_MEAN_CURVATURE_DIFFERENCE,
        _METADATAKEY_S2_VOLUME_INTEGRAL,
        _METADATAKEY_STRESS_CELL,
        _METADATAKEY_STRESS_CELL_ALL_PAIR_ANISO,
        _METADATAKEY_STRESS_CELL_ALL_PAIR_DIST,
        _METADATAKEY_STRESS_CELL_NEAREST_PAIR_ANISO,
        _METADATAKEY_STRESS_CELL_NEAREST_PAIR_DIST,
        _METADATAKEY_STRESS_ELLIPSOID_ANISO_E12,
        _METADATAKEY_STRESS_ELLIPSOID_ANISO_E13,
        _METADATAKEY_STRESS_ELLIPSOID_ANISO_E23,
        _METADATAKEY_STRESS_TENSOR_CARTESIAN_E11,
        _METADATAKEY_STRESS_TENSOR_CARTESIAN_E12,
        _METADATAKEY_STRESS_TENSOR_CARTESIAN_E13,
        _METADATAKEY_STRESS_TENSOR_CARTESIAN_E22,
        _METADATAKEY_STRESS_TENSOR_CARTESIAN_E23,
        _METADATAKEY_STRESS_TENSOR_CARTESIAN_E33,
        _METADATAKEY_STRESS_TENSOR_ELLI_E11,
        _METADATAKEY_STRESS_TENSOR_ELLI_E22,
        _METADATAKEY_STRESS_TENSOR_ELLI_E33,
        _METADATAKEY_STRESS_TISSUE,
        _METADATAKEY_STRESS_TISSUE_ANISO,
        _METADATAKEY_STRESS_TOTAL,
        _METADATAKEY_STRESS_TOTAL_RADIAL,
    )

    # =====================================================================
    # Spherical harmonics expansion
    # =====================================================================
    # Cartesian
    Expander_SH_cartesian = approximation.SphericalHarmonicsExpander(
        max_degree=max_degree, expansion_type="cartesian"
    )

    Expander_Lebedev_droplet = approximation.LebedevExpander(
        max_degree=max_degree,
        n_quadrature_points=n_quadrature_points,
        use_minimal_point_set=False,
    )
    fitted_pointcloud = Expander_SH_cartesian.fit_expand(pointcloud)
    Expander_Lebedev_droplet.coefficients_ = (
        Expander_SH_cartesian.coefficients_
    )
    quadrature_surface = Expander_Lebedev_droplet.expand()

    # Radial
    Expander_SH_radial = approximation.SphericalHarmonicsExpander(
        max_degree=max_degree, expansion_type="radial"
    )
    Expander_Lebedev_droplet_radial = approximation.LebedevExpander(
        max_degree=max_degree,
        n_quadrature_points=n_quadrature_points,
        use_minimal_point_set=False,
        expansion_type="radial",
    )
    Expander_SH_radial.fit(pointcloud)
    Expander_Lebedev_droplet_radial.coefficients_ = (
        Expander_SH_radial.coefficients_
    )
    _ = (
        Expander_Lebedev_droplet_radial.expand()
    )  # triggers properties calculation

    # =====================================================================
    # Ellipsoid fit
    # =====================================================================

    Expander_ellipsoid = approximation.EllipsoidExpander()
    Expander_ellipsoid_Lebedev = approximation.LebedevExpander(
        max_degree=max_degree,
        n_quadrature_points=n_quadrature_points,
        use_minimal_point_set=False,
    )

    ellipsoid_points = Expander_ellipsoid.fit_expand(pointcloud)
    quadrature_points_ellipsoid = Expander_ellipsoid_Lebedev.fit_expand(
        ellipsoid_points
    )

    # =========================================================================
    # Evaluate fit quality
    # =========================================================================
    # Spherical harmonics
    residue_spherical_harmonics = vectors.pairwise_point_distances(
        pointcloud, fitted_pointcloud
    )
    residue_spherical_harmonics_norm = np.linalg.norm(
        residue_spherical_harmonics[:, 1], axis=1
    )

    # Ellipsoid
    residue_ellipsoid = vectors.pairwise_point_distances(
        pointcloud, ellipsoid_points
    )
    residue_ellipsoid_norm = np.linalg.norm(residue_ellipsoid[:, 1], axis=1)

    # =========================================================================
    # (mean) curvature on droplet and ellipsoid
    # =========================================================================
    # Droplet (cartesian)
    H0_arithmetic_droplet = Expander_Lebedev_droplet.properties[
        _METADATAKEY_H0_ARITHMETIC
    ]

    # Droplet (radial)
    from napari_stress._stress.euclidian_k_form_SPB import (
        Integral_on_Manny,
    )

    H0_volume_droplet = Expander_Lebedev_droplet_radial.properties[
        _METADATAKEY_H0_VOLUME_INTEGRAL
    ]
    S2_volume_droplet = Expander_Lebedev_droplet_radial.properties[
        _METADATAKEY_S2_VOLUME_INTEGRAL
    ]
    H0_radial_surface = Expander_Lebedev_droplet_radial.properties[
        _METADATAKEY_H0_RADIAL_SURFACE
    ]
    mean_curvature_radial = Expander_Lebedev_droplet_radial.properties[
        "mean_curvature"
    ]

    stress_total_radial = (
        2 * gamma * (mean_curvature_radial - abs(H0_radial_surface))
    )

    delta_mean_curvature = np.mean(
        [
            Integral_on_Manny(
                mean_curvature_radial - mean_curvature_radial,
                Expander_Lebedev_droplet_radial._manifold,
                Expander_Lebedev_droplet_radial._manifold.lebedev_info,
            ),
            Integral_on_Manny(
                mean_curvature_radial - mean_curvature_radial,
                Expander_Lebedev_droplet._manifold,
                Expander_Lebedev_droplet._manifold.lebedev_info,
            ),
        ]
    )

    # Ellipsoid
    mean_curvature_ellipsoid = Expander_ellipsoid_Lebedev.properties[
        _METADATAKEY_MEAN_CURVATURE
    ]
    H0_surface_ellipsoid = Expander_ellipsoid.properties[
        _METADATAKEY_H0_ELLIPSOID
    ]

    # =========================================================================
    # Stresses
    # =========================================================================
    stress_total, stress_tissue, stress_cell = measurements.anisotropic_stress(
        mean_curvature_droplet=Expander_Lebedev_droplet.properties[
            _METADATAKEY_MEAN_CURVATURE
        ],
        H0_droplet=Expander_Lebedev_droplet.properties[
            _METADATAKEY_H0_SURFACE_INTEGRAL
        ],
        mean_curvature_ellipsoid=Expander_ellipsoid_Lebedev.properties[
            _METADATAKEY_MEAN_CURVATURE
        ],
        H0_ellipsoid=H0_surface_ellipsoid,
        gamma=gamma,
    )

    max_min_anisotropy = (
        2
        * gamma
        * (
            Expander_ellipsoid.properties["maximum_mean_curvature"]
            - Expander_ellipsoid.properties["minimum_mean_curvature"]
        )
    )

    result = measurements.tissue_stress_tensor(
        Expander_ellipsoid.coefficients_, H0_surface_ellipsoid, gamma=gamma
    )
    stress_tensor_ellipsoidal = result[0]
    stress_tensor_cartesian = result[1]

    # calculate angles
    angles = []
    for j in range(3):
        dot_product = np.dot(
            stress_tensor_cartesian[:, j], stress_tensor_ellipsoidal[:, 0]
        )
        norm = np.linalg.norm(stress_tensor_cartesian[:, j]) * np.linalg.norm(
            stress_tensor_ellipsoidal[:, 0]
        )
        angles.append(np.arccos(dot_product / norm) * 180 / np.pi)

    # =============================================================================
    # Geodesics
    # =============================================================================

    surface_cell_stress = list(quadrature_surface) + [stress_cell]
    surface_total_stress = list(quadrature_surface) + [stress_total]
    surface_tissue_stress = list(quadrature_surface) + [stress_tissue]

    GDM = None
    if GDM is None:
        GDM = measurements.geodesic_distance_matrix(surface_cell_stress)

    if maximal_distance is None:
        maximal_distance = int(np.floor(np.nanmax(GDM[np.inf != GDM])))

    # Compute Overall total stress spatial correlations
    autocorrelations_total = measurements.correlation_on_surface(
        surface_total_stress,
        surface_total_stress,
        distance_matrix=GDM,
        maximal_distance=maximal_distance,
    )

    # Compute cellular Stress spatial correlations
    autocorrelations_cell = measurements.correlation_on_surface(
        surface_cell_stress,
        surface_cell_stress,
        distance_matrix=GDM,
        maximal_distance=maximal_distance,
    )

    # Compute tissue Stress spatial correlations
    autocorrelations_tissue = measurements.correlation_on_surface(
        surface_tissue_stress,
        surface_tissue_stress,
        distance_matrix=GDM,
        maximal_distance=maximal_distance,
    )

    #########################################################################
    # Do Local Max/Min analysis on 2\gamma*(H - H0) and 2\gamma*(H - H_ellps):
    (
        extrema_total_stress,
        max_min_geodesics_total,
        min_max_geodesics_total,
    ) = measurements.local_extrema_analysis(surface_total_stress, GDM)
    (
        extrema_cellular_stress,
        max_min_geodesics_cell,
        min_max_geodesics_cell,
    ) = measurements.local_extrema_analysis(surface_cell_stress, GDM)

    # =========================================================================
    # Ellipsoid deviation analysis
    # =========================================================================
    results_deviation = measurements.deviation_from_ellipsoidal_mode(
        pointcloud, max_degree=max_degree
    )
    deviation_heatmap = results_deviation[1]["metadata"][
        _METADATAKEY_ELIPSOID_DEVIATION_CONTRIB
    ]

    # =========================================================================
    # Create views as layerdatatuples
    # =========================================================================

    size = 0.5
    # spherical harmonics expansion
    properties = {
        "name": f"Result of fit spherical harmonics (deg = {max_degree})",
        "features": {"fit_residue": residue_spherical_harmonics_norm},
        "metadata": {
            _METADATAKEY_ELIPSOID_DEVIATION_CONTRIB: deviation_heatmap
        },
        "face_colormap": "inferno",
        "face_color": "fit_residue",
        "size": size,
    }
    layer_spherical_harmonics = (fitted_pointcloud, properties, "points")

    # ellipsoid expansion
    features = {_METADATAKEY_FIT_RESIDUE: residue_ellipsoid_norm}
    properties = {
        "name": "Result of expand points on ellipsoid",
        "features": features,
        "face_colormap": "inferno",
        "face_color": _METADATAKEY_FIT_RESIDUE,
        "size": size,
    }
    layer_fitted_ellipsoid_points = (ellipsoid_points, properties, "points")

    # Ellipsoid major axes
    properties = {
        "name": "Result of least squares ellipsoid",
        "edge_width": size,
    }
    layer_fitted_ellipsoid = (
        Expander_ellipsoid.coefficients_,
        properties,
        "vectors",
    )

    # Quadrature points on ellipsoid
    features = {
        _METADATAKEY_MEAN_CURVATURE: mean_curvature_ellipsoid,
        _METADATAKEY_STRESS_TISSUE: stress_tissue,
    }

    sigma_e13 = (
        stress_tensor_ellipsoidal[0, 0] - stress_tensor_ellipsoidal[2, 2]
    )
    sigma_e23 = (
        stress_tensor_ellipsoidal[1, 1] - stress_tensor_ellipsoidal[2, 2]
    )
    sigma_e12 = (
        stress_tensor_ellipsoidal[0, 0] - stress_tensor_ellipsoidal[1, 1]
    )
    metadata = {
        _METADATAKEY_STRESS_TENSOR_ELLI_E11: stress_tensor_ellipsoidal[0, 0],
        _METADATAKEY_STRESS_TENSOR_ELLI_E22: stress_tensor_ellipsoidal[1, 1],
        _METADATAKEY_STRESS_TENSOR_ELLI_E33: stress_tensor_ellipsoidal[2, 2],
        _METADATAKEY_STRESS_TENSOR_CARTESIAN_E11: stress_tensor_cartesian[
            0, 0
        ],
        _METADATAKEY_STRESS_TENSOR_CARTESIAN_E22: stress_tensor_cartesian[
            1, 1
        ],
        _METADATAKEY_STRESS_TENSOR_CARTESIAN_E33: stress_tensor_cartesian[
            2, 2
        ],
        _METADATAKEY_STRESS_TENSOR_CARTESIAN_E12: stress_tensor_cartesian[
            0, 1
        ],
        _METADATAKEY_STRESS_TENSOR_CARTESIAN_E13: stress_tensor_cartesian[
            0, 2
        ],
        _METADATAKEY_STRESS_TENSOR_CARTESIAN_E23: stress_tensor_cartesian[
            1, 2
        ],
        _METADATAKEY_STRESS_ELLIPSOID_ANISO_E12: sigma_e12,
        _METADATAKEY_STRESS_ELLIPSOID_ANISO_E23: sigma_e23,
        _METADATAKEY_STRESS_ELLIPSOID_ANISO_E13: sigma_e13,
        _METADATAKEY_ANGLE_ELLIPSOID_CART_E1_X1: angles[0],
        _METADATAKEY_ANGLE_ELLIPSOID_CART_E1_X2: angles[1],
        _METADATAKEY_ANGLE_ELLIPSOID_CART_E1_X3: angles[2],
        _METADATAKEY_STRESS_TISSUE_ANISO: max_min_anisotropy,
    }
    properties = {
        "name": "Result of lebedev quadrature on ellipsoid",
        "features": features,
        "metadata": metadata,
    }
    layer_quadrature_ellipsoid = (
        quadrature_points_ellipsoid,
        properties,
        "surface",
    )

    # Curvatures and stresses: Show on droplet surface
    features = {
        _METADATAKEY_MEAN_CURVATURE: Expander_Lebedev_droplet.properties[
            _METADATAKEY_MEAN_CURVATURE
        ],
        _METADATAKEY_STRESS_CELL: stress_cell,
        _METADATAKEY_STRESS_TOTAL: stress_total,
        _METADATAKEY_STRESS_TOTAL_RADIAL: stress_total_radial,
        _METADATAKEY_EXTREMA_CELL_STRESS: extrema_cellular_stress[1][
            "features"
        ]["local_max_and_min"],
        _METADATAKEY_EXTREMA_TOTAL_STRESS: extrema_total_stress[1]["features"][
            "local_max_and_min"
        ],
    }
    metadata = {
        _METADATAKEY_GAUSS_BONNET_REL: Expander_Lebedev_droplet.properties[
            _METADATAKEY_GAUSS_BONNET_REL
        ],
        _METADATAKEY_GAUSS_BONNET_ABS: Expander_Lebedev_droplet.properties[
            _METADATAKEY_GAUSS_BONNET_ABS
        ],
        _METADATAKEY_GAUSS_BONNET_ABS_RAD: Expander_Lebedev_droplet_radial.properties[
            _METADATAKEY_GAUSS_BONNET_ABS
        ],
        _METADATAKEY_GAUSS_BONNET_REL_RAD: Expander_Lebedev_droplet_radial.properties[
            _METADATAKEY_GAUSS_BONNET_REL
        ],
        _METADATAKEY_MEAN_CURVATURE_DIFFERENCE: delta_mean_curvature,
        _METADATAKEY_H0_VOLUME_INTEGRAL: H0_volume_droplet,
        _METADATAKEY_H0_ARITHMETIC: H0_arithmetic_droplet,
        _METADATAKEY_H0_SURFACE_INTEGRAL: Expander_Lebedev_droplet.properties[
            _METADATAKEY_H0_SURFACE_INTEGRAL
        ],
        _METADATAKEY_S2_VOLUME_INTEGRAL: S2_volume_droplet,
        _METADATAKEY_STRESS_CELL_NEAREST_PAIR_ANISO: extrema_cellular_stress[
            1
        ]["metadata"]["nearest_pair_anisotropy"],
        _METADATAKEY_STRESS_CELL_NEAREST_PAIR_DIST: extrema_cellular_stress[1][
            "metadata"
        ]["nearest_pair_distance"],
        _METADATAKEY_STRESS_CELL_ALL_PAIR_ANISO: extrema_cellular_stress[1][
            "metadata"
        ]["all_pair_anisotropy"],
        _METADATAKEY_STRESS_CELL_ALL_PAIR_DIST: extrema_cellular_stress[1][
            "metadata"
        ]["all_pair_distance"],
        _METADATAKEY_AUTOCORR_SPATIAL_TOTAL: autocorrelations_total,
        _METADATAKEY_AUTOCORR_SPATIAL_CELL: autocorrelations_cell,
        _METADATAKEY_AUTOCORR_SPATIAL_TISSUE: autocorrelations_tissue,
    }

    properties = {
        "name": "Result of lebedev quadrature (droplet)",
        "features": features,
        "metadata": metadata,
    }
    layer_quadrature = (quadrature_surface, properties, "surface")

    max_min_geodesics_total[1]["name"] = (
        "Total stress: " + max_min_geodesics_total[1]["name"]
    )
    min_max_geodesics_total[1]["name"] = (
        "Total stress: " + min_max_geodesics_total[1]["name"]
    )
    max_min_geodesics_cell[1]["name"] = (
        "Cell stress: " + max_min_geodesics_cell[1]["name"]
    )
    min_max_geodesics_cell[1]["name"] = (
        "Cell stress: " + min_max_geodesics_cell[1]["name"]
    )

    return [
        layer_spherical_harmonics,
        layer_fitted_ellipsoid_points,
        layer_fitted_ellipsoid,
        layer_quadrature_ellipsoid,
        layer_quadrature,
        max_min_geodesics_total,
        min_max_geodesics_total,
        max_min_geodesics_cell,
        min_max_geodesics_cell,
    ]
