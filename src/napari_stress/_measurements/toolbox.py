# -*- coding: utf-8 -*-

import numpy as np

from qtpy.QtWidgets import QWidget
from pathlib import Path
import os

from qtpy.QtCore import QEvent, QObject
from qtpy import uic
from magicgui.widgets import create_widget

from napari.layers import Points, Layer
from napari.types import PointsData, LayerDataTuple

from typing import List

from .._stress import lebedev_info_SPB
from .._spherical_harmonics.spherical_harmonics import (
    stress_spherical_harmonics_expansion,
    lebedev_quadrature,
    create_manifold)

from .._utils.frame_by_frame import frame_by_frame
from napari_tools_menu import register_dock_widget


@register_dock_widget(menu='Measurement > Measure stresses on droplet pointcloud (n-STRESS)')
class stress_analysis_toolbox(QWidget):
    """Comprehensive stress analysis of droplet points layer."""

    def __init__(self, napari_viewer):
        super().__init__()

        self.viewer = napari_viewer
        self.layer_select = create_widget(annotation=Points, label="points_layer")
        uic.loadUi(os.path.join(Path(__file__).parent, './toolbox.ui'), self)

        self.layout().addWidget(self.layer_select.native, 0, 1)
        self.installEventFilter(self)

        # populate quadrature dropdown: Only specific n_quadrature points
        # are allowed
        points_lookup = lebedev_info_SPB.quad_deg_lookUp
        for n_points in points_lookup.keys():
            self.comboBox_quadpoints.addItem(str(n_points), n_points)

        # select default value corresponding to current max_degree
        minimal_point_number = lebedev_info_SPB.pts_of_lbdv_lookup[
            self.spinBox_max_degree.value()]
        index = self.comboBox_quadpoints.findData(minimal_point_number)
        self.comboBox_quadpoints.setCurrentIndex(index)

        # connect buttons
        self.pushButton_run.clicked.connect(self._run)
        self.spinBox_max_degree.valueChanged.connect(self._check_minimal_point_number)
        self.pushButton_import.clicked.connect(self._import_settings)
        self.pushButton_export.clicked.connect(self._export_settings)

    def _import_settings(self, file_name: str = None):
        """
        Import settings from yaml file.
        """
        from .._utils.import_export_settings import import_settings

        settings = import_settings(parent=self, file_name=file_name)
        if settings:
            self.spinBox_max_degree.setValue(settings['max_degree'])
            self.comboBox_quadpoints.setCurrentIndex(
                self.comboBox_quadpoints.findData(settings['n_quadrature_points']))
            self.doubleSpinBox_gamma.setValue(settings['gamma'])

    def _export_settings(self, file_name: str = None):
        """
        Export settings to yaml file.
        """
        from .._utils.import_export_settings import export_settings

        settings = {'max_degree': self.spinBox_max_degree.value(),
                    'n_quadrature_points': self.comboBox_quadpoints.currentData(),
                    'gamma': self.doubleSpinBox_gamma.value()}
        export_settings(settings, parent=self, file_name=file_name)

    def eventFilter(self, obj: QObject, event: QEvent):
        """https://forum.image.sc/t/composing-workflows-in-napari/61222/3."""
        if event.type() == QEvent.ParentChange:
            self.layer_select.parent_changed.emit(self.parent())

        return super().eventFilter(obj, event)

    def _check_minimal_point_number(self) -> None:
        """Check if number of quadrature point complies with max_degree."""
        minimal_point_number = lebedev_info_SPB.pts_of_lbdv_lookup[
            self.spinBox_max_degree.value()]

        if self.comboBox_quadpoints.currentData() < minimal_point_number:
            index = self.comboBox_quadpoints.findData(minimal_point_number)
            self.comboBox_quadpoints.setCurrentIndex(index)

        return None

    def _run(self):
        """Call analysis function."""
        # Prepare before analysis
        from .. import stress_backend
        _ = stress_backend.lbdv_info(Max_SPH_Deg=self.spinBox_max_degree.value(),
                                     Num_Quad_Pts=int(self.comboBox_quadpoints.currentData()))
        from napari_stress import TimelapseConverter

        # calculate number of frames
        Converter = TimelapseConverter()
        list_of_points = Converter.data_to_list_of_data(self.layer_select.value.data,
                                                        layertype='napari.types.PointsData')
        self.n_frames = len(list_of_points)

        # Run analysis
        results = comprehensive_analysis(
            self.layer_select.value.data,
            max_degree=self.spinBox_max_degree.value(),
            n_quadrature_points=int(self.comboBox_quadpoints.currentData()),
            gamma=self.doubleSpinBox_gamma.value(),
            use_dask=self.checkBox_use_dask.isChecked()
            )

        for layer in results:
            _layer = Layer.create(data=layer[0],
                                  meta=layer[1],
                                  layer_type=layer[2])
            self.viewer.add_layer(_layer)


def aggregate_singular_values(results_stress_analysis: List[LayerDataTuple],
                              n_frames: int,
                              time_step: float) -> tuple:
    import pandas as pd
    from ..types import (
        _METADATAKEY_STRESS_TOTAL,
        _METADATAKEY_STRESS_TISSUE,
        _METADATAKEY_STRESS_CELL,
        _METADATAKEY_AUTOCORR_TEMPORAL_TOTAL,
        _METADATAKEY_AUTOCORR_TEMPORAL_CELL,
        _METADATAKEY_AUTOCORR_TEMPORAL_TISSUE
    )

    from .temporal_correlation import temporal_autocorrelation

    # Single values over time
    _metadata = [layer[1]['metadata'] for layer in results_stress_analysis if 'metadata' in layer[1].keys()]
    df_over_time = {}
    df_over_time['frame'] = np.arange(n_frames)
    for meta in _metadata:
        for key in meta.keys():
            if type(meta[key][0]) == dict:
                for key2 in meta[key][0].keys():
                    v = [dic[key2] for dic in meta[key]]
                    df_over_time[key2] = v
            else:
                df_over_time[key] = meta[key]
    df_over_time = pd.DataFrame(df_over_time)
    df_over_time['time'] = df_over_time['frame'].values * time_step

    # Find layer with stress_tissue in features
    for layer in results_stress_analysis:
        if 'features' not in layer[1].keys():
            continue
        if _METADATAKEY_STRESS_TOTAL in layer[1]['features'].keys():
            df_total_stress = pd.DataFrame(layer[1]['features'])
            df_total_stress['time'] = layer[0][:, 0] * time_step

        if _METADATAKEY_STRESS_TISSUE in layer[1]['features'].keys():
            df_tissue_stress = pd.DataFrame(layer[1]['features'])
            df_tissue_stress['time'] = layer[0][:, 0] * time_step

    df_over_time[_METADATAKEY_AUTOCORR_TEMPORAL_TOTAL] = temporal_autocorrelation(
        df_total_stress, 'stress_total_radial', frame_column_name='time')
    df_over_time[_METADATAKEY_AUTOCORR_TEMPORAL_CELL] = temporal_autocorrelation(
        df_total_stress, 'stress_cell', frame_column_name='time')
    df_over_time[_METADATAKEY_AUTOCORR_TEMPORAL_TISSUE] = temporal_autocorrelation(
        df_tissue_stress, 'stress_tissue', frame_column_name='time')

    return df_over_time


def aggregate_extrema_results(results_stress_analysis: List[LayerDataTuple],
                              n_frames: int,
                              time_step: float) -> tuple:
    import pandas as pd
    from ..types import (
        _METADATAKEY_STRESS_CELL_NEAREST_PAIR_ANISO,
        _METADATAKEY_STRESS_CELL_NEAREST_PAIR_DIST,
        _METADATAKEY_STRESS_CELL_ALL_PAIR_ANISO,
        _METADATAKEY_STRESS_CELL_ALL_PAIR_DIST
    )

    # Find layer with NEAREST EXTREMA data
    for layer in results_stress_analysis:
        if 'metadata' not in layer[1].keys():
            continue
        if _METADATAKEY_STRESS_CELL_NEAREST_PAIR_ANISO in layer[1]['metadata'].keys():
            break

    # stack keys of metadata into dataframe and add frame column
    metadata = layer[1]['metadata']
    frames = np.concatenate(
        [[i] * len(metadata[_METADATAKEY_STRESS_CELL_NEAREST_PAIR_ANISO][i]
                   ) for i in range(n_frames)]
        ) * time_step
    min_max_pair_distances = np.concatenate(
        metadata[_METADATAKEY_STRESS_CELL_NEAREST_PAIR_DIST])
    min_max_pair_anisotropies = np.concatenate(
        metadata[_METADATAKEY_STRESS_CELL_NEAREST_PAIR_ANISO])

    df_nearest_pair = pd.DataFrame(
        {'frame': frames,
         'time': frames * time_step,
         'nearest_pair_distance': min_max_pair_distances,
         'nearest_pair_anisotropy': min_max_pair_anisotropies})

    # Find layer with ALL PAIR EXTREMA data
    for layer in results_stress_analysis:
        if not 'metadata' in layer[1].keys():
            continue
        if _METADATAKEY_STRESS_CELL_ALL_PAIR_ANISO in layer[1]['metadata'].keys():
            break

    # stack keys of metadata into dataframe and add frame column
    metadata = layer[1]['metadata']
    frames = np.concatenate(
        [[i] * len(metadata[_METADATAKEY_STRESS_CELL_ALL_PAIR_ANISO][i]
                     ) for i in range(n_frames)]
        ) * time_step
    all_pair_distances = np.concatenate(
        metadata[_METADATAKEY_STRESS_CELL_ALL_PAIR_DIST])
    all_pair_anisotropies = np.concatenate(
        metadata[_METADATAKEY_STRESS_CELL_ALL_PAIR_ANISO])

    df_all_pair = pd.DataFrame(
        {'frame': frames,
         'time': frames * time_step,
         'all_pair_distance': all_pair_distances,
         'all_pair_anisotropy': all_pair_anisotropies})

    return df_nearest_pair, df_all_pair


def aggregate_spatial_autocorrelations_results(results_stress_analysis: List[LayerDataTuple],
                                               n_frames: int,
                                               time_step: float) -> tuple:
    import pandas as pd
    from ..types import (
        _METADATAKEY_AUTOCORR_SPATIAL_CELL,
        _METADATAKEY_AUTOCORR_SPATIAL_TISSUE,
        _METADATAKEY_AUTOCORR_SPATIAL_TOTAL
        
    )

    # Find layer with SPATIAL AUTOCORRELATIONS
    for layer in results_stress_analysis:
        if 'metadata' not in layer[1].keys():
            continue
        if _METADATAKEY_AUTOCORR_SPATIAL_CELL in layer[1]['metadata'].keys():
            break

    # TOTAL STRESS
    metadata = layer[1]['metadata'][_METADATAKEY_AUTOCORR_SPATIAL_TOTAL]
    distances = [metadata[t]['auto_correlations_distances'] for t in range(n_frames)]
    normalized_autocorrelation_total = [metadata[t]['auto_correlations_averaged_normalized'] for t in range(n_frames)]
    frames = [[t] * len(metadata[t]['auto_correlations_averaged_normalized']) for t in range(n_frames)]

    df_autocorrelations_total = pd.DataFrame(
        {'time': np.concatenate(frames).squeeze() * time_step,
         'distances': np.concatenate(distances).squeeze(),
         'autocorrelation_total': np.concatenate(normalized_autocorrelation_total).squeeze()
         })

    # CELL STRESS
    metadata = layer[1]['metadata'][_METADATAKEY_AUTOCORR_SPATIAL_CELL]
    distances = [metadata[t]['auto_correlations_distances'] for t in range(n_frames)]
    normalized_autocorrelation_cell = [metadata[t]['auto_correlations_averaged_normalized'] for t in range(n_frames)]
    frames = [[t] * len(metadata[t]['auto_correlations_averaged_normalized']) for t in range(n_frames)]

    df_autocorrelations_cell = pd.DataFrame(
        {'time': np.concatenate(frames).squeeze() * time_step,
         'distances': np.concatenate(distances).squeeze(),
         'autocorrelation_cell': np.concatenate(normalized_autocorrelation_cell).squeeze()
        })

    # TISSUE STRESS
    metadata = layer[1]['metadata'][_METADATAKEY_AUTOCORR_SPATIAL_TISSUE]
    distances = [metadata[t]['auto_correlations_distances'] for t in range(n_frames)]
    normalized_autocorrelation_tissue = [metadata[t]['auto_correlations_averaged_normalized'] for t in range(n_frames)]
    frames = [[t] * len(metadata[t]['auto_correlations_averaged_normalized']) for t in range(n_frames)]

    df_autocorrelations_tissue = pd.DataFrame(
        {'time': np.concatenate(frames).squeeze() * time_step,
         'distances': np.concatenate(distances).squeeze(),
         'autocorrelation_tissue': np.concatenate(normalized_autocorrelation_tissue).squeeze()
        })
    
    df_autocorrelations = pd.merge(
        df_autocorrelations_total,
        df_autocorrelations_tissue,
        'left',
        on=['time', 'distances']
        )
    df_autocorrelations = pd.merge(
        df_autocorrelations,
        df_autocorrelations_cell, 
        'left',
        on=['time', 'distances']
        )
    
    return df_autocorrelations



@frame_by_frame
def comprehensive_analysis(pointcloud: PointsData,
                           max_degree: int = 5,
                           n_quadrature_points: int = 110,
                           maximal_distance: int = None,
                           gamma: float = 26.0,
                           verbose=False) -> List[LayerDataTuple]:
    """
    Run a comprehensive stress analysis on a given pointcloud.

    Parameters
    ----------
    pointcloud : PointsData
    max_degree : int, optional
        Maximal used degree of the spherical harmonics expansion.
        The default is 5.
    n_quadrature_points : int, optional
        Number of used quadrature points. The default is 110.
    gamma : float, optional
        Interfacial surface tension in mN/m. The default is 26.0 mN/m.

    Returns
    -------
    List[LayerDataTuple]
        List of `LayerDataTuple` objects. The order of elements in the list is
        as follows:
            * `layer_spherical_harmonics`: Points layer containing result of
            the spherical harmonics expansion of the pointcloud.
            * `layer_fitted_ellipsoid_points`: Points layer with points on the
            surface of the fitted least-squares ellipsoid.
            * `layer_fitted_ellipsoid`: Vectors layers with major axes of the
            fitted ellipsoid.
            * `layer_quadrature_ellipsoid`: Points layer with quadrature points
            projected on the surface of the ellipsoid.
            * `layer_quadrature`: Points layer with quadrature points on the
            surface of the spherical harmonics expansion.

    See Also
    --------

    [0]
    """
    from .. import approximation
    from .. import measurements
    from .. import reconstruction
    from .. import vectors

    from ..types import (_METADATAKEY_MEAN_CURVATURE,
                         _METADATAKEY_MEAN_CURVATURE_DIFFERENCE,
                         _METADATAKEY_H_E123_ELLIPSOID,
                         _METADATAKEY_STRESS_TISSUE,
                         _METADATAKEY_STRESS_CELL,
                         _METADATAKEY_STRESS_TOTAL,
                         _METADATAKEY_STRESS_TISSUE_ANISO,
                         _METADATAKEY_STRESS_TOTAL_RADIAL,
                         _METADATAKEY_H0_ARITHMETIC,
                         _METADATAKEY_H0_SURFACE_INTEGRAL,
                         _METADATAKEY_S2_VOLUME_INTEGRAL,
                         _METADATAKEY_H0_VOLUME_INTEGRAL,
                         _METADATAKEY_GAUSS_BONNET_ABS,
                         _METADATAKEY_GAUSS_BONNET_REL,
                         _METADATAKEY_GAUSS_BONNET_ABS_RAD,
                         _METADATAKEY_GAUSS_BONNET_REL_RAD,
                         _METADATAKEY_STRESS_TENSOR_CART,
                         _METADATAKEY_STRESS_TENSOR_ELLI,
                         _METADATAKEY_STRESS_TENSOR_ELLI_E1,
                         _METADATAKEY_STRESS_TENSOR_ELLI_E2,
                         _METADATAKEY_STRESS_TENSOR_ELLI_E3,
                         _METADATAKEY_STRESS_ELLIPSOID_ANISO_E12,
                         _METADATAKEY_STRESS_ELLIPSOID_ANISO_E23,
                         _METADATAKEY_STRESS_ELLIPSOID_ANISO_E13,
                         _METADATAKEY_ANGLE_ELLIPSOID_CART_E1,
                         _METADATAKEY_ANGLE_ELLIPSOID_CART_E2,
                         _METADATAKEY_ANGLE_ELLIPSOID_CART_E3,
                         _METADATAKEY_EXTREMA_CELL_STRESS,
                         _METADATAKEY_EXTREMA_TOTAL_STRESS,
                         _METADATAKEY_STRESS_CELL_NEAREST_PAIR_ANISO,
                         _METADATAKEY_STRESS_CELL_NEAREST_PAIR_DIST,
                         _METADATAKEY_STRESS_CELL_ALL_PAIR_ANISO,
                         _METADATAKEY_STRESS_CELL_ALL_PAIR_DIST,
                         _METADATAKEY_AUTOCORR_SPATIAL_CELL,
                         _METADATAKEY_AUTOCORR_SPATIAL_TISSUE,
                         _METADATAKEY_AUTOCORR_SPATIAL_TOTAL,
                         _METADATAKEY_FIT_RESIDUE,
                         _METADATAKEY_ELIPSOID_DEVIATION_CONTRIB)
    # =====================================================================
    # Spherical harmonics expansion
    # =====================================================================

    # CARTESIAN
    fitted_pointcloud, coefficients = stress_spherical_harmonics_expansion(
            pointcloud,max_degree=max_degree)

    quadrature_points, lebedev_info = lebedev_quadrature(
        coefficients=coefficients,
        number_of_quadrature_points=n_quadrature_points,
        use_minimal_point_set=False)

    manifold_droplet = create_manifold(quadrature_points, lebedev_info, max_degree)

    # RADIAL
    fitted_pointcloud_radial, coefficients_radial = stress_spherical_harmonics_expansion(
            pointcloud,max_degree=max_degree,
            expansion_type='radial')

    quadrature_points_radial, _ = lebedev_quadrature(
        coefficients=coefficients_radial,
        number_of_quadrature_points=n_quadrature_points,
        use_minimal_point_set=False)

    manifold_droplet_radial = create_manifold(quadrature_points_radial, lebedev_info, max_degree)

    # Gauss Bonnet test
    gauss_bonnet_absolute, gauss_bonnet_relative = measurements.gauss_bonnet_test(manifold_droplet)
    gauss_bonnet_absolute_radial, gauss_bonnet_relative_radial = measurements.gauss_bonnet_test(manifold_droplet_radial)

    # =====================================================================
    # Ellipsoid fit
    # =====================================================================

    ellipsoid = approximation.least_squares_ellipsoid(pointcloud)
    ellipsoid_points = approximation.expand_points_on_ellipse(ellipsoid,
                                                              pointcloud)

    fitted_ellipsoid, coefficients_ell = stress_spherical_harmonics_expansion(
        ellipsoid_points, max_degree=max_degree)

    quadrature_points_ellipsoid, lebedev_info_ellipsoid = lebedev_quadrature(
        coefficients=coefficients_ell,
        number_of_quadrature_points=n_quadrature_points,
        use_minimal_point_set=False)

    manifold_ellipsoid = create_manifold(quadrature_points_ellipsoid,
                                         lebedev_info_ellipsoid,
                                         max_degree)

    # expand quadrature points on droplet on ellipsoid surface
    quadrature_points_ellipsoid = approximation.expand_points_on_ellipse(
        ellipsoid, quadrature_points)

    # =========================================================================
    # Evaluate fit quality
    # =========================================================================
    # Spherical harmonics
    residue_spherical_harmonics = vectors.pairwise_point_distances(
        pointcloud, fitted_pointcloud)
    residue_spherical_harmonics_norm = np.linalg.norm(
        residue_spherical_harmonics[:, 1], axis=1)

    # Ellipsoid
    residue_ellipsoid = vectors.pairwise_point_distances(
        pointcloud, ellipsoid_points)
    residue_ellipsoid_norm = np.linalg.norm(
        residue_ellipsoid[:, 1], axis=1)

    # =========================================================================
    # (mean) curvature on droplet and ellipsoid
    # =========================================================================
    # Droplet (cartesian)
    curvature_droplet = measurements.calculate_mean_curvature_on_manifold(manifold_droplet)
    mean_curvature_droplet = curvature_droplet[0]

    averaged_curvatures = measurements.average_mean_curvatures_on_manifold(manifold_droplet)
    H0_arithmetic_droplet = averaged_curvatures[0]
    H0_surface_droplet = averaged_curvatures[1]

    # Droplet (radial)
    mean_curvature_radial = measurements.mean_curvature_on_radial_manifold(manifold_droplet_radial)
    averaged_curvatures_radial = measurements.average_mean_curvatures_on_manifold(manifold_droplet_radial)

    H0_volume_droplet = averaged_curvatures_radial[2]
    S2_volume_droplet = averaged_curvatures_radial[3]
    H0_radial_surface = averaged_curvatures_radial[4]
    stress_total_radial = gamma * (mean_curvature_radial - H0_radial_surface)

    delta_mean_curvature = measurements.mean_curvature_differences_radial_cartesian_manifolds(manifold_droplet,
                                                                                              manifold_droplet_radial)

    # Ellipsoid
    curvature_ellipsoid = measurements.curvature_on_ellipsoid(
        ellipsoid, quadrature_points_ellipsoid)[1]
    curvature_ellipsoid_sh = measurements.calculate_mean_curvature_on_manifold(manifold_ellipsoid)

    features = curvature_ellipsoid['features']
    metadata = curvature_ellipsoid['metadata']
    mean_curvature_ellipsoid = features[_METADATAKEY_MEAN_CURVATURE]
    H_major_minor = metadata[_METADATAKEY_H_E123_ELLIPSOID]
    H0_arithmetic_ellipsoid = curvature_ellipsoid_sh[1]
    H0_surface_ellipsoid = curvature_ellipsoid_sh[2]

    # =========================================================================
    # Stresses
    # =========================================================================
    stress_total, stress_tissue, stress_cell = measurements.anisotropic_stress(
        mean_curvature_droplet=mean_curvature_droplet,
        H0_droplet=H0_surface_droplet,
        mean_curvature_ellipsoid=mean_curvature_ellipsoid,
        H0_ellipsoid=H0_surface_ellipsoid,
        gamma=gamma)

    max_min_anisotropy = measurements.maximal_tissue_anisotropy(ellipsoid, gamma=gamma)

    result = measurements.tissue_stress_tensor(ellipsoid, H0_surface_ellipsoid, gamma=gamma)
    stress_tensor_ellipsoidal = result[0]
    stress_tensor_cartesian = result[1]

    # calculate angles
    angles = []
    for j in range(3):
        dot_product = np.dot(stress_tensor_cartesian[:, j], stress_tensor_ellipsoidal[:, 0])
        norm = np.linalg.norm(stress_tensor_cartesian[:, j]) * np.linalg.norm(stress_tensor_ellipsoidal[:, 0])
        angles.append(np.arccos(dot_product/norm)*180/np.pi)

    # =============================================================================
    # Geodesics
    # =============================================================================

    # Find the surface triangles for the quadrature points and create
    # SurfaceData from it
    surface_droplet = reconstruction.reconstruct_surface_from_quadrature_points(
        quadrature_points)

    surface_cell_stress = list(surface_droplet) + [stress_cell]
    surface_total_stress = list(surface_droplet) + [stress_total]
    surface_tissue_stress = list(surface_droplet) + [stress_tissue]

    GDM = None
    if GDM is None:
        GDM = measurements.geodesic_distance_matrix(surface_cell_stress,
                                                    show_progress=verbose)

    if maximal_distance is None:
        maximal_distance = int(np.floor(np.nanmax(GDM[GDM != np.inf])))

    # Compute Overall total stress spatial correlations
    autocorrelations_total = measurements.correlation_on_surface(
        surface_total_stress,
        surface_total_stress,
        distance_matrix=GDM,
        maximal_distance=maximal_distance)

    # Compute cellular Stress spatial correlations
    autocorrelations_cell = measurements.correlation_on_surface(
        surface_cell_stress,
        surface_cell_stress,
        distance_matrix=GDM,
        maximal_distance=maximal_distance)

    # Compute tissue Stress spatial correlations
    autocorrelations_tissue = measurements.correlation_on_surface(
        surface_tissue_stress,
        surface_tissue_stress,
        distance_matrix=GDM,
        maximal_distance=maximal_distance)

    #########################################################################
    # Do Local Max/Min analysis on 2\gamma*(H - H0) and 2\gamma*(H - H_ellps) data:
    extrema_total_stress, max_min_geodesics_total, min_max_geodesics_total = measurements.local_extrema_analysis(
        surface_total_stress, GDM)
    extrema_cellular_stress, max_min_geodesics_cell, min_max_geodesics_cell = measurements.local_extrema_analysis(
        surface_cell_stress, GDM)

    # =========================================================================
    # Ellipsoid deviation analysis
    # =========================================================================
    results_deviation = measurements.deviation_from_ellipsoidal_mode(pointcloud,
        max_degree=max_degree)
    deviation_heatmap = results_deviation[1]['metadata'][_METADATAKEY_ELIPSOID_DEVIATION_CONTRIB]

    # =========================================================================
    # Create views as layerdatatuples
    # =========================================================================

    size = 0.5
    # spherical harmonics expansion
    properties = {'name': f'Result of fit spherical harmonics (deg = {max_degree})',
                  'features': {'fit_residue': residue_spherical_harmonics_norm},
                  'metadata': {_METADATAKEY_ELIPSOID_DEVIATION_CONTRIB: deviation_heatmap},
                  'face_colormap': 'inferno',
                  'face_color': 'fit_residue',
                  'size': size}
    layer_spherical_harmonics = (fitted_pointcloud, properties, 'points')

    # ellipsoid expansion
    features = {_METADATAKEY_FIT_RESIDUE: residue_ellipsoid_norm}
    properties = {'name': 'Result of expand points on ellipsoid',
                  'features': features,
                  'face_colormap': 'inferno',
                  'face_color': _METADATAKEY_FIT_RESIDUE,
                  'size': size}
    layer_fitted_ellipsoid_points = (ellipsoid_points, properties, 'points')

    # Ellipsoid major axes
    properties = {'name': 'Result of least squares ellipsoid',
                  'edge_width': size}
    layer_fitted_ellipsoid = (ellipsoid, properties, 'vectors')

    # Quadrature points on ellipsoid
    features = {_METADATAKEY_MEAN_CURVATURE: mean_curvature_ellipsoid,
                _METADATAKEY_STRESS_TISSUE: stress_tissue}
    metadata = {_METADATAKEY_STRESS_TENSOR_CART: stress_tensor_cartesian,
                _METADATAKEY_STRESS_TENSOR_ELLI: stress_tensor_ellipsoidal,
                _METADATAKEY_STRESS_TENSOR_ELLI_E1: stress_tensor_ellipsoidal[0, 0],
                _METADATAKEY_STRESS_TENSOR_ELLI_E2: stress_tensor_ellipsoidal[1, 1],
                _METADATAKEY_STRESS_TENSOR_ELLI_E3: stress_tensor_ellipsoidal[2, 2],
                _METADATAKEY_STRESS_ELLIPSOID_ANISO_E12: stress_tensor_ellipsoidal[0, 0] - stress_tensor_ellipsoidal[1, 1],
                _METADATAKEY_STRESS_ELLIPSOID_ANISO_E23: stress_tensor_ellipsoidal[1, 1] - stress_tensor_ellipsoidal[2, 2],
                _METADATAKEY_STRESS_ELLIPSOID_ANISO_E13: stress_tensor_ellipsoidal[0, 0] - stress_tensor_ellipsoidal[2, 2],
                _METADATAKEY_ANGLE_ELLIPSOID_CART_E1: angles[0],
                _METADATAKEY_ANGLE_ELLIPSOID_CART_E2: angles[1],
                _METADATAKEY_ANGLE_ELLIPSOID_CART_E3: angles[2],
                _METADATAKEY_STRESS_TISSUE_ANISO: max_min_anisotropy}
    properties = {'name': 'Result of lebedev quadrature on ellipsoid',
                  'features': features,
                  'metadata': metadata,
                  'face_color': _METADATAKEY_STRESS_TISSUE,
                  'face_colormap': 'twilight',
                  'size': size}
    layer_quadrature_ellipsoid =(quadrature_points_ellipsoid, properties, 'points')

    # Curvatures and stresses: Show on droplet surface (points)
    features = {_METADATAKEY_MEAN_CURVATURE: mean_curvature_droplet,
                _METADATAKEY_MEAN_CURVATURE_DIFFERENCE: delta_mean_curvature,
                _METADATAKEY_STRESS_CELL: stress_cell,
                _METADATAKEY_STRESS_TOTAL: stress_total,
                _METADATAKEY_STRESS_TOTAL_RADIAL: stress_total_radial,
                _METADATAKEY_EXTREMA_CELL_STRESS: extrema_cellular_stress[1]['features']['local_max_and_min'],
                _METADATAKEY_EXTREMA_TOTAL_STRESS: extrema_total_stress[1]['features']['local_max_and_min']}
    metadata = {_METADATAKEY_GAUSS_BONNET_REL: gauss_bonnet_relative,
                _METADATAKEY_GAUSS_BONNET_ABS: gauss_bonnet_absolute,
                _METADATAKEY_GAUSS_BONNET_ABS_RAD: gauss_bonnet_absolute_radial,
                _METADATAKEY_GAUSS_BONNET_REL_RAD: gauss_bonnet_relative_radial,
                _METADATAKEY_H0_VOLUME_INTEGRAL: H0_volume_droplet,
                _METADATAKEY_H0_ARITHMETIC: H0_arithmetic_droplet,
                _METADATAKEY_H0_SURFACE_INTEGRAL: H0_surface_droplet,
                _METADATAKEY_S2_VOLUME_INTEGRAL: S2_volume_droplet,
                _METADATAKEY_STRESS_CELL_NEAREST_PAIR_ANISO: extrema_cellular_stress[1]['metadata']['nearest_pair_anisotropy'],
                _METADATAKEY_STRESS_CELL_NEAREST_PAIR_DIST: extrema_cellular_stress[1]['metadata']['nearest_pair_distance'],
                _METADATAKEY_STRESS_CELL_ALL_PAIR_ANISO: extrema_cellular_stress[1]['metadata']['all_pair_anisotropy'],
                _METADATAKEY_STRESS_CELL_ALL_PAIR_DIST: extrema_cellular_stress[1]['metadata']['all_pair_distance'],
                }

    properties = {'name': 'Result of lebedev quadrature (droplet)',
                  'features': features,
                  'metadata': metadata,
                  'face_colormap': 'twilight',
                  'face_color': _METADATAKEY_STRESS_CELL,
                  'size': size}
    layer_quadrature = (quadrature_points, properties, 'points')

    # Geodesics and autocorrelations
    layer_extrema_total_stress = extrema_total_stress
    layer_extrema_total_stress[1]['name'] = 'Extrema total stress'
    layer_extrema_cellular_stress = extrema_cellular_stress
    layer_extrema_cellular_stress[1]['name'] = 'Extrema cell stress'

    max_min_geodesics_total[1]['name'] = 'Total stress: ' + max_min_geodesics_total[1]['name']
    min_max_geodesics_total[1]['name'] = 'Total stress: ' + min_max_geodesics_total[1]['name']
    max_min_geodesics_cell[1]['name'] = 'Cell stress: ' + max_min_geodesics_cell[1]['name']
    min_max_geodesics_cell[1]['name'] = 'Cell stress: ' + min_max_geodesics_cell[1]['name']

    metadata = {_METADATAKEY_AUTOCORR_SPATIAL_TOTAL: autocorrelations_total,
                _METADATAKEY_AUTOCORR_SPATIAL_CELL: autocorrelations_cell,
                _METADATAKEY_AUTOCORR_SPATIAL_TISSUE: autocorrelations_tissue}
    properties = {'name': 'stress_autocorrelations',
                  'metadata':  metadata}
    layer_surface_autocorrelation = (surface_droplet, properties, 'surface')

    return [layer_spherical_harmonics,
            layer_fitted_ellipsoid_points,
            layer_fitted_ellipsoid,
            layer_quadrature_ellipsoid,
            layer_quadrature,
            layer_extrema_total_stress,
            layer_extrema_cellular_stress,
            max_min_geodesics_total,
            min_max_geodesics_total,
            max_min_geodesics_cell,
            min_max_geodesics_cell,
            layer_surface_autocorrelation]
