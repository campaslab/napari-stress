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

@register_dock_widget(menu = 'Measurement > Measure stresses on droplet pointcloud (n-STRESS)')
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
        results = comprehensive_analysis(
            self.layer_select.value.data,
            max_degree=self.spinBox_max_degree.value(),
            n_quadrature_points=int(self.comboBox_quadpoints.currentData()),
            gamma=self.doubleSpinBox_gamma.value()
            )

        for layer in results:
            _layer = Layer.create(data=layer[0],
                                  meta=layer[1],
                                  layer_type=layer[2])
            self.viewer.add_layer(_layer)

@frame_by_frame
def comprehensive_analysis(pointcloud: PointsData,
                           max_degree: int=5,
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

    from ..types import (_METADATAKEY_MEAN_CURVATURE,
                         _METADATAKEY_MEAN_CURVATURE_DIFFERENCE,
                         _METADATAKEY_H_E123_ELLIPSOID,
                         _METADATAKEY_ANISO_STRESS_TISSUE,
                         _METADATAKEY_ANISO_STRESS_CELL,
                         _METADATAKEY_ANISO_STRESS_TOTAL,
                         _METADATAKEY_ANISO_STRESS_TOTAL_RADIAL,
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
                         _METADATAKEY_MAX_TISSUE_ANISOTROPY,
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
    residue_spherical_harmonics = approximation.pairwise_point_distances(
        pointcloud, fitted_pointcloud)
    residue_spherical_harmonics_norm = np.linalg.norm(
        residue_spherical_harmonics[:, 1], axis=1)

    # Ellipsoid
    residue_ellipsoid = approximation.pairwise_point_distances(
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
        maximal_distance = int(np.floor(GDM.max()))

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
    properties = {'name': f'Result of fit spherical harmonics (deg = {max_degree}',
                  'features': {'fit_residue': residue_spherical_harmonics_norm},
                  'metadata': {_METADATAKEY_ELIPSOID_DEVIATION_CONTRIB: deviation_heatmap},
                  'face_colormap': 'inferno',
                  'face_color': 'fit_residue',
                  'size': size}
    layer_spherical_harmonics = (fitted_pointcloud, properties, 'points')

    properties = {'name': 'Result of lebedev quadrature',
                  'size': size}

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
                _METADATAKEY_ANISO_STRESS_TISSUE: stress_tissue}
    metadata = {_METADATAKEY_STRESS_TENSOR_CART: stress_tensor_cartesian,
                _METADATAKEY_STRESS_TENSOR_ELLI: stress_tensor_ellipsoidal,
                _METADATAKEY_MAX_TISSUE_ANISOTROPY: max_min_anisotropy}
    properties = {'name': 'Result of lebedev quadrature on ellipsoid',
                  'features': features,
                  'metadata': metadata,
                  'face_color': _METADATAKEY_ANISO_STRESS_TISSUE,
                  'face_colormap': 'twilight',
                  'size': size}
    layer_quadrature_ellipsoid =(quadrature_points_ellipsoid, properties, 'points')

    # Curvatures and stresses: Show on droplet surface (points)
    features = {_METADATAKEY_MEAN_CURVATURE: mean_curvature_droplet,
                _METADATAKEY_MEAN_CURVATURE_DIFFERENCE: delta_mean_curvature,
                 _METADATAKEY_ANISO_STRESS_CELL: stress_cell,
                 _METADATAKEY_ANISO_STRESS_TOTAL: stress_total,
                 _METADATAKEY_ANISO_STRESS_TOTAL_RADIAL: stress_total_radial}
    metadata = {_METADATAKEY_GAUSS_BONNET_REL: gauss_bonnet_relative,
                _METADATAKEY_GAUSS_BONNET_ABS: gauss_bonnet_absolute,
                _METADATAKEY_GAUSS_BONNET_ABS_RAD: gauss_bonnet_absolute_radial,
                _METADATAKEY_GAUSS_BONNET_REL_RAD: gauss_bonnet_relative_radial,
                _METADATAKEY_H0_VOLUME_INTEGRAL: H0_volume_droplet,
                _METADATAKEY_H0_ARITHMETIC: H0_arithmetic_droplet,
                _METADATAKEY_H0_SURFACE_INTEGRAL: H0_surface_droplet,
                _METADATAKEY_S2_VOLUME_INTEGRAL: S2_volume_droplet}

    properties = {'name': 'Result of lebedev quadrature (droplet)',
                  'features': features,
                  'metadata': metadata,
                  'face_colormap': 'twilight',
                  'face_color': _METADATAKEY_ANISO_STRESS_CELL,
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

    metadata = {'autocorrelations_total': autocorrelations_total,
                'autocorrelations_cell': autocorrelations_cell,
                'autocorrelations_tissue': autocorrelations_tissue}
    properties = {'name': 'stress_autocorrelations',
                  'metadata':  metadata}
    layer_surface_autocorrelation = (surface_droplet, properties, 'surface')

    # # Fit residues
    # properties = {'name': 'Spherical harmonics fit residues',
    #               'edge_width': size,
    #               'features': {'fit_residue': residue_spherical_harmonics_norm},
    #               'edge_color': 'fit_residue',
    #               'edge_colormap': 'twilight'}
    # layer_spherical_harmonics_residues = (residue_spherical_harmonics, properties, 'vectors')

    # properties = {'name': 'Ellipsoid fit residues',
    #               'edge_width': size,
    #               'features': {'fit_residue': residue_ellipsoid_norm},
    #               'edge_color': 'fit_residue',
    #               'edge_colormap': 'twilight'}
    # layer_ellipsoid_residues = (residue_ellipsoid, properties, 'vectors')

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
