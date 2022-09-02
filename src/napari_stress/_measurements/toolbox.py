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
                           gamma: float = 26.0) -> List[LayerDataTuple]:
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

    from ..types import (_METADATAKEY_MEAN_CURVATURE,
                         _METADATAKEY_H_E123_ELLIPSOID,
                         _METADATAKEY_ANISO_STRESS_TISSUE,
                         _METADATAKEY_ANISO_STRESS_CELL,
                         _METADATAKEY_ANISO_STRESS_TOTAL,
                         _METADATAKEY_GAUSS_BONNET_ABS,
                         _METADATAKEY_GAUSS_BONNET_REL,
                         _METADATAKEY_STRESS_TENSOR_CART,
                         _METADATAKEY_STRESS_TENSOR_ELLI,
                         _METADATAKEY_MAX_TISSUE_ANISOTROPY,
                         _METADATAKEY_FIT_RESIDUE)
    # =====================================================================
    # Spherical harmonics expansion
    # =====================================================================

    fitted_pointcloud, coefficients = stress_spherical_harmonics_expansion(
            pointcloud,max_degree=max_degree)

    quadrature_points, lebedev_info = lebedev_quadrature(
        coefficients=coefficients,
        number_of_quadrature_points=n_quadrature_points,
        use_minimal_point_set=False)

    manifold = create_manifold(quadrature_points, lebedev_info, max_degree)

    # Gauss Bonnet test
    gauss_bonnet_absolute, gauss_bonnet_relative = measurements.gauss_bonnet_test(manifold)

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
    # Droplet
    curvature_droplet = measurements.calculate_mean_curvature_on_manifold(manifold)
    mean_curvature_droplet = curvature_droplet[0]
    H0_arithmetic_droplet = curvature_droplet[1]
    H0_surface_droplet = curvature_droplet[2]

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
    stress, stress_tissue, stress_droplet = measurements.anisotropic_stress(
        mean_curvature_droplet=mean_curvature_droplet,
        H0_droplet=H0_surface_droplet,
        mean_curvature_ellipsoid=mean_curvature_ellipsoid,
        H0_ellipsoid=H0_surface_ellipsoid,
        gamma=gamma)

    max_min_anisotropy = measurements.maximal_tissue_anisotropy(ellipsoid, gamma=gamma)

    result = measurements.tissue_stress_tensor(ellipsoid, H0_surface_ellipsoid, gamma=gamma)
    stress_tensor_ellipsoidal = result[0]
    stress_tensor_cartesian = result[1]

    # =========================================================================
    # Create views as layerdatatuples
    # =========================================================================

    size = 0.5
    # spherical harmonics expansion
    properties = {'name': f'Result of fit spherical harmonics (deg = {max_degree}',
                  'features': {'fit_residue': residue_spherical_harmonics_norm},
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
                _METADATAKEY_STRESS_TENSOR_ELLI: stress_tensor_ellipsoidal}
    properties = {'name': 'Result of lebedev quadrature on ellipsoid',
                  'features': features,
                  'metadata': metadata,
                  'face_color': _METADATAKEY_ANISO_STRESS_TISSUE,
                  'face_colormap': 'twilight',
                  'size': size}
    layer_quadrature_ellipsoid =(quadrature_points_ellipsoid, properties, 'points')

    # Curvatures and stresses: Show on droplet surface (points)
    features = {_METADATAKEY_MEAN_CURVATURE: mean_curvature_droplet,
                 _METADATAKEY_ANISO_STRESS_CELL: stress_droplet,
                 _METADATAKEY_ANISO_STRESS_TOTAL: stress}
    metadata = {_METADATAKEY_GAUSS_BONNET_REL: gauss_bonnet_relative,
                _METADATAKEY_GAUSS_BONNET_ABS: gauss_bonnet_absolute,
                _METADATAKEY_MAX_TISSUE_ANISOTROPY: max_min_anisotropy}

    properties = {'name': 'Result of lebedev quadrature (droplet)',
                  'features': features,
                  'metadata': metadata,
                  'face_colormap': 'twilight',
                  'face_color': _METADATAKEY_ANISO_STRESS_CELL,
                  'size': size}
    layer_quadrature = (quadrature_points, properties, 'points')


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
            layer_quadrature]
