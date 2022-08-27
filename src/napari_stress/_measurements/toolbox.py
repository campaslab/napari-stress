# -*- coding: utf-8 -*-

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
            pointcloud=self.layer_select.value.data,
            max_degree=self.spinBox_max_degree.value(),
            n_quadrature_points=int(self.comboBox_quadpoints.currentData()),
            gamma=self.doubleSpinBox_gamma.value()
            )

        for layer in results:
            _layer = Layer.create(data=layer[0],
                                  meta=layer[1],
                                  layer_type=layer[2])
            self.viewer.add_layer(_layer)


def comprehensive_analysis(pointcloud: PointsData,
                           max_degree: int=5,
                           n_quadrature_points: int = 110,
                           gamma: float = 26.0) -> List[LayerDataTuple]:
    from .. import approximation
    from ..measurements import (calculate_mean_curvature_on_manifold,
                                curvature_on_ellipsoid,
                                anisotropic_stress,
                                maximal_tissue_anisotropy,
                                tissue_stress_tensor)
    from ..types import (_METADATAKEY_MEAN_CURVATURE,
                         _METADATAKEY_H_E123_ELLIPSOID)
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
    # (mean) curvature on droplet and ellipsoid
    # =========================================================================
    # Droplet
    curvature_droplet = calculate_mean_curvature_on_manifold(manifold)
    mean_curvature_droplet = curvature_droplet[0]
    H0_arithmetic_droplet = curvature_droplet[1]
    H0_surface_droplet = curvature_droplet[2]

    # Ellipsoid
    curvature_ellipsoid = curvature_on_ellipsoid(
        ellipsoid, quadrature_points_ellipsoid)[1]
    curvature_ellipsoid_sh = calculate_mean_curvature_on_manifold(manifold_ellipsoid)

    features = curvature_ellipsoid['features']
    metadata = curvature_ellipsoid['metadata']
    mean_curvature_ellipsoid = features[_METADATAKEY_MEAN_CURVATURE]
    H_major_minor = metadata[_METADATAKEY_H_E123_ELLIPSOID]
    H0_arithmetic_ellipsoid = curvature_ellipsoid_sh[1]
    H0_surface_ellipsoid = curvature_ellipsoid_sh[2]

    # =========================================================================
    # Stresses
    # =========================================================================
    stress, stress_tissue, stress_droplet = anisotropic_stress(
        mean_curvature_drople=mean_curvature_droplet,
        H0_droplet=H0_surface_droplet,
        mean_curvature_ellipsoid=mean_curvature_ellipsoid,
        H0_ellipsoid=H0_surface_ellipsoid,
        gamma=gamma)

    max_min_anisotropy = maximal_tissue_anisotropy(ellipsoid, gamma=gamma)

    tissue_stress_tensor(H_major_minor)

    # =========================================================================
    # Create views as layerdatatuples
    # =========================================================================

    size = 0.5
    # spherical harmonics expansion
    properties = {'name': f'Result of fit spherical harmonics (deg = {max_degree}',
                  'size': size}
    layer_spherical_harmonics = (fitted_pointcloud, properties, 'points')

    properties = {'name': 'Result of lebedev quadrature',
                  'size': size}

    # ellipsoid expansion
    properties = {'name': 'Result of expand points on ellipsoid',
                  'size': size}
    layer_fitted_ellipsoid_points = (ellipsoid_points, properties, 'points')
    properties = {'name': 'Result of least squares ellipsoid',
                  'edge_width': size}
    layer_fitted_ellipsoid = (ellipsoid, properties, 'vectors')

    return [layer_spherical_harmonics,
            layer_fitted_ellipsoid_points,
            layer_fitted_ellipsoid]
