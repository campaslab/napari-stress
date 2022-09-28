# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import numpy as np

from qtpy.QtWidgets import QWidget
from pathlib import Path
import os

from qtpy.QtCore import QEvent, QObject
from qtpy import uic
from magicgui.widgets import create_widget

from napari.layers import Image, Layer, Surface
from napari.types import PointsData, LayerDataTuple, ImageData, LabelsData

from typing import List

from .._stress import lebedev_info_SPB
from .._spherical_harmonics.spherical_harmonics import (
    stress_spherical_harmonics_expansion,
    lebedev_quadrature,
    create_manifold)

from .._utils.frame_by_frame import frame_by_frame

class droplet_reconstruction_toolbox(QWidget):
    """Comprehensive stress analysis of droplet points layer."""

    def __init__(self, napari_viewer):
        super().__init__()

        self.viewer = napari_viewer

        uic.loadUi(os.path.join(Path(__file__).parent, './toolbox.ui'), self)

        # add input dropdowns to plugin
        self.image_layer_select = create_widget(annotation=Image, label="Image_layer")
        self.surface_layer_select = create_widget(annotation=Surface, label="Surface_layer")
        self.layout().addWidget(self.image_layer_select.native, 0, 1)
        self.layout().addWidget(self.labels_layer_select.native, 1, 1)
        self.installEventFilter(self)

        # populate comboboxes with allowed values
        self.comboBox_fittype.addItems(['Fancy', 'Quick'])
        self.comboBox_fluorescence_type.addItems(['Interior', 'Surface'])

        self.pushButton_run.clicked.connect(self._run)


    def eventFilter(self, obj: QObject, event: QEvent):
        """https://forum.image.sc/t/composing-workflows-in-napari/61222/3."""
        if event.type() == QEvent.ParentChange:
            self.image_layer_select.parent_changed.emit(self.parent())
            self.labels_layer_select.parent_changed.emit(self.parent())

        return super().eventFilter(obj, event)

    def _run(self):
        """Call analysis function."""
        results = reconstruct_droplet(
            self.image_layer_select.value.data,
            self.surface_layer_select.value.data,
            n_smoothing_iterations=self.spinBox_n_smoothing.value(),
            n_points=self.spinBox_n_vertices.value()
            )

        for layer in results:
            _layer = Layer.create(data=layer[0],
                                  meta=layer[1],
                                  layer_type=layer[2])
            self.viewer.add_layer(_layer)

@frame_by_frame
def reconstruct_droplet(image: ImageData,
                        surface: LabelsData,
                        n_smoothing_iterations: int = 10,
                        n_points: int = 256,
                        fit_type: str = 'fancy',
                        edge_type: str = 'interior',
                        trace_length: float = 10,
                        sampling_distance: float = 0.5
                        ) -> List[LayerDataTuple]:
    import napari_process_points_and_surfaces as nppas
    from napari_stress import reconstruction

    # Smooth surface
    surface_smoothed = nppas.filter_smooth_laplacian(
        surface, number_of_iterations=n_smoothing_iterations)

    points_first_guess = nppas.sample_points_poisson_disk(
        surface_smoothed, number_of_points=n_points)

    # Tracing
    traced_points, trace_vectors = reconstruction.trace_refinement_of_surface(
        image,
        points_first_guess,
        selected_fit_type=fit_type,
        selected_edge=edge_type,
        trace_length=trace_length,
        sampling_distance=sampling_distance)


    # =========================================================================
    # Returns
    # =========================================================================

    properties = {'name': 'Surface_smoothed'}
    layer_surface_smoothed = (surface_smoothed, properties, 'surface')

    properties = {'name': 'points_first_guess',
                  'size': 1}
    layer_points_first_guess = (points_first_guess, properties, 'points')

    return [layer_surface_smoothed,
            layer_points_first_guess,
            traced_points,
            trace_vectors
            ]
