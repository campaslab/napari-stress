# -*- coding: utf-8 -*-

from qtpy.QtWidgets import QWidget
from qtpy.QtCore import QEvent, QObject
from qtpy import uic
from napari_matplotlib.base import NapariMPLWidget

import napari
from napari.layers import Points
from magicgui.widgets import create_widget

import pathlib, os
import numpy as np

class feature_visualizer(QWidget):
    """Dockwidget for spherical harmonics analysis."""

    def __init__(self, napari_viewer: napari.viewer.Viewer,
                 points_layer: Points):
        super(feature_visualizer, self).__init__()
        ui_file = os.path.join(pathlib.Path(__file__).parent.absolute(),
                               'feature_visualizer.ui')
        uic.loadUi(ui_file, self)
        self.viewer = napari_viewer
        self.layer_name = points_layer.name
        self.colormap = 'viridis'

        self.layer_select = create_widget(annotation=Points, label="Points_layer")

        self.layout().addWidget(self.layer_select.native, 0, 1)
        self.installEventFilter(self)

        self.layer_select.native.currentIndexChanged.connect(self._populate_features)
        self.layer_select.native.activated.connect(self._populate_features)
        self.feature_select.currentIndexChanged.connect(self._set_color)

    def _populate_features(self):
        selected_layer = self.layer_select.value
        self.feature_select.clear()
        self.feature_select.addItems(list(selected_layer.features.keys()))

    def _set_color(self):
        selected_layer = self.layer_select.value
        selected_key = self.feature_select.currentText()
        print('Feature: ', selected_key)

        selected_layer.properties = selected_layer.features
        selected_layer.face_color = selected_key
        selected_layer.face_colormap = self.colormap

    def eventFilter(self, obj: QObject, event: QEvent):
        # See https://forum.image.sc/t/composing-workflows-in-napari/61222/3
        if event.type() == QEvent.ParentChange:
            self.layer_select.parent_changed.emit(self.parent())

        return super().eventFilter(obj, event)
