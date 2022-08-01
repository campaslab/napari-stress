# -*- coding: utf-8 -*-

from qtpy.QtWidgets import QWidget
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

        self.layer_selector = create_widget(annotation=Points, label="Points_layer")
