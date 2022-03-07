"""
This module is an example of a barebones QWidget plugin for napari

It implements the ``napari_experimental_provide_dock_widget`` hook specification.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below according to your needs.
"""
from napari_plugin_engine import napari_hook_implementation
from magicgui.widgets import create_widget

import napari

import numpy as np
import tifffile as tf
import os
from pathlib import Path

from napari_tools_menu import register_dock_widget

from napari_stress._surface import reconstruct_surface, calculate_curvatures, surface2layerdata
from napari_stress._preprocess import preprocessing, fit_ellipse
from napari_stress._tracing import get_traces_4d
from napari_stress._utils import (pointcloud_to_vertices4D,
                                  vertices4d_to_pointcloud)

from qtpy.QtWidgets import QWidget, QMessageBox, QPushButton
from qtpy.QtCore import QEvent, QObject
from qtpy import uic

@register_dock_widget(
    menu="Measurement > Measure curvature (stress)"
)
class stress_widget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()

        self.image_select = create_widget(annotation=napari.layers.Image, label="image_layer")

        self.viewer = napari_viewer
        uic.loadUi(os.path.join(Path(__file__).parent, './widget.ui'), self)

        self.group_input.layout().addWidget(self.image_select.native)
        self.installEventFilter(self)

        self.StartButton.clicked.connect(self.run)
        
        self.surfs = None
        self.max_vertices = 10000  # surfaces should not have more than this number of vertices

    def run(self):
        
        # TODO: Move the contents of this function outside the widget to make
        # it testable
        # Retrieve config parameters
        img_layer = self.image_select.value
        vsx = self.spinbox_vsx.value()
        vsy = self.spinbox_vsy.value()
        vsz = self.spinbox_vsz.value()

        n_rays = self.spinbox_n_rays.value()
        curvature_radius = self.spinbox_curv_radius.value()

        # Add empty time dimension if it doesn't exist
        if len(img_layer.data.shape) ==3:
                img_layer.data = img_layer.data[None, :]  # add empty time dimension if it doesn't exist

        n_frames = img_layer.data.shape[0]

        # Preprocessing: Mask image and fit ellipse
        image_resampled, mask_resampled = preprocessing(img_layer.data,
                                                        vsx, vsy, vsz)
        pts_ellipse = fit_ellipse(mask_resampled, n_samples=n_rays)  # returns list of vedo pointclouds

        # Add intermediary image data to viewer if desired
        if self.checkbox_verbose.checkState():
            self.viewer.add_image(image_resampled)
            self.viewer.add_labels(mask_resampled)
            self.viewer.add_points(pointcloud_to_vertices4D(pts_ellipse),
                                   face_color='magenta',
                                   size=0.5,
                                   edge_width=0.1,
                                   name='Fitted ellipse')
        

    
        # Do first tracing
        pts_surf = get_traces_4d(image_resampled,
                                 [pt.centerOfMass() for pt in pts_ellipse],
                                 pts_ellipse)

        # Reconstruct the surface and calculate curvatures and check results
        self.surfs = reconstruct_surface(pts_surf, dims=image_resampled[0].shape,
                                         surf_density=self.spinbox_vertex_density.value())
        if self.has_excessive_surface_size(): return 0
        
        # Calculate curvatures and check results
        self.surfs = calculate_curvatures(self.surfs, radius=curvature_radius)
        if not self.check_surface_curvature_integrity(): return 0

        # Add to viewer
        surf_data = surface2layerdata(self.surfs)
        self.surface_layer = self.viewer.add_surface(surf_data,
                                                      colormap='viridis',
                                                      contrast_limits=(np.quantile(surf_data[2], 0.2),
                                                                      np.quantile(surf_data[2], 0.8))
                                                      )
        
        # Turn on visualization layer widgets
        self.groupBox_visualization.setEnabled(True)
        self.combobox_vis_layers.clear()
        self.combobox_vis_layers.addItems([x for x in list(self.surfs[0].pointdata.keys()) if x != ' Normals'])
        self.combobox_vis_layers
        
        # Connect widgets when all computation is done
        self.combobox_vis_layers.currentIndexChanged.connect(self.change_visualization_layer)
        self.checkBox_show_normals.stateChanged .connect(self.show_normals)
        
    def check_surface_curvature_integrity(self) -> bool:
        "Check whether the curvature was properly calculated."
        
        if self.surfs is None: return True
        
        surf_data = surface2layerdata(self.surfs)
        
        if 0 in surf_data[2]:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setWindowTitle("Warning")
            msgBox.setText("The chosen curvature radius "
                           f"({self.spinbox_curv_radius.value()}) " 
                           "was too small to calculate curvatures. Increase " 
                           "the value to silence this error.")
            msgBox.addButton(QPushButton('Ok'), QMessageBox.YesRole)
            msgBox.exec()
            return False
        else:
            return True

    def has_excessive_surface_size(self) -> bool:
        "Check whether the current surface has too many points."
        
        if self.surfs is None: return True

        density = self.spinbox_vertex_density.value()
        n_vertices = np.array([surf.N() for surf in self.surfs])
        
        if any(n_vertices > self.max_vertices):
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setWindowTitle("Warning")
            msgBox.setText(f"The chosen vertex density on the surface ({density}) " + 
                           f"would lead to a high number of vertices (n = {max(n_vertices)}). " +
                           "Consider lowering the vertex density.")
            msgBox.addButton(QPushButton('Proceed'), QMessageBox.YesRole)
            msgBox.addButton(QPushButton('Abort'), QMessageBox.NoRole)
            
            msgBox.exec()
            reply = msgBox.buttonRole(msgBox.clickedButton())
            
            if reply == QMessageBox.YesRole:
                return False
            else:
                return True

    def change_visualization_layer(self):
        "Change the values encoded in the surface color to different layer"
        key = self.combobox_vis_layers.currentText()
        
        if key == 'Normals':
            return 0
        
        layerdata = surface2layerdata(self.surfs, key)
        
        self.surface_layer.data = layerdata
        self.surface_layer.contrast_limits = [np.quantile(layerdata[2], 0.2),
                                              np.quantile(layerdata[2], 0.8)]
        
    def show_normals(self):
        
        if self.surfs is None:
            return 0
        
        if self.checkBox_show_normals.checkState():
            _bases = pointcloud_to_vertices4D(self.surfs)
            _normals = _bases.copy()
            for surf in self.surfs:
                surf.computeNormals()
                
            _normals[:, 1:] = np.vstack([surf.pointdata['Normals'] for surf in self.surfs])
            
            normals = np.stack([_bases, _normals]).transpose((1, 0, 2))   # Make it N x 2 x D
            self.viewer.add_vectors(normals, edge_width=0.1)
            

    def eventFilter(self, obj: QObject, event: QEvent):
        # See https://forum.image.sc/t/composing-workflows-in-napari/61222/3
        if event.type() == QEvent.ParentChange:
            self.image_select.parent_changed.emit(self.parent())
        return super().eventFilter(obj, event)
