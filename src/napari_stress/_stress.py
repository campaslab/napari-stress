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
from napari_stress._preprocess import preprocessing, fit_ellipse_4D
from napari_stress._tracing import get_traces

from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget
)
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

    def run(self):

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

        pts_ellipse = fit_ellipse_4D(mask_resampled, n_rays=n_rays)

        if self.checkbox_verbose.checkState():

            self.viewer.add_image(image_resampled)
            self.viewer.add_labels(mask_resampled)
            self.viewer.add_points(pts_ellipse, face_color='magenta', size=0.5, edge_width=0.1)

        # Do first tracing
        # Calculate the center of mass of determined points for every frame
        pts_surf = np.zeros([n_rays * n_frames, 4])
        for t in range(n_frames):
            _pts = pts_ellipse[t * n_rays : (t + 1) * n_rays, :]
            CoM = _pts.mean(axis=0)

            _pts_surf, _err, _FitP = get_traces(image = image_resampled[t],
                                                start_pts=CoM[1:], target_pts=_pts[:, 1:])

            pts_surf[t * n_rays : (t + 1) * n_rays, 1:] = _pts_surf
            pts_surf[t * n_rays : (t + 1) * n_rays, 0] = t

        # Reconstruct the surface and calculate curvatures
        self.surfs = reconstruct_surface(pts_surf, dims=image_resampled[0].shape)
        self.surfs = calculate_curvatures(self.surfs, radius=curvature_radius)

        # Add to viewer
        surf_data = surface2layerdata(self.surfs)
        self.surface_layer = self.viewer.add_surface(surf_data,
                                                     colormap='viridis',
                                                     contrast_limits=(np.quantile(surf_data[2], 0.2),
                                                                      np.quantile(surf_data[2], 0.8))
                                                     )
        
        # Turn on visualization layer combobox
        print(list(self.surfs[0].pointdata.keys()))
        self.combobox_vis_layers.setEnabled(True)
        self.combobox_vis_layers.addItems(list(self.surfs[0].pointdata.keys()).remove('Normals'))
        self.combobox_vis_layers
        
        self.combobox_vis_layers.currentIndexChanged.connect(self.change_visualization_layer)


    def change_visualization_layer(self):
        "Change the values encoded in the surface color to different layer"
        key = self.combobox_vis_layers.currentText()
        layerdata = surface2layerdata(self.surfs, key)
        
        self.surface_layer.data = layerdata
        self.surface_layer.contrast_limits = [np.quantile(layerdata[2], 0.2),
                                              np.quantile(layerdata[2], 0.8)]


    def eventFilter(self, obj: QObject, event: QEvent):
        # See https://forum.image.sc/t/composing-workflows-in-napari/61222/3
        if event.type() == QEvent.ParentChange:
            parent = self.parent()
            print('parent Changed!, now:', parent)
            self.image_select.parent_changed.emit(self.parent())
        return super().eventFilter(obj, event)



# def _stress_widget(viewer: napari.Viewer,
#                   img_layer: "napari.layers.Image",
#                   vsx: float = 2.076,
#                   vsy: float = 2.076,
#                   vsz: float = 3.99,
#                   vt:float = 3,
#                   N_points: np.uint16 = 256,
#                   curvature_radius: float = 2):

#     image = img_layer.data
    
#     if image.shape ==3:
#         img_layer.data = img_layer.data[None, :]  # add empty time dimension if it doesn't exist

#     # Preprocessing
#     img_layer.scale = [vt, vsz, vsy, vsx]
#     image_resampled, mask_resampled = preprocessing(img_layer.data,
#                                                     vsx=vsx, vsy=vsy, vsz=vsz)
#     # scale = [vt] + [np.min([vsx, vsy, vsz])] * 3
#     n_frames = img_layer.data.shape[0]

#     image_resampled = viewer.add_image(image_resampled, name='Resampled image')
#     mask_resampled = viewer.add_labels(mask_resampled, name='Resampled mask')

#     # Fit ellipse
#     pts = np.zeros([N_points * n_frames, 4])
#     for t in range(image_resampled.data.shape[0]):
#         _pts = fit_ellipse(binary_image=mask_resampled.data[t])
#         pts[t * N_points : (t + 1) * N_points, 1:] = _pts
#         pts[t * N_points : (t + 1) * N_points, 0] = t

#     # Do first tracing
#     # Calculate the center of mass of determined points for every frame
#     pts_surf = np.zeros([N_points * n_frames, 4])
#     for t in range(n_frames):
#         _pts = pts[t * N_points : (t + 1) * N_points, :]
#         CoM = _pts.mean(axis=0)

#         _pts_surf, _err, _FitP = get_traces(image = image_resampled.data[t],
#                                             start_pts=CoM[1:], target_pts=_pts[:, 1:])

#         pts_surf[t * N_points : (t + 1) * N_points, 1:] = _pts_surf
#         pts_surf[t * N_points : (t + 1) * N_points, 0] = t

#     # Reconstruct the surface and calculate curvatures
#     surfs = reconstruct_surface(pts_surf, dims=image_resampled.data[0].shape)
#     surfs = calculate_curvatures(surfs, radius=curvature_radius)

#     # Add to viewer
#     surf_data = surface2layerdata(surfs)
#     viewer.add_surface(surf_data,
#                        colormap='viridis',
#                        contrast_limits=(np.quantile(surf_data[2], 0.2),
#                                         np.quantile(surf_data[2], 0.8))
#                        )
