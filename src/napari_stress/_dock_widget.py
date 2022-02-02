"""
This module is an example of a barebones QWidget plugin for napari

It implements the ``napari_experimental_provide_dock_widget`` hook specification.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below according to your needs.
"""
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget, QHBoxLayout, QPushButton
from magicgui import magic_factory

import napari
from napari.qt import create_worker

import numpy as np
import tifffile as tf

from napari_stress._preprocess import preprocessing, fit_ellipse
from napari_stress._tracing import get_traces
from napari.qt.threading import thread_worker

@magic_factory
def stress_widget(viewer: napari.Viewer,
                  img_layer: "napari.layers.Image",
                  vsx: float = 2.076,
                  vsy: float = 2.076,
                  vsz: float = 3.99,
                  vt:float = 3,
                  N_points: np.uint16 = 256):

    # Preprocessing
    img_layer.scale = [vt, vsz, vsy, vsx]
    image_resampled, mask_resampled = preprocessing(img_layer.data,
                                                    vsx=vsx, vsy=vsy, vsz=vsz)
    scale = [vt] + [np.min([vsx, vsy, vsz])] * 3
    n_frames = img_layer.data.shape[0]

    image_resampled = viewer.add_image(image_resampled, name='Resampled image', scale=scale)
    mask_resampled = viewer.add_labels(mask_resampled, name='Resampled mask', scale=scale)

    # Fit ellipse
    pts = np.zeros([N_points * n_frames, 4])
    for t in range(image_resampled.data.shape[0]):
        _pts = fit_ellipse(binary_image=mask_resampled.data[t])
        pts[t * N_points : (t + 1) * N_points, 1:] = _pts
        pts[t * N_points : (t + 1) * N_points, 0] = t

    viewer.add_points(pts, ndim=4, face_color='orange', opacity=0.4, scale=scale,
                      edge_width=0.1, size=0.5, edge_color='white', name='Fitted elipse')

    # Do first tracing
    # Calculate the center of mass of determined points for every frame
    pts_surf = np.zeros([N_points * n_frames, 4])
    for t in range(n_frames):
        _pts = pts[t * N_points : (t + 1) * N_points, :]
        CoM = _pts.mean(axis=0)

        _pts_surf, _err, _FitP = get_traces(image = image_resampled.data[t],
                                            start_pts=CoM[1:], target_pts=_pts[:, 1:])

        pts_surf[t * N_points : (t + 1) * N_points, 1:] = _pts_surf
        pts_surf[t * N_points : (t + 1) * N_points, 0] = t

    viewer.add_points(pts_surf, ndim=4, face_color='blue', opacity=0.4, scale=scale,
                      edge_width=0.1, size=0.5, edge_color='white', name='Fitted surface')


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return [stress_widget]


if __name__ == "__main__":

    filename = r'E:\BiAPoL\Projects\napari-stress\data\synthetic3.tif'
    image = tf.imread(filename)

    if len(image.shape) == 3:
        image = image[None, :]

    viewer = napari.Viewer()
    viewer.add_image(image)

    vsx: float = 1
    vsy: float = 1
    vsz: float = 1
    vt: float = 3
    N_points: np.uint16 = 256

    img_layer = viewer.layers[0]

    # Preprocessing
    img_layer.scale = [vt, vsz, vsy, vsx]
    image_resampled, mask_resampled = preprocessing(img_layer.data,
                                                    vsx=vsx, vsy=vsy, vsz=vsz)
    scale = [vt] + [np.min([vsx, vsy, vsz])] * 3
    n_frames = img_layer.data.shape[0]

    image_resampled = viewer.add_image(image_resampled, name='Resampled image', scale=scale)
    mask_resampled = viewer.add_labels(mask_resampled, name='Resampled mask', scale=scale)

    # Fit ellipse
    pts = np.zeros([N_points * n_frames, 4])
    for t in range(image_resampled.data.shape[0]):
        _pts = fit_ellipse(binary_image=mask_resampled.data[t])
        pts[t * N_points : (t + 1) * N_points, 1:] = _pts
        pts[t * N_points : (t + 1) * N_points, 0] = t

    viewer.add_points(pts, ndim=4, face_color='orange', opacity=0.4, scale=scale,
                      edge_width=0.1, size=0.5, edge_color='white', name='Fitted elipse')

    # Do first tracing
    # Calculate the center of mass of determined points for every frame
    pts_surf = np.zeros([N_points * n_frames, 4])
    for t in range(n_frames):
        _pts = pts[t * N_points : (t + 1) * N_points, :]
        CoM = _pts.mean(axis=0)

        _pts_surf, _err, _FitP = get_traces(image = image_resampled.data[t],
                                            start_pts=CoM[1:], target_pts=_pts[:, 1:])

        pts_surf[t * N_points : (t + 1) * N_points, 1:] = _pts_surf
        pts_surf[t * N_points : (t + 1) * N_points, 0] = t

    viewer.add_points(pts_surf, ndim=4, face_color='blue', opacity=0.4, scale=scale,
                      edge_width=0.1, size=0.5, edge_color='white', name='Fitted surface')
