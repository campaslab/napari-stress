# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import numpy as np
import vedo

from qtpy.QtWidgets import QWidget
from pathlib import Path
import os

from qtpy.QtCore import QEvent, QObject
from qtpy import uic
from magicgui.widgets import create_widget

from napari.layers import Image, Layer, Surface
from napari.types import LayerDataTuple, ImageData, SurfaceData, PointsData

from typing import List

from .._stress import lebedev_info_SPB
from .._spherical_harmonics.spherical_harmonics import (
    stress_spherical_harmonics_expansion,
    lebedev_quadrature,
    create_manifold)

from .._utils.frame_by_frame import frame_by_frame
from napari_tools_menu import register_dock_widget

@register_dock_widget(menu="Surfaces > Droplet reconstruction toolbox (n-STRESS)")
class droplet_reconstruction_toolbox(QWidget):
    """Comprehensive stress analysis of droplet points layer."""

    def __init__(self, napari_viewer):
        super().__init__()

        self.viewer = napari_viewer

        uic.loadUi(os.path.join(Path(__file__).parent, './toolbox.ui'), self)

        # add input dropdowns to plugin
        self.image_layer_select = create_widget(annotation=Image, label="Image_layer")
        self.layout().addWidget(self.image_layer_select.native, 0, 1)
        self.installEventFilter(self)

        # populate comboboxes with allowed values
        self.comboBox_fittype.addItems(['fancy', 'quick'])
        self.comboBox_fluorescence_type.addItems(['interior', 'surface'])

        # calculate density/point number
        self.spinBox_n_vertices.valueChanged.connect(self._on_update_n_points)
        self.doubleSpinBox_pointdensity.valueChanged.connect(self._on_update_density)
        self.spinBox_n_vertices.setValue(256)

        self.pushButton_run.clicked.connect(self._run)

    def _on_update_density(self):
        """Recalculate point number if point density is changed."""
        surface = self.surface_layer_select.value.data
        mesh = vedo.mesh.Mesh((surface[0], surface[1]))
        area = mesh.area()

        n_points = int(self.doubleSpinBox_pointdensity.value() * area)
        self.spinBox_n_vertices.setValue(n_points)

    def _on_update_n_points(self):
        """Recalculate point density if point number is changed."""
        surface = self.surface_layer_select.value.data
        mesh = vedo.mesh.Mesh((surface[0], surface[1]))
        area = mesh.area()

        density = self.spinBox_n_vertices.value()/area
        self.doubleSpinBox_pointdensity.setValue(density)


    def eventFilter(self, obj: QObject, event: QEvent):
        """https://forum.image.sc/t/composing-workflows-in-napari/61222/3."""
        if event.type() == QEvent.ParentChange:
            self.image_layer_select.parent_changed.emit(self.parent())

        return super().eventFilter(obj, event)

    def _run(self):
        """Call analysis function."""

        current_voxel_size = np.asarray([
            self.doubleSpinBox_voxelsize_z.value(),
            self.doubleSpinBox_voxelsize_y.value(),
            self.doubleSpinBox_voxelsize_x.value()
            ])

        results = reconstruct_droplet(
            self.image_layer_select.value.data,
            voxelsize=current_voxel_size,
            target_voxelsize=self.doubleSpinBox_target_voxelsize.value(),
            n_smoothing_iterations=self.spinBox_n_smoothing.value(),
            n_tracing_iterations=self.spinBox_n_refinement_steps.value(),
            resampling_length=self.doubleSpinBox_sampling_length.value(),
            n_points=self.spinBox_n_vertices.value(),
            fit_type=self.comboBox_fittype.currentText(),
            edge_type=self.comboBox_fluorescence_type.currentText(),
            trace_length=self.doubleSpinBox_trace_length.value(),
            sampling_distance=self.doubleSpinBox_sampling_distance.value()
            )

        for layer in results:
            _layer = Layer.create(data=layer[0],
                                  meta=layer[1],
                                  layer_type=layer[2])
            self.viewer.add_layer(_layer)

@frame_by_frame
def reconstruct_droplet(image: ImageData,
                        voxelsize: np.ndarray = None,
                        target_voxelsize: float = 1.0,
                        n_smoothing_iterations: int = 10,
                        n_points: int = 256,
                        n_tracing_iterations: int = 1,
                        resampling_length: float = 5,
                        fit_type: str = 'fancy',
                        edge_type: str = 'interior',
                        trace_length: float = 10,
                        sampling_distance: float = 0.5,
                        verbose=False
                        ) -> List[LayerDataTuple]:
    import napari_process_points_and_surfaces as nppas
    import napari_segment_blobs_and_things_with_membranes as nsbatwm
    from napari_stress import reconstruction
    from .._preprocess import rescale
    import copy

    # rescale
    scaling_factors = voxelsize/target_voxelsize
    rescaled_image = rescale(image,
                             scale_z=scaling_factors[0],
                             scale_y=scaling_factors[1],
                             scale_x=scaling_factors[2]).squeeze()

    # convert to surface
    binarized_image = nsbatwm.threshold_otsu(rescaled_image)
    label_image = nsbatwm.connected_component_labeling(binarized_image)
    surface = nppas.largest_label_to_surface(label_image)

    # Smooth surface
    surface_smoothed = nppas.filter_smooth_laplacian(
        surface, number_of_iterations=n_smoothing_iterations)

    points_first_guess = nppas.sample_points_poisson_disk(
        surface_smoothed, number_of_points=n_points)
    points = copy.deepcopy(points_first_guess)

    # repeat tracing `n_tracing_iterations` times
    for i in range(n_tracing_iterations):
        resampled_points = _resample_pointcloud(
            points, sampling_length=resampling_length)

        traced_points, trace_vectors = reconstruction.trace_refinement_of_surface(
            rescaled_image,
            resampled_points,
            selected_fit_type=fit_type,
            selected_edge=edge_type,
            trace_length=trace_length,
            sampling_distance=sampling_distance,
            remove_outliers=True,
            scale_x=1, scale_y=1, scale_z=1,
            show_progress=verbose)

        points = traced_points[0]

    # =========================================================================
    # Returns
    # =========================================================================

    properties = {'name': 'points_first_guess',
                  'size': 1}
    layer_points_first_guess = (points_first_guess*target_voxelsize, properties, 'points')

    properties = {'name': 'Rescaled image',
                  'scale': [target_voxelsize] * 3}
    layer_image_rescaled = (rescaled_image, properties, 'image')

    properties = {'name': 'Label image',
                  'scale': [target_voxelsize] * 3}
    layer_label_image = (label_image, properties, 'labels')

    traced_points = list(traced_points)
    traced_points[0] *= target_voxelsize

    trace_vectors = list(trace_vectors)
    trace_vectors[0] *= target_voxelsize

    return [layer_image_rescaled,
            layer_label_image,
            layer_points_first_guess,
            traced_points,
            trace_vectors
            ]

def _fibonacci_sampling(number_of_points: int = 256)->PointsData:
    """
    Sample points on unit sphere according to fibonacci-scheme.

    See Also
    --------
    http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/

    Parameters
    ----------
    number_of_points : int, optional
        Number of points to be sampled. The default is 256.

    Returns
    -------
    PointsData

    """
    goldenRatio = (1 + 5**0.5)/2
    i = np.arange(0, number_of_points)
    theta = 2 *np.pi * i / goldenRatio
    phi = np.arccos(1 - 2 * (i + 0.5)/number_of_points)
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)

    return np.stack([x, y, z]).T

def _resample_pointcloud(points: PointsData,
                         sampling_length: float = 5):
    """
    Resampe a spherical-like pointcloud on fibonacci grid.

    Parameters
    ----------
    points : PointsData
    sampling_length : float, optional
        Distance between sampled point locations. The default is 5.

    Returns
    -------
    resampled_points : TYPE

    """
    from scipy.interpolate import Rbf

    # convert to spherical, relative coordinates
    center = np.mean(points, axis=0)
    points_centered = points - center
    points_spherical = vedo.cart2spher(points_centered[:, 0],
                                       points_centered[:, 1],
                                       points_centered[:, 2]).T

    # estimate point number according to passed sampling length
    mean_radius = points_spherical[:, 0].mean()
    surface_area = mean_radius**2 * 2 * np.pi
    n = int(surface_area/sampling_length**2)

    # sample points on unit-sphere according to fibonacci-scheme
    sampled_points = _fibonacci_sampling(n)
    sampled_points = vedo.utils.cart2spher(sampled_points[:, 0],
                                           sampled_points[:, 1],
                                           sampled_points[:, 2]).T

    # interpolate cartesian coordinates on (theta, phi) grid
    theta_interpolation = np.concatenate([points_spherical[:, 1] + 2 * np.pi,
                                          points_spherical[:, 1],
                                          points_spherical[:, 1] - 2 * np.pi])
    phi_interpolation = np.concatenate([points_spherical[:, 2],
                                        points_spherical[:, 2],
                                        points_spherical[:, 2]])
    rbf_x = Rbf(theta_interpolation,
                phi_interpolation,
                list(points_centered[:, 0])*3)
    rbf_y = Rbf(theta_interpolation,
                phi_interpolation,
                list(points_centered[:, 1])*3)
    rbf_z = Rbf(theta_interpolation,
                phi_interpolation,
                list(points_centered[:, 2])*3)

    new_x = rbf_x(sampled_points[:, 1], sampled_points[:, 2])
    new_y = rbf_y(sampled_points[:, 1], sampled_points[:, 2])
    new_z = rbf_z(sampled_points[:, 1], sampled_points[:, 2])

    resampled_points = np.stack([new_x, new_y, new_z]).T + center
    return resampled_points
