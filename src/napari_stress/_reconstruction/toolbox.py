import os
from pathlib import Path
from typing import List

import numpy as np
import vedo
from magicgui.widgets import create_widget
from napari.layers import Image, Layer
from napari.types import ImageData, LayerDataTuple, PointsData
from napari_tools_menu import register_dock_widget
from qtpy import uic
from qtpy.QtCore import QEvent, QObject
from qtpy.QtWidgets import QWidget

from .._utils.frame_by_frame import frame_by_frame


@register_dock_widget(menu="Surfaces > Droplet reconstruction toolbox (n-STRESS)")
class droplet_reconstruction_toolbox(QWidget):
    """Comprehensive stress analysis of droplet points layer."""

    def __init__(self, napari_viewer):
        super().__init__()

        self.viewer = napari_viewer

        uic.loadUi(os.path.join(Path(__file__).parent, "./toolbox.ui"), self)

        # add input dropdowns to plugin
        self.image_layer_select = create_widget(annotation=Image, label="Image_layer")
        self.layout().addWidget(self.image_layer_select.native, 0, 1)
        self.installEventFilter(self)

        # populate comboboxes with allowed values
        self.comboBox_fittype.addItems(["fancy", "quick"])
        self.comboBox_fluorescence_type.addItems(["interior", "surface"])
        self.comboBox_interpolation_method.addItems(["linear", "cubic"])

        # calculate density/point number
        self.spinBox_n_vertices.setValue(256)

        self.pushButton_run.clicked.connect(self._run)

    def eventFilter(self, obj: QObject, event: QEvent):
        """https://forum.image.sc/t/composing-workflows-in-napari/61222/3."""
        if event.type() == QEvent.ParentChange:
            self.image_layer_select.parent_changed.emit(self.parent())

        return super().eventFilter(obj, event)

    def _run(self):
        """Call analysis function."""
        import webbrowser

        current_voxel_size = np.asarray(
            [
                self.doubleSpinBox_voxelsize_z.value(),
                self.doubleSpinBox_voxelsize_y.value(),
                self.doubleSpinBox_voxelsize_x.value(),
            ]
        )

        if self.checkBox_use_dask.isChecked():
            webbrowser.open("http://localhost:8787")

        results = reconstruct_droplet(
            self.image_layer_select.value.data,
            voxelsize=current_voxel_size,
            target_voxelsize=self.doubleSpinBox_target_voxelsize.value(),
            smoothing_sigma=self.doubleSpinBox_gaussian_blur.value(),
            n_smoothing_iterations=self.spinBox_n_smoothing.value(),
            n_tracing_iterations=self.spinBox_n_refinement_steps.value(),
            resampling_length=self.doubleSpinBox_sampling_length.value(),
            n_points=self.spinBox_n_vertices.value(),
            fit_type=self.comboBox_fittype.currentText(),
            edge_type=self.comboBox_fluorescence_type.currentText(),
            trace_length=self.doubleSpinBox_trace_length.value(),
            remove_outliers=self.checkBox_remove_outliers.isChecked(),
            outlier_tolerance=self.doubleSpinBox_outlier_tolerance.value(),
            sampling_distance=self.doubleSpinBox_sampling_distance.value(),
            interpolation_method=self.comboBox_interpolation_method.currentText(),
            use_dask=self.checkBox_use_dask.isChecked()
        )

        for layer in results:
            _layer = Layer.create(data=layer[0], meta=layer[1], layer_type=layer[2])
            self.viewer.add_layer(_layer)


@frame_by_frame
def reconstruct_droplet(
    image: ImageData,
    voxelsize: np.ndarray = None,
    target_voxelsize: float = 1.0,
    smoothing_sigma: float = 1.0,
    n_smoothing_iterations: int = 10,
    n_points: int = 256,
    n_tracing_iterations: int = 1,
    resampling_length: float = 5,
    fit_type: str = "fancy",
    edge_type: str = "interior",
    trace_length: float = 10,
    remove_outliers: bool = True,
    outlier_tolerance: float = 1.5,
    sampling_distance: float = 0.5,
    interpolation_method: str = "cubic",
    verbose=False,
) -> List[LayerDataTuple]:
    import copy
    import napari_process_points_and_surfaces as nppas
    import napari_segment_blobs_and_things_with_membranes as nsbatwm
    from napari_stress import reconstruction

    try:
        import pyclesperanto_prototype as cle
        cle.get_device()  # run to check whether GPU exists

        cle_installed = True
    except Exception:
        cle_installed = False
        from skimage import filters, transform

    scaling_factors = voxelsize / target_voxelsize

    # rescale
    if cle_installed:
        rescaled_image = np.asarray(
            cle.scale(
                image,
                None,
                factor_z=scaling_factors[0],
                factor_y=scaling_factors[1],
                factor_x=scaling_factors[2],
                auto_size=True,
                )
            ).squeeze()

        # Blur
        rescaled_image = np.asarray(
            cle.gaussian_blur(
                rescaled_image,
                sigma_x=smoothing_sigma,
                sigma_y=smoothing_sigma,
                sigma_z=smoothing_sigma,
            )
        )
        binarized_image = cle.threshold_otsu(rescaled_image)

    else:
        rescaled_image = transform.rescale(
            image,
            scaling_factors)
        rescaled_image = filters.gaussian(
            rescaled_image,
            sigma=smoothing_sigma)
        threshold = filters.threshold_otsu(rescaled_image)
        binarized_image = rescaled_image > threshold

    # convert to surface
    label_image = nsbatwm.connected_component_labeling(binarized_image)
    surface = nppas.largest_label_to_surface(label_image)
    mesh_vedo = vedo.mesh.Mesh((surface[0], surface[1])).clean()

    # Smooth surface
    surface_smoothed = nppas.smooth_surface(
        (mesh_vedo.points(), mesh_vedo.faces()),
        number_of_iterations=n_smoothing_iterations,
    )
    points_low = nppas.sample_points_from_surface(
        surface_smoothed, distance_fraction=0.01
    )
    points_high = nppas.sample_points_from_surface(
        surface_smoothed, distance_fraction=0.25
    )
    points_per_fraction = (0.01 - 0.25) / (len(points_low) - len(points_high))

    points_first_guess = nppas.sample_points_from_surface(
        surface_smoothed, distance_fraction=points_per_fraction * n_points
    )

    points = copy.deepcopy(points_first_guess)

    # repeat tracing `n_tracing_iterations` times
    for i in range(n_tracing_iterations):
        resampled_points = _resample_pointcloud(
            points, sampling_length=resampling_length
        )

        traced_points, trace_vectors = reconstruction.trace_refinement_of_surface(
            rescaled_image,
            resampled_points,
            selected_fit_type=fit_type,
            selected_edge=edge_type,
            trace_length=trace_length,
            sampling_distance=sampling_distance,
            remove_outliers=remove_outliers,
            outlier_tolerance=outlier_tolerance,
            interpolation_method=interpolation_method,
        )

        points = traced_points[0]

    # =========================================================================
    # Returns
    # =========================================================================

    properties = {"name": "points_first_guess", "size": 1}
    layer_points_first_guess = (
        points_first_guess * target_voxelsize,
        properties,
        "points",
    )

    properties = {"name": "Rescaled image", "scale": [target_voxelsize] * 3}
    layer_image_rescaled = (rescaled_image, properties, "image")

    properties = {"name": "Label image", "scale": [target_voxelsize] * 3}
    layer_label_image = (label_image, properties, "labels")

    traced_points = list(traced_points)
    traced_points[0] *= target_voxelsize

    trace_vectors = list(trace_vectors)
    trace_vectors[0] *= target_voxelsize

    properties = {"name": "Center", "symbol": "ring", "face_color": "yellow", "size": 3}
    droplet_center = (traced_points[0].mean(axis=0)[None, :], properties, "points")

    return [
        layer_image_rescaled,
        layer_label_image,
        layer_points_first_guess,
        traced_points,
        trace_vectors,
        droplet_center,
    ]


def _fibonacci_sampling(number_of_points: int = 256) -> PointsData:
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
    goldenRatio = (1 + 5**0.5) / 2
    i = np.arange(0, number_of_points)
    theta = 2 * np.pi * i / goldenRatio
    phi = np.arccos(1 - 2 * (i + 0.5) / number_of_points)
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)

    return np.stack([x, y, z]).T


def _resample_pointcloud(points: PointsData, sampling_length: float = 5):
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
    from scipy.interpolate import griddata

    # convert to spherical, relative coordinates
    center = np.mean(points, axis=0)
    points_centered = points - center
    points_spherical = vedo.cart2spher(
        points_centered[:, 0], points_centered[:, 1], points_centered[:, 2]
    ).T

    # estimate point number according to passed sampling length
    mean_radius = points_spherical[:, 0].mean()
    surface_area = mean_radius**2 * 4 * np.pi
    n = int(surface_area / sampling_length**2)

    # sample points on unit-sphere according to fibonacci-scheme
    sampled_points = _fibonacci_sampling(n)
    sampled_points = vedo.utils.cart2spher(
        sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2]
    ).T

    # interpolate cartesian coordinates on (theta, phi) grid
    theta_interpolation = np.concatenate(
        [points_spherical[:, 1], points_spherical[:, 1], points_spherical[:, 1]]
    )
    phi_interpolation = np.concatenate(
        [
            points_spherical[:, 2] + 2 * np.pi,
            points_spherical[:, 2],
            points_spherical[:, 2] - 2 * np.pi,
        ]
    )

    new_x = griddata(
        np.stack([theta_interpolation, phi_interpolation]).T,
        list(points_centered[:, 0]) * 3,
        sampled_points[:, 1:],
        method="cubic",
    )

    new_y = griddata(
        np.stack([theta_interpolation, phi_interpolation]).T,
        list(points_centered[:, 1]) * 3,
        sampled_points[:, 1:],
        method="cubic",
    )

    new_z = griddata(
        np.stack([theta_interpolation, phi_interpolation]).T,
        list(points_centered[:, 2]) * 3,
        sampled_points[:, 1:],
        method="cubic",
    )

    resampled_points = np.stack([new_x, new_y, new_z]).T + center

    no_nan_idx = np.where(~np.isnan(resampled_points[:, 0]))[0]
    return resampled_points[no_nan_idx, :]
