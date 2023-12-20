import os
from pathlib import Path
from typing import List

import numpy as np
from magicgui.widgets import create_widget
from napari.layers import Image, Layer
from napari.types import ImageData, LayerDataTuple
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
        self.image_layer_select.changed.connect(self._set_scales)
        self.pushButton_export.clicked.connect(self._export_settings)
        self.pushButton_import.clicked.connect(self._import_settings)

        self._set_scales()

    def eventFilter(self, obj: QObject, event: QEvent):
        """https://forum.image.sc/t/composing-workflows-in-napari/61222/3."""
        if event.type() == QEvent.ParentChange:
            self.image_layer_select.parent_changed.emit(self.parent())

        return super().eventFilter(obj, event)

    def _set_scales(self):
        """Get scales from loaded data."""
        try:
            scales = self.image_layer_select.value.scale
            scales = scales[-3:]  # scales may be 4D
            self.doubleSpinBox_voxelsize_x.setValue(scales[2])
            self.doubleSpinBox_voxelsize_y.setValue(scales[1])
            self.doubleSpinBox_voxelsize_z.setValue(scales[0])
        except Exception:
            pass

    def _export_settings(self, file_name: str = None):
        """
        Export reconstruction parameters to YAML file.
        """
        from .._utils.import_export_settings import export_settings

        reconstruction_parameters = {
            "voxelsize": np.asarray(
                [
                    self.doubleSpinBox_voxelsize_z.value(),
                    self.doubleSpinBox_voxelsize_y.value(),
                    self.doubleSpinBox_voxelsize_x.value(),
                ]
            ),
            "target_voxelsize": self.doubleSpinBox_target_voxelsize.value(),
            "smoothing_sigma": self.doubleSpinBox_gaussian_blur.value(),
            "n_smoothing_iterations": self.spinBox_n_smoothing.value(),
            "n_points": self.spinBox_n_vertices.value(),
            "n_tracing_iterations": self.spinBox_n_refinement_steps.value(),
            "resampling_length": self.doubleSpinBox_sampling_length.value(),
            "fit_type": self.comboBox_fittype.currentText(),
            "edge_type": self.comboBox_fluorescence_type.currentText(),
            "trace_length": self.doubleSpinBox_trace_length.value(),
            "sampling_distance": self.doubleSpinBox_sampling_distance.value(),
            "interpolation_method": self.comboBox_interpolation_method.currentText(),
            "outlier_tolerance": self.doubleSpinBox_outlier_tolerance.value(),
            "remove_outliers": self.checkBox_remove_outliers.isChecked(),
        }

        export_settings(reconstruction_parameters, self, file_name=file_name)

    def _import_settings(self, file_name: str = None):
        """
        Import reconstruction parameters from YAML file.
        """
        from .._utils.import_export_settings import import_settings

        reconstruction_parameters = import_settings(self, file_name=file_name)

        self.doubleSpinBox_voxelsize_z.setValue(
            reconstruction_parameters["voxelsize"][0]
        )
        self.doubleSpinBox_voxelsize_y.setValue(
            reconstruction_parameters["voxelsize"][1]
        )
        self.doubleSpinBox_voxelsize_x.setValue(
            reconstruction_parameters["voxelsize"][2]
        )
        self.doubleSpinBox_target_voxelsize.setValue(
            reconstruction_parameters["target_voxelsize"]
        )
        self.doubleSpinBox_gaussian_blur.setValue(
            reconstruction_parameters["smoothing_sigma"]
        )
        self.spinBox_n_smoothing.setValue(
            reconstruction_parameters["n_smoothing_iterations"]
        )
        self.spinBox_n_vertices.setValue(reconstruction_parameters["n_points"])
        self.spinBox_n_refinement_steps.setValue(
            reconstruction_parameters["n_tracing_iterations"]
        )
        self.doubleSpinBox_sampling_length.setValue(
            reconstruction_parameters["resampling_length"]
        )
        self.comboBox_fittype.setCurrentText(reconstruction_parameters["fit_type"])
        self.comboBox_fluorescence_type.setCurrentText(
            reconstruction_parameters["edge_type"]
        )
        self.doubleSpinBox_trace_length.setValue(
            reconstruction_parameters["trace_length"]
        )
        self.doubleSpinBox_sampling_distance.setValue(
            reconstruction_parameters["sampling_distance"]
        )
        self.comboBox_interpolation_method.setCurrentText(
            reconstruction_parameters["interpolation_method"]
        )
        self.doubleSpinBox_outlier_tolerance.setValue(
            reconstruction_parameters["outlier_tolerance"]
        )
        self.checkBox_remove_outliers.setChecked(
            reconstruction_parameters["remove_outliers"]
        )

    def _run(self):
        """Call analysis function."""
        import webbrowser
        from dask.distributed import get_client

        current_voxel_size = np.asarray(
            [
                self.doubleSpinBox_voxelsize_z.value(),
                self.doubleSpinBox_voxelsize_y.value(),
                self.doubleSpinBox_voxelsize_x.value(),
            ]
        )

        if self.checkBox_use_dask.isChecked():
            client = get_client()
            webbrowser.open_new_tab(client.dashboard_link)

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
            return_intermediate_results=self.checkBox_return_intermediate.isChecked(),
            use_dask=self.checkBox_use_dask.isChecked(),
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
    return_intermediate_results: bool = False,
) -> List[LayerDataTuple]:
    """
    Reconstruct droplet surface from points layer.

    Parameters
    ----------
    image : ImageData
        Image data.
    voxelsize : np.ndarray, optional
        Voxel size of image. The default is None.
    target_voxelsize : float, optional
        Target voxel size for reconstruction. The default is 1.0.
    smoothing_sigma : float, optional
        Sigma for gaussian smoothing. The default is 1.0.
    n_smoothing_iterations : int, optional
        Number of smoothing iterations. The default is 10.
    n_points : int, optional
        Number of points to be sampled. The default is 256.
    n_tracing_iterations : int, optional
        Number of tracing iterations. The default is 1.
    resampling_length : float, optional
        Distance between sampled point locations. The default is 5.
        Choose smaller values for more accurate reconstruction.
    fit_type : str, optional
        Type of fit to be used. The default is "fancy". Can also be "quick".
    edge_type : str, optional
        Type of edge to be used. The default is "interior". Can also be "surface".
    trace_length : float, optional
        Length of traces along which to measure intensity. The default is 10.
    remove_outliers : bool, optional
        Whether to remove outliers. The default is True.
    outlier_tolerance : float, optional
        Tolerance for outlier removal. The default is 1.5.
    sampling_distance : float, optional
        Distance between points sampled on traces. The default is 0.5.
    interpolation_method : str, optional
        Interpolation method to be used. The default is "cubic". Can also be "linear".
        "cubic" is more accurate but slower.
    use_dask: bool, optional
        Whether to use dask for parallelization. The default is False.
    return_intermediate_results: bool, optional
        Whether to return intermediate results. The default is False.

    Returns
    -------
    List[LayerDataTuple]
        List of napari layers:
            - rescaled image: image rescaled to target voxel size
            - label image: connected components of rescaled image
            - points first guess: points sampled on fibonacci grid
            - traced points: points after tracing
            - trace vectors: vectors along which intensity was measured
            - droplet center: center of droplet

    """
    import copy
    import napari_process_points_and_surfaces as nppas
    import napari_segment_blobs_and_things_with_membranes as nsbatwm
    from napari_stress import reconstruction
    from skimage import filters, transform
    from .refine_surfaces import resample_pointcloud
    from .patches import iterative_curvature_adaptive_patch_fitting

    scaling_factors = voxelsize / target_voxelsize
    rescaled_image = transform.rescale(
        image, scaling_factors, preserve_range=True, anti_aliasing=True
    )
    rescaled_image = filters.gaussian(rescaled_image, sigma=smoothing_sigma)
    threshold = filters.threshold_otsu(rescaled_image)
    binarized_image = rescaled_image > threshold

    # convert to surface
    label_image = nsbatwm.connected_component_labeling(binarized_image)
    surface = nppas.largest_label_to_surface(label_image)
    surface = nppas.remove_duplicate_vertices(surface)

    # Smooth and decimate
    surface = nppas.smooth_surface(
        surface, n_smoothing_iterations, feature_angle=120, edge_angle=90
    )
    surface = nppas.decimate_quadric(surface, number_of_vertices=n_points)
    points_first_guess = surface[0]
    points = copy.deepcopy(points_first_guess)

    # repeat tracing `n_tracing_iterations` times
    for i in range(n_tracing_iterations):
        resampled_points = resample_pointcloud(
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

    # adaptive patch fitting
    points = iterative_curvature_adaptive_patch_fitting(traced_points[0])

    # =========================================================================
    # Returns
    # =========================================================================

    properties = {"name": "surface_first_guess"}
    layer_first_guess = (
        (surface[0] * target_voxelsize, surface[1]),
        properties,
        "surface",
    )

    properties = {"name": "points_patch_fitted", "size": 0.5}
    layer_patch_fitted = (
        points * target_voxelsize,
        properties,
        "points",
    )

    properties = {"name": "Label image", "scale": [target_voxelsize] * 3}
    layer_label_image = (label_image, properties, "labels")

    traced_points = list(traced_points)
    traced_points[0] *= target_voxelsize

    trace_vectors = list(trace_vectors)
    trace_vectors[0] *= target_voxelsize

    properties = {"name": "Center", "symbol": "ring", "face_color": "yellow", "size": 3}
    droplet_center = (traced_points[0].mean(axis=0)[None, :], properties, "points")

    properties = {
        "name": "Rescaled image",
        "blending": "additive",
        "scale": [target_voxelsize] * 3,
    }
    layer_rescaled_image = (rescaled_image, properties, "image")

    if return_intermediate_results:
        return [
            layer_label_image,
            layer_first_guess,
            layer_patch_fitted,
            traced_points,
            trace_vectors,
            droplet_center,
            layer_rescaled_image,
        ]
    else:
        return [
            layer_first_guess,
            layer_patch_fitted,
        ]
