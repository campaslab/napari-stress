# -*- coding: utf-8 -*-
from magicgui.widgets.protocols import CategoricalWidgetProtocol
from typing import List

from napari import layers
from napari.utils._magicgui import find_viewer_ancestor
from magicgui import register_type

from ._stress.manifold_SPB import manifold

_METADATAKEY_MANIFOLD = "manifold"
_METADATAKEY_MEAN_CURVATURE = "mean_curvature"
_METADATAKEY_PRINCIPAL_CURVATURES1 = "principal_curvatures_k1"
_METADATAKEY_PRINCIPAL_CURVATURES2 = "principal_curvatures_k2"
_METADATAKEY_MEAN_CURVATURE_DIFFERENCE = (
    "difference_mean_curvature_cartesian_radial_manifold"
)
_METADATAKEY_H0_ELLIPSOID = "H0_ellipsoid"
_METADATAKEY_H0_SURFACE_INTEGRAL = "H0_surface_integral"
_METADATAKEY_H0_VOLUME_INTEGRAL = "H0_volume_integral"
_METADATAKEY_H0_RADIAL_SURFACE = "H0_radial_surface_integral"
_METADATAKEY_S2_VOLUME_INTEGRAL = "S2_volume_integral"
_METADATAKEY_H0_ARITHMETIC = "H0_arithmetic"
_METADATAKEY_H_E123_ELLIPSOID = "H_ellipsoid_major_medial_minor"
_METADATAKEY_GAUSS_BONNET_ABS = "Gauss_Bonnet_error"
_METADATAKEY_GAUSS_BONNET_REL = "Gauss_Bonnet_relative_error"
_METADATAKEY_GAUSS_BONNET_ABS_RAD = "Gauss_Bonnet_error_radial"
_METADATAKEY_GAUSS_BONNET_REL_RAD = "Gauss_Bonnet_relative_error_radial"

_METADATAKEY_STRESS_TISSUE = "stress_tissue"
_METADATAKEY_STRESS_TISSUE_ANISO = "stress_tissue_anisotropy"

_METADATAKEY_STRESS_CELL = "stress_cell"
_METADATAKEY_STRESS_CELL_ANISO = "stress_cell_anisotropy"

_METADATAKEY_STRESS_TOTAL = "stress_total"
_METADATAKEY_STRESS_TOTAL_ANISO = "stress_total_anisotropy"
_METADATAKEY_STRESS_TOTAL_RADIAL = "stress_total_radial"

_METADATAKEY_STRESS_CELL_NEAREST_PAIR_ANISO = "stress_cell_nearest_pair_anisotropy"
_METADATAKEY_STRESS_CELL_NEAREST_PAIR_DIST = "stress_cell_nearest_pair_distance"
_METADATAKEY_STRESS_CELL_ALL_PAIR_ANISO = "stress_cell_all_pair_anisotropy"
_METADATAKEY_STRESS_CELL_ALL_PAIR_DIST = "stress_cell_all_pair_distance"

_METADATAKEY_AUTOCORR_SPATIAL_TOTAL = "autocorrelations_spatial_total"
_METADATAKEY_AUTOCORR_SPATIAL_CELL = "autocorrelations_spatial_cell"
_METADATAKEY_AUTOCORR_SPATIAL_TISSUE = "autocorrelations_spatial_tissue"
_METADATAKEY_AUTOCORR_TEMPORAL_TOTAL = "autocorrelations_temporal_total"
_METADATAKEY_AUTOCORR_TEMPORAL_CELL = "autocorrelations_temporal_cell"
_METADATAKEY_AUTOCORR_TEMPORAL_TISSUE = "autocorrelations_temporal_tissue"

_METADATAKEY_EXTREMA_CELL_STRESS = "stress_cell_extrema"
_METADATAKEY_EXTREMA_TOTAL_STRESS = "stress_total_extrema"

_METADATAKEY_STRESS_TENSOR_ELLI = "Tissue_stress_tensor_elliptical"
_METADATAKEY_STRESS_TENSOR_ELLI_E1 = "Tissue_stress_tensor_elliptical_e1"
_METADATAKEY_STRESS_TENSOR_ELLI_E2 = "Tissue_stress_tensor_elliptical_e2"
_METADATAKEY_STRESS_TENSOR_ELLI_E3 = "Tissue_stress_tensor_elliptical_e3"
_METADATAKEY_STRESS_ELLIPSOID_ANISO_E12 = "stress_ellipsoid_anisotropy_e12"
_METADATAKEY_STRESS_ELLIPSOID_ANISO_E13 = "stress_ellipsoid_anisotropy_e13"
_METADATAKEY_STRESS_ELLIPSOID_ANISO_E23 = "stress_ellipsoid_anisotropy_e23"
_METADATAKEY_STRESS_TENSOR_CART = "Tissue_stress_tensor_cartesian"
_METADATAKEY_ANGLE_ELLIPSOID_CART_E1 = "angle_ellipsoid_cartesian_e1_x1"
_METADATAKEY_ANGLE_ELLIPSOID_CART_E2 = "angle_ellipsoid_cartesian_e1_x2"
_METADATAKEY_ANGLE_ELLIPSOID_CART_E3 = "angle_ellipsoid_cartesian_e1_x3"

_METADATAKEY_FIT_RESIDUE = "fit_residue"
_METADATAKEY_ELIPSOID_DEVIATION_CONTRIB = "Elipsoid_deviation_contribution_matrix"


def _get_layers_features(gui: CategoricalWidgetProtocol) -> List[layers.Layer]:
    """Retrieve layers matching gui.annotation, from the Viewer the gui is in.

    Parameters
    ----------
    gui : magicgui.widgets.Widget
        The instantiated MagicGui widget.  May or may not be docked in a
        dock widget.
    Returns
    -------
    tuple
        Tuple of layers of type ``gui.annotation``
    Examples
    --------
    This allows the user to do this, and get a dropdown box in their GUI
    that shows the available image layers.
    >>> @magicgui
    ... def get_layer_mean(layer: napari.layers.Image) -> float:
    ...     return layer.data.mean()
    """
    if not (viewer := find_viewer_ancestor(gui.native)):
        return ()

    search_key = gui.annotation.__name__

    return [
        layer
        for layer in viewer.layers
        if search_key in list(layer.features.keys()) + list(layer.metadata.keys())
    ]


register_type(manifold, choices=_get_layers_features)
