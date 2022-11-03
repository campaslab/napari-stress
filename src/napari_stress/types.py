# -*- coding: utf-8 -*-
from magicgui.widgets._bases import CategoricalWidget
from typing import List, NewType

from napari import layers
from napari.utils._magicgui import find_viewer_ancestor
from magicgui import register_type

from ._stress.manifold_SPB import manifold

_METADATAKEY_MANIFOLD = 'manifold'
_METADATAKEY_MEAN_CURVATURE = 'mean_curvature'
_METADATAKEY_MEAN_CURVATURE_DIFFERENCE = 'difference_mean_curvature_cartesian_radial_manifold'
_METADATAKEY_H0_ELLIPSOID = 'H0_ellipsoid'
_METADATAKEY_H0_SURFACE_INTEGRAL = 'H0_surface_integral'
_METADATAKEY_H0_VOLUME_INTEGRAL = 'H0_volume_integral'
_METADATAKEY_H0_RADIAL_SURFACE = 'H0_radial_surface_integral'
_METADATAKEY_S2_VOLUME_INTEGRAL = 'S2_volume_integral'
_METADATAKEY_H0_ARITHMETIC = 'H0_arithmetic'
_METADATAKEY_H_E123_ELLIPSOID = 'H_ellipsoid_major_medial_minor'
_METADATAKEY_GAUSS_BONNET_ABS = 'Gauss_Bonnet_error'
_METADATAKEY_GAUSS_BONNET_REL = 'Gauss_Bonnet_relative_error'
_METADATAKEY_GAUSS_BONNET_ABS_RAD = 'Gauss_Bonnet_error_radial'
_METADATAKEY_GAUSS_BONNET_REL_RAD = 'Gauss_Bonnet_relative_error_radial'
_METADATAKEY_ANISO_STRESS_TISSUE = 'anisotropic_stress_tissue'
_METADATAKEY_ANISO_STRESS_CELL = 'anisotropic_stress_cell'
_METADATAKEY_ANISO_STRESS_TOTAL = 'anisotropic_stress'
_METADATAKEY_ANISO_STRESS_TOTAL_RADIAL = 'anisotropic_stress_total_radial'
_METADATAKEY_STRESS_TENSOR_ELLI = 'Tissue_stress_tensor_elliptical'
_METADATAKEY_STRESS_TENSOR_CART = 'Tissue_stress_tensor_cartesian'
_METADATAKEY_MAX_TISSUE_ANISOTROPY = 'max_issue_stress_anisotropy'
_METADATAKEY_FIT_RESIDUE = 'fit_residue'
_METADATAKEY_ELIPSOID_DEVIATION_CONTRIB = 'Elipsoid_deviation_contribution_matrix'

def _get_layers_features(gui: CategoricalWidget) -> List[layers.Layer]:
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
        layer for layer in viewer.layers if search_key in list(layer.features.keys()) + list(layer.metadata.keys())
        ]

register_type(
    manifold,
    choices = _get_layers_features
)
