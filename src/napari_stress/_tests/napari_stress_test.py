# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 16:42:04 2022

@author: mazo260d
"""

import napari
from napari_stress._spherical_harmonics.spherical_harmonics_napari import fit_spherical_harmonics
# from napari_stress._spherical_harmomics.toolbox import spherical_harmonics_toolbox
from magicgui import magicgui
import pandas as pd
import numpy as np
from napari_matplotlib import HistogramWidget 
from napari_matplotlib.util import Interval
from magicgui.widgets import ComboBox
from typing import List, Optional, Tuple

class FeaturesHistogramWidget(HistogramWidget):
    n_layers_input = Interval(1, 1)
    # All layers that have a .features attributes
    input_layer_types = (
        napari.layers.Labels,
        napari.layers.Points,
        napari.layers.Shapes,
        napari.layers.Tracks,
        napari.layers.Vectors,
    )

    def __init__(self, napari_viewer: napari.viewer.Viewer):
        super().__init__(napari_viewer)
        self._key_selection_widget = magicgui(
            self._set_axis_keys,
            x_axis_key={"choices": self._get_valid_axis_keys},
            n_bins={"value": 100, "widget_type": "SpinBox"},
            call_button="plot",
        )

        self.layout().addWidget(self._key_selection_widget.native)

    @property
    def x_axis_key(self) -> Optional[str]:
        """Key to access x axis data from the FeaturesTable"""
        return self._x_axis_key

    @x_axis_key.setter
    def x_axis_key(self, key: Optional[str]) -> None:
        self._x_axis_key = key
        self._draw()

    @property
    def n_bins(self) -> Optional[str]:
        """Key to access y axis data from the FeaturesTable"""
        return self._n_bins

    # @n_bins.setter
    # def n_bins(self, key: Optional[str]) -> None:
    #     # self._y_axis_key = key
    #     self._draw()

    def _set_axis_keys(self, x_axis_key: str, n_bins: int) -> None:
        """Set both axis keys and then redraw the plot"""
        self._x_axis_key = x_axis_key
        self._n_bins = n_bins
        self._draw()

    def _get_valid_axis_keys(
        self, combo_widget: Optional[ComboBox] = None
    ) -> List[str]:
        """
        Get the valid axis keys from the layer FeatureTable.
        Returns
        -------
        axis_keys : List[str]
            The valid axis keys in the FeatureTable. If the table is empty
            or there isn't a table, returns an empty list.
        """
        if len(self.layers) == 0 or not (hasattr(self.layers[0], "features")):
            return []
        else:
            return self.layers[0].features.keys()

    def _get_data(self) -> Tuple[List[np.ndarray], str, int]:
        """Get the plot data.
        Returns
        -------
        data : List[np.ndarray]
            List contains X and Y columns from the FeatureTable. Returns
            an empty array if nothing to plot.
        x_axis_name : str
            The title to display on the x axis. Returns
            an empty string if nothing to plot.
        y_axis_name: int
            The title to display on the y axis. Returns
            an empty string if nothing to plot.
        """
        if not hasattr(self.layers[0], "features"):
            # if the selected layer doesn't have a featuretable,
            # skip draw
            return [], "", ""

        feature_table = self.layers[0].features

        if (
            (len(feature_table) == 0)
            or (self.x_axis_key is None)
        ):
            return [], "", 0

        data_x = feature_table[self.x_axis_key]
        bins = np.linspace(np.min(data_x), np.max(data_x), self.n_bins+1)
        # data_y = feature_table[self.y_axis_key]
        data = [data_x, bins]

        x_axis_name = self.x_axis_key.replace("_", " ")
        y_axis_name = '# of occurrences'#self.y_axis_key.replace("_", " ")

        return data, x_axis_name, y_axis_name

    def _on_update_layers(self) -> None:
        """
        This is called when the layer selection changes by
        ``self.update_layers()``.
        """
        if hasattr(self, "_key_selection_widget"):
            self._key_selection_widget.reset_choices()

        # reset the axis keys
        self._x_axis_key = None
        self._n_bins = None
    
    def draw(self) -> None:
        """
        Clear the axes and histogram the currently selected layer/slice.
        """
        data, x_axis_name, y_axis_name = self._get_data()

        if len(data) == 0:
            # don't plot if there isn't data
            return

        
        self.axes.hist(data[0], bins=data[1])#, label=layer.name)

        self.axes.set_xlabel(x_axis_name)
        self.axes.set_ylabel(y_axis_name)

# from napari_matplotlib.base import NapariMPLWidget
# from napari_matplotlib.util import Interval
# import numpy as np

# _COLORS = {"r": "tab:red", "g": "tab:green", "b": "tab:blue"}


# class HistogramWidget(NapariMPLWidget):
#     """
#     Display a histogram of the currently selected layer.
#     """

#     n_layers_input = Interval(1, 1)
#     input_layer_types = (napari.layers.Image,
#                          napari.layers.Points,)
#     # Include also Surface and Labels
#     # Decide where to specify the property to be taken: data, roperties, feature or metadata
#     # for now, using feature

#     def __init__(self, napari_viewer: napari.viewer.Viewer):
#         super().__init__(napari_viewer)
#         self.axes = self.canvas.figure.subplots()
#         self.update_layers(None)

#     def clear(self) -> None:
#         self.axes.clear()

#     def draw(self) -> None:
#         """
#         Clear the axes and histogram the currently selected layer/slice.
#         """
#         layer = self.layers[0]
#         values = layer.features['error']
#         bins = np.linspace(np.min(values), np.max(values), 100)

#         if layer.data.ndim - layer.rgb == 3:
#             # 3D data, can be single channel or RGB
#             data = layer.data[self.current_z]
#             self.axes.set_title(f"z={self.current_z}")
#         else:
#             data = layer.data

#         if layer.rgb:
#             # Histogram RGB channels independently
#             for i, c in enumerate("rgb"):
#                 self.axes.hist(
#                     data[..., i].ravel(),
#                     bins=bins,
#                     label=c,
#                     histtype="step",
#                     color=_COLORS[c],
#                 )
#         else:
#             self.axes.hist(data.ravel(), bins=bins, label=layer.name)

#         self.axes.legend()


viewer = napari.Viewer()

df = pd.read_csv('../sample_data/dropplet_point_cloud.csv')

# Add points data to napari
viewer.add_points(df.iloc[:,1:].values, size=1, visible=False)
# Loads spherical harmonics widget
widget_fit_harmonics = magicgui(fit_spherical_harmonics)
viewer.window.add_dock_widget(widget_fit_harmonics)
# Run spherical harmonics widget with default parameters
widget_fit_harmonics()
viewer.dims.ndisplay=3 # Show data in 3D

widget_toolbox = FeaturesHistogramWidget(viewer)
viewer.window.add_dock_widget(widget_toolbox)


