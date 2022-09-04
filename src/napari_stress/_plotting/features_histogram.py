import napari
from magicgui import magicgui
import numpy as np
import pandas as pd
from napari_matplotlib import HistogramWidget
from napari_matplotlib.util import Interval
from matplotlib.widgets  import RectangleSelector
from matplotlib.patches import Rectangle
from magicgui.widgets import ComboBox
from typing import List, Optional, Tuple
from qtpy.QtWidgets import QFileDialog, QHBoxLayout, QPushButton
from scipy import stats


class FeaturesHistogramWidget(HistogramWidget):
    """Plot widget to display histogram of selected layer features."""

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
            n_bins={"value": 50, "widget_type": "SpinBox"},
            call_button="Update",
        )
        self._export_button = magicgui(
            self.export,
            call_button='Export plot as csv'
            )

        # buttons to switch between histogram/CDF
        self.enable_histogram = QPushButton('Histogram')
        self.enable_histogram.setCheckable(True)
        self.enable_histogram.setChecked(True)
        self.enable_cdf = QPushButton('CDF')
        self.enable_cdf.setCheckable(True)
        self.enable_histogram.clicked.connect(self._draw)
        self.enable_cdf.clicked.connect(self._draw)

        container = QHBoxLayout()
        container.addWidget(self.enable_histogram)
        container.addWidget(self.enable_cdf)
        self.layout().addLayout(container)

        self.layout().addWidget(self._key_selection_widget.native)
        self.layout().addWidget(self._export_button.native)

        self.viewer = napari_viewer

        # create a second y-axis in the plot
        self.axes2 = self.axes.twinx()  # for cdf
        self.axes3 = self.axes.twinx()  # for selector rectangle
        self.axes3.get_xaxis().set_visible(False)
        self.axes3.get_yaxis().set_visible(False)

        # hook up rectangle selector
        self.rectangle_selector = RectangleSelector(
            self.axes3,
            self._on_area_select,
            drawtype="box",
            useblit=True,
            rectprops=dict(edgecolor="white", fill=False),
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=False,
        )
        self.highlight_rectangle = None
        self.highlight_layer = None

    @property
    def x_axis_key(self) -> Optional[str]:
        """Key to access x axis data from the FeaturesTable."""
        return self._x_axis_key

    @x_axis_key.setter
    def x_axis_key(self, key: Optional[str]) -> None:
        self._x_axis_key = key
        self._draw()

    @property
    def n_bins(self) -> Optional[str]:
        """Key to access y axis data from the FeaturesTable."""
        return self._n_bins

    # @n_bins.setter
    # def n_bins(self, key: Optional[str]) -> None:
    #     # self._y_axis_key = key
    #     self._draw()

    def _set_axis_keys(self, x_axis_key: str, n_bins: int) -> None:
        """Set both axis keys and then redraw the plot."""
        self._x_axis_key = x_axis_key
        self._n_bins = n_bins
        self._draw()

        self.viewer.layers[self.layers[0].name].refresh_colors(True)
        self.viewer.layers[self.layers[0].name].face_color = x_axis_key

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
        y_axis_name = 'Occurrences [#]'

        return data, x_axis_name, y_axis_name

    def _on_update_layers(self) -> None:
        """This is called when the layer selection changes by
        ``self.update_layers()``.

        """
        if hasattr(self, "_key_selection_widget"):
            self._key_selection_widget.reset_choices()

        # reset the axis keys
        self._x_axis_key = None
        self._n_bins = None

    def _on_area_select(self, eclick, erelease):
        """Triggered when user clicks within axes"""
        # get click event coordinates
        x1, _ = eclick.xdata, eclick.ydata
        x2, _ = erelease.xdata, erelease.ydata

        y1 = 0
        y2 = self.axes.get_ylim()[1]
        self._draw_highlight_rectangle(x1, x2, y1, y2)

    def _draw_highlight_rectangle(self, x1, x2, y1, y2):

        # put highlight rectangle in histogram plot
        if self.highlight_rectangle is None:
            self.highlight_rectangle = Rectangle(
                (min(x1,x2),min(y1,y2)), np.abs(x1-x2), np.abs(y1-y2),
                alpha=0.35, facecolor='white')
            self.axes3.add_patch(self.highlight_rectangle)

        else:
            self.highlight_rectangle.set_x(x1)
            self.highlight_rectangle.set_width(np.abs(x1-x2))

        self.canvas.draw()

        # highlight selected points in viewer
        shown_range = self.layers[0].features[self.x_axis_key]
        left = min(x1,x2)
        right = max(x1, x2)
        shown_range = np.asarray((shown_range > left) * (shown_range < right))
        # print(f'Selected {sum(shown_range)} out of {len(shown_range)}')

        # self.viewer.layers[self.layers[0].name].features['shown_range'] = shown_range
        self.viewer.layers[self.layers[0].name].refresh_colors(True)

        colors_highlight = np.ones((self.layers[0].data.shape[0], 4))
        colors_highlight[np.argwhere(shown_range == False), :3] = 0
        self.viewer.layers[self.layers[0].name].edge_color = colors_highlight

    def draw(self) -> None:
        """Clear the axes and histogram the currently selected layer/slice."""
        data, x_axis_name, y_axis_name = self._get_data()

        if len(data) == 0:
            # don't plot if there isn't data
            return

        self.N, bins = np.histogram(data[0], data[1])

        colormapping = self.layers[0].face_colormap
        self.bins_norm = (bins - bins.min())/(bins.max() - bins.min())
        colors = colormapping.map(self.bins_norm)

        if self.enable_histogram.isChecked():
            self.N, bins, patches = self.axes.hist(data[0],
                                                   bins=data[1],
                                                   edgecolor='white',
                                                   linewidth=0.3,
                                                   label=self.layers[0].name)
            # Set histogram style:
            for idx, patch in enumerate(patches):
                patch.set_facecolor(colors[idx])
        else:
            self.axes.clear()

        if self.enable_cdf.isChecked():
            cdf_histogram = stats.rv_histogram((self.N, bins))
            self.axes2.step(bins, cdf_histogram.cdf(data[1]), color='black',
                            where='pre', zorder=1)
        else:
            self.axes2.clear()

        self.axes.set_xlabel(x_axis_name)
        self.axes.set_ylabel(y_axis_name)
        self.axes2.set_ylabel('Cumulative density')

        self.canvas.draw()

    def export(self) -> None:
        """Export plotted data as csv."""
        if not self.axes.has_data():
            print('No data plotted')
            return
        # Not including last bin because len(bins) = len(N) + 1
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.hist.html
        df_to_save = pd.DataFrame({self.axes.get_xlabel(): self.bins_norm[:-1],
                                   self.axes.get_ylabel(): self.N})
        fname = QFileDialog.getSaveFileName(self, 'Save plotted data',
                                            'c:\\',
                                            "Csv files (*.csv)")
        df_to_save.to_csv(fname[0])
        return
