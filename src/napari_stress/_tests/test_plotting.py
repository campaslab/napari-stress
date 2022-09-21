def test_plotting(make_napari_viewer):
    from napari_stress import FeaturesHistogramWidget
    from napari_stress import get_droplet_point_cloud
    from napari_stress._spherical_harmonics.spherical_harmonics_napari \
        import fit_spherical_harmonics

    # Load sample data
    data = get_droplet_point_cloud()[0][0]

    # Calculate spherical harmonics
    layer_data_tuple = fit_spherical_harmonics(data)

    viewer = make_napari_viewer()
    # Add sample points to viewer
    viewer.add_points(layer_data_tuple[0], **layer_data_tuple[1])
    # Add plot widget to viewer
    plot_widget = FeaturesHistogramWidget(viewer)
    viewer.window.add_dock_widget(plot_widget)
    # Run plot widget
    plot_widget._key_selection_widget()
    # Check if plot has data
    assert plot_widget.axes.has_data()

    plot_widget.enable_cdf.setChecked(True)
    plot_widget._key_selection_widget()

    # check the data highlighting
    plot_widget._draw_highlight_rectangle(0,1,0,1)


if __name__ == '__main__':
    import napari
    test_plotting(napari.Viewer)
