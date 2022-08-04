(point_and_click.spherical_harmonics_expansion)=
# Measure curvature

This tutorial explains how to measure [curvature](spherical_harmonics:measurements:mean_curvature) from a sperhical harmonics expansion. To do so, first create a spherical harmonics expansion as shownw in [this tutorial](./demo_spherical_harmonics.ipynb). To get the curvature from this you'll first have to perform a lebedev quadrature and then measure curvature.

## Lebedev quadrature

A lebedev quadrature determines points on the surface of a sperhical harmonics expansion that allow to calculate downstream parameters with high accuracy. To do so, chose the appropriate command from the napari tools menu (`Tools > Points > Perform lebedev quadrature (n-STRESS)` and select the layer with the expansion data from the drowdown:

![](./imgs/demo_measure_curvature1.png)

## Measure curvature

Next, you can already measure the mean curvature on this surface with `Tools > Measurement > Measure mean curvature on manifold (n-STRESS)"`. Make sure to select the previously generated layer that contains the Lebedev quadrature points in the dropdown:

![](./imgs/demo_measure_curvature2.png)

The results are then stored in the `layer.features`. To retrieve the actual curvature values, navigate to the console of the viewer (button in the lower left corner) and type `viewer.layers[-1].features` to display all features that are associated with the data:

![](./imgs/demo_measure_curvature3.png)

Type `viewer.layers[-1].features['Mean_curvature_at_lebedev_points']` to retrieve specifically the curvature values.