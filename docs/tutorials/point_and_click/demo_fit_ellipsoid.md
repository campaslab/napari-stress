(point_and_click.fit_ellipsoid)=
# Ellipse fitting

This tutorial shows how to handle ellipse-fitting in napari-stress. To get started, create a pointcloud according to the workflow suggestions in this repository or load the sample data from napari-stress (`File > Open Sample > napari-stress: Dropplet pointcloud`).

![](../../imgs/viewer_screenshots/open_sample_droplet.png)
![](../../imgs/viewer_screenshots/open_sample_droplet1.png)

(point_and_click.fit_ellipsoid.least_squares)=
## Napari-stress implementation
This section describes the implementation taken from the [stress repository](https://github.com/campaslab/STRESS). It provides a least-squares approach to obtain an ellipsoid which is represented by its major axes. A pointcloud can be obtained by combining the input points and the ellipsoid object. First, fit an ellipse by selecting `Tools > Points > Fit ellipsoid to pointcloud (n-STRESS)` and click on `Run`. The resulting image should look like this:

![](./imgs/demo_fit_ellipsoid5.png)

If you want to retrieve a pointcloud on the surface of the fitted ellipsoid, you can calculate the corresponding point locations of the input pointcloud on the surface of the fitted ellipsoid with `Tools > Points > Points > Expand point locations on ellipsoid (n-STRESS)`:

![](./imgs/demo_fit_ellipsoid6.png)


## Vedo implementation
This section describes the implementation taken from the [vedo repository](https://vedo.embl.es/). It uses a [pca-algorithm](https://en.wikipedia.org/wiki/Principal_component_analysis) to calculate the minor/major axes. Use it via `Tools > Points > Points > Fit ellipsoid to pointcloud (vedo, n-STRESS)`. As an additional parameter, it requires you to set which percentage of the input points should be contained by the surface of the fitted ellipse, which is controlled by the `inside fraction` parameter. 

![](./imgs/demo_fit_ellipsoid1.png)

Similar to the napari-stress implementation, this returns a vectors layer:

![](./imgs/demo_fit_ellipsoid4.png)

Again, you can use the expansion widget (`Tools > Points > Points > Expand point locations on ellipsoid (n-STRESS)`) to create a pointcloud from this:

![](./imgs/demo_fit_ellipsoid2.png)

## Mean curvature

Lastly, you can measure mean curvature on the surface of this ellipse. To do so, use the `Tools > Measurement > Mean curvature on ellipsoid (n-STRESS)` plugin from the tools menu. As input, you should select the fitted ellipsoid as well as a set of points on the surface of the ellipsoid. You can use the previously obtained points (`Expand point locations on ellipsoid (n-STRESS)`) as input. The result will be the following:

![](./imgs/demo_fit_ellipsoid7.png)

You can also use the built-in [feature visualization widget](point_and_click:visualize_features) to show, for instance, a hiostogram of the curvature on the surface.
