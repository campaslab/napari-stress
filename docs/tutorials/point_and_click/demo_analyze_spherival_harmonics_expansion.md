# Analyzing a spherical harmonics expansion

After approximating a pointcloud with a spherical harmonics expansion, we may want to extract quantitative features (e.g., curvature, surface area, etc.). The stress-code provides a multitude of quantitative features for this purpose, which shall be demonstrated in this document. To get started, create a pointcloud according to the workflow suggestions in this repository or load the sample data from napari-stress (`File > Open Sample > napari-stress: Dropplet pointcloud`).

<img src="../../imgs/viewer_screenshots/open_sample_droplet.png" width="45%"> <img src="../../imgs/viewer_screenshots/open_sample_droplet1.png" width="45%">

## Analyze the approximated surface

napari-stress provides a number of quantitative parameters for a spherical harmonics surface expansion. To access these measurements, select the respective plugin from the tools menu (`Tools > Measurement > Surface curvature from points (n-STRESS)`). This plugin performs a spherical harmonics expansion with the parameters described in the [respective tutorial](./demo_spherical_harmonics.md). Additionally, the plugin calculates so-called `quadrature points`, which allow for fast and accurate measurement of surface characteristics. For more information on these points (valid values, etc), check out the [notebook](../demo/demo_analyze_spherical_harmonics.ipynb). 

Make sure to check the `run analysis toolbox` checkbox to proceed to the analysis toolbox:

<img src="./imgs/demo_analyze_spherical_harmonics1.png" width="45%">

The result is a toolbox plugin showing the histogram of the calculated curvature and the averaged mean curvature:

<img src="./imgs/demo_analyze_spherical_harmonics2.png" width="100%">