# STRESS

This plugin allows to segment fluorescence-labelled droplets , determine the surface with a ray-casting approach and calculate the surface curvatures.

## Installation

Create a new conda environment with

```
conda create -n napari-stress Python=3.9
conda activate napari stress
```

Install a few necessary plugins:

```
conda install -c conda-forge napari jupyterlab
```

To install the plugin, clone the repository and install it:

```
git clone https://github.com/BiAPoL/napari-stress.git
cd napari-stress
pip install -e .
```

## Usage
Napari-stress is intended to work for 3D and 3D+t datasets in `[TZYX]` or `[ZYX]` format. This is an example of input data from a confocal microscope (taken from [here](https://github.com/campaslab/STRESS/blob/main/ExampleTifSequence-InteriorLabel-vsx_2.076um-vsz_3.998um-TimeInterval_3.00min-21timesteps.tif):

![](./docs/imgs/1_input_raw.png)

ts_2.png" width=45% height=45%> <img src="./docs/imgs/3_int_results_1.png" width=45% height=45%>

The resulting surface will look like this:

|Low curvature radius (r=2.5)| Medium curvature radius (r =5) | Higher curvature radius (r=10) |
|---|---|---|
|<img src="./docs/imgs/4_result_3.png" width=100% height=100%>|<img src="./docs/imgs/4_result_1.png" width=100% height=100%>|<img src="./docs/imgs/4_result_2.png" width=100% height=100%>|



