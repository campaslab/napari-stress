# napari-stress

This plugin allows to segment fluorescence-labelled droplets , determine the surface with a ray-casting approach and calculate the surface curvatures. It re-implements code in Napari that was written for [Gross et al. (2021): STRESS, an automated geometrical characterization of deformable particles for in vivo measurements of cell and tissue mechanical stresses](https://www.biorxiv.org/content/10.1101/2021.03.26.437148v1) and has been made public in [this repository](https://www.biorxiv.org/content/10.1101/2021.03.26.437148v1).

## Installation

Create a new conda environment with

```
conda create -n napari-stress Python=3.9
conda activate napari-stress
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

### General timelapse-processing

Data to be used for this plugin is typically of the form `[TZYX]` (e.g., 3D + time). Napari-stress offers some convenient way to extent other function's functionality (which are often made for 3D data) to timelapse data using the `frame_by_frame` function and the `TimelapseConverter` class, both of which are described in more detail in [this notebook]([url](https://github.com/BiAPoL/napari-stress/blob/add-timelapse-decorator-for-points-and-surfaces/docs/notebooks/demo/TimeLapse_processing.ipynb)).

### Recipes

Napari-stress provides jupyter notebooks with complete workflows for different types of input data. Napari-stress currently provides notebooks for the following data/image types:

* Confocal data (*.tif*), 3D+t: This type of data can be processed with napari-stressed as show in [this notebook](https://github.com/BiAPoL/napari-stress/blob/main/docs/notebooks/Process_confocal.ipynb)
* Lightsheet data (*.czi*), 3D + t: coming soon....



