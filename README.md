# napari-stress

This plugin allows to segment fluorescence-labelled droplets , determine the surface with a ray-casting approach and calculate the surface curvatures. It re-implements code in Napari that was written for [Gross et al. (2021): STRESS, an automated geometrical characterization of deformable particles for in vivo measurements of cell and tissue mechanical stresses](https://www.biorxiv.org/content/10.1101/2021.03.26.437148v1) and has been made public in [this repository](https://www.biorxiv.org/content/10.1101/2021.03.26.437148v1).

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
Napari-stress is intended to work for 3D+t datasets in `[TZYX]` format. This is an example of input data from a confocal microscope (taken from [here](https://github.com/campaslab/STRESS/blob/main/ExampleTifSequence-InteriorLabel-vsx_2.076um-vsz_3.998um-TimeInterval_3.00min-21timesteps.tif):

<img src="./docs/imgs/confocal/1_raw_confocal.png" width=45% height=45%>

Napari-stress currently provides notebooks for the following data/image types:

* Confical data (*.tif*), 3D+t: This type of data can be processed with napari-stressed as show in [this notebook](./docs/notebooks/Process_confocal.ipynb)
* Lightsheet data (*.czi*), 3D + t: coming soon....

The resulting surface will look like this:

||Low curvature radius (r=5)| Medium curvature radius (r=10) | Higher curvature radius (r=20) |
| --- | --- | --- | --- |
|Curvature | <img src="./docs/imgs/confocal/2_result_curvature_5radius0.png" width=100% height=100%> | <img src="./docs/imgs/confocal/2_result_curvature_10radius0.png" width=100% height=100%> | <img src="./docs/imgs/confocal/2_result_curvature_20radius0.png" width=100% height=100%> |
|Fit residue|<img src="./docs/imgs/confocal/2_result_fit_residues_5radius0.png" width=100% height=100%>|<img src="./docs/imgs/confocal/2_result_fit_residues_10radius0.png" width=100% height=100%>|<img src="./docs/imgs/confocal/2_result_fit_residues_20radius0.png" width=100% height=100%>|

Depending on the set curvature radius, the calculation captures the global or the local curvature.


