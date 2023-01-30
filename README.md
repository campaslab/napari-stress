[![License](https://img.shields.io/pypi/l/napari-stress.svg?color=green)](https://github.com/campaslab/napari-stress/raw/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-stress.svg?color=green)](https://pypi.org/project/napari-stress)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-stress.svg?color=green)](https://python.org)
[![tests](https://github.com/campaslab/napari-stress/actions/workflows/test_and_deploy.yml/badge.svg)](https://github.com/campaslab/napari-stress/actions/workflows/test_and_deploy.yml)
[![codecov](https://codecov.io/gh/campaslab/napari-stress/branch/main/graph/badge.svg?token=ZXQGREJAT9)](https://codecov.io/gh/campaslab/napari-stress)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/napari-stress.svg)](https://pypistats.org/packages/napari-stress)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-stress)](https://www.napari-hub.org/plugins/napari-stress)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6607329.svg)](https://doi.org/10.5281/zenodo.6607329)

# napari-stress

This plugin provides tools for the analysis of surfaces in Napari, such as utilities to determine and refine the surface-representation of objects using a ray-casting approach and calculate the curvature of surfaces. 
It re-implements code in Napari that was written for [Gross et al. (2021): STRESS, an automated geometrical characterization of deformable particles for in vivo measurements of cell and tissue mechanical stresses](https://www.biorxiv.org/content/10.1101/2021.03.26.437148v1) 
and has been made open source in [this repository](https://github.com/campaslab/STRESS).

![](https://github.com/campaslab/napari-stress/raw/main/docs/imgs/function_gifs/spherical_harmonics.gif)

## Usage

For documentation on how to use napari-stress both interactively from the napari-viewer or from code, please visit the [**documentation**](https://campaslab.github.io/napari-stress/intro.html)


## Installation

Create a new conda environment with the following command. 
If you have never used conda before, please [read this guide first](https://campaslab.github.io/blog/johannes_mueller/anaconda_getting_started/).

```
conda create -n napari-stress Python=3.9 napari jupyterlab -c conda-forge
conda activate napari-stress
```

You can then install napari-stress using pip:

```
pip install napari-stress
```

## Issues

To report bugs, request new features or get in touch, please [open an issue](https://github.com/campaslab/napari-stress/issues) or tag `@EL_Pollo_Diablo` on [image.sc](https://forum.image.sc/).

## See also

There are other napari plugins with similar / overlapping functionality

* [morphometrics](https://www.napari-hub.org/plugins/morphometrics)
* [napari-pymeshlab](https://www.napari-hub.org/plugins/napari-pymeshlab)
* [napari-process-points-and-surfaces](https://www.napari-hub.org/plugins/napari-process-points-and-surfaces)

## Contributing

Contributions are very welcome. Tests can be run with [pytest], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-stress" is free and open source software

## Acknowledgements
This project was supported by the Deutsche Forschungsgemeinschaft under Germany’s Excellence Strategy – EXC2068 - Cluster of Excellence "Physics of Life" of TU Dresden.

[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[pytest]: https://docs.pytest.org/en/7.0.x/
