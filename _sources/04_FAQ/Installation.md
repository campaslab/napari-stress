(FAQ:installation)=
# Installation issues
In this section you find some known installation issues and how to fix them.

## Setting up Python

Depending on how you have your Python set up on your machine, different preparation may be advisable. Here are some suggestions:
- [Getting started with Python and Anaconda](https://biapol.github.io/blog/mara_lampert/getting_started_with_mambaforge_and_python/readme.html): If you have not yet installed Python or Anaconda on your computer **at all**, this explains how to set it up and create an environment that contain the most basic functionality ([napari](https://napari.org/stable/) & [Jupyterlab](https://jupyter.org/))
- Setting up a new environment: If you already have Python installed and want to create a new environment for napari-stress, follow these steps to create a new environment and install napari-stress into it.

```bash
conda create -n stress Python=3.9
conda acticate stress
conda install mamba -c conda-forge

mamba install napari pyqt devbio-napari
pip install napari-stress
```

If you want to make sure you are using the latest version, replace `pip install napari-stress` with `pip install napari-stress==version.number` (e.g., `pip install napari-stress==0.1.0`). You find the current version number on the [pypi page](https://pypi.org/project/napari-stress/) and on top of the [documentation frontpage](https://campaslab.github.io/napari-stress/intro.html).

### Related packages

Some packages work well in conjunction with napari-stress, e.g., for better 3D interactivity, visualization of the results and/or data export. Here are some suggestions:
- [Devbio-napari](https://github.com/haesleinhuepf/devbio-napari): Not strictly necessary but strongly recommended - this package brings many handy functionalities to an otherwise quite plain napari-viewer.
- [Napari-threedee](https://www.napari-hub.org/plugins/napari-threedee): Enhance rendering options in napari with a set of interesting tools for mesh lighting, plane rendering, etc.install type `pip install napari-threedee`.
- [napari-process-points-and-surfaces](https://www.napari-hub.org/plugins/napari-process-points-and-surfaces): A plugin that allows you to process points and surfaces in 3D. To install, type `pip install napari-process-points-and-surfaces`.


## Installation failing on Mac
(FAQ:installation:xcode)=

When pip-installing napari-stress on M1 or M2 Macs, you may encounter the following error message ([link to report](https://forum.image.sc/t/napari-stress-problem-loading-some-dependancies/73758)):

```bash
ImportError: dlopen(/Users/manon/opt/anaconda3/envs/napari-stress/lib/python3.9/site-packages/open3d/cpu/pybind.cpython-39-darwin.so, 0x0002): Library not loaded: '/usr/local/opt/libomp/lib/libomp.dylib'
Referenced from: '/Users/xxxx/opt/anaconda3/envs/napari-stress/lib/python3.9/site-packages/open3d/cpu/pybind.cpython-39-darwin.so'
Reason: tried: '/usr/local/opt/libomp/lib/libomp.dylib' (no such file), '/usr/local/lib/libomp.dylib' (no such file), '/usr/lib/libomp.dylib' (no such file)
```

In order to fix this issue, please update the XCode tools on your Mac as described [here](https://stackoverflow.com/questions/52522565/git-is-not-working-after-macos-update-xcrun-error-invalid-active-developer-pa).

## Pygeodesic installation fail

On some Macs, the `pygeodesic` package fails to install with this error message:

```bash
building ‘pygeodesic.geodesic’ extension
creating build/temp.macosx-10.9-x86_64-cpython-39
creating build/temp.macosx-10.9-x86_64-cpython-39/pygeodesic
clang -Wno-unused-result -Wsign-compare -Wunreachable-code -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /Users/xxxx/opt/anaconda3/envs/mamba/envs/napari-stress-open3d/include -fPIC -O2 -isystem /Users/rachna.narayanan/opt/anaconda3/envs/mamba/envs/napari-stress-open3d/include -I/Users/xxxx/opt/anaconda3/envs/mamba/envs/napari-stress-open3d/include/python3.9 -I/Users/xxxx/opt/anaconda3/envs/mamba/envs/napari-stress-open3d/lib/python3.9/site-packages/numpy/core/include -Ipygeodesic\geodesic_kirsanov -c pygeodesic/geodesic.cpp -o build/temp.macosx-10.9-x86_64-cpython-39/pygeodesic/geodesic.o
xcrun: error: invalid active developer path (/Library/Developer/CommandLineTools), missing xcrun at: /Library/Developer/CommandLineTools/usr/bin/xcrun
error: command ‘/usr/bin/clang’ failed with exit code 1
[end of output]

note: This error originates from a subprocess, and is likely not a problem with pip.
error: legacy-install-failure

× Encountered error while trying to install package.
╰─> pygeodesic

note: This is an issue with the package mentioned above, not pip.
hint: See above for output from the failure.
```

Similar to [this issue](FAQ:installation:xcode), this problem can be fixed by updating the XCode command line tools on your Mac as described above.

## OSError (Windows) during installation

This describes how to fix the following error:

```
ERROR: Could not install packages due to an OSError: [WinError 5] Access is denied: 'C:\\Users\\xxx\\AppData\\Local\\Temp\\pip-uninstall-f7qla__t\\jupyter-trust.exe'
Consider using the `--user` option or check the permissions.
```

If this occurs, close all open Python instances (e.g., anaconda command line prompts, Jupyterlab instances, etc) and try again - this should fix the issue.
