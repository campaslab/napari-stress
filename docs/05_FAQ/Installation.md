(FAQ:installation)=
# Installation issues
In this section you find some known installation issues and how to fix them.


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
