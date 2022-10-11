# Droplet reconstruction toolbox

In order to save you the trouble of walking through each of the steps manually for the reconstruction of oil droplets, napari-stress provides you the droplet reconstruction toolbox (`Plugins > napari-stress > Droplet reconstruction toolbox`). Let's go through the princiiipal steps of the reconstruction:

## Principal steps
The principal steps of the reconstruction procedure and the relevant settings are:

**Preprocessing**
* Rescaling: Making data isotropic. The principal choice here is, to which voxel size the data should be rescaled to.

![](./imgs/demo_reconstruction_toolbox1.png) 

The fields `Voxel size [x,y,z]` denote the *current* voxel sizes of the data and the `Target voxel size` refers to the voxel size you'd like the data to have. The lower you set the latter, the slower the process will be.

* Binarization: Getting a first estimate of the droplet volume as a [label image](https://napari.org/stable/howtos/layers/labels.html)
* Conversion to surface: The [marching cubes](https://en.wikipedia.org/wiki/Marching_cubes) algorithm is used to extract the surface from the label image
* Surface smoothing: To avoid voxel-artifacts, the surface is smoothed with the [Laplacian smoothing algorithm](https://en.wikipedia.org/wiki/Laplacian_smoothing). The behavior of this step is controlled with the `Smoothing iterations` parameter. Note that this is only a first guess for the point locations.

![](./imgs/demo_reconstruction_toolbox2.png) 

* Extracting points: The [Poisson disk algorithm](https://en.wikipedia.org/wiki/Supersampling#Poisson_disk) is used to extract a set of points on the surface as a first guess at the reconstructed droplets from the smoothed surface. The number of points drawn (and the respective density of points on the surface) can  be controlled via the `Initial number of points` and `Point density on surface` parameters. The initially sampled points are shown after the toolbox has finished, so if you find the points too few, you can increase this number.

**Refinement**

The refinement of the first selection of points consists of two principal steps and is performed several times - the number of iterations can be controlled with the `number of iterations` parameter:

![](./imgs/demo_reconstruction_toolbox3.png) 

In each iteration, the following steps are done:
* Resampling: The cartesian coordinates of the points are interpolated according to latitude and longitude. The resampled points are drawn according to a fibonacci-scheme. Again, you can control the number of points that are sampled with the `Point density` field.
* Trace-refinement: Normals are cast outwards from the sample point and the intesity in the rescaled image along the vectors is mmeasured. The surface point is then moved along this line to best fit the intensity distribution. For more details, see [this notebook](../demo/demo_surface_tracing.ipynb)