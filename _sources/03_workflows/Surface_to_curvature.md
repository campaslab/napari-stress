# From surface data to curvatures

This tutorial will demonstrate how to convert an existing intensity image with a corresponding surface layer into a points layer with indicated curvatures in Napari.

| Image + surface data | Resulting pointcloud with curvatures |
| --- | --- |
|<img src="./_surface_to_curvature_imgs/surface_to_curvature1.png" width="100%"> |<img src="./_surface_to_curvature_imgs/surface_to_curvature5.gif" width="100%">|

This tutorial consists of three elemental steps:

- Retrieving points from the surface. There are multiple methods for this, some of which are described in this tutorial.
- Spherical harmonics expansion: This will approximate the obtained point cloud with a spherical harmonics expansion.
- Measureing curvature: The obtained spherical harmonics solution can then be used to measure the mean curvatures (and hence, tissue stresses) on the surface of the fitted expansion.

## Getting points from the surface

One of the most suitable functions for this (especially for odd-shaped objects) is the Poission-disk algorithm. In Napari, it is available from `Tools > Points > Create points from surface using poisson disk sampling (open3d, nppas)`. It has the particular advantage of putting more points in regions of complicated shape.

<img src="./_surface_to_curvature_imgs/surface_to_curvature2.png" width="50%">

You can also use this menu to increase the point density on the surface. The result should look like this:

<img src="./_surface_to_curvature_imgs/surface_to_curvature3.png" width="100%">

## Spherical harmonics expansion

The basic idea of a [spherical harmonics](https://en.wikipedia.org/wiki/Spherical_harmonics) expansion is to approximate a pointcloud by a superposition of spherical harmonics functions of different degree and order. Thus, complex shapes can be considered to be a superposition of several basic shapes (e.g., a sphere) and more complex shapes. The resulting representation can then be represented by a small number of coefficients.

Napari stress provides functionality for such an approximation in the tools-menu (`Tools > Points > Fit spherical harmonics (n-STRESS)`). The `max_degree` parameter controls the quality of the approximation: High values will lead to a better approximation of the input pointcloud, but eventually lead to [overfitting](https://en.wikipedia.org/wiki/Overfitting) and high computation expense. Conversely, low values only captures the general shape of the pointcloud.

Napari-stress provides two different implementations of the approximation, `stress` and `shtools`. The former typically converges better.

<img src="./_surface_to_curvature_imgs/surface_to_curvature4.png" width="50%">

It can make sense to play a bit with the `max_degree` parameter to get an understanding of the resulting quality. The output should look something like this (you may have to change the color of the pointcloud for better vision):

<img src="../../imgs/function_gifs/spherical_harmonics.gif" width="100%">

## Measureing curvature

Lastly, you can use the spherical harmonics expansion to calculate curvatures (which directly translate to [anistropic stress](https://www.biorxiv.org/content/10.1101/2021.03.26.437148v1.abstract) on the surface). To apply this measurement, select the measurement function from the tools menu (`Tools > Measurement > Surface curvature from points (n-STRESS)`).

<img src="./_surface_to_curvature_imgs/surface_to_curvature5.png" width="50%">

This measurement function provides the same option as the spherical harmonics expansion described above (`max_degree`, `implementation`) as well as an option to control the `number of quadrature points`. This controls how many points will be used to show curvature on the resulting pointcloud. For simple shapes (small `max_degree`), a small set of points sufices to calculate correct global mean curvatures. To select the minimal amount of points required for correct calculation, select the `use minimal point set` option.

To determine accurate curvatures, a minimal amount of quadrature points is required. This number depends on the chosen `max_degree`. You can, however, chose a higher number for nicer visualization.

*Note*: Chosing a higher number of points (>3000) will lead to significant computational expense. It is theoretically impossible to use more than 5180 surface points

*Background*: In order to calculate curvatures on the surface of a set of spherical harmonics functions $Y_{m, n}^l (\theta , \phi)$, where $l$ corresponds to `max_degree`, it is required to calculate to calculate the derivates of $Y_{m, n}^l$ with respect to the latitude and longitude of points on the surface. This problem is closely related to calculating surface integrals - a mathematical procedure which is also referred to as [quadrature](https://en.wikipedia.org/wiki/Quadrature_(mathematics)). There are several methods for this procedure, one of which is the so-called [Lebedev-quadrature](https://en.wikipedia.org/wiki/Lebedev_quadrature). The Lebedev method identifies specific points on the spherical harmonic surface, the so-called lebedev points. Surprisingly, it is sufficient to calculate properties of the surface (e.g., curvature, etc) at these points to determine global properties with [high accuracy](https://link.springer.com/article/10.1007/s10915-017-0617-2#Bib1). The lebedev points have the further property of being symetric under octahedral rotation.

The output will look like this:

<img src="./_surface_to_curvature_imgs/surface_to_curvature6.png" width="100%">
